import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from io import BytesIO
import time
from numba import njit, prange
import zipfile
import base64
import os
import itertools
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, List, Optional

# Set default Streamlit layout
st.set_page_config(layout="wide", page_title="Intergranular Penetration Simulator")

# =====================
# PHYSICAL PARAMETERS
# =====================
# Default parameters - will be adjustable via UI
DEFAULT_PARAMS = {
    'grid_size': 128,           # Grid dimension (will be square)
    'n_grains': 8,              # Number of inert grains
    'interface_width': 3.0,     # Diffuse interface width
    'kappa': 0.5,               # Gradient energy coefficient
    'mobility': 1.0,            # Mobility for substrate phase
    'gb_mobility': 10.0,        # Grain boundary mobility enhancement
    'gb_energy_bias': 0.8,      # Energy bias for GB growth
    'supersaturation': 0.6,     # Initial supersaturation
    'time_step': 0.01,          # Time step for integration
    'total_steps': 1000,        # Total simulation steps
    'grain_arrangement': 'voronoi',  # Options: 'voronoi', 'random_seeds', 'regular_grid'
    'gb_indicator_type': 'standard', # Options: 'standard', 'gradient_based'
}

# =====================
# MOELANS INTERPOLATION FUNCTIONS
# =====================
@njit
def moelans_standard_interp(eta: np.ndarray) -> np.ndarray:
    """
    Standard Moelans interpolation function for single phase:
    h(η) = η²(3 - 2η)
    
    Args:
        eta: Order parameter field
        
    Returns:
        Interpolated field
    """
    return eta * eta * (3.0 - 2.0 * eta)

@njit
def moelans_generalized_interp(eta_k: np.ndarray, etas: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Generalized Moelans interpolation function for multiple phases:
    h_k = η_k² / (η_k² + α∑_{j≠k} η_j²)
    
    Args:
        eta_k: Order parameter for phase k
        etas: Array of all order parameters
        alpha: Weighting parameter (default 1.0)
        
    Returns:
        Interpolated field for phase k
    """
    numerator = eta_k * eta_k
    denominator = numerator.copy()
    
    for i in range(etas.shape[0]):
        if i >= etas.shape[0]:
            continue
        denominator += alpha * etas[i] * etas[i]
    
    # Avoid division by zero
    result = np.zeros_like(numerator)
    for i in range(numerator.shape[0]):
        for j in range(numerator.shape[1]):
            if denominator[i, j] > 1e-10:
                result[i, j] = numerator[i, j] / denominator[i, j]
            else:
                result[i, j] = 0.0 if numerator[i, j] < 0.5 else 1.0
                
    return result

@njit
def compute_gb_indicator(etas: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Compute grain boundary indicator function
    
    Args:
        etas: Array of grain order parameters (n_grains, nx, ny)
        method: 'standard' for η_i²η_j² sum, 'gradient' for gradient-based
        
    Returns:
        GB indicator field (0 = grain interior, 1 = grain boundary)
    """
    n_grains, nx, ny = etas.shape
    gb_indicator = np.zeros((nx, ny), dtype=np.float64)
    
    if method == 'standard':
        # Standard method: sum of η_i²η_j² for all i<j
        for i in range(n_grains):
            eta_i_sq = etas[i] * etas[i]
            for j in range(i + 1, n_grains):
                eta_j_sq = etas[j] * etas[j]
                gb_indicator += eta_i_sq * eta_j_sq
        
        # Normalize to [0, 1]
        max_val = 0.75 * (n_grains / 2.0)  # Theoretical maximum
        for i in range(nx):
            for j in range(ny):
                if gb_indicator[i, j] > max_val:
                    gb_indicator[i, j] = max_val
                gb_indicator[i, j] = gb_indicator[i, j] / max_val if max_val > 0 else 0.0
    
    elif method == 'gradient':
        # Gradient-based method: sum of |∇η_i|²
        for i in range(n_grains):
            grad_sq = np.zeros((nx, ny))
            for x in range(1, nx-1):
                for y in range(1, ny-1):
                    dx = (etas[i, x+1, y] - etas[i, x-1, y]) / 2.0
                    dy = (etas[i, x, y+1] - etas[i, x, y-1]) / 2.0
                    grad_sq[x, y] = dx*dx + dy*dy
            gb_indicator += grad_sq
        
        # Normalize
        max_val = np.max(gb_indicator)
        if max_val > 1e-10:
            gb_indicator = gb_indicator / max_val
    
    return gb_indicator

# =====================
# NUMERICAL OPERATORS
# =====================
@njit(parallel=True)
def laplacian_2d(field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Compute 2D Laplacian with periodic boundaries in y, fixed in x
    
    Args:
        field: Input field (nx, ny)
        dx: Grid spacing
        
    Returns:
        Laplacian of field
    """
    nx, ny = field.shape
    lap = np.zeros_like(field)
    
    for i in prange(nx):
        for j in range(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1 + ny) % ny
            
            if i == 0:
                # Left boundary: fixed value (Dirichlet)
                lap[i, j] = (field[i+1, j] - field[i, j] + 
                            field[i, jp1] + field[i, jm1] - 2 * field[i, j]) / dx**2
            elif i == nx - 1:
                # Right boundary: fixed value (Dirichlet)
                lap[i, j] = (field[i-1, j] - field[i, j] + 
                            field[i, jp1] + field[i, jm1] - 2 * field[i, j]) / dx**2
            else:
                # Interior points
                lap[i, j] = (field[i+1, j] + field[i-1, j] + 
                            field[i, jp1] + field[i, jm1] - 4 * field[i, j]) / dx**2
    
    return lap

@njit(parallel=True)
def update_substrate(etas: np.ndarray, eta_sp: np.ndarray, c: np.ndarray, 
                    params: Dict, step: int) -> np.ndarray:
    """
    Update substrate phase with preferential GB growth
    
    Args:
        etas: Inert grain order parameters (n_grains, nx, ny)
        eta_sp: Substrate phase order parameter (nx, ny)
        c: Concentration field (nx, ny)
        params: Simulation parameters dictionary
        step: Current simulation step
        
    Returns:
        Updated substrate phase field
    """
    nx, ny = eta_sp.shape
    new_eta_sp = np.copy(eta_sp)
    
    # Compute GB indicator
    gb_indicator = compute_gb_indicator(etas, params['gb_indicator_type'])
    
    # Compute time-dependent GB bias strength (ramp up over first 10% of simulation)
    max_steps = params['total_steps']
    bias_strength = params['gb_energy_bias'] * min(1.0, step / max(1, 0.1 * max_steps))
    
    for i in prange(nx):
        for j in range(ny):
            # Apply boundary conditions first
            if i == 0:  # Left boundary: fixed to substrate
                new_eta_sp[i, j] = 1.0
                continue
            elif i == nx - 1:  # Right boundary: fixed to matrix
                new_eta_sp[i, j] = 0.0
                continue
            
            # Chemical driving force (simplified)
            chem_force = params['supersaturation'] - c[i, j]
            
            # GB preference term
            gb_force = bias_strength * gb_indicator[i, j]
            
            # Total driving force
            driving_force = chem_force + gb_force
            
            # Laplacian term
            jp1 = (j + 1) % ny
            jm1 = (j - 1 + ny) % ny
            lap_eta_sp = (eta_sp[i+1, j] + eta_sp[i-1, j] + 
                         eta_sp[i, jp1] + eta_sp[i, jm1] - 4 * eta_sp[i, j]) / params['dx']**2
            
            # Update equation: ∂η_sp/∂t = M(Δη_sp - dF/dη_sp)
            # Simplified free energy derivative
            dF_deta = eta_sp[i, j] * (eta_sp[i, j] - 1.0) * (eta_sp[i, j] - 0.5)  # Double well
            
            # Mobility field (GB enhanced)
            mobility_local = params['mobility'] * (1.0 + params['gb_mobility'] * gb_indicator[i, j])
            
            # Euler integration
            new_val = eta_sp[i, j] + params['time_step'] * mobility_local * (
                params['kappa'] * lap_eta_sp - dF_deta + driving_force
            )
            
            # Stabilize
            if new_val < 0.0:
                new_val = 0.0
            elif new_val > 1.0:
                new_val = 1.0
            
            new_eta_sp[i, j] = new_val
    
    return new_eta_sp

@njit(parallel=True)
def update_concentration(etas: np.ndarray, eta_sp: np.ndarray, c: np.ndarray, 
                        params: Dict) -> np.ndarray:
    """
    Update concentration field with GB-enhanced diffusion
    
    Args:
        etas: Inert grain order parameters
        eta_sp: Substrate phase order parameter
        c: Current concentration field
        params: Simulation parameters
        
    Returns:
        Updated concentration field
    """
    nx, ny = c.shape
    new_c = np.copy(c)
    
    # Compute GB indicator for mobility enhancement
    gb_indicator = compute_gb_indicator(etas, params['gb_indicator_type'])
    
    for i in prange(nx):
        for j in range(ny):
            # Skip boundaries
            if i == 0:
                new_c[i, j] = 1.0  # Fixed concentration at substrate boundary
                continue
            elif i == nx - 1:
                new_c[i, j] = params['supersaturation']  # Fixed at far field
                continue
            
            jp1 = (j + 1) % ny
            jm1 = (j - 1 + ny) % ny
            
            # Mobility at this point (GB enhanced)
            mobility_local = params['mobility'] * (1.0 + params['gb_mobility'] * gb_indicator[i, j])
            
            # Concentration gradient
            dc_dx_pos = (c[i+1, j] - c[i, j]) / params['dx']
            dc_dx_neg = (c[i, j] - c[i-1, j]) / params['dx']
            dc_dy_pos = (c[i, jp1] - c[i, j]) / params['dx']
            dc_dy_neg = (c[i, j] - c[i, jm1]) / params['dx']
            
            # Flux divergence
            div_flux = (mobility_local * (dc_dx_pos - dc_dx_neg) + 
                       mobility_local * (dc_dy_pos - dc_dy_neg)) / params['dx']
            
            # Update concentration
            new_val = c[i, j] - params['time_step'] * div_flux
            
            # Stabilize
            if new_val < 0.0:
                new_val = 0.0
            elif new_val > 1.5:  # Allow some supersaturation
                new_val = 1.5
            
            new_c[i, j] = new_val
    
    return new_c

# =====================
# INITIALIZATION FUNCTIONS
# =====================
def create_grain_structure(nx: int, ny: int, n_grains: int, 
                          method: str = 'voronoi') -> np.ndarray:
    """
    Create grain structure with specified arrangement
    
    Args:
        nx, ny: Grid dimensions
        n_grains: Number of grains
        method: 'voronoi', 'random_seeds', or 'regular_grid'
        
    Returns:
        Order parameter fields for grains (n_grains, nx, ny)
    """
    etas = np.zeros((n_grains, nx, ny), dtype=np.float64)
    
    if method == 'voronoi':
        # Create Voronoi tessellation
        seeds = np.random.rand(n_grains, 2) * [nx, ny]
        for i in range(nx):
            for j in range(ny):
                distances = np.sqrt((seeds[:, 0] - i)**2 + (seeds[:, 1] - j)**2)
                closest_grain = np.argmin(distances)
                etas[closest_grain, i, j] = 1.0
    
    elif method == 'random_seeds':
        # Place random seed points for each grain
        for k in range(n_grains):
            n_seeds = max(1, int(0.01 * nx * ny / n_grains))
            seed_x = np.random.randint(0, nx, n_seeds)
            seed_y = np.random.randint(0, ny, n_seeds)
            for sx, sy in zip(seed_x, seed_y):
                etas[k, sx, sy] = 1.0
    
    elif method == 'regular_grid':
        # Regular grid arrangement
        grid_size = int(np.ceil(np.sqrt(n_grains)))
        grain_width = nx // grid_size
        grain_height = ny // grid_size
        
        grain_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if grain_idx < n_grains:
                    x_start = i * grain_width
                    x_end = min((i + 1) * grain_width, nx)
                    y_start = j * grain_height
                    y_end = min((j + 1) * grain_height, ny)
                    etas[grain_idx, x_start:x_end, y_start:y_end] = 1.0
                    grain_idx += 1
    
    # Apply diffuse interfaces
    interface_width = DEFAULT_PARAMS['interface_width']
    for k in range(n_grains):
        # Distance transform approximation
        for i in range(nx):
            for j in range(ny):
                if etas[k, i, j] < 0.5:
                    continue
                # Simple diffuse interface
                for di in range(-int(interface_width), int(interface_width)+1):
                    for dj in range(-int(interface_width), int(interface_width)+1):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < nx and 0 <= nj < ny:
                            dist = np.sqrt(di*di + dj*dj)
                            if dist < interface_width:
                                weight = np.exp(-dist**2 / (2 * (interface_width/2)**2))
                                if etas[k, ni, nj] < weight:
                                    etas[k, ni, nj] = weight
    
    # Normalize to ensure sum(etas) = 1 everywhere
    sum_eta = np.sum(etas, axis=0)
    for k in range(n_grains):
        etas[k] = np.divide(etas[k], sum_eta, out=np.zeros_like(etas[k]), where=sum_eta>0)
    
    return etas

def initialize_fields(params: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize all fields based on parameters
    
    Args:
        params: Simulation parameters dictionary
        
    Returns:
        etas: Grain order parameters (n_grains, nx, ny)
        eta_sp: Substrate phase (nx, ny)
        c: Concentration field (nx, ny)
    """
    nx = params['grid_size']
    ny = params['grid_size']
    n_grains = params['n_grains']
    
    # Initialize inert grains
    etas = create_grain_structure(nx, ny, n_grains, params['grain_arrangement'])
    
    # Initialize substrate phase (left side)
    eta_sp = np.zeros((nx, ny), dtype=np.float64)
    substrate_width = max(1, int(0.1 * nx))  # 10% of domain width
    eta_sp[:substrate_width, :] = 1.0
    
    # Initialize concentration field
    c = np.full((nx, ny), params['supersaturation'], dtype=np.float64)
    c[:substrate_width, :] = 1.0  # Substrate concentration
    
    return etas, eta_sp, c

# =====================
# VISUALIZATION FUNCTIONS
# =====================
def create_microstructure_plot(etas: np.ndarray, eta_sp: np.ndarray, 
                              show_grains: bool = True, show_substrate: bool = True) -> plt.Figure:
    """
    Create microstructure visualization
    
    Args:
        etas: Grain order parameters
        eta_sp: Substrate phase
        show_grains: Whether to show grains
        show_substrate: Whether to show substrate
        
    Returns:
        Matplotlib figure
    """
    n_grains, nx, ny = etas.shape
    
    # Create grain coloring
    grain_ids = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            if show_substrate and eta_sp[i, j] > 0.5:
                grain_ids[i, j] = -1  # Substrate marker
            else:
                max_idx = 0
                max_val = etas[0, i, j]
                for k in range(1, n_grains):
                    if etas[k, i, j] > max_val:
                        max_val = etas[k, i, j]
                        max_idx = k
                grain_ids[i, j] = max_idx
    
    # Create colormap
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_grains + 1)))
    grain_colors = [(0.0, 0.0, 0.0, 1.0)]  # Black for substrate
    for i in range(n_grains):
        grain_colors.append(tuple(colors[i % 20]))
    
    cmap = ListedColormap(grain_colors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot microstructure
    im = ax.imshow(grain_ids.T, cmap=cmap, origin='lower', interpolation='nearest')
    
    # Add colorbar with labels
    cbar = plt.colorbar(im, ax=ax, ticks=range(-1, n_grains))
    cbar_labels = ['Substrate'] + [f'Grain {i+1}' for i in range(n_grains)]
    cbar.ax.set_yticklabels(cbar_labels)
    
    ax.set_title('Microstructure with Substrate Phase')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig

def create_field_visualization(etas: np.ndarray, eta_sp: np.ndarray, c: np.ndarray, 
                             energy_history: List[float], step: int) -> go.Figure:
    """
    Create Plotly visualization of all fields
    
    Args:
        etas: Grain order parameters
        eta_sp: Substrate phase
        c: Concentration field
        energy_history: Energy evolution history
        step: Current simulation step
        
    Returns:
        Plotly figure
    """
    n_grains, nx, ny = etas.shape
    
    # Create subplot figure
    fig = sp.make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Grains & Substrate', f'Substrate Phase (Step {step})', 'Concentration Field',
            'GB Indicator', 'Chemical Potential', 'Free Energy'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Grains & Substrate
    grain_ids = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            if eta_sp[i, j] > 0.5:
                grain_ids[i, j] = -1  # Substrate
            else:
                max_idx = 0
                max_val = etas[0, i, j]
                for k in range(1, n_grains):
                    if etas[k, i, j] > max_val:
                        max_val = etas[k, i, j]
                        max_idx = k
                grain_ids[i, j] = max_idx
    
    # Custom colormap for grains
    grain_colors = ['black'] + [f'rgb({int(255*i/n_grains)}, {int(255*(1-i/n_grains))}, {int(128)})' 
                              for i in range(n_grains)]
    grain_cmap = [[i/(len(grain_colors)-1), color] for i, color in enumerate(grain_colors)]
    
    fig.add_trace(
        go.Heatmap(
            z=grain_ids.T,
            colorscale=grain_cmap,
            showscale=False,
            name='Microstructure'
        ),
        row=1, col=1
    )
    
    # 2. Substrate Phase
    fig.add_trace(
        go.Heatmap(
            z=eta_sp.T,
            colorscale='Blues',
            zmin=0,
            zmax=1,
            showscale=True,
            name='Substrate'
        ),
        row=1, col=2
    )
    
    # 3. Concentration Field
    fig.add_trace(
        go.Heatmap(
            z=c.T,
            colorscale='Hot',
            zmin=0,
            zmax=1.5,
            showscale=True,
            name='Concentration'
        ),
        row=1, col=3
    )
    
    # 4. GB Indicator
    gb_indicator = compute_gb_indicator(etas)
    fig.add_trace(
        go.Heatmap(
            z=gb_indicator.T,
            colorscale='Viridis',
            zmin=0,
            zmax=1,
            showscale=True,
            name='GB Indicator'
        ),
        row=2, col=1
    )
    
    # 5. Chemical Potential (simplified)
    chem_pot = c - 0.5  # Simplified for visualization
    fig.add_trace(
        go.Heatmap(
            z=chem_pot.T,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            showscale=True,
            name='Chem Pot'
        ),
        row=2, col=2
    )
    
    # 6. Free Energy History
    if len(energy_history) > 0:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(energy_history))),
                y=energy_history,
                mode='lines+markers',
                name='Free Energy',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=2, col=3
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"Intergranular Penetration Simulation - Step {step}",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="X", row=2, col=3)
    fig.update_yaxes(title_text="Energy", row=2, col=3)
    
    return fig

def create_interpolation_visualization(etas: np.ndarray, eta_sp: np.ndarray) -> plt.Figure:
    """
    Visualize Moelans interpolation functions
    
    Args:
        etas: Grain order parameters
        eta_sp: Substrate phase
        
    Returns:
        Matplotlib figure
    """
    n_grains, nx, ny = etas.shape
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 1. Substrate interpolation function
    h_sp = moelans_standard_interp(eta_sp)
    im0 = axes[0].imshow(h_sp.T, cmap='viridis', origin='lower')
    axes[0].set_title('Substrate Interpolation h(η_sp)')
    plt.colorbar(im0, ax=axes[0])
    
    # 2-4. First three grain interpolation functions
    for i in range(min(3, n_grains)):
        h_grain = np.zeros((nx, ny))
        for x in range(nx):
            for y in range(ny):
                # Extract all etas at this point
                etas_at_point = np.zeros(n_grains)
                for k in range(n_grains):
                    etas_at_point[k] = etas[k, x, y]
                # Compute generalized interpolation
                h_grain[x, y] = moelans_generalized_interp(etas[i, x, y], etas_at_point)
        
        im = axes[i+1].imshow(h_grain.T, cmap='plasma', origin='lower')
        axes[i+1].set_title(f'Grain {i+1} Interpolation h_{i+1}')
        plt.colorbar(im, ax=axes[i+1])
    
    # 5. Sum of all grain interpolations
    sum_h = np.zeros((nx, ny))
    for i in range(n_grains):
        for x in range(nx):
            for y in range(ny):
                etas_at_point = np.zeros(n_grains)
                for k in range(n_grains):
                    etas_at_point[k] = etas[k, x, y]
                sum_h[x, y] += moelans_generalized_interp(etas[i, x, y], etas_at_point)
    
    im4 = axes[4].imshow(sum_h.T, cmap='coolwarm', origin='lower', vmin=0, vmax=1.1)
    axes[4].set_title('Sum of Grain Interpolations')
    plt.colorbar(im4, ax=axes[4])
    
    # 6. Combined view
    combined = np.zeros((nx, ny))
    for x in range(nx):
        for y in range(ny):
            if eta_sp[x, y] > 0.5:
                combined[x, y] = 0  # Substrate
            else:
                max_idx = 0
                max_val = etas[0, x, y]
                for k in range(1, n_grains):
                    if etas[k, x, y] > max_val:
                        max_val = etas[k, x, y]
                        max_idx = k
                combined[x, y] = max_idx + 1
    
    grain_colors = ['black'] + [plt.cm.tab20(i) for i in range(n_grains)]
    grain_cmap = ListedColormap(grain_colors[:n_grains+1])
    im5 = axes[5].imshow(combined.T, cmap=grain_cmap, origin='lower')
    axes[5].set_title('Combined Microstructure')
    plt.colorbar(im5, ax=axes[5])
    
    plt.tight_layout()
    return fig

# =====================
# STREAMLIT APPLICATION
# =====================
def main():
    st.title("Phase Field Simulation: Substrate Growth Through Grain Boundaries")
    
    # Sidebar for parameter control
    st.sidebar.header("Simulation Parameters")
    
    # Grid and geometry parameters
    st.sidebar.subheader("Domain & Geometry")
    grid_size = st.sidebar.select_slider(
        "Grid Size", 
        options=[64, 96, 128, 160, 192, 256],
        value=DEFAULT_PARAMS['grid_size'],
        help="Larger grids provide better resolution but slower simulation"
    )
    
    n_grains = st.sidebar.slider(
        "Number of Inert Grains", 
        min_value=2, max_value=32, value=DEFAULT_PARAMS['n_grains'],
        help="Number of fixed grains through which substrate will grow"
    )
    
    grain_arrangement = st.sidebar.selectbox(
        "Grain Arrangement", 
        ['voronoi', 'random_seeds', 'regular_grid'],
        index=0,
        help="Method to arrange the inert grains"
    )
    
    # Interface parameters
    st.sidebar.subheader("Interface Properties")
    interface_width = st.sidebar.slider(
        "Interface Width", 
        min_value=1.0, max_value=10.0, value=DEFAULT_PARAMS['interface_width'],
        step=0.5,
        help="Width of diffuse interfaces between phases"
    )
    
    kappa = st.sidebar.slider(
        "Gradient Energy (κ)", 
        min_value=0.1, max_value=2.0, value=DEFAULT_PARAMS['kappa'],
        step=0.1,
        help="Controls interface energy and width"
    )
    
    # Kinetics parameters
    st.sidebar.subheader("Kinetics")
    mobility = st.sidebar.slider(
        "Base Mobility", 
        min_value=0.1, max_value=2.0, value=DEFAULT_PARAMS['mobility'],
        step=0.1,
        help="Base mobility for phase evolution"
    )
    
    gb_mobility = st.sidebar.slider(
        "GB Mobility Enhancement", 
        min_value=1.0, max_value=50.0, value=DEFAULT_PARAMS['gb_mobility'],
        step=1.0,
        help="Enhancement of mobility along grain boundaries"
    )
    
    gb_energy_bias = st.sidebar.slider(
        "GB Energy Bias", 
        min_value=0.1, max_value=2.0, value=DEFAULT_PARAMS['gb_energy_bias'],
        step=0.1,
        help="Energy bias favoring growth along grain boundaries"
    )
    
    gb_indicator_type = st.sidebar.selectbox(
        "GB Detection Method", 
        ['standard', 'gradient'],
        index=0,
        help="Method to detect grain boundaries"
    )
    
    # Thermodynamics parameters
    st.sidebar.subheader("Thermodynamics")
    supersaturation = st.sidebar.slider(
        "Supersaturation", 
        min_value=0.1, max_value=1.0, value=DEFAULT_PARAMS['supersaturation'],
        step=0.05,
        help="Initial supersaturation driving substrate growth"
    )
    
    # Simulation control
    st.sidebar.subheader("Simulation Control")
    time_step = st.sidebar.slider(
        "Time Step", 
        min_value=0.001, max_value=0.1, value=DEFAULT_PARAMS['time_step'],
        step=0.001,
        format="%.3f",
        help="Time integration step (smaller = more stable but slower)"
    )
    
    total_steps = st.sidebar.slider(
        "Total Steps", 
        min_value=100, max_value=5000, value=DEFAULT_PARAMS['total_steps'],
        step=100,
        help="Total number of simulation time steps"
    )
    
    # Create parameter dictionary
    params = {
        'grid_size': grid_size,
        'n_grains': n_grains,
        'interface_width': interface_width,
        'kappa': kappa,
        'mobility': mobility,
        'gb_mobility': gb_mobility,
        'gb_energy_bias': gb_energy_bias,
        'supersaturation': supersaturation,
        'time_step': time_step,
        'total_steps': total_steps,
        'grain_arrangement': grain_arrangement,
        'gb_indicator_type': gb_indicator_type,
        'dx': 1.0,  # Fixed grid spacing
    }
    
    # Simulation control buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        run_sim = st.button("Run Simulation", type="primary")
    with col2:
        pause_sim = st.button("Pause")
    with col3:
        reset_sim = st.button("Reset")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Simulation", "Microstructure", "Interpolation Functions", "Results", "Theory"
    ])
    
    # =====================
    # SIMULATION TAB
    # =====================
    with tab1:
        st.header("Real-time Simulation")
        
        if run_sim or 'simulation_running' in st.session_state:
            if reset_sim:
                if 'simulation_running' in st.session_state:
                    del st.session_state.simulation_running
                st.experimental_rerun()
            
            # Initialize or continue simulation
            if 'simulation_running' not in st.session_state:
                st.session_state.simulation_running = True
                st.session_state.current_step = 0
                st.session_state.etas, st.session_state.eta_sp, st.session_state.c = initialize_fields(params)
                st.session_state.energy_history = []
                st.session_state.field_history = {
                    'etas': [], 'eta_sp': [], 'c': [], 'gb_indicator': [], 'step_indices': []
                }
            
            # Simulation controls
            col1, col2 = st.columns([3, 1])
            with col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
                plot_placeholder = st.empty()
            
            with col2:
                st.subheader("Controls")
                vis_step_freq = st.slider("Visualization Frequency", 1, 100, 10)
                save_step_freq = st.slider("Save Frequency", 1, 200, 50)
                show_gb_indicator = st.checkbox("Show GB Indicator", True)
                show_energy = st.checkbox("Show Energy", True)
            
            # Run simulation
            start_time = time.time()
            total_steps = params['total_steps']
            
            for step in range(st.session_state.current_step, total_steps):
                if pause_sim:
                    st.session_state.current_step = step
                    break
                
                # Update fields
                st.session_state.eta_sp = update_substrate(
                    st.session_state.etas, 
                    st.session_state.eta_sp, 
                    st.session_state.c,
                    params,
                    step
                )
                
                st.session_state.c = update_concentration(
                    st.session_state.etas,
                    st.session_state.eta_sp,
                    st.session_state.c,
                    params
                )
                
                # Calculate energy (simplified)
                energy = np.mean(st.session_state.eta_sp**2 * (1 - st.session_state.eta_sp)**2)
                st.session_state.energy_history.append(energy)
                
                # Save fields for visualization and export
                if step % vis_step_freq == 0 or step == total_steps - 1:
                    fig = create_field_visualization(
                        st.session_state.etas,
                        st.session_state.eta_sp,
                        st.session_state.c,
                        st.session_state.energy_history,
                        step
                    )
                    plot_placeholder.plotly_chart(fig, use_container_width=True)
                
                if step % save_step_freq == 0 or step == total_steps - 1:
                    st.session_state.field_history['etas'].append(np.copy(st.session_state.etas))
                    st.session_state.field_history['eta_sp'].append(np.copy(st.session_state.eta_sp))
                    st.session_state.field_history['c'].append(np.copy(st.session_state.c))
                    st.session_state.field_history['gb_indicator'].append(
                        compute_gb_indicator(st.session_state.etas, params['gb_indicator_type'])
                    )
                    st.session_state.field_history['step_indices'].append(step)
                
                # Update progress
                progress = (step + 1) / total_steps
                progress_bar.progress(progress)
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (step + 1) * total_steps
                remaining_time = estimated_total - elapsed_time
                status_text.text(
                    f"Step {step+1}/{total_steps} | "
                    f"Time: {elapsed_time:.1f}s | "
                    f"Est. remaining: {remaining_time:.1f}s | "
                    f"Energy: {energy:.6f}"
                )
                
                st.session_state.current_step = step + 1
            
            if st.session_state.current_step >= total_steps:
                st.success("✅ Simulation completed successfully!")
                st.session_state.simulation_running = False
        
        else:
            st.info("👆 Configure parameters in the sidebar and click 'Run Simulation' to start")
            
            # Show initial microstructure preview
            st.subheader("Initial Microstructure Preview")
            with st.spinner("Generating preview..."):
                etas_init, eta_sp_init, c_init = initialize_fields(params)
                fig_preview = create_microstructure_plot(etas_init, eta_sp_init)
                st.pyplot(fig_preview)
            
            # Show Moelans interpolation preview
            st.subheader("Moelans Interpolation Functions")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Standard Interpolation:**
                $h(η) = η^2(3 - 2η)$
                
                This function has the properties:
                - $h(0) = 0$, $h(1) = 1$
                - $h'(0) = h'(1) = 0$ (zero derivatives at bounds)
                - Monotonic between 0 and 1
                """)
            
            with col2:
                # Plot the standard interpolation function
                eta_vals = np.linspace(0, 1, 100)
                h_vals = eta_vals**2 * (3 - 2*eta_vals)
                
                fig_interp, ax = plt.subplots(figsize=(6, 4))
                ax.plot(eta_vals, h_vals, 'b-', linewidth=2, label='h(η) = η²(3-2η)')
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y = η')
                ax.set_xlabel('η')
                ax.set_ylabel('h(η)')
                ax.set_title('Standard Moelans Interpolation')
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig_interp)
    
    # =====================
    # MICROSTRUCTURE TAB
    # =====================
    with tab2:
        st.header("Microstructure Evolution")
        
        if 'field_history' in st.session_state and st.session_state.field_history['eta_sp']:
            # Create animation slider
            step_idx = st.slider(
                "Simulation Step", 
                0, 
                len(st.session_state.field_history['step_indices']) - 1,
                len(st.session_state.field_history['step_indices']) - 1
            )
            
            step_num = st.session_state.field_history['step_indices'][step_idx]
            eta_sp = st.session_state.field_history['eta_sp'][step_idx]
            c = st.session_state.field_history['c'][step_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Microstructure at Step {step_num}")
                fig_micro = create_microstructure_plot(
                    st.session_state.etas, 
                    eta_sp,
                    show_grains=True,
                    show_substrate=True
                )
                st.pyplot(fig_micro)
            
            with col2:
                st.subheader(f"Substrate Phase at Step {step_num}")
                fig_substrate, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(eta_sp.T, cmap='Blues', origin='lower', vmin=0, vmax=1)
                ax.set_title(f'Substrate Phase Field (Step {step_num})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax, label='η_sp')
                st.pyplot(fig_substrate)
            
            # GB penetration analysis
            st.subheader("Grain Boundary Penetration Analysis")
            
            gb_indicator = st.session_state.field_history['gb_indicator'][step_idx]
            penetration = eta_sp * gb_indicator
            
            fig_analysis, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # GB indicator
            im1 = axes[0].imshow(gb_indicator.T, cmap='viridis', origin='lower')
            axes[0].set_title('Grain Boundary Indicator')
            plt.colorbar(im1, ax=axes[0])
            
            # Penetration field
            im2 = axes[1].imshow(penetration.T, cmap='hot', origin='lower')
            axes[1].set_title('GB Penetration')
            plt.colorbar(im2, ax=axes[1])
            
            # Cross-section
            mid_y = params['grid_size'] // 2
            axes[2].plot(eta_sp[:, mid_y], 'b-', label='Substrate', linewidth=2)
            axes[2].plot(gb_indicator[:, mid_y], 'r--', label='GB Indicator', linewidth=2)
            axes[2].set_title(f'Cross-section at y={mid_y}')
            axes[2].set_xlabel('X position')
            axes[2].set_ylabel('Value')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_analysis)
        
        else:
            st.info("Run the simulation first to view microstructure evolution")
    
    # =====================
    # INTERPOLATION TAB
    # =====================
    with tab3:
        st.header("Moelans Interpolation Functions")
        
        st.markdown("""
        ### Moelans Interpolation Functions for Multi-Phase Systems
        
        The Moelans interpolation functions ensure proper coupling between multiple order parameters in phase-field models. Two key forms are used:
        
        **1. Standard Interpolation (for substrate phase):**
        $$
        h(η_{sp}) = η_{sp}^2(3 - 2η_{sp})
        $$
        
        **2. Generalized Interpolation (for multiple grains):**
        $$
        h_k = \\frac{η_k^2}{η_k^2 + α\\sum_{j≠k} η_j^2}
        $$
        where $α$ is a weighting parameter that controls the interface properties.
        """)
        
        if 'etas' in st.session_state:
            with st.spinner("Computing interpolation functions..."):
                fig_interp = create_interpolation_visualization(
                    st.session_state.etas,
                    st.session_state.eta_sp
                )
                st.pyplot(fig_interp)
            
            # Mathematical properties
            st.subheader("Key Properties")
            st.markdown("""
            **Standard Interpolation Properties:**
            - $h(0) = 0$, $h(1) = 1$ (bounds preserved)
            - $h'(0) = h'(1) = 0$ (zero derivatives at bounds)
            - $h(η) + h(1-η) = 1$ (partition of unity)
            - Monotonic increasing between 0 and 1
            
            **Generalized Interpolation Properties:**
            - $\\sum_k h_k = 1$ (partition of unity)
            - $h_k = 1$ when $η_k = 1$ and all other $η_j = 0$
            - Smooth transition between phases
            - Controls interface width through parameter $α$
            """)
        
        else:
            st.info("Run the simulation first to visualize interpolation functions")
    
    # =====================
    # RESULTS TAB
    # =====================
    with tab4:
        st.header("Simulation Results and Export")
        
        if 'field_history' in st.session_state and st.session_state.field_history['eta_sp']:
            col1, col2 = st.columns(2)
            
            with col1:
                # Energy evolution
                st.subheader("Energy Evolution")
                fig_energy, ax = plt.subplots(figsize=(8, 5))
                ax.plot(st.session_state.energy_history, 'b-', linewidth=2)
                ax.set_xlabel('Simulation Step')
                ax.set_ylabel('Free Energy Density')
                ax.set_title('Free Energy Evolution')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_energy)
            
            with col2:
                # Penetration depth analysis
                st.subheader("Penetration Depth Analysis")
                
                # Calculate penetration depth over time
                penetration_depths = []
                for eta_sp in st.session_state.field_history['eta_sp']:
                    # Find the rightmost point where substrate > 0.5
                    max_x = 0
                    for x in range(params['grid_size']):
                        if np.any(eta_sp[x, :] > 0.5):
                            max_x = x
                    penetration_depths.append(max_x)
                
                fig_depth, ax = plt.subplots(figsize=(8, 5))
                steps = st.session_state.field_history['step_indices']
                ax.plot(steps, penetration_depths, 'g-', linewidth=2, marker='o')
                ax.set_xlabel('Simulation Step')
                ax.set_ylabel('Penetration Depth (grid units)')
                ax.set_title('Substrate Penetration Depth vs Time')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_depth)
            
            # Export section
            st.subheader("Export Simulation Data")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("Export as NumPy Arrays", type="primary"):
                    with st.spinner("Preparing NumPy export..."):
                        export_data = {
                            'parameters': params,
                            'etas': np.array(st.session_state.field_history['etas']),
                            'eta_sp': np.array(st.session_state.field_history['eta_sp']),
                            'c': np.array(st.session_state.field_history['c']),
                            'gb_indicator': np.array(st.session_state.field_history['gb_indicator']),
                            'step_indices': np.array(st.session_state.field_history['step_indices']),
                            'energy_history': np.array(st.session_state.energy_history)
                        }
                        
                        # Save to buffer
                        buf = BytesIO()
                        np.savez_compressed(buf, **export_data)
                        buf.seek(0)
                        
                        st.download_button(
                            label="Download NumPy Data",
                            data=buf.getvalue(),
                            file_name="intergranular_penetration_data.npz",
                            mime="application/octet-stream"
                        )
            
            with export_col2:
                st.markdown("""
                **Export Contents:**
                - `parameters`: Simulation parameters dictionary
                - `etas`: Grain order parameters over time (steps, n_grains, nx, ny)
                - `eta_sp`: Substrate phase over time (steps, nx, ny)
                - `c`: Concentration field over time (steps, nx, ny)
                - `gb_indicator`: Grain boundary indicator over time (steps, nx, ny)
                - `step_indices`: Corresponding simulation steps
                - `energy_history`: Free energy evolution
                
                **To load in Python:**
                ```python
                import numpy as np
                data = np.load('intergranular_penetration_data.npz')
                params = data['parameters'].item()
                eta_sp = data['eta_sp']
                ```
                """)
        
        else:
            st.info("Run the simulation first to access results and export options")
    
    # =====================
    # THEORY TAB
    # =====================
    with tab5:
        st.header("Theoretical Foundation")
        
        st.markdown("""
        ## Phase Field Model for Substrate Growth Through Grain Boundaries
        
        This model simulates the preferential growth of a substrate phase along grain boundaries in a polycrystalline material where the grains themselves are inert.
        
        ### Key Model Features
        
        **1. Inert Grains:**
        - The grain structure is fixed after initialization
        - No grain boundary migration or grain growth occurs
        - This represents a scenario where the matrix grains are thermodynamically stable
        
        **2. Substrate Phase Evolution:**
        - The substrate phase grows from the left boundary
        - Growth is driven by chemical supersaturation
        - Preferential growth along grain boundaries is achieved through:
          - Energy bias: Lower energy barrier along GBs
          - Mobility enhancement: Higher diffusivity along GBs
        
        **3. Moelans Interpolation Functions:**
        - Ensure proper partition of unity between phases
        - Control interface properties and width
        - Enable multi-phase coupling without spurious phases
        
        ### Mathematical Formulation
        
        **Free Energy Functional:**
        $$
        F = \\int_V \\left[ f_{bulk} + \\frac{\\kappa_{sp}}{2}|\\nabla\\eta_{sp}|^2 + 
        \\frac{\\kappa_c}{2}|\\nabla c|^2 \\right] dV
        $$
        
        where the bulk energy density is:
        $$
        f_{bulk} = w\\left[h(\\eta_{sp})f_{sp}(c) + (1-h(\\eta_{sp}))f_{matrix}(c)\\right] - 
        \\epsilon h(\\eta_{sp})\\mathcal{I}_{GB}
        $$
        
        **Evolution Equations:**
        
        - **Substrate Phase** (non-conserved):
        $$
        \\frac{\\partial\\eta_{sp}}{\\partial t} = -L_{sp}\\frac{\\delta F}{\\delta\\eta_{sp}} + \\xi(\\mathbf{r},t)
        $$
        
        - **Concentration Field** (conserved, Cahn-Hilliard):
        $$
        \\frac{\\partial c}{\\partial t} = \\nabla\\cdot\\left[M(\\mathbf{r})\\nabla\\frac{\\delta F}{\\delta c}\\right]
        $$
        
        where the mobility is enhanced along grain boundaries:
        $$
        M(\\mathbf{r}) = M_0(1 + \\beta\\mathcal{I}_{GB}(\\mathbf{r}))
        $$
        
        **Grain Boundary Indicator:**
        $$
        \\mathcal{I}_{GB} = \\sum_{i<j}\\eta_i^2\\eta_j^2
        $$
        
        This function peaks at grain boundaries (where multiple η_i are non-zero) and is zero within grain interiors.
        
        ### Numerical Implementation
        
        - **Spatial Discretization**: Finite differences on uniform grid
        - **Time Integration**: Explicit Euler method
        - **Boundary Conditions**:
          - Left boundary (x=0): Fixed substrate phase (η_sp=1) and concentration (c=1)
          - Right boundary (x=L): Fixed matrix phase (η_sp=0) and concentration (c=c_∞)
          - Periodic boundaries in y-direction
        - **Performance**: Numba JIT compilation for critical functions
        """)
        
        # Parameter guide
        st.subheader("Parameter Guide")
        
        params_explanation = {
            'Grid Size': 'Spatial resolution of the simulation. Larger grids provide better resolution but require more computation time.',
            'Number of Inert Grains': 'Number of fixed grains through which the substrate will grow. More grains create more grain boundary pathways.',
            'Interface Width': 'Width of diffuse interfaces between phases (in grid units). Controls numerical stability and physical accuracy.',
            'Gradient Energy (κ)': 'Controls the energy penalty for phase interfaces. Higher values lead to sharper interfaces.',
            'GB Mobility Enhancement': 'Factor by which mobility is increased along grain boundaries. Higher values promote faster GB growth.',
            'GB Energy Bias': 'Energy reduction along grain boundaries that favors substrate nucleation and growth.',
            'Supersaturation': 'Initial concentration excess that drives the substrate growth. Higher values increase driving force.',
            'Time Step': 'Numerical integration step size. Must be small enough for stability (typically < 0.01).'
        }
        
        for param, explanation in params_explanation.items():
            with st.expander(param):
                st.markdown(explanation)
        
      
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note**: This simulation uses a specialized model where grains are inert and only the substrate phase evolves. 
    The Moelans interpolation functions ensure proper multi-phase coupling. For research applications, 
    consider validating against analytical solutions or experimental data.
    
    **Performance Tips**:
    - Start with smaller grid sizes (64×64) for testing
    - Increase grid size gradually for production runs
    - Use fewer grains initially to verify model behavior
    - Monitor energy evolution for numerical stability
    """)

if __name__ == "__main__":
    main()
