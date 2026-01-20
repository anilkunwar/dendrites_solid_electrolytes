import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from io import BytesIO
import time
from numba import njit, prange
import zipfile
import base64
import os
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
def moelans_generalized_interp_point(eta_k: float, etas_at_point: np.ndarray, alpha: float = 1.0) -> float:
    """
    Generalized Moelans interpolation function for multiple phases at a single point:
    h_k = η_k² / (η_k² + α∑_{j≠k} η_j²)
    
    Args:
        eta_k: Order parameter for phase k at this point
        etas_at_point: Array of all order parameters at this point
        alpha: Weighting parameter (default 1.0)
        
    Returns:
        Interpolated value for phase k at this point
    """
    numerator = eta_k * eta_k
    denominator = numerator
    
    for i in range(etas_at_point.shape[0]):
        if i < etas_at_point.shape[0]:
            denominator += alpha * etas_at_point[i] * etas_at_point[i]
    
    if denominator > 1e-10:
        return numerator / denominator
    else:
        return 0.0 if numerator < 0.5 else 1.0

@njit(parallel=True)
def compute_gb_indicator(etas: np.ndarray, method: str = 'standard', n_grains: int = 0) -> np.ndarray:
    """
    Compute grain boundary indicator function
    
    Args:
        etas: Array of grain order parameters (n_grains, nx, ny)
        method: 'standard' for η_i²η_j² sum, 'gradient' for gradient-based
        n_grains: Number of grains (explicitly passed for Numba compatibility)
        
    Returns:
        GB indicator field (0 = grain interior, 1 = grain boundary)
    """
    n_grains_calc = etas.shape[0] if n_grains == 0 else n_grains
    nx, ny = etas.shape[1], etas.shape[2]
    gb_indicator = np.zeros((nx, ny), dtype=np.float64)
    
    if method == 'standard':
        # Standard method: sum of η_i²η_j² for all i<j
        for i in range(n_grains_calc):
            eta_i_sq = etas[i] * etas[i]
            for j in range(i + 1, n_grains_calc):
                eta_j_sq = etas[j] * etas[j]
                for x in prange(nx):
                    for y in range(ny):
                        gb_indicator[x, y] += eta_i_sq[x, y] * eta_j_sq[x, y]
        
        # Normalize to [0, 1]
        max_val = 0.75 * (n_grains_calc / 2.0)  # Theoretical maximum
        if max_val > 1e-10:
            for x in prange(nx):
                for y in range(ny):
                    if gb_indicator[x, y] > max_val:
                        gb_indicator[x, y] = max_val
                    gb_indicator[x, y] = gb_indicator[x, y] / max_val
    
    elif method == 'gradient':
        # Gradient-based method: sum of |∇η_i|²
        for i in range(n_grains_calc):
            grad_sq = np.zeros((nx, ny))
            for x in prange(1, nx-1):
                for y in range(1, ny-1):
                    dx = (etas[i, x+1, y] - etas[i, x-1, y]) / 2.0
                    dy = (etas[i, x, y+1] - etas[i, x, y-1]) / 2.0
                    grad_sq[x, y] = dx*dx + dy*dy
            for x in prange(nx):
                for y in range(ny):
                    gb_indicator[x, y] += grad_sq[x, y]
        
        # Normalize
        max_val = 0.0
        for x in range(nx):
            for y in range(ny):
                if gb_indicator[x, y] > max_val:
                    max_val = gb_indicator[x, y]
        
        if max_val > 1e-10:
            for x in prange(nx):
                for y in range(ny):
                    gb_indicator[x, y] = gb_indicator[x, y] / max_val
    
    return gb_indicator

# =====================
# NUMERICAL OPERATORS - FIXED FOR NUMBA COMPATIBILITY
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

# FIX: Replaced dictionary access with individual parameters for Numba compatibility
@njit(parallel=True)
def update_substrate(etas: np.ndarray, eta_sp: np.ndarray, c: np.ndarray, 
                    kappa: float, mobility: float, gb_mobility: float, 
                    gb_energy_bias: float, supersaturation: float,
                    time_step: float, dx: float, step: int, 
                    n_grains: int, gb_indicator_type: str) -> np.ndarray:
    """
    Update substrate phase with preferential GB growth - FIXED FOR NUMBA
    
    Args:
        etas: Inert grain order parameters (n_grains, nx, ny)
        eta_sp: Substrate phase order parameter (nx, ny)
        c: Concentration field (nx, ny)
        kappa: Gradient energy coefficient
        mobility: Base mobility
        gb_mobility: GB mobility enhancement
        gb_energy_bias: GB energy bias strength
        supersaturation: Initial supersaturation
        time_step: Time step
        dx: Grid spacing
        step: Current simulation step
        n_grains: Number of grains
        gb_indicator_type: GB indicator method
        
    Returns:
        Updated substrate phase field
    """
    nx, ny = eta_sp.shape
    new_eta_sp = np.copy(eta_sp)
    
    # Pre-compute GB indicator outside the main loop
    gb_indicator = compute_gb_indicator(etas, gb_indicator_type, n_grains)
    
    # Compute time-dependent GB bias strength (ramp up over first 10% of simulation)
    max_steps = 1000  # Fixed maximum for normalization
    bias_strength = gb_energy_bias * min(1.0, step / max(1, 0.1 * max_steps))
    
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
            chem_force = supersaturation - c[i, j]
            
            # GB preference term
            gb_force = bias_strength * gb_indicator[i, j]
            
            # Total driving force
            driving_force = chem_force + gb_force
            
            # Laplacian term
            jp1 = (j + 1) % ny
            jm1 = (j - 1 + ny) % ny
            lap_eta_sp = (eta_sp[i+1, j] + eta_sp[i-1, j] + 
                         eta_sp[i, jp1] + eta_sp[i, jm1] - 4 * eta_sp[i, j]) / dx**2
            
            # Update equation: ∂η_sp/∂t = M(Δη_sp - dF/dη_sp)
            # Simplified free energy derivative
            dF_deta = eta_sp[i, j] * (eta_sp[i, j] - 1.0) * (eta_sp[i, j] - 0.5)  # Double well
            
            # Mobility field (GB enhanced)
            mobility_local = mobility * (1.0 + gb_mobility * gb_indicator[i, j])
            
            # Euler integration
            new_val = eta_sp[i, j] + time_step * mobility_local * (
                kappa * lap_eta_sp - dF_deta + driving_force
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
                        mobility: float, gb_mobility: float, time_step: float, dx: float,
                        supersaturation: float, n_grains: int, gb_indicator_type: str) -> np.ndarray:
    """
    Update concentration field with GB-enhanced diffusion - FIXED FOR NUMBA
    
    Args:
        etas: Inert grain order parameters
        eta_sp: Substrate phase order parameter
        c: Current concentration field
        mobility: Base mobility
        gb_mobility: GB mobility enhancement
        time_step: Time step
        dx: Grid spacing
        supersaturation: Far-field concentration
        n_grains: Number of grains
        gb_indicator_type: GB indicator method
        
    Returns:
        Updated concentration field
    """
    nx, ny = c.shape
    new_c = np.copy(c)
    
    # Compute GB indicator for mobility enhancement
    gb_indicator = compute_gb_indicator(etas, gb_indicator_type, n_grains)
    
    for i in prange(nx):
        for j in range(ny):
            # Skip boundaries
            if i == 0:
                new_c[i, j] = 1.0  # Fixed concentration at substrate boundary
                continue
            elif i == nx - 1:
                new_c[i, j] = supersaturation  # Fixed at far field
                continue
            
            jp1 = (j + 1) % ny
            jm1 = (j - 1 + ny) % ny
            
            # Mobility at this point (GB enhanced)
            mobility_local = mobility * (1.0 + gb_mobility * gb_indicator[i, j])
            
            # Concentration gradient
            dc_dx_pos = (c[i+1, j] - c[i, j]) / dx
            dc_dx_neg = (c[i, j] - c[i-1, j]) / dx
            dc_dy_pos = (c[i, jp1] - c[i, j]) / dx
            dc_dy_neg = (c[i, j] - c[i, jm1]) / dx
            
            # Flux divergence
            div_flux = (mobility_local * (dc_dx_pos - dc_dx_neg) + 
                       mobility_local * (dc_dy_pos - dc_dy_neg)) / dx
            
            # Update concentration
            new_val = c[i, j] - time_step * div_flux
            
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
                          method: str = 'voronoi', interface_width: float = 3.0) -> np.ndarray:
    """
    Create grain structure with specified arrangement
    
    Args:
        nx, ny: Grid dimensions
        n_grains: Number of grains
        method: 'voronoi', 'random_seeds', or 'regular_grid'
        interface_width: Width of diffuse interfaces
        
    Returns:
        Order parameter fields for grains (n_grains, nx, ny)
    """
    etas = np.zeros((n_grains, nx, ny), dtype=np.float64)
    
    if method == 'voronoi':
        # Create Voronoi tessellation
        seeds = np.random.rand(n_grains, 2) * [nx, ny]
        for i in range(nx):
            for j in range(ny):
                distances = np.sqrt((seeds[:, 0] - i)**2 + **(seeds[:, 1] - j)2)
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
    if interface_width > 0:
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
    etas = create_grain_structure(nx, ny, n_grains, params['grain_arrangement'], params['interface_width'])
    
    # Initialize substrate phase (left side)
    eta_sp = np.zeros((nx, ny), dtype=np.float64)
    substrate_width = max(1, int(0.1 * nx))  # 10% of domain width
    eta_sp[:substrate_width, :] = 1.0
    
    # Initialize concentration field
    c = np.full((nx, ny), params['supersaturation'], dtype=np.float64)
    c[:substrate_width, :] = 1.0  # Substrate concentration
    
    return etas, eta_sp, c

# =====================
# VISUALIZATION FUNCTIONS - ENHANCED FOR COMPOSITE VIEW
# =====================
def create_composite_microstructure(etas: np.ndarray, eta_sp: np.ndarray, 
                                   show_interpolation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create composite microstructure visualization combining both interpolation functions
    
    Args:
        etas: Grain order parameters (n_grains, nx, ny)
        eta_sp: Substrate phase (nx, ny)
        show_interpolation: Whether to apply Moelans interpolation
        
    Returns:
        composite_grain: Composite grain field (showing both substrate and matrix)
        grain_ids: Grain ID field for coloring
    """
    n_grains, nx, ny = etas.shape
    
    # Create grain IDs (0 = substrate, 1+ = grain numbers)
    grain_ids = np.zeros((nx, ny), dtype=int)
    
    # Create composite field combining substrate and grains
    composite_grain = np.zeros((nx, ny))
    
    # Substrate interpolation function
    h_sp = moelans_standard_interp(eta_sp) if show_interpolation else eta_sp
    
    for i in range(nx):
        for j in range(ny):
            if h_sp[i, j] > 0.5:  # Substrate region
                grain_ids[i, j] = 0
                composite_grain[i, j] = 0.0  # Substrate value
            else:  # Matrix region with grains
                # Find dominant grain
                max_idx = 0
                max_val = etas[0, i, j]
                for k in range(1, n_grains):
                    if etas[k, i, j] > max_val:
                        max_val = etas[k, i, j]
                        max_idx = k
                
                grain_ids[i, j] = max_idx + 1
                
                # Apply generalized interpolation for grains
                if show_interpolation:
                    etas_at_point = np.zeros(n_grains)
                    for k in range(n_grains):
                        etas_at_point[k] = etas[k, i, j]
                    composite_grain[i, j] = moelans_generalized_interp_point(etas[max_idx, i, j], etas_at_point)
                else:
                    composite_grain[i, j] = max_val
    
    return composite_grain, grain_ids

def visualize_initial_conditions(etas: np.ndarray, eta_sp: np.ndarray, params: Dict):
    """
    Create comprehensive visualization of initial conditions showing both phases
    
    Args:
        etas: Grain order parameters
        eta_sp: Substrate phase
        params: Simulation parameters
    """
    n_grains, nx, ny = etas.shape
    
    # Create composite microstructure
    composite_grain, grain_ids = create_composite_microstructure(etas, eta_sp, show_interpolation=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Composite microstructure with interpolation
    ax1 = fig.add_subplot(221)
    grain_colors = ['black'] + [plt.cm.tab20(i) for i in range(n_grains)]
    grain_cmap = ListedColormap(grain_colors[:n_grains+1])
    
    im1 = ax1.imshow(grain_ids.T, cmap=grain_cmap, origin='lower', interpolation='nearest')
    ax1.set_title('Initial Microstructure (Composite View)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, ticks=range(n_grains+1), 
                label='Phase: 0=Substrate, 1+=Grains')
    
    # 2. Substrate phase with interpolation
    ax2 = fig.add_subplot(222)
    h_sp = moelans_standard_interp(eta_sp)
    im2 = ax2.imshow(h_sp.T, cmap='Blues', origin='lower', vmin=0, vmax=1)
    ax2.set_title('Substrate Interpolation Function h(η_sp)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='h(η_sp)')
    
    # 3. GB indicator
    ax3 = fig.add_subplot(223)
    gb_indicator = compute_gb_indicator(etas, params['gb_indicator_type'], params['n_grains'])
    im3 = ax3.imshow(gb_indicator.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax3.set_title('Initial Grain Boundary Indicator')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3, label='GB Indicator')
    
    # 4. Concentration field preview
    ax4 = fig.add_subplot(224)
    c_preview = np.full((nx, ny), params['supersaturation'])
    substrate_width = max(1, int(0.1 * nx))
    c_preview[:substrate_width, :] = 1.0
    im4 = ax4.imshow(c_preview.T, cmap='hot', origin='lower', vmin=0, vmax=1.5)
    ax4.set_title('Initial Concentration Field')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4, label='Concentration')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Create separate figure for interpolation functions
    fig2 = plt.figure(figsize=(12, 8))
    
    # Plot standard interpolation function
    ax5 = fig2.add_subplot(121)
    eta_vals = np.linspace(0, 1, 100)
    h_vals = eta_vals**2 * (3 - 2*eta_vals)
    ax5.plot(eta_vals, h_vals, 'b-', linewidth=3, label='h(η) = η²(3-2η)')
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Reference')
    ax5.set_xlabel('η', fontsize=12)
    ax5.set_ylabel('h(η)', fontsize=12)
    ax5.set_title('Standard Moelans Interpolation', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot generalized interpolation example
    ax6 = fig2.add_subplot(122)
    n_points = 50
    eta1_vals = np.linspace(0, 1, n_points)
    eta2_vals = np.linspace(0, 1, n_points)
    h_matrix = np.zeros((n_points, n_points))
    
    for i, e1 in enumerate(eta1_vals):
        for j, e2 in enumerate(eta2_vals):
            etas_at_point = np.array([e1, e2])
            h_matrix[i, j] = moelans_generalized_interp_point(e1, etas_at_point, alpha=1.0)
    
    im6 = ax6.imshow(h_matrix.T, extent=[0, 1, 0, 1], origin='lower', 
                    cmap='viridis', aspect='equal')
    ax6.set_xlabel('η₁', fontsize=12)
    ax6.set_ylabel('η₂', fontsize=12)
    ax6.set_title('Generalized Interpolation (2 phases)', fontsize=14)
    plt.colorbar(im6, ax=ax6, label='h₁')
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Summary statistics
    st.subheader("Initial Conditions Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        substrate_area = np.sum(eta_sp > 0.5) / (nx * ny) * 100
        st.metric("Substrate Area", f"{substrate_area:.1f}%")
    
    with col2:
        avg_gb_indicator = np.mean(gb_indicator)
        st.metric("Average GB Indicator", f"{avg_gb_indicator:.3f}")
    
    with col3:
        st.metric("Grid Resolution", f"{nx}×{ny}")
    
    st.info("""
    **Initial Conditions Explained:**
    - **Composite Microstructure**: Shows both substrate phase (black) and polycrystalline grains with different colors
    - **Substrate Interpolation**: Uses standard Moelans function h(η_sp) = η_sp²(3-2η_sp) for smooth transition
    - **GB Indicator**: Highlights grain boundaries where η_i²η_j² is maximum (values close to 1)
    - **Concentration**: Shows initial supersaturation driving force (c > equilibrium in matrix)
    
    The simulation will evolve the substrate phase while keeping grains inert, with preferential growth along grain boundaries.
    """)

def create_field_visualization(etas: np.ndarray, eta_sp: np.ndarray, c: np.ndarray, 
                             energy_history: List[float], step: int, params: Dict) -> go.Figure:
    """
    Create Plotly visualization of all fields
    
    Args:
        etas: Grain order parameters
        eta_sp: Substrate phase
        c: Concentration field
        energy_history: Energy evolution history
        step: Current simulation step
        params: Simulation parameters
        
    Returns:
        Plotly figure
    """
    n_grains, nx, ny = etas.shape
    
    # Create subplot figure
    fig = sp.make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            f'Microstructure (Step {step})', 'Substrate Phase', 'Concentration Field',
            'GB Indicator', 'Composite Field', 'Free Energy'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Composite Microstructure
    composite_grain, grain_ids = create_composite_microstructure(etas, eta_sp, show_interpolation=True)
    
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
    gb_indicator = compute_gb_indicator(etas, params['gb_indicator_type'], params['n_grains'])
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
    
    # 5. Composite Field (interpolated)
    fig.add_trace(
        go.Heatmap(
            z=composite_grain.T,
            colorscale='Plasma',
            zmin=0,
            zmax=1,
            showscale=True,
            name='Composite Field'
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Initial Conditions", "Simulation", "Results", "Theory"
    ])
    
    # =====================
    # INITIAL CONDITIONS TAB
    # =====================
    with tab1:
        st.header("Initial Microstructure and Interpolation Functions")
        
        st.markdown("""
        This tab shows the initial conditions of the simulation, including:
        - **Composite microstructure** showing both substrate phase and polycrystalline grains
        - **Moelans interpolation functions** for both substrate and grain phases
        - **Grain boundary indicator** highlighting preferential growth paths
        - **Initial concentration field** showing the driving force for penetration
        
        The composite view combines both interpolation functions to show the complete initial state.
        """)
        
        # Generate and show initial conditions
        with st.spinner("Generating initial microstructure..."):
            etas_init, eta_sp_init, _ = initialize_fields(params)
            
            # Show comprehensive initial conditions visualization
            visualize_initial_conditions(etas_init, eta_sp_init, params)
        
        # Interactive parameter effects
        st.subheader("Parameter Effects Preview")
        
        st.markdown("""
        Adjust the sidebar parameters to see how they affect the initial microstructure:
        - **Interface Width**: Controls the smoothness of phase boundaries
        - **Number of Grains**: Changes the complexity of the polycrystalline structure
        - **Grain Arrangement**: Different patterns of grain distribution
        - **GB Detection Method**: Different ways to identify grain boundaries
        
        The visualization updates automatically when you change parameters.
        """)
        
        if st.button("Refresh Initial Conditions", type="secondary"):
            st.experimental_rerun()
    
    # =====================
    # SIMULATION TAB
    # =====================
    with tab2:
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
                st.info("✅ Simulation initialized successfully!")
            
            # Show current state
            st.subheader("Current Simulation State")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                plot_placeholder = st.empty()
            with col2:
                progress_bar = st.progress(0)
                status_text = st.empty()
                st.metric("Current Step", st.session_state.current_step)
                st.metric("Grid Size", f"{params['grid_size']}×{params['grid_size']}")
            
            # Simulation controls
            st.subheader("Simulation Controls")
            col1, col2, col3 = st.columns(3)
            with col1:
                vis_step_freq = st.slider("Visualization Frequency", 1, 100, 10)
            with col2:
                save_step_freq = st.slider("Save Frequency", 1, 200, 50)
            with col3:
                st.write("Paused" if pause_sim else "Running")
            
            # Run simulation
            start_time = time.time()
            total_steps = params['total_steps']
            
            for step in range(st.session_state.current_step, total_steps):
                if pause_sim:
                    st.session_state.current_step = step
                    break
                
                # Update fields - FIXED CALL WITH INDIVIDUAL PARAMETERS
                st.session_state.eta_sp = update_substrate(
                    st.session_state.etas, 
                    st.session_state.eta_sp, 
                    st.session_state.c,
                    params['kappa'],
                    params['mobility'],
                    params['gb_mobility'],
                    params['gb_energy_bias'],
                    params['supersaturation'],
                    params['time_step'],
                    params['dx'],
                    step,
                    params['n_grains'],
                    params['gb_indicator_type']
                )
                
                st.session_state.c = update_concentration(
                    st.session_state.etas,
                    st.session_state.eta_sp,
                    st.session_state.c,
                    params['mobility'],
                    params['gb_mobility'],
                    params['time_step'],
                    params['dx'],
                    params['supersaturation'],
                    params['n_grains'],
                    params['gb_indicator_type']
                )
                
                # Calculate energy (simplified)
                energy = np.mean(st.session_state.eta_sp**2 * (1 - st.session_state.eta_sp)**2)
                st.session_state.energy_history.append(energy)
                
                # Save fields for visualization and export
                if step % save_step_freq == 0 or step == total_steps - 1:
                    st.session_state.field_history['etas'].append(np.copy(st.session_state.etas))
                    st.session_state.field_history['eta_sp'].append(np.copy(st.session_state.eta_sp))
                    st.session_state.field_history['c'].append(np.copy(st.session_state.c))
                    gb_ind = compute_gb_indicator(st.session_state.etas, params['gb_indicator_type'], params['n_grains'])
                    st.session_state.field_history['gb_indicator'].append(gb_ind)
                    st.session_state.field_history['step_indices'].append(step)
                
                # Update visualization
                if step % vis_step_freq == 0 or step == total_steps - 1:
                    fig = create_field_visualization(
                        st.session_state.etas,
                        st.session_state.eta_sp,
                        st.session_state.c,
                        st.session_state.energy_history,
                        step,
                        params
                    )
                    plot_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Update progress
                progress = (step + 1) / total_steps
                progress_bar.progress(progress)
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (step + 1) * total_steps if step > 0 else 0
                remaining_time = max(0, estimated_total - elapsed_time)
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
            
            # Show initial conditions preview
            with st.spinner("Generating preview..."):
                etas_init, eta_sp_init, _ = initialize_fields(params)
                fig_preview = plt.figure(figsize=(10, 8))
                composite_grain, grain_ids = create_composite_microstructure(etas_init, eta_sp_init)
                
                grain_colors = ['black'] + [plt.cm.tab20(i) for i in range(params['n_grains'])]
                grain_cmap = ListedColormap(grain_colors[:params['n_grains']+1])
                
                ax = fig_preview.add_subplot(111)
                im = ax.imshow(grain_ids.T, cmap=grain_cmap, origin='lower')
                ax.set_title('Initial Microstructure Preview')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax, label='Phase ID')
                plt.tight_layout()
                st.pyplot(fig_preview)
    
    # =====================
    # RESULTS TAB
    # =====================
    with tab3:
        st.header("Simulation Results and Analysis")
        
        if 'field_history' in st.session_state and st.session_state.field_history['eta_sp']:
            # Step selection
            st.subheader("Select Simulation Step")
            step_idx = st.slider(
                "Simulation Step", 
                0, 
                len(st.session_state.field_history['step_indices']) - 1,
                len(st.session_state.field_history['step_indices']) - 1
            )
            
            step_num = st.session_state.field_history['step_indices'][step_idx]
            eta_sp = st.session_state.field_history['eta_sp'][step_idx]
            c = st.session_state.field_history['c'][step_idx]
            etas = st.session_state.etas  # Grains remain inert
            
            # Create composite visualization
            composite_grain, grain_ids = create_composite_microstructure(etas, eta_sp)
            
            # Display results
            st.subheader(f"Results at Step {step_num}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Composite Microstructure**")
                fig_micro, ax = plt.subplots(figsize=(8, 6))
                
                grain_colors = ['black'] + [plt.cm.tab20(i) for i in range(params['n_grains'])]
                grain_cmap = ListedColormap(grain_colors[:params['n_grains']+1])
                
                im = ax.imshow(grain_ids.T, cmap=grain_cmap, origin='lower')
                ax.set_title(f'Composite Microstructure (Step {step_num})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax, label='Phase ID')
                st.pyplot(fig_micro)
            
            with col2:
                st.markdown("**Substrate Phase Field**")
                fig_sub, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(eta_sp.T, cmap='Blues', origin='lower', vmin=0, vmax=1)
                ax.set_title(f'Substrate Phase (Step {step_num})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                plt.colorbar(im, ax=ax, label='η_sp')
                st.pyplot(fig_sub)
            
            # GB penetration analysis
            st.subheader("Grain Boundary Penetration Analysis")
            
            gb_indicator = st.session_state.field_history['gb_indicator'][step_idx]
            penetration = eta_sp * gb_indicator
            
            fig_analysis, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # GB indicator
            im1 = axes[0].imshow(gb_indicator.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
            axes[0].set_title('Grain Boundary Indicator')
            plt.colorbar(im1, ax=axes[0])
            
            # Penetration field
            im2 = axes[1].imshow(penetration.T, cmap='hot', origin='lower', vmin=0, vmax=1)
            axes[1].set_title('GB Penetration Field')
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
            
            # Export section
            st.subheader("Export Simulation Data")
            
            if st.button("Export as NumPy Arrays", type="primary"):
                with st.spinner("Preparing export..."):
                    export_data = {
                        'parameters': params,
                        'etas': np.array(st.session_state.etas),  # Inert grains
                        'eta_sp_history': np.array(st.session_state.field_history['eta_sp']),
                        'c_history': np.array(st.session_state.field_history['c']),
                        'gb_indicator_history': np.array(st.session_state.field_history['gb_indicator']),
                        'step_indices': np.array(st.session_state.field_history['step_indices']),
                        'energy_history': np.array(st.session_state.energy_history)
                    }
                    
                    # Save to buffer
                    buf = BytesIO()
                    np.savez_compressed(buf, **export_data)
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Simulation Data",
                        data=buf.getvalue(),
                        file_name="intergranular_penetration_data.npz",
                        mime="application/octet-stream"
                    )
            
            st.info("""
            **Export Data Contents:**
            - `parameters`: Simulation parameters dictionary
            - `etas`: Inert grain structure (n_grains, nx, ny)
            - `eta_sp_history`: Substrate phase evolution over time
            - `c_history`: Concentration field evolution
            - `gb_indicator_history`: Grain boundary indicator evolution
            - `step_indices`: Corresponding simulation steps
            - `energy_history`: Free energy evolution
            
            This data can be loaded in Python for further analysis:
            ```python
            import numpy as np
            data = np.load('intergranular_penetration_data.npz')
            params = data['parameters'].item()
            eta_sp_history = data['eta_sp_history']
            ```
            """)
        
        else:
            st.info("Run the simulation first to view results and export data")
    
    # =====================
    # THEORY TAB
    # =====================
    with tab4:
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
        - **Standard form**: $h(η_{sp}) = η_{sp}^2(3 - 2η_{sp})$ for substrate phase
        - **Generalized form**: $h_k = \\frac{η_k^2}{η_k^2 + α\\sum_{j≠k} η_j^2}$ for grain phases
        - Ensures partition of unity and proper interface properties
        - Controls interface width through parameter $α$
        
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
        
        This function peaks at grain boundaries (where multiple $η_i$ are non-zero) and is zero within grain interiors.
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
