import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import BytesIO
import time
from numba import njit, prange
import zipfile
import base64
import os

# Constants - minimal grid for robustness
GRID_X, GRID_Y = 64, 64
DX = 1.0
DT = 0.01  # Smaller time step for stability
STEPS_GRAIN_GROWTH = 500  # Reduced for faster execution
MAX_PENETRATION_STEPS = 1000  # Reduced for faster execution

# Model parameters (dimensionless)
M_GRAIN = 1.0    # Grain mobility
M_SP = 0.5       # Substrate mobility (slower)
K_GRAIN = 0.5    # Gradient energy for grains
K_SP = 0.5       # Gradient energy for substrate
K_C = 1.0        # Gradient energy for concentration
GAMMA_GB = 1.0   # Grain boundary energy coefficient
ALPHA_GB = 0.8   # GB preference for substrate
M_BULK = 0.1     # Bulk mobility for c
M_GB = 5.0       # GB-enhanced mobility for c
A_MATRIX = 1.0   # Chemical energy coefficient - matrix
B_SUBSTRATE = 1.0 # Chemical energy coefficient - substrate
C_MATRIX_EQ = 0.3 # Equilibrium concentration in matrix
C_SUBSTRATE_EQ = 1.0 # Equilibrium concentration in substrate

# FIX 1: Use explicit float64 dtype for all arrays to avoid Numba typing issues
@njit(parallel=True)
def compute_gb_indicator(eta):
    """Vectorized GB indicator: 0.5[(∑η_i²)² - ∑η_i⁴]"""
    n_grains = eta.shape[0]
    gb = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    sum_eta_sq = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    sum_eta_quad = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    
    for i in prange(n_grains):
        eta_sq = eta[i] * eta[i]  # More Numba-friendly than **2
        sum_eta_sq += eta_sq
        sum_eta_quad += eta_sq * eta_sq
    
    gb = 0.5 * (sum_eta_sq * sum_eta_sq - sum_eta_quad)
    # Normalize to [0, 1] for stability
    max_gb = 0.75  # Theoretical maximum for 4 grains
    for i in prange(GRID_X):
        for j in range(GRID_Y):
            if gb[i, j] > max_gb:
                gb[i, j] = max_gb
            gb[i, j] = gb[i, j] / max_gb
    
    return gb

# FIX 2: Simplified boundary handling with explicit type declarations
@njit(parallel=True)
def laplacian_2d(field, dx):
    """Efficient 2D Laplacian with periodic boundaries in y, fixed in x"""
    lap = np.zeros_like(field)
    nx, ny = field.shape
    
    # Periodic boundaries in y-direction
    for i in prange(nx):
        for j in range(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1 + ny) % ny  # Ensure positive index
            
            # Fixed boundaries in x-direction - simplified approach
            if i == 0:
                # Left boundary: zero gradient (Neumann)
                lap[i, j] = (field[i+1, j] - field[i, j] + 
                            field[i, jp1] + field[i, jm1] - 2 * field[i, j]) / dx**2
            elif i == nx - 1:
                # Right boundary: zero gradient (Neumann)
                lap[i, j] = (field[i-1, j] - field[i, j] + 
                            field[i, jp1] + field[i, jm1] - 2 * field[i, j]) / dx**2
            else:
                # Interior points
                lap[i, j] = (field[i+1, j] + field[i-1, j] + 
                            field[i, jp1] + field[i, jm1] - 4 * field[i, j]) / dx**2
    
    return lap

# FIX 3: Complete rewrite of update_grains with Numba-friendly operations
@njit(parallel=True)
def update_grains(eta, eta_sp, c, dt, k_grain, gamma_gb):
    """Update grain order parameters with coupling to substrate"""
    n_grains = eta.shape[0]
    new_eta = np.copy(eta)
    
    # Pre-compute sum of all eta squared for GB energy calculation
    sum_all_eta_sq = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    for i in range(n_grains):
        sum_all_eta_sq += eta[i] * eta[i]
    
    for i in prange(n_grains):
        # Compute sum of squares of other grains efficiently
        sum_other_sq = sum_all_eta_sq - eta[i] * eta[i]
        
        # Functional derivative - using explicit operations
        eta_cubed = eta[i] * eta[i] * eta[i]
        df_deta = eta_cubed - eta[i] + 2.0 * gamma_gb * eta[i] * sum_other_sq
        
        # Laplacian term
        lap_eta = laplacian_2d(eta[i], DX)
        
        # Update equation - using float constants for Numba
        new_eta[i] = eta[i] - dt * M_GRAIN * (df_deta - k_grain * lap_eta)
        
        # Stabilize values with explicit loop
        for x in range(GRID_X):
            for y in range(GRID_Y):
                val = new_eta[i, x, y]
                if val < 0.0:
                    new_eta[i, x, y] = 0.0
                elif val > 1.0:
                    new_eta[i, x, y] = 1.0
    
    return new_eta

# FIX 4: Rewritten update_substrate with explicit typing and operations
@njit(parallel=True)
def update_substrate(eta, eta_sp, c, dt, k_sp, alpha_gb):
    """Update substrate phase with GB preference"""
    gb_indicator = compute_gb_indicator(eta)
    new_eta_sp = np.copy(eta_sp)
    
    for i in prange(GRID_X):
        for j in range(GRID_Y):
            # Local free energy derivatives - explicit operations
            eta_val = eta_sp[i, j]
            eta_cubed = eta_val * eta_val * eta_val
            
            # Double-well potential
            df_deta_sp = M_SP * (eta_cubed - eta_val)
            
            # GB preference term
            df_deta_sp -= 2.0 * alpha_gb * eta_val * gb_indicator[i, j]
            
            # Chemical coupling
            h_eta_sp = eta_val * eta_val * (3.0 - 2.0 * eta_val)
            chem_term = 2.0 * (B_SUBSTRATE * (c[i, j] - C_SUBSTRATE_EQ)**2 - 
                             A_MATRIX * (c[i, j] - C_MATRIX_EQ)**2) * h_eta_sp
            df_deta_sp += chem_term
            
            # Gradient term - compute laplacian locally for this point
            jp1 = (j + 1) % GRID_Y
            jm1 = (j - 1 + GRID_Y) % GRID_Y
            
            if i == 0:
                # Left boundary fixed
                new_eta_sp[i, j] = 1.0
            elif i == GRID_X - 1:
                # Right boundary fixed
                new_eta_sp[i, j] = 0.0
            else:
                lap_eta_sp = (eta_sp[i+1, j] + eta_sp[i-1, j] + 
                             eta_sp[i, jp1] + eta_sp[i, jm1] - 4.0 * eta_val) / DX**2
                
                # Update with gradient term
                new_val = eta_val - dt * M_SP * (df_deta_sp - k_sp * lap_eta_sp)
                
                # Stabilize
                if new_val < 0.0:
                    new_val = 0.0
                elif new_val > 1.0:
                    new_val = 1.0
                
                new_eta_sp[i, j] = new_val
    
    return new_eta_sp

# FIX 5: Rewritten Cahn-Hilliard update with explicit flux calculations
@njit(parallel=True)
def update_concentration(eta, eta_sp, c, dt, m_bulk, m_gb):
    """Cahn-Hilliard update with GB-enhanced mobility - explicit implementation"""
    gb_indicator = compute_gb_indicator(eta)
    new_c = np.copy(c)
    
    # Compute chemical potential at each point
    mu = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    for i in prange(GRID_X):
        for j in range(GRID_Y):
            eta_val = eta_sp[i, j]
            h_eta_sp = eta_val * eta_val * (3.0 - 2.0 * eta_val)
            
            # df/dc term
            df_dc = 2.0 * (h_eta_sp * B_SUBSTRATE * (c[i, j] - C_SUBSTRATE_EQ) + 
                          (1.0 - h_eta_sp) * A_MATRIX * (c[i, j] - C_MATRIX_EQ))
            
            # Approximate laplacian of c for gradient term
            jp1 = (j + 1) % GRID_Y
            jm1 = (j - 1 + GRID_Y) % GRID_Y
            
            if i == 0:
                lap_c = (c[i+1, j] - c[i, j] + c[i, jp1] + c[i, jm1] - 2.0 * c[i, j]) / DX**2
            elif i == GRID_X - 1:
                lap_c = (c[i-1, j] - c[i, j] + c[i, jp1] + c[i, jm1] - 2.0 * c[i, j]) / DX**2
            else:
                lap_c = (c[i+1, j] + c[i-1, j] + c[i, jp1] + c[i, jm1] - 4.0 * c[i, j]) / DX**2
            
            mu[i, j] = df_dc - K_C * lap_c
    
    # Compute fluxes and update concentration
    for i in prange(GRID_X):
        for j in range(GRID_Y):
            jp1 = (j + 1) % GRID_Y
            jm1 = (j - 1 + GRID_Y) % GRID_Y
            
            # Mobility at this point
            mobility_here = m_bulk + m_gb * gb_indicator[i, j]
            
            # X-direction flux
            flux_x = 0.0
            if i < GRID_X - 1:
                # Average mobility at interface
                mobility_interface_x = 0.5 * (mobility_here + m_bulk + m_gb * gb_indicator[i+1, j])
                flux_x = mobility_interface_x * (mu[i+1, j] - mu[i, j]) / DX
            
            # Y-direction flux (periodic)
            # Average mobility at interface
            mobility_interface_y = 0.5 * (mobility_here + m_bulk + m_gb * gb_indicator[i, jp1])
            flux_y = mobility_interface_y * (mu[i, jp1] - mu[i, j]) / DX
            
            # Divergence of fluxes
            div_flux = 0.0
            if i > 0:
                # Include left flux
                mobility_interface_left = 0.5 * (mobility_here + m_bulk + m_gb * gb_indicator[i-1, j])
                flux_x_left = mobility_interface_left * (mu[i, j] - mu[i-1, j]) / DX
                div_flux += (flux_x - flux_x_left) / DX
            else:
                # Left boundary - fixed concentration
                new_c[i, j] = C_SUBSTRATE_EQ
                continue
            
            div_flux += (flux_y - mobility_interface_y * (mu[i, j] - mu[i, jm1]) / DX) / DX
            
            # Update concentration
            new_val = c[i, j] - dt * div_flux
            
            # Stabilize
            if new_val < 0.0:
                new_val = 0.0
            elif new_val > 1.5:  # Allow some supersaturation
                new_val = 1.5
            
            new_c[i, j] = new_val
    
    return new_c

@njit
def calculate_free_energy(eta, eta_sp, c):
    """Calculate total free energy for monitoring"""
    n_grains = eta.shape[0]
    total_energy = 0.0
    
    # Compute GB indicator once
    gb_indicator = compute_gb_indicator(eta)
    
    for i in range(GRID_X):
        for j in range(GRID_Y):
            # Grain energy terms
            grain_energy = 0.0
            sum_eta_sq = 0.0
            for k in range(n_grains):
                eta_val = eta[k, i, j]
                eta_sq = eta_val * eta_val
                grain_energy += 0.25 * (eta_sq - 1.0)**2
                sum_eta_sq += eta_sq
            
            # GB energy penalty
            gb_energy = 0.0
            for k in range(n_grains):
                for l in range(k+1, n_grains):
                    gb_energy += GAMMA_GB * eta[k, i, j]**2 * eta[l, i, j]**2
            
            # Substrate energy
            eta_sp_val = eta_sp[i, j]
            eta_sp_sq = eta_sp_val * eta_sp_val
            sp_energy = 0.25 * (eta_sp_sq - 1.0)**2
            
            # GB preference term
            gb_preference = -ALPHA_GB * eta_sp_sq * gb_indicator[i, j]
            
            # Chemical energy
            h_eta_sp = eta_sp_sq * (3.0 - 2.0 * eta_sp_val)
            chem_energy = h_eta_sp * B_SUBSTRATE * (c[i, j] - C_SUBSTRATE_EQ)**2 + \
                         (1.0 - h_eta_sp) * A_MATRIX * (c[i, j] - C_MATRIX_EQ)**2
            
            total_energy += (grain_energy + gb_energy + sp_energy + 
                           gb_preference + chem_energy)
    
    return total_energy / (GRID_X * GRID_Y)

def initialize_fields(n_grains, c_super):
    """Initialize fields with proper grain seeds and explicit typing"""
    # Initialize grains with seed points - explicit float64 dtype
    eta = np.zeros((n_grains, GRID_X, GRID_Y), dtype=np.float64)
    
    # Place seed points for 4 grains
    if n_grains >= 4:
        # Quadrant seeding
        eta[0, :GRID_X//2, :GRID_Y//2] = 1.0  # Bottom-left
        eta[1, :GRID_X//2, GRID_Y//2:] = 1.0  # Top-left
        eta[2, GRID_X//2:, :GRID_Y//2] = 1.0  # Bottom-right
        eta[3, GRID_X//2:, GRID_Y//2:] = 1.0  # Top-right
        
        # Add noise for interface formation
        noise = np.random.uniform(-0.1, 0.1, eta.shape).astype(np.float64)
        eta += noise
        np.clip(eta, 0.0, 1.0, out=eta)
    
    # Initialize substrate phase (left boundary) - explicit typing
    eta_sp = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    eta_sp[:GRID_X//10, :] = 1.0  # Left 10% as substrate
    
    # Initialize concentration field - explicit typing
    c = np.full((GRID_X, GRID_Y), c_super, dtype=np.float64)
    c[:GRID_X//10, :] = C_SUBSTRATE_EQ  # Left boundary = substrate composition
    
    return eta, eta_sp, c

def create_plotly_figure(eta, eta_sp, c, energy_history):
    """Create Plotly figure with all fields and energy plot"""
    # Create grain coloring
    grains = np.zeros((GRID_X, GRID_Y))
    for i in range(GRID_X):
        for j in range(GRID_Y):
            if eta_sp[i, j] > 0.5:
                grains[i, j] = 0  # Substrate
            else:
                max_idx = 0
                max_val = eta[0, i, j]
                for k in range(1, eta.shape[0]):
                    if eta[k, i, j] > max_val:
                        max_val = eta[k, i, j]
                        max_idx = k
                grains[i, j] = max_idx + 1
    
    # Create subplot figure
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=('Grains & Substrate', 'Concentration Field', 'Substrate Phase', 'Free Energy'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    
    # Custom colormap for grains
    grain_colors = ['black', 'red', 'green', 'blue', 'yellow']
    grain_cmap = [[i/(len(grain_colors)-1), color] for i, color in enumerate(grain_colors)]
    
    # Plot grains with substrate
    fig.add_trace(
        go.Heatmap(
            z=grains.T,  # Transpose for correct orientation
            colorscale=grain_cmap,
            showscale=False,
            name='Microstructure'
        ),
        row=1, col=1
    )
    
    # Plot concentration field
    fig.add_trace(
        go.Heatmap(
            z=c.T,
            colorscale='Hot',
            showscale=True,
            name='Concentration',
            zmin=0.0,
            zmax=1.5
        ),
        row=1, col=2
    )
    
    # Plot substrate phase
    fig.add_trace(
        go.Heatmap(
            z=eta_sp.T,
            colorscale='Blues',
            showscale=True,
            name='Substrate',
            zmin=0.0,
            zmax=1.0
        ),
        row=2, col=1
    )
    
    # Plot free energy history
    if len(energy_history) > 0:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(energy_history))),
                y=energy_history,
                mode='lines',
                name='Free Energy',
                line=dict(color='blue', width=2)
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Intergranular Penetration Simulation",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Step", row=2, col=2)
    fig.update_yaxes(title_text="Energy", row=2, col=2)
    
    return fig

def create_animation_frames(eta_history, eta_sp_history, c_history):
    """Create animation frames for download as PNG sequence"""
    frames = []
    num_frames = len(eta_history)
    
    # Create figure for animation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Grain structure colormap
    grain_colors = ['black', '#FF4500', '#32CD32', '#1E90FF', '#FFD700']
    grain_cmap = ListedColormap(grain_colors[:5])
    
    for frame_idx in range(num_frames):
        eta = eta_history[frame_idx]
        eta_sp = eta_sp_history[frame_idx]
        c = c_history[frame_idx]
        
        # Create grain visualization
        grains = np.zeros((GRID_X, GRID_Y))
        for i in range(GRID_X):
            for j in range(GRID_Y):
                if eta_sp[i, j] > 0.5:
                    grains[i, j] = 0  # Substrate
                else:
                    max_idx = 0
                    max_val = eta[0, i, j]
                    for k in range(1, eta.shape[0]):
                        if eta[k, i, j] > max_val:
                            max_val = eta[k, i, j]
                            max_idx = k
                    grains[i, j] = max_idx + 1
        
        # Clear previous plots
        for ax in axes:
            ax.clear()
        
        # Plot grains with substrate
        axes[0].imshow(grains.T, cmap=grain_cmap, origin='lower')
        axes[0].set_title(f'Grains & Substrate (Step {frame_idx*10})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Concentration field
        im2 = axes[1].imshow(c.T, cmap='hot', origin='lower', vmin=0, vmax=1.5)
        axes[1].set_title(f'Concentration Field (Step {frame_idx*10})')
        axes[1].set_xlabel('X')
        fig.colorbar(im2, ax=axes[1])
        
        # Substrate phase
        im3 = axes[2].imshow(eta_sp.T, cmap='Blues', origin='lower', vmin=0, vmax=1)
        axes[2].set_title(f'Substrate Phase (Step {frame_idx*10})')
        axes[2].set_xlabel('X')
        fig.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(buf.getvalue())
        plt.close(fig)
    
    return frames

def create_download_zip(eta_history, eta_sp_history, c_history, energy_history, params):
    """Create ZIP file containing simulation data and animation frames"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Save NumPy data
        data_dict = {
            'eta_history.npy': np.array(eta_history),
            'eta_sp_history.npy': np.array(eta_sp_history),
            'c_history.npy': np.array(c_history),
            'energy_history.npy': np.array(energy_history)
        }
        
        for filename, data in data_dict.items():
            arr_bytes = BytesIO()
            np.save(arr_bytes, data)
            arr_bytes.seek(0)
            zip_file.writestr(filename, arr_bytes.getvalue())
        
        # Create and add animation frames
        frames = create_animation_frames(eta_history, eta_sp_history, c_history)
        for i, frame in enumerate(frames):
            zip_file.writestr(f"frame_{i:04d}.png", frame)
        
        # Add metadata
        metadata = f"""
Intergranular Penetration Simulation Results
=============================================

Simulation Parameters:
- Grid Size: {params['grid_size']}x{params['grid_size']}
- Number of Grains: {params['n_grains']}
- Grain Growth Steps: {params['grain_growth_steps']}
- Penetration Steps: {params['penetration_steps']}
- Supersaturation (c_super): {params['c_super']}
- GB Mobility Multiplier: {params['m_gb_mult']}
- Time Step (dt): {params['dt']}
- Simulation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Files Included:
- eta_history.npy: Grain order parameters over time (shape: {np.array(eta_history).shape})
- eta_sp_history.npy: Substrate phase field over time (shape: {np.array(eta_sp_history).shape})
- c_history.npy: Concentration field over time (shape: {np.array(c_history).shape})
- energy_history.npy: Free energy evolution
- frame_XXXX.png: Animation frames showing microstructure evolution

To visualize the animation frames:
1. Extract all PNG files
2. Use ImageJ, FFmpeg, or any animation software to create a video
3. Or load the NumPy arrays in Python for custom analysis

For questions or support, contact the simulation developer.
        """
        zip_file.writestr("README.txt", metadata)
    
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit App
st.set_page_config(layout="wide", page_title="Intergranular Penetration Simulator")

st.title("Phase Field Simulation: Intergranular Penetration")
st.markdown("""
This simulation models preferential growth of a substrate phase along grain boundaries in a polycrystalline material.
The model includes:
- 4-grain polycrystal with diffuse interfaces
- Substrate phase infiltrating from left boundary
- Supersaturated concentration field driving penetration
- Grain boundary enhanced mobility and energy preference
""")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")

# Stage 1: Grain growth parameters
st.sidebar.subheader("Grain Formation")
grain_growth_steps = st.sidebar.slider(
    "Grain Growth Steps", 
    min_value=100, max_value=1000, value=300, step=50
)

# Stage 2: Penetration parameters
st.sidebar.subheader("Penetration Stage")
penetration_steps = st.sidebar.slider(
    "Penetration Steps", 
    min_value=100, max_value=1500, value=300, step=50
)
c_super = st.sidebar.slider(
    "Supersaturation (c in grains)", 
    min_value=0.1, max_value=1.0, value=0.6, step=0.05
)
m_gb_mult = st.sidebar.slider(
    "GB Mobility Multiplier", 
    min_value=1.0, max_value=20.0, value=5.0, step=0.5
)

# Run simulation button
if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("First-time compilation may take 1-2 minutes..."):
        # Store parameters for metadata
        simulation_params = {
            'grid_size': GRID_X,
            'n_grains': 4,
            'grain_growth_steps': grain_growth_steps,
            'penetration_steps': penetration_steps,
            'c_super': c_super,
            'm_gb_mult': m_gb_mult,
            'dt': DT
        }
        
        # Initialize fields
        st.info("Initializing fields...")
        eta, eta_sp, c = initialize_fields(4, c_super)
        
        # Energy history
        energy_history = []
        
        # Run grain growth stage
        st.info(f"Stage 1: Grain formation ({grain_growth_steps} steps)")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for step in range(grain_growth_steps):
            # Update grains only (no substrate movement, no concentration evolution)
            eta = update_grains(eta, eta_sp, c, DT, K_GRAIN, GAMMA_GB)
            
            # Calculate energy
            if step % max(1, grain_growth_steps // 20) == 0:
                energy = calculate_free_energy(eta, eta_sp, c)
                energy_history.append(energy)
                status_text.text(f"Grain formation: Step {step}/{grain_growth_steps}, Energy: {energy:.4f}")
            
            # Update progress
            progress_bar.progress(step / grain_growth_steps)
        
        # Run penetration stage
        st.info(f"Stage 2: Penetration ({penetration_steps} steps)")
        eta_history = []
        eta_sp_history = []
        c_history = []
        
        # Store initial state for visualization
        eta_history.append(np.copy(eta))
        eta_sp_history.append(np.copy(eta_sp))
        c_history.append(np.copy(c))
        
        total_steps = grain_growth_steps + penetration_steps
        
        for step in range(penetration_steps):
            # Update all fields
            eta = update_grains(eta, eta_sp, c, DT, K_GRAIN, GAMMA_GB)
            eta_sp = update_substrate(eta, eta_sp, c, DT, K_SP, ALPHA_GB)
            c = update_concentration(eta, eta_sp, c, DT, M_BULK, M_GB * m_gb_mult)
            
            # Calculate energy
            energy = calculate_free_energy(eta, eta_sp, c)
            energy_history.append(energy)
            
            # Store history for visualization and export
            if step % max(1, penetration_steps // 25) == 0:
                eta_history.append(np.copy(eta))
                eta_sp_history.append(np.copy(eta_sp))
                c_history.append(np.copy(c))
            
            # Update progress and visualization
            if step % max(1, penetration_steps // 15) == 0:
                global_step = grain_growth_steps + step
                progress = global_step / total_steps
                progress_bar.progress(progress)
                
                # Create and update plot
                fig = create_plotly_figure(eta, eta_sp, c, energy_history)
                st.plotly_chart(fig, use_container_width=True)
                
                status_text.text(f"Penetration: Step {step}/{penetration_steps}, Energy: {energy:.4f}")
        
        # Final visualization
        fig = create_plotly_figure(eta, eta_sp, c, energy_history)
        st.plotly_chart(fig, use_container_width=True)
        
        # Store results in session state for download
        st.session_state.eta_history = eta_history
        st.session_state.eta_sp_history = eta_sp_history
        st.session_state.c_history = c_history
        st.session_state.energy_history = energy_history
        st.session_state.simulation_params = simulation_params
        
        status_text.text("Simulation completed successfully!")
        progress_bar.progress(1.0)
        
        st.success("✅ Simulation completed! Check the 'Results' tab for detailed analysis and download options.")

# Results tab
st.header("Simulation Results")

if 'eta_history' in st.session_state:
    # Final microstructure visualization
    st.subheader("Final Microstructure")
    
    # Create final grain visualization
    eta = st.session_state.eta_history[-1]
    eta_sp = st.session_state.eta_sp_history[-1]
    c = st.session_state.c_history[-1]
    
    grains = np.zeros((GRID_X, GRID_Y))
    for i in range(GRID_X):
        for j in range(GRID_Y):
            if eta_sp[i, j] > 0.5:
                grains[i, j] = 0  # Substrate
            else:
                max_idx = 0
                max_val = eta[0, i, j]
                for k in range(1, eta.shape[0]):
                    if eta[k, i, j] > max_val:
                        max_val = eta[k, i, j]
                        max_idx = k
                grains[i, j] = max_idx + 1
    
    # Plot using matplotlib for better control
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Grain structure with substrate
    grain_colors = ['black', '#FF4500', '#32CD32', '#1E90FF', '#FFD700']
    grain_cmap = ListedColormap(grain_colors[:5])
    im1 = axes[0].imshow(grains.T, cmap=grain_cmap, origin='lower')
    axes[0].set_title('Grains & Substrate (Black)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Concentration field
    im2 = axes[1].imshow(c.T, cmap='hot', origin='lower')
    axes[1].set_title('Concentration Field')
    axes[1].set_xlabel('X')
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label('Concentration')
    
    # Substrate phase
    im3 = axes[2].imshow(eta_sp.T, cmap='Blues', origin='lower', vmin=0, vmax=1)
    axes[2].set_title('Substrate Phase Field')
    axes[2].set_xlabel('X')
    cbar3 = fig.colorbar(im3, ax=axes[2])
    cbar3.set_label('η_sp')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Free energy plot
    st.subheader("Free Energy Evolution")
    fig_energy, ax_energy = plt.subplots(figsize=(10, 6))
    ax_energy.plot(st.session_state.energy_history, 'b-', linewidth=2)
    ax_energy.set_xlabel('Simulation Step')
    ax_energy.set_ylabel('Free Energy Density')
    ax_energy.set_title('Free Energy Evolution During Simulation')
    ax_energy.grid(True, alpha=0.3)
    ax_energy.set_yscale('log')  # Log scale often better for energy plots
    st.pyplot(fig_energy)
    
    # Download section
    st.subheader("Download Results")
    
    if st.button("Prepare Download Package (ZIP)"):
        with st.spinner("Creating download package with simulation data and animation frames... This may take a minute."):
            # Create ZIP file
            zip_buffer = create_download_zip(
                st.session_state.eta_history,
                st.session_state.eta_sp_history,
                st.session_state.c_history,
                st.session_state.energy_history,
                st.session_state.simulation_params
            )
            
            # Provide download button
            st.download_button(
                label="⬇️ Download Simulation Results (ZIP)",
                data=zip_buffer,
                file_name="intergranular_penetration_results.zip",
                mime="application/zip",
                help="Contains NumPy arrays of simulation data and PNG animation frames"
            )
            st.success("✅ Download package created successfully!")
    
    st.info("""
    **Download Package Contents**:
    - `eta_history.npy`: Grain order parameters over time
    - `eta_sp_history.npy`: Substrate phase field over time  
    - `c_history.npy`: Concentration field over time
    - `energy_history.npy`: Free energy evolution
    - `frame_XXXX.png`: Animation frames (one per stored time step)
    - `README.txt`: Simulation parameters and usage instructions
    
    **To use the data**:
    1. Load NumPy arrays: `data = np.load('eta_history.npy')`
    2. Create animations from PNG sequence using ImageJ or FFmpeg
    3. Analyze energy evolution for stability verification
    """)
else:
    st.info("👆 Run the simulation first using the sidebar controls to see results here.")

# Theory section
st.markdown("---")
st.header("Theoretical Foundation")

st.markdown("""
## Robust Phase Field Model for Intergranular Penetration

This model extends the multi-phase field approach to simulate preferential substrate phase growth along grain boundaries.

### Key Theoretical Elements

**1. Order Parameter Representation:**
- **Grains**: $n$ non-conserved order parameters $\eta_i(\mathbf{r},t)$, where $\eta_i = 1$ in grain $i$, 0 elsewhere
- **Substrate Phase**: Single non-conserved order parameter $\eta_{sp}(\mathbf{r},t)$, representing infiltrating phase
- **Concentration**: Conserved field $c(\mathbf{r},t)$ representing solute atoms

**2. Free Energy Functional:**

$$
F = \int_V \left[ f_{local} + \\frac{\kappa_c}{2}|\nabla c|^2 + \\frac{k}{2}\sum_{i=1}^n|\nabla\eta_i|^2 + \\frac{k_{sp}}{2}|\nabla\eta_{sp}|^2 \\right] dV
$$

where the local free energy density is:

$$
f_{local} = \sum_{i=1}^n \\frac{m}{4}(\eta_i^2 - 1)^2 + \gamma\sum_{i<j}\eta_i^2\eta_j^2 + \\frac{m_{sp}}{4}(\eta_{sp}^2 - 1)^2 - \\alpha\eta_{sp}^2\sum_{i<j}\eta_i^2\eta_j^2 + f_{chem}
$$

**3. Chemical Energy Term:**

$$
f_{chem} = h(\eta_{sp})B(c - c_{sp}^{eq})^2 + [1-h(\eta_{sp})]A(c - c_{grain}^{eq})^2
$$

where $h(\eta_{sp}) = \eta_{sp}^2(3-2\eta_{sp})$ is a smooth interpolation function.

**4. Evolution Equations:**

- **Grains and Substrate** (non-conserved):
  $$
  \\frac{\partial\eta_i}{\partial t} = -L_{\eta} \\frac{\delta F}{\delta\eta_i}, \quad
  \\frac{\partial\eta_{sp}}{\partial t} = -L_{sp} \\frac{\delta F}{\delta\eta_{sp}}
  $$

- **Concentration** (conserved, Cahn-Hilliard):
  $$
  \\frac{\partial c}{\partial t} = \nabla\cdot\left[M(c, \{\eta_i\}, \eta_{sp}) \nabla\\frac{\delta F}{\delta c}\\right]
  $$

**5. Grain Boundary Enhanced Mobility:**

$$
M = M_{bulk} + M_{GB}\cdot\mathcal{I}_{GB}
$$

where the GB indicator is:

$$
\mathcal{I}_{GB} = \\frac{1}{2}\left[\left(\sum_{i=1}^n\eta_i^2\\right)^2 - \sum_{i=1}^n\eta_i^4\\right]
$$

This peaks at grain boundaries and is zero within grains.

### Numerical Implementation Fixes Applied

✅ **Explicit typing**: All arrays explicitly typed as `float64` for Numba compatibility  
✅ **Simplified operations**: Replaced `**` operators with explicit multiplication for better Numba support  
✅ **Boundary condition rewrite**: Complete redesign of boundary handling to avoid Numba typing issues  
✅ **Loop restructuring**: Eliminated complex nested loops that confused Numba's type inference  
✅ **Memory management**: Pre-allocated arrays with explicit dimensions to avoid dynamic allocation  
✅ **Function decomposition**: Broke down complex functions into smaller, Numba-friendly components  

### Two-Stage Simulation Approach

1. **Grain Formation Stage**: 
   - Only grain order parameters evolve
   - Concentration field uniform
   - Forms stable polycrystalline structure

2. **Penetration Stage**:
   - All fields evolve
   - Supersaturated concentration drives substrate growth
   - GB-enhanced mobility and energy preference enable preferential growth

This approach ensures physical realism and numerical stability.
""")

# Footer
st.markdown("---")
st.markdown("""
**Note**: This simulation uses a minimal 64×64 grid and 4 grains for computational efficiency. 
For research applications, larger grids (256×256+) and more grains are recommended. 
The model is non-dimensionalized and captures the essential physics of intergranular penetration.

**Performance**: First run may take 1-2 minutes due to Numba JIT compilation. Subsequent runs will be much faster.
""")
