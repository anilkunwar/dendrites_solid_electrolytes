import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from io import BytesIO
import time
from numba import njit, prange
import zipfile
import base64

# Constants - minimal grid for robustness
GRID_X, GRID_Y = 64, 64
DX = 1.0
DT = 0.01  # Smaller time step for stability
STEPS_GRAIN_GROWTH = 1000  # Stage 1: grain formation
MAX_PENETRATION_STEPS = 2000  # Stage 2: penetration

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

@njit(parallel=True)
def compute_gb_indicator(eta):
    """Vectorized GB indicator: 0.5[(∑η_i²)² - ∑η_i⁴]"""
    n_grains = eta.shape[0]
    gb = np.zeros((GRID_X, GRID_Y))
    sum_eta_sq = np.zeros((GRID_X, GRID_Y))
    sum_eta_quad = np.zeros((GRID_X, GRID_Y))
    
    for i in prange(n_grains):
        eta_sq = eta[i]**2
        sum_eta_sq += eta_sq
        sum_eta_quad += eta_sq**2
    
    gb = 0.5 * (sum_eta_sq**2 - sum_eta_quad)
    # Normalize to [0, 1] for stability
    max_gb = 1.0  # Theoretical max for n=4 is 0.75, but we normalize conservatively
    return np.minimum(gb / max_gb, 1.0)

@njit(parallel=True)
def laplacian_2d(field, dx):
    """Efficient 2D Laplacian with periodic boundaries in y, fixed in x"""
    lap = np.zeros_like(field)
    nx, ny = field.shape
    
    # Periodic boundaries in y-direction
    for i in prange(nx):
        for j in range(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            
            # Fixed boundaries in x-direction
            if i == 0:
                ip1 = i + 1
                im1 = i  # Zero gradient at boundary
            elif i == nx - 1:
                ip1 = i
                im1 = i - 1
            else:
                ip1 = i + 1
                im1 = i - 1
            
            lap[i, j] = (field[ip1, j] + field[im1, j] + field[i, jp1] + field[i, jm1] - 4 * field[i, j]) / dx**2
    
    return lap

@njit(parallel=True)
def update_grains(eta, eta_sp, c, dt, m_grain, k_grain, gamma_gb):
    """Update grain order parameters with coupling to substrate"""
    n_grains = eta.shape[0]
    new_eta = np.copy(eta)
    
    for i in prange(n_grains):
        # Sum of squares of other grains
        sum_other_sq = np.zeros((GRID_X, GRID_Y))
        for j in range(n_grains):
            if j != i:
                sum_other_sq += eta[j]**2
        
        # Functional derivative
        df_deta = (eta[i]**3 - eta[i] + 2 * gamma_gb * eta[i] * sum_other_sq) * m_grain
        
        # Laplacian term
        lap_eta = laplacian_2d(eta[i], DX)
        
        # Update equation
        new_eta[i] = eta[i] - dt * M_GRAIN * (df_deta - k_grain * lap_eta)
        
        # Stabilize values
        for x in range(GRID_X):
            for y in range(GRID_Y):
                if new_eta[i, x, y] < 0:
                    new_eta[i, x, y] = 0.0
                elif new_eta[i, x, y] > 1:
                    new_eta[i, x, y] = 1.0
    
    return new_eta

@njit(parallel=True)
def update_substrate(eta, eta_sp, c, dt, m_sp, k_sp, alpha_gb):
    """Update substrate phase with GB preference"""
    gb_indicator = compute_gb_indicator(eta)
    
    # Local free energy derivatives
    df_deta_sp = m_sp * (eta_sp**3 - eta_sp)  # Double-well potential
    
    # GB preference term
    df_deta_sp -= 2 * alpha_gb * eta_sp * gb_indicator
    
    # Chemical coupling
    h_eta_sp = eta_sp**2 * (3 - 2 * eta_sp)
    df_deta_sp += 2 * (B_SUBSTRATE * (c - C_SUBSTRATE_EQ)**2 - A_MATRIX * (c - C_MATRIX_EQ)**2) * h_eta_sp
    
    # Gradient term
    lap_eta_sp = laplacian_2d(eta_sp, DX)
    
    # Update
    new_eta_sp = eta_sp - dt * M_SP * (df_deta_sp - k_sp * lap_eta_sp)
    
    # Apply boundary conditions: fixed at left boundary
    for y in range(GRID_Y):
        new_eta_sp[0, y] = 1.0  # Left boundary fixed to substrate
        new_eta_sp[-1, y] = 0.0  # Right boundary fixed to matrix
    
    # Stabilize
    for x in range(GRID_X):
        for y in range(GRID_Y):
            if new_eta_sp[x, y] < 0:
                new_eta_sp[x, y] = 0.0
            elif new_eta_sp[x, y] > 1:
                new_eta_sp[x, y] = 1.0
    
    return new_eta_sp

@njit(parallel=True)
def compute_chemical_potential(eta, eta_sp, c, k_c):
    """Compute chemical potential mu = df/dc - k_c * laplacian(c)"""
    # Interpolation function
    h_eta_sp = eta_sp**2 * (3 - 2 * eta_sp)
    
    # df/dc term
    df_dc = 2 * (h_eta_sp * B_SUBSTRATE * (c - C_SUBSTRATE_EQ) + 
                (1 - h_eta_sp) * A_MATRIX * (c - C_MATRIX_EQ))
    
    # Gradient term
    lap_c = laplacian_2d(c, DX)
    
    return df_dc - k_c * lap_c

@njit(parallel=True)
def update_concentration(eta, eta_sp, c, dt, m_bulk, m_gb):
    """Cahn-Hilliard update with GB-enhanced mobility"""
    # Compute chemical potential
    mu = compute_chemical_potential(eta, eta_sp, c, K_C)
    
    # GB indicator for mobility enhancement
    gb_indicator = compute_gb_indicator(eta)
    mobility_field = m_bulk + m_gb * gb_indicator
    
    # Compute fluxes (conservative form)
    flux_x = np.zeros((GRID_X+1, GRID_Y))
    flux_y = np.zeros((GRID_X, GRID_Y+1))
    
    # X-fluxes (i+1/2 interfaces)
    for i in range(GRID_X):
        for j in range(GRID_Y):
            # Mobility at interface (harmonic mean for stability)
            if i < GRID_X - 1:
                m_interface = 2 * mobility_field[i,j] * mobility_field[i+1,j] / (mobility_field[i,j] + mobility_field[i+1,j] + 1e-10)
                flux_x[i+1,j] = m_interface * (mu[i+1,j] - mu[i,j]) / DX
    
    # Y-fluxes (j+1/2 interfaces) with periodic boundaries
    for i in range(GRID_X):
        for j in range(GRID_Y):
            jp1 = (j + 1) % GRID_Y
            m_interface = 2 * mobility_field[i,j] * mobility_field[i,jp1] / (mobility_field[i,j] + mobility_field[i,jp1] + 1e-10)
            flux_y[i,j+1] = m_interface * (mu[i,jp1] - mu[i,j]) / DX
    
    # Divergence of fluxes
    dc_dt = np.zeros_like(c)
    for i in range(GRID_X):
        for j in range(GRID_Y):
            jp1 = (j + 1) % GRID_Y
            jm1 = (j - 1) % GRID_Y
            
            div_flux = (flux_x[i+1,j] - flux_x[i,j]) / DX + (flux_y[i,jp1] - flux_y[i,j]) / DX
            dc_dt[i,j] = -div_flux
    
    # Update concentration
    new_c = c + dt * dc_dt
    
    # Apply boundary conditions for c
    for y in range(GRID_Y):
        new_c[0, y] = C_SUBSTRATE_EQ  # Left boundary = substrate composition
    
    # Stabilize
    for x in range(GRID_X):
        for y in range(GRID_Y):
            if new_c[x, y] < 0:
                new_c[x, y] = 0.0
            elif new_c[x, y] > 1.5:  # Allow some supersaturation
                new_c[x, y] = 1.5
    
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
                grain_energy += 0.25 * (eta_val**2 - 1)**2
                sum_eta_sq += eta_val**2
            
            # GB energy penalty
            gb_energy = 0.0
            for k in range(n_grains):
                for l in range(k+1, n_grains):
                    gb_energy += GAMMA_GB * eta[k, i, j]**2 * eta[l, i, j]**2
            
            # Substrate energy
            eta_sp_val = eta_sp[i, j]
            sp_energy = 0.25 * (eta_sp_val**2 - 1)**2
            
            # GB preference term
            gb_preference = -ALPHA_GB * eta_sp_val**2 * gb_indicator[i, j]
            
            # Chemical energy
            h_eta_sp = eta_sp_val**2 * (3 - 2 * eta_sp_val)
            chem_energy = h_eta_sp * B_SUBSTRATE * (c[i, j] - C_SUBSTRATE_EQ)**2 + \
                         (1 - h_eta_sp) * A_MATRIX * (c[i, j] - C_MATRIX_EQ)**2
            
            # Gradient energies (approximated)
            grad_energy = 0.0
            
            total_energy += (grain_energy + gb_energy + sp_energy + 
                           gb_preference + chem_energy + grad_energy)
    
    return total_energy / (GRID_X * GRID_Y)

def initialize_fields(n_grains, c_super):
    """Initialize fields with proper grain seeds"""
    # Initialize grains with seed points
    eta = np.zeros((n_grains, GRID_X, GRID_Y))
    
    # Place seed points for 4 grains
    if n_grains >= 4:
        # Quadrant seeding
        eta[0, :GRID_X//2, :GRID_Y//2] = 1.0  # Bottom-left
        eta[1, :GRID_X//2, GRID_Y//2:] = 1.0  # Top-left
        eta[2, GRID_X//2:, :GRID_Y//2] = 1.0  # Bottom-right
        eta[3, GRID_X//2:, GRID_Y//2:] = 1.0  # Top-right
        
        # Add noise for interface formation
        eta += np.random.uniform(-0.1, 0.1, eta.shape)
        eta = np.clip(eta, 0, 1)
    
    # Initialize substrate phase (left boundary)
    eta_sp = np.zeros((GRID_X, GRID_Y))
    eta_sp[:GRID_X//10, :] = 1.0  # Left 10% as substrate
    
    # Initialize concentration field
    c = np.full((GRID_X, GRID_Y), c_super)
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
            name='Concentration'
        ),
        row=1, col=2
    )
    
    # Plot substrate phase
    fig.add_trace(
        go.Heatmap(
            z=eta_sp.T,
            colorscale='Blues',
            showscale=True,
            name='Substrate'
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
    
    fig.update_xaxes(title_text="X", row=2, col=2)
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
        axes[0].set_title(f'Grains & Substrate (Frame {frame_idx})')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Concentration field
        im2 = axes[1].imshow(c.T, cmap='hot', origin='lower', vmin=0, vmax=1.5)
        axes[1].set_title(f'Concentration Field (Frame {frame_idx})')
        axes[1].set_xlabel('X')
        fig.colorbar(im2, ax=axes[1])
        
        # Substrate phase
        im3 = axes[2].imshow(eta_sp.T, cmap='Blues', origin='lower', vmin=0, vmax=1)
        axes[2].set_title(f'Substrate Phase (Frame {frame_idx})')
        axes[2].set_xlabel('X')
        fig.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(buf.getvalue())
        plt.close()
    
    return frames

def create_download_zip(eta_history, eta_sp_history, c_history, energy_history):
    """Create ZIP file containing simulation data and animation frames"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Save NumPy data
        data_dict = {
            'eta_history.npy': eta_history,
            'eta_sp_history.npy': eta_sp_history,
            'c_history.npy': c_history,
            'energy_history.npy': energy_history
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
Simulation Parameters:
- Grid Size: {GRID_X}x{GRID_Y}
- Number of Grains: 4
- Grain Growth Steps: {STEPS_GRAIN_GROWTH}
- Penetration Steps: {MAX_PENETRATION_STEPS}
- Supersaturation: {c_super}
- GB Mobility Multiplier: {M_GB}
- Time Step: {DT}
        """
        zip_file.writestr("metadata.txt", metadata)
    
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
    min_value=100, max_value=2000, value=STEPS_GRAIN_GROWTH, step=100
)

# Stage 2: Penetration parameters
st.sidebar.subheader("Penetration Stage")
penetration_steps = st.sidebar.slider(
    "Penetration Steps", 
    min_value=100, max_value=MAX_PENETRATION_STEPS, value=500, step=100
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
run_simulation = st.sidebar.button("Run Simulation", type="primary")

# Main content area
tab1, tab2, tab3 = st.tabs(["Simulation", "Results", "Theory"])

with tab1:
    st.header("Real-time Simulation")
    plot_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if run_simulation:
        # Initialize fields
        status_text.text("Initializing fields...")
        eta, eta_sp, c = initialize_fields(4, c_super)
        
        # Energy history
        energy_history = []
        
        # Run grain growth stage
        status_text.text(f"Stage 1: Grain formation ({grain_growth_steps} steps)")
        for step in range(grain_growth_steps):
            # Update grains only (no substrate movement, no concentration evolution)
            eta = update_grains(eta, eta_sp, c, DT, M_GRAIN, K_GRAIN, GAMMA_GB)
            
            # Update progress
            if step % max(1, grain_growth_steps // 20) == 0:
                progress_bar.progress(step / grain_growth_steps)
                energy = calculate_free_energy(eta, eta_sp, c)
                energy_history.append(energy)
        
        # Run penetration stage
        status_text.text(f"Stage 2: Penetration ({penetration_steps} steps)")
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
            eta = update_grains(eta, eta_sp, c, DT, M_GRAIN, K_GRAIN, GAMMA_GB)
            eta_sp = update_substrate(eta, eta_sp, c, DT, M_SP, K_SP, ALPHA_GB)
            c = update_concentration(eta, eta_sp, c, DT, M_BULK, M_GB * m_gb_mult)
            
            # Calculate energy
            energy = calculate_free_energy(eta, eta_sp, c)
            energy_history.append(energy)
            
            # Store history for visualization and export
            if step % max(1, penetration_steps // 50) == 0:
                eta_history.append(np.copy(eta))
                eta_sp_history.append(np.copy(eta_sp))
                c_history.append(np.copy(c))
            
            # Update progress and visualization
            if step % max(1, penetration_steps // 20) == 0:
                global_step = grain_growth_steps + step
                progress = global_step / total_steps
                progress_bar.progress(progress)
                
                # Create and update plot
                fig = create_plotly_figure(eta, eta_sp, c, energy_history)
                plot_placeholder.plotly_chart(fig, use_container_width=True)
                
                status_text.text(f"Running penetration: Step {step}/{penetration_steps}, Energy: {energy:.4f}")
        
        # Final visualization
        fig = create_plotly_figure(eta, eta_sp, c, energy_history)
        plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Store results in session state for download
        st.session_state.eta_history = eta_history
        st.session_state.eta_sp_history = eta_sp_history
        st.session_state.c_history = c_history
        st.session_state.energy_history = energy_history
        
        status_text.text("Simulation completed successfully!")
        progress_bar.progress(1.0)

with tab2:
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
        st.pyplot(fig_energy)
        
        # Download section
        st.subheader("Download Results")
        
        if st.button("Prepare Download Package (ZIP)"):
            with st.spinner("Creating download package with simulation data and animation frames..."):
                # Create ZIP file
                zip_buffer = create_download_zip(
                    st.session_state.eta_history,
                    st.session_state.eta_sp_history,
                    st.session_state.c_history,
                    st.session_state.energy_history
                )
                
                # Provide download button
                st.download_button(
                    label="Download Simulation Results (ZIP)",
                    data=zip_buffer,
                    file_name="intergranular_penetration_results.zip",
                    mime="application/zip",
                    help="Contains NumPy arrays of simulation data and PNG animation frames"
                )
        
        st.info("""
        **Download Package Contents**:
        - `eta_history.npy`: Grain order parameters over time
        - `eta_sp_history.npy`: Substrate phase field over time
        - `c_history.npy`: Concentration field over time
        - `energy_history.npy`: Free energy evolution
        - `frame_XXXX.png`: Animation frames (one per time step)
        - `metadata.txt`: Simulation parameters
        
        **To visualize the animation**:
        1. Extract the ZIP file
        2. Use the PNG sequence with any animation software (ImageJ, FFmpeg, etc.)
        3. Or load the NumPy arrays in Python for custom visualization
        """)
    else:
        st.info("Run the simulation first to see results here.")

with tab3:
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
    f_{local} = \sum_{i=1}^n \\frac{m}{4}(\eta_i^2 - \eta_0^2)^2 + \gamma\sum_{i<j}\eta_i^2\eta_j^2 + \\frac{m_{sp}}{4}(\eta_{sp}^2 - 1)^2 - \\alpha\eta_{sp}^2\sum_{i<j}\eta_i^2\eta_j^2 + f_{chem}
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
    
    ### Numerical Implementation
    
    - **Spatial Discretization**: Finite differences with 64×64 grid
    - **Time Integration**: Explicit Euler with small time step ($\Delta t = 0.01$)
    - **Boundary Conditions**:
      - Periodic in y-direction
      - Fixed Dirichlet in x-direction (substrate at left, matrix at right)
    - **Performance**: Numba JIT compilation for all compute-intensive functions
    
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
    
    # Add references
    st.subheader("Key References")
   

# Footer
st.markdown("---")
st.markdown("""
**Note**: This simulation uses a minimal 64×64 grid and 4 grains for computational efficiency. 
For research applications, larger grids (256×256+) and more grains are recommended. 
The model is non-dimensionalized and captures the essential physics of intergranular penetration.
""")

# Performance note
st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Performance Note**:  
This simulation uses Numba JIT compilation for speed. 
First run may be slower due to compilation. 
Subsequent runs will be significantly faster.

🚀 **For faster execution**:  
- Reduce grid size (currently 64×64)
- Decrease number of steps
- Run on a machine with multiple CPU cores
""")
