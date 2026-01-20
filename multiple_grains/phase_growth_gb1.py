import streamlit as st
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
import plotly.graph_objects as go
import plotly.express as px
import zipfile
import io

# Function to compute free energy density (simplified for one grain η1 and substrate η_sp)
def free_energy_density(eta1, eta_sp, gamma=1.5):
    return (eta1**4 / 4 - eta1**2 / 2) + (eta_sp**4 / 4 - eta_sp**2 / 2) + gamma * eta1**2 * eta_sp**2 + 1/4

# Generate Plotly surface for free energy
def plot_free_energy():
    eta1 = np.linspace(-1.5, 1.5, 100)
    eta_sp = np.linspace(-1.5, 1.5, 100)
    Eta1, Eta_sp = np.meshgrid(eta1, eta_sp)
    F = free_energy_density(Eta1, Eta_sp)
    fig = go.Figure(data=[go.Surface(z=F, x=eta1, y=eta_sp, colorscale='Viridis')])
    fig.update_layout(
        title='Local Free Energy Density f(η_grain, η_substrate)',
        scene=dict(xaxis_title='η_grain', yaxis_title='η_substrate', zaxis_title='Free Energy Density')
    )
    return fig

# Simplified 2D phase field simulation (Fourier spectral method for efficiency)
def run_simulation(nx, ny, dt, max_steps, save_interval, L_eta, kappa_eta, gamma, c0, c_eq, M, A):
    # Initialize grids
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize multiple grains (simplified: 2 grains with random boundaries)
    eta1 = np.tanh(5 * (0.5 - np.sqrt((X-0.3)**2 + (Y-0.5)**2)))  # Grain 1
    eta2 = np.tanh(5 * (0.5 - np.sqrt((X-0.7)**2 + (Y-0.5)**2)))  # Grain 2
    eta_grains = [eta1, eta2]  # List of grain etas
    
    # Substrate eta_sp starts near bottom
    eta_sp = np.zeros((ny, nx))
    eta_sp[:10, :] = 1.0  # Initial substrate layer
    
    # Concentration: supersaturated
    c = np.ones((ny, nx)) * c0
    
    # Fourier space setup
    kx = 2 * np.pi * fftfreq(nx, 1/nx)
    ky = 2 * np.pi * fftfreq(ny, 1/ny)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    
    snapshots = []  # List to store time snapshots
    
    for step in range(max_steps):
        # Compute functional derivatives (simplified model)
        sum_eta2 = sum(eta**2 for eta in eta_grains)
        for i in range(len(eta_grains)):
            df_deta_i = eta_grains[i]**3 - eta_grains[i] + 2 * gamma * eta_grains[i] * (sum_eta2 - eta_grains[i]**2)
            eta_hat = fft2(eta_grains[i])
            lap_eta = ifft2(-K2 * eta_hat).real
            delta_eta = -L_eta * (df_deta_i + kappa_eta * lap_eta)
            eta_grains[i] += dt * delta_eta
        
        # Substrate eta_sp evolution
        df_deta_sp = eta_sp**3 - eta_sp + gamma * eta_sp * (2 - sum_eta2) + A * (c - c_eq)**2 * (-2 * (1 - eta_sp))  # Coupling to grains and c
        eta_sp_hat = fft2(eta_sp)
        lap_eta_sp = ifft2(-K2 * eta_sp_hat).real
        delta_eta_sp = -L_eta * (df_deta_sp + kappa_eta * lap_eta_sp)
        eta_sp += dt * delta_eta_sp
        
        # Concentration c evolution (Cahn-Hilliard-like)
        df_dc = 2 * A * (c - c_eq) * (1 - eta_sp)**2
        mu = df_dc  # Chemical potential (simplified, no gradient for c itself)
        mu_hat = fft2(mu)
        delta_c = ifft2(-M * K2 * mu_hat).real
        c += dt * delta_c
        
        # Save snapshot every interval
        if step % save_interval == 0:
            snapshots.append({
                'step': step,
                'eta_sp': eta_sp.copy(),
                'c': c.copy(),
                'sum_eta_grains': sum_eta2.copy()
            })
    
    return snapshots

# Function to generate VTI content as string (ASCII format for simplicity)
def generate_vti(data, nx, ny, field_name='eta_sp'):
    vti_content = '<?xml version="1.0"?>\n'
    vti_content += '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
    vti_content += f'<ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
    vti_content += f'<Piece Extent="0 {nx-1} 0 {ny-1} 0 0">\n'
    vti_content += '<PointData>\n'
    vti_content += f'<DataArray type="Float32" Name="{field_name}" format="ascii">\n'
    for y in range(ny):
        for x in range(nx):
            vti_content += f'{data[y, x]:.6f} '
        vti_content += '\n'
    vti_content += '</DataArray>\n'
    vti_content += '</PointData>\n'
    vti_content += '</Piece>\n'
    vti_content += '</ImageData>\n'
    vti_content += '</VTKFile>\n'
    return vti_content

# Streamlit app
st.title("Modified Phase Field Simulation: Substrate Penetration through Grain Boundaries")

# Show free energy visualization
st.subheader("Free Energy Landscape for Grains and Substrate")
free_energy_fig = plot_free_energy()
st.plotly_chart(free_energy_fig)

# Input parameters
st.sidebar.header("Simulation Parameters")
nx = st.sidebar.slider("Grid size X", 50, 200, 100)
ny = st.sidebar.slider("Grid size Y", 50, 200, 100)
dt = st.sidebar.number_input("Time step dt", 0.001, 0.1, 0.01)
max_steps = st.sidebar.number_input("Max steps", 100, 5000, 1000)
save_interval = st.sidebar.number_input("Save snapshot every n steps", 10, 200, 50)
L_eta = st.sidebar.number_input("Mobility L_eta", 0.1, 10.0, 1.0)
kappa_eta = st.sidebar.number_input("Gradient coeff kappa_eta", 0.1, 10.0, 1.0)
gamma = st.sidebar.number_input("Interaction gamma", 1.0, 3.0, 1.5)
c0 = st.sidebar.number_input("Initial concentration c0 (> c_eq for supersaturation)", 0.1, 1.0, 0.6)
c_eq = st.sidebar.number_input("Equilibrium concentration c_eq", 0.1, 0.5, 0.3)
M = st.sidebar.number_input("Diffusion mobility M", 0.1, 10.0, 1.0)
A = st.sidebar.number_input("Concentration coupling A", 1.0, 10.0, 5.0)

if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        snapshots = run_simulation(nx, ny, dt, max_steps, save_interval, L_eta, kappa_eta, gamma, c0, c_eq, M, A)
    st.success("Simulation complete!")
    
    # Slider for time
    st.subheader("Time-Based Simulation Results")
    time_idx = st.slider("Select timestep index", 0, len(snapshots) - 1, 0)
    selected_snap = snapshots[time_idx]
    st.write(f"Showing step: {selected_snap['step']}")
    
    # Plotly visualization of substrate eta_sp
    fig_sp = px.imshow(selected_snap['eta_sp'], color_continuous_scale='plasma', title='Substrate Phase (η_sp)')
    fig_sp.update_layout(coloraxis_colorbar=dict(title='η_sp'))
    st.plotly_chart(fig_sp)
    
    # Additional viz: concentration c
    fig_c = px.imshow(selected_snap['c'], color_continuous_scale='viridis', title='Concentration (c)')
    fig_c.update_layout(coloraxis_colorbar=dict(title='c'))
    st.plotly_chart(fig_c)
    
    # Download PVD
    st.subheader("Download Simulation Data in PVD Format")
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Generate PVD content
        pvd_content = '<?xml version="1.0"?>\n'
        pvd_content += '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        pvd_content += '<Collection>\n'
        for idx, snap in enumerate(snapshots):
            vti_filename = f'sim_step_{snap["step"]}.vti'
            vti_content = generate_vti(snap['eta_sp'], nx, ny, field_name='eta_sp')
            zipf.writestr(vti_filename, vti_content)
            pvd_content += f'<DataSet timestep="{snap["step"]}" part="0" file="{vti_filename}"/>\n'
        pvd_content += '</Collection>\n'
        pvd_content += '</VTKFile>\n'
        zipf.writestr('simulation.pvd', pvd_content)
    
    st.download_button(
        label="Download PVD Zip (for ParaView)",
        data=zip_bytes.getvalue(),
        file_name="phase_field_simulation.zip",
        mime="application/zip"
    )
