import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="CFD Fluid Solver", layout="wide")
st.title(" 2D Navier-Stokes Fluid Simulator")
st.sidebar.header("Simulation Settings")

# --- INPUTS ---
nx = 41  # Grid points
ny = 41
nt = st.sidebar.slider("Time Steps", 10, 500, 100)
nit = 50 # Iterations for pressure solver
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

rho = 1.0  # Density
viscosity = st.sidebar.select_slider("Fluid Viscosity (nu)", options=[0.01, 0.05, 0.1, 0.5])
dt = 0.001

# --- CFD CORE FUNCTIONS ---
def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                    2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) -
                    ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, b, dx, dy):
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        p[:, -1] = p[:, -2] ; p[0, :] = p[1, :] ; p[:, 0] = p[:, 1] ; p[-1, :] = 0
    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, viscosity):
    b = np.zeros((ny, nx))
    for n in range(nt):
        un = u.copy() ; vn = v.copy()
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, b, dx, dy)
        
        # Momentum Equations
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         viscosity * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                     dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         viscosity * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                     dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :], u[-1, :], u[:, 0], u[:, -1] = 0, 1, 0, 0 # Top lid moves (u=1)
        v[0, :], v[-1, :], v[:, 0], v[:, -1] = 0, 0, 0, 0
    return u, v, p

# --- EXECUTION ---
if st.button("Simulate Flow"):
    u = np.zeros((ny, nx)) ; v = np.zeros((ny, nx)) ; p = np.zeros((ny, nx))
    u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, viscosity)

    # Visualization
    fig = plt.figure(figsize=(11, 7), dpi=100)
    X, Y = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 2, ny))
    plt.contourf(X, Y, p, alpha=0.5, cmap='viridis')
    plt.colorbar(label="Pressure Field")
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    plt.streamplot(X, Y, u, v, color='black')
    plt.xlabel('X') ; plt.ylabel('Y')
    st.pyplot(fig)
