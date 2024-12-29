import numpy as np
import plotly.graph_objects as go
from scipy import signal

def rect(x):
    """Define the rect function."""
    return np.where(np.abs(x) <= 0.5, 1, 0)

# Fourier Transform: Rectangular Function
def generate_rectangular_plot():
    """Generate a 2D Plotly Heatmap for a rectangular function."""
    L = 0.5
    M = 100
    x = np.linspace(-L / 2, L / 2, M)
    y = x
    X, Y = np.meshgrid(x, y)
    w = 0.5
    funct_to_be_convolved_1 = rect(X / (2 * w)) * rect(Y / (2 * w))
    fig = go.Figure(data=go.Heatmap(z=funct_to_be_convolved_1, x=x, y=y, colorscale='Viridis'))
    fig.update_layout(
        title="Rectangular Function",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=False,
        width=800,
        height=800
    )
    return fig

# Fourier Transform: Convolution
def generate_convolution_plot():
    """Generate a 2D Plotly Heatmap for the convolution of a rectangular function with itself."""
    L = 0.5
    M = 100
    x = np.linspace(-L / 2, L / 2, M)
    y = x
    X, Y = np.meshgrid(x, y)
    w = 0.5
    funct_to_be_convolved_1 = rect(X / (2 * w)) * rect(Y / (2 * w))
    convolved1 = signal.convolve2d(funct_to_be_convolved_1, funct_to_be_convolved_1, mode='same')
    fig = go.Figure(data=go.Heatmap(z=convolved1, x=x, y=y, colorscale='Viridis'))
    fig.update_layout(
        title="Convolution of Rectangular Function",
        xaxis_title="X",
        yaxis_title="Y",
        autosize=False,
        width=800,
        height=800
    )
    return fig

# Seidel Polynomial Visualization
def generate_seidel_plot():
    """Generate a 3D Plotly Surface for Seidel polynomial visualization."""
    def circ_(x, y, r):
        return np.where(np.sqrt(x**2 + y**2) <= r, 1, 0)

    def seidel_5(u0, v0, X, Y, wd, w040, w131, w222, w220, w311):
        beta = np.arctan2(v0, u0)
        u0r = np.sqrt(u0**2 + v0**2)
        Xr = X * np.cos(beta) + Y * np.sin(beta)
        Yr = -X * np.sin(beta) + Y * np.cos(beta)
        rho2 = Xr**2 + Yr**2
        w = (wd * rho2 +
             w040 * rho2**2 +
             w131 * u0r * rho2 * Xr +
             w222 * u0r**2 * Xr**2 +
             w220 * u0r**2 * rho2 +
             w311 * u0r**3 * Xr)
        return w

    u0, v0 = 1, 0
    wd, w040, w131, w222, w220, w311 = 0, 1, 0, 0, 0, 0
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    w = seidel_5(u0, v0, X, Y, wd, w040, w131, w222, w220, w311)

    # Apply circular mask
    P = circ_(X, Y, 1)
    w[P == 0] = np.nan

    fig = go.Figure(data=[go.Surface(z=w, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title="Seidel Polynomial Visualization",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Wavefront OPD"),
        autosize=False,
        width=800,
        height=800
    )
    return fig
