
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
from scipy import ndimage
import skimage as ski
from scipy import ndimage
import plotly.subplots as sp

def kernel_density_estimation():

    cv = np.array([[50., 25.],
                   [59., 12.],
                   [50., 10.],
                   [57., 2.],
                   [40., 4.],
                   [40., 14.]])

    degrees = [1, 2, 3]
    fig = go.Figure()

    # Add control points
    fig.add_trace(go.Scatter(x=cv[:, 0], y=cv[:, 1], mode='lines+markers', name="Control Points"))

    # Generate B-splines for different degrees
    for degree in degrees:
        count = cv.shape[0]
        kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)
        max_param = count - degree
        spl = interpolate.BSpline(kv, cv, degree)
        spline_data = spl(np.linspace(0, max_param, 100))
        fig.add_trace(go.Scatter(x=spline_data[:, 0], y=spline_data[:, 1], mode='lines',
                                 name=f'Degree {degree}'))

    fig.update_layout(
        title="Kernel Density Estimation with B-Splines",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(range=[35, 70]),
        yaxis=dict(range=[0, 30]),
        width=800,
        height=600
    )

    return fig

import numpy as np
from scipy import ndimage
import skimage as ski
import matplotlib.pyplot as plt

def image_filtering():

    img = ski.data.camera()
    img_f = np.array(img, dtype=float)

    # Filters for averaging and edge detection
    filters = [
        np.ones((11, 11)) / 121,  # Averaging filter
        np.array([np.ones(11), np.zeros(11), -1 * np.ones(11)]),  # Edge detection (horizontal)
    ]
    filters.append(filters[-1].T)  # Edge detection (vertical)

    # Filter the images
    filtered = [ndimage.correlate(img_f, filt) for filt in filters]

    # Binary threshold images
    binary_horizontal = filtered[1] > 125
    binary_vertical = filtered[2] > 125

    # Plot the images
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    axs[0, 0].imshow(img, cmap="gray")
    axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(filtered[0], cmap="gray")
    axs[0, 1].set_title("Averaged Image")
    axs[1, 0].imshow(filtered[1], cmap="gray")
    axs[1, 0].set_title("Horizontal Edges")
    axs[1, 1].imshow(filtered[2], cmap="gray")
    axs[1, 1].set_title("Vertical Edges")
    axs[2, 0].imshow(binary_horizontal, cmap="gray")
    axs[2, 0].set_title("Horizontal Binary Edges")
    axs[2, 1].imshow(binary_vertical, cmap="gray")
    axs[2, 1].set_title("Vertical Binary Edges")

    # Remove axes ticks and labels
    for ax in axs.ravel():
        ax.axis("off")

    # Adjust layout
    plt.tight_layout()

    return fig
