import streamlit as st
from processing import generate_rectangular_plot, generate_convolution_plot, generate_seidel_plot
import plotly.graph_objs as go
from collections import deque
import streamlit.components.v1 as components
from Signals import kernel_density_estimation, image_filtering

# Configure the page
st.set_page_config(
    page_title="Projects - Kaan Başpınar",
    page_icon="💻",
    layout="wide"
)

# Initialize session state for each plot
if "show_rectangular_plot" not in st.session_state:
    st.session_state["show_rectangular_plot"] = False
if "show_convolution_plot" not in st.session_state:
    st.session_state["show_convolution_plot"] = False
if "show_seidel_plot" not in st.session_state:
    st.session_state["show_seidel_plot"] = False

# Initialize session state
if "live_data" not in st.session_state:
    st.session_state["live_data"] = deque(maxlen=50)  # Store last 50 data points
if "real_time_plot" not in st.session_state:
    st.session_state["real_time_plot"] = False


# Main page content
st.title("My Projects")

# Tabs for project categories
tabs = st.tabs(["Python Projects", "Zemax Projects", "Others"])
# Tabs for project categories

# Python Projects
with tabs[0]:
    st.header("Python Projects")
    st.markdown("""
    Here are some of my Python projects. You can find detailed descriptions, code snippets, and related visualizations for each project below.
    """)
    st.divider()
    # Subtabs for individual Python projects
    python_subtabs = st.tabs(["Fourier Transform and Fourier Optics", "Signals", "Project 3"])
    # Fourier Transform and Fourier Optics Project
    with python_subtabs[0]:
        # Fourier Transform: Rectangular Function
        st.markdown("### 1. Fourier Transform and Fourier Optics")
        st.write("github page: https://github.com/KaanBaspinar00/Fourier-Optics/blob/main/All_at_Once_FourierOptics.ipynb")
        st.divider()
        st.markdown("#### 1.a Rectangular Function")
        st.write("The rectangular function is a basic mathematical function used in Fourier optics.")
        st.code("""
            L = 0.5
        M = 100
        x = np.linspace(-L / 2, L / 2, M)
        y = x
        X, Y = np.meshgrid(x, y)
        w = 0.5
    
    
        Funct_to_be_convolved_1 = rect(X/2*w)* rect(Y/2*w)
        plt.contourf(X, Y, Funct_to_be_convolved_1)
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])
        plt.title("Rect function")
        plt.show()
    
        #Funct_to_be_convolved_2 = rect(X/2*w)* rect(Y/2*w)""")

        if st.button("Generate Rectangular Function Plot"):
            st.session_state["show_rectangular_plot"] = True
        if st.button("Close Rectangular Function Plot"):
            st.session_state["show_rectangular_plot"] = False
        if st.session_state["show_rectangular_plot"]:
            fig = generate_rectangular_plot()
            st.plotly_chart(fig)
        st.divider()

        # Fourier Transform: Convolution
        st.markdown("#### 1.b Fourier Transform: Convolution")
        st.write(
            "The convolution of the rectangular function with itself demonstrates its properties in the Fourier domain.")

        st.code("""
        convolved1 = signal.convolve2d(Funct_to_be_convolved_1,Funct_to_be_convolved_1)
        # visualize convulation result
        plt.figure(figsize=(10, 10))
        plt.imshow(convolved1)
        plt.colorbar()
        plt.show()
        """)
        if st.button("Generate Convolution Plot"):
            st.session_state["show_convolution_plot"] = True
        if st.button("Close Convolution Plot"):
            st.session_state["show_convolution_plot"] = False
        if st.session_state["show_convolution_plot"]:
            fig = generate_convolution_plot()
            st.plotly_chart(fig)
        st.divider()

        # Seidel Polynomial Visualization
        st.markdown("#### 1.c Seidel Polynomial Visualization")
        st.write("Seidel polynomials are used to represent wavefront aberrations in optical systems.")
        st.code('''
            convolved1 = signal.convolve2d(Funct_to_be_convolved_1,Funct_to_be_convolved_1)
            # visualize convulation result
            plt.figure(figsize=(10, 10))
            plt.imshow(convolved1)
            plt.colorbar()
            plt.show()
            """)
    
            st.image("show2.png", caption = """Function is convelved with itself.""")
    
    
            st.markdown("#### Chapter 8 - Wavefront Aberrations")
            st.code("""
            #################################################################
        
            # 8.2.1 Seidel Polynomials - Definition and primary aberrations
        
            #################################################################
        
            def seidel_5(u0, v0, X, Y, wd, w040, w131, w222, w220, w311):
            """
            Compute wavefront OPD for the first 5 Seidel wavefront aberration coefficients + defocus.
    
            Parameters:
            u0, v0 : float
                Normalized image plane coordinates.
            X, Y : array_like
                Normalized pupil coordinate arrays (from meshgrid).
            wd : float
                Defocus coefficient.
            w040 : float
                Spherical aberration coefficient.
            w131 : float
                Coma aberration coefficient.
            w222 : float
                Astigmatism coefficient.
            w220 : float
                Field curvature coefficient.
            w311 : float
                Distortion coefficient.
    
            Returns:
            w : array_like
                Wavefront OPD (Optical Path Difference).
            """
    
            # Image rotation angle
            beta = np.arctan2(v0, u0)
    
            # Image height
            u0r = np.sqrt(u0 ** 2 + v0 ** 2)
    
            # Rotate grid
            Xr = X * np.cos(beta) + Y * np.sin(beta)
            Yr = -X * np.sin(beta) + Y * np.cos(beta)
    
            # Seidel polynomials
            rho2 = Xr ** 2 + Yr ** 2
            w = (wd * rho2 +
                 w040 * rho2 ** 2 +
                 w131 * u0r * rho2 * Xr +
                 w222 * u0r ** 2 * Xr ** 2 +
                 w220 * u0r ** 2 * rho2 +
                 w311 * u0r ** 3 * Xr)
    
            return w
    
        import numpy as np
    
        # Example input
        u0, v0 = 0.5, 0.5
        X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        wd, w040, w131, w222, w220, w311 = 1, 0.5, 0.1, 0.2, 0.05, 0.03
    
        # Call the function
        w = seidel_5(u0, v0, X, Y, wd, w040, w131, w222, w220, w311)
    
        print(w)
    
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    
        # Define the seidel_5 function
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
    
    
    
        # Define the variables
        u0, v0 = 1, 0
        wd, w040, w131, w222, w220, w311 = 0, 1, 0, 0, 0, 0
    
        # Create a meshgrid for X and Y
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)
        X, Y = np.meshgrid(x, y)
    
        # Calculate the wavefront OPD
        w = seidel_5(u0, v0, X, Y, wd, w040, w131, w222, w220, w311)
    
        # Create the circular pupil function
        P = circ_(X,Y,1)
    
        # Mask where P == 0
        mask = (P == 0)
        w[mask] = np.nan
    
        # Plot the result
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Plot the surface
        surf = ax.plot_surface(X, Y, w, cmap='gray', edgecolor='none')
    
        # Add lighting effect
        ax.view_init(30, 30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlim([0,1.5])
    
        plt.show()
    
        ''')
        if st.button("Generate Seidel Polynomial Plot"):
            st.session_state["show_seidel_plot"] = True
        if st.button("Close Seidel Polynomial Plot"):
            st.session_state["show_seidel_plot"] = False
        if st.session_state["show_seidel_plot"]:
            fig = generate_seidel_plot()
            st.plotly_chart(fig)
            st.divider()

    # Signals Subtab
    with python_subtabs[1]:
        st.markdown("### 2. Signal Processing Examples")
        st.write("github page: https://github.com/KaanBaspinar00/SignalProcess/blob/main/Signal-Notes.ipynb")
        st.divider()
        # Initialize session states
        if "show_kde_plot" not in st.session_state:
            st.session_state["show_kde_plot"] = False
        if "show_image_filter_plot" not in st.session_state:
            st.session_state["show_image_filter_plot"] = False

        # Example 1: Kernel Density Estimation
        st.markdown("#### 2.a Kernel Density Estimation (B-Splines)")
        st.write("This example demonstrates Kernel Density Estimation using B-splines.")
        st.code("""
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

        """)
        if st.button("Show KDE Plot"):
            st.session_state["show_kde_plot"] = True
        if st.button("Close KDE Plot"):
            st.session_state["show_kde_plot"] = False
        if st.session_state["show_kde_plot"]:
            fig = kernel_density_estimation()
            st.plotly_chart(fig)
        st.divider()

        # Example 2: Filtering Images
        st.markdown("#### 2.b Filtering Images")
        st.write("This example demonstrates filtering on grayscale images using averaging and edge detection filters.")
        st.code("""
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

        """)
        if st.button("Show Image Filtering Plot"):
            st.session_state["show_image_filter_plot"] = True
        if st.button("Close Image Filtering Plot"):
            st.session_state["show_image_filter_plot"] = False
        if st.session_state["show_image_filter_plot"]:
            fig = image_filtering()
            st.pyplot(fig)
        st.divider()

    # Signals Subtab
    with python_subtabs[2]:
        st.markdown("### 3. Others")
        st.title("Martin Puplett Interferometer Presentation")
        components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vQ6xA654sg5Lvn2XIFgQFAdVz2VZSPIVerCqwx3kQjaMqDmPayT1_-GpI29JpEhaA/embed?start=false&loop=false&delayms=3000", height= 480)
        st.write("github page: https://github.com/KaanBaspinar00")
        st.write("Other projects I've done will be shown here soon!")
        st.divider()
        
# Zemax Projects
with tabs[1]:
    st.header("Zemax Projects")
    st.markdown("""
    These are some of the optical system design projects I have worked on using Zemax.
    """)
    st.subheader("Project 1")
    st.markdown("""
    **Description:** A detailed explanation of the Zemax project.
    """)
    st.image("https://via.placeholder.com/600x400", caption="Zemax simulation image")
    st.markdown("---")

# Other Projects
with tabs[2]:
    st.header("Other Projects")
    st.markdown("""
    Here you can find details about my other projects and contributions.
    """)
    st.subheader("Project 1")
    st.markdown("""
    **Description:** A brief explanation of this project.
    """)
    st.markdown("---")
