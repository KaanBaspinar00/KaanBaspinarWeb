import streamlit as st
from processing import generate_rectangular_plot, generate_convolution_plot, generate_seidel_plot
from sensor import start_acquisition, stop_acquisition, save_sensor_data, get_real_time_data
import asyncio
import plotly.graph_objs as go
from collections import deque
import os

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
    python_subtabs = st.tabs(["Fourier Transform and Fourier Optics", "Using Sensors", "Project 3"])
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
        # Sensor Data Project
            # Sensor Data Project
            with python_subtabs[1]:
                st.markdown("### Sensor Data Project")
                st.write("This project requires using a phone to provide real-time sensor data via WebSocket.")
                st.divider()
        
                # Real-time data acquisition controls
                if st.button("Start Acquisition"):
                    asyncio.run(start_acquisition())
                    st.session_state["real_time_plot"] = True
                    st.success("Data acquisition started! Use your phone to send data via WebSocket.")
        
                if st.button("Stop Acquisition"):
                    asyncio.run(stop_acquisition())
                    st.session_state["real_time_plot"] = False
                    st.success("Data acquisition stopped!")
        
                # Save data as CSV
                if st.button("Save Data"):
                    try:
                        filename = asyncio.run(save_sensor_data())
                        st.success(f"Data saved to {filename}")
                        with open(filename, "rb") as file:
                            btn = st.download_button(
                                label="Download CSV",
                                data=file,
                                file_name=filename,
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error saving data: {e}")
        
                # Real-time Plotly graph
                st.markdown("#### Real-Time Sensor Data")
                st.write("The graph below shows live data from the phone's sensors.")
        
                # Initialize Plotly graph
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="X-Axis"))
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Y-Axis"))
                fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Z-Axis"))
                fig.update_layout(
                    title="Real-Time Sensor Data",
                    xaxis_title="Timestamp",
                    yaxis_title="Sensor Values",
                    autosize=True,
                    height=500,
                    width=800
                )
        
                # Live data visualization loop
                if st.session_state["real_time_plot"]:
                    placeholder = st.empty()  # Placeholder for the Plotly graph
        
                    async def update_plot():
                        async for data in get_real_time_data(None):  # Pass WebSocket instance here if testing with live WebSocket
                            st.session_state["live_data"].append(data)
        
                            # Extract data for Plotly
                            timestamps = [d["timestamp"] for d in st.session_state["live_data"]]
                            x_values = [d["x"] for d in st.session_state["live_data"]]
                            y_values = [d["y"] for d in st.session_state["live_data"]]
                            z_values = [d["z"] for d in st.session_state["live_data"]]
        
                            # Update Plotly graph
                            fig.data[0].x = timestamps
                            fig.data[0].y = x_values
                            fig.data[1].x = timestamps
                            fig.data[1].y = y_values
                            fig.data[2].x = timestamps
                            fig.data[2].y = z_values
        
                            # Render updated graph
                            placeholder.plotly_chart(fig, use_container_width=True)
        
                    asyncio.run(update_plot())

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
