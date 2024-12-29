import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Kaan Başpınar's Portfolio",
    page_icon="🎓",
    layout="wide"
)


# Main page content
st.title("Welcome to My Portfolio")

st.markdown("""
### About Me

Hello! I'm **Kaan Başpınar**, a physicist with a strong background in accelerator technologies and Martin-Puplett Interferometer (MPI). I have experience in Python programming and optical system design with ZEMAX. During my time at the Turkish Accelerator Radiation Laboratory (TARLA-FEL), I developed data analysis codes and worked on commissioning diagnostic cabinets.

I am deeply interested in topics such as deep learning, neuroscience, photoplethysmogram (PPG), and opto-electronics. I strive to combine my skills in physics and computer science to tackle challenging problems and contribute to innovative projects.

Feel free to explore my projects and connect with me!
""")

# CV Section
st.markdown("---")
st.header("My CV")
st.markdown("""
You can view my CV details below or download it as a PDF file.
""")

st.write("""
**Name:** Kaan Başpınar

**Profession:** Physicist

**Education:** Middle East Technical University (Physics, GPA: 3.20/4)

**Experience:**
- Candidate Engineer at TARLA-FEL (2023 Nov - 2024 Jun)
  - Developing data analysis code in Python for the Martin-Puplett Interferometer.
  - Commissioning the diagnostic cabinet.
- Co-Founder at Zetetis (2024 May - 2024 Nov)
  - Developing Python codes for volatility measure models, risk evaluations, anomaly and similarity detections, and time-series analysis for assets.

**Skills:** Python, ZEMAX, Fourier Optics, Data Analysis

**Languages:** Turkish, English

**Interests:** Deep Learning, Neuroscience, Photoplethysmogram (PPG), Opto-electronics

**GitHub:** [Kaan's GitHub](https://github.com/KaanBaspinar00)
""")

with open("Kaan Başpınar - CV - 12-2024.pdf", "rb") as pdf_file:
    pdf_data = pdf_file.read()
    st.download_button(
        label="📄 Download CV",
        data=pdf_data,
        file_name="Kaan_Baspinar_CV.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown("---")
st.markdown("""
Developed with ❤️ by Kaan Başpınar
""")

st.markdown("---")
st.write("""## Contact 

**Phone number:** +90 507 871 13 47

**e-mail address:** baspinarlee@gmail.com
""")
