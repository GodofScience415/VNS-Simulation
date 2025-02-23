import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="VNS Seizure Prevention",
    layout="wide"
)

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        color: #f7fafc;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use the custom CSS class to center align the title
st.markdown('<div class="centered-title">VNS Seizure Prevention Simulation</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Project Overview", 
    "Data & Graphs",
    "Future Research"
])

# --- Load Images ---
vns_ppt = Image.open("vns-ppt-ss.png")
training_loss_img = Image.open("Figure_1.png")
rat_vs_predicted_img = Image.open("Figure_2.png")
chimp_vs_predicted_img = Image.open("Figure_3.png")
human_vs_predicted_img = Image.open("Figure_4.png")

with tab1:
    st.image(vns_ppt, use_container_width=True)

with tab2:
    st.write("""
    - **Seizure Data (24 Months)**: Showed a 20% reduction in seizure frequency among 20 participants using the device.
    - **Depression Data (24 Months)**: Indicated a 25% decrease in self-reported depression, suggesting potential additional benefits of VNS therapy.
    """)
    st.image(training_loss_img, caption="Training Loss Over Epochs", use_container_width=True)
    st.image(rat_vs_predicted_img, caption="Rat Seizure Reduction", use_container_width=True)
    st.image(chimp_vs_predicted_img, caption="Chimpanzee Seizure Reduction", use_container_width=True)
    st.image(human_vs_predicted_img, caption="Human Seizure Reduction", use_container_width=True)

with tab3:
    st.write("""
    - **Refinement**: Further improve comfort, detection accuracy, and overall user experience.
    - **Expanded Trials**: Test with a larger and more diverse population for robust validation.
    - **Advanced ML**: Implement more sophisticated algorithms or personalized models to adapt to each patient's seizure patterns.
    - **Telemedicine Integration**: Enable remote monitoring and updates for patients with limited access to healthcare facilities.
    """)