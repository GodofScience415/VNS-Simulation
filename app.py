import streamlit as st
from PIL import Image

# ================================
# Streamlit Setup
# ================================
st.title("VNS Seizure Prevention Simulation")

# ================================
# Project Description
# ================================
st.header("Project Overview")
st.write("""
This application simulates a **Vagus Nerve Stimulation (VNS) system** that helps in seizure reduction. 
It generates synthetic physiological data for different species (Rat, Chimpanzee, Human) and trains a 
deep learning model to predict seizure reduction efficiency based on stimulation parameters.

The simulation involves:
- **Data Generation**: Creating synthetic heart rate, blood pressure, neural activity, respiration rate, and oxygen levels.
- **Model Training**: A neural network is trained to predict the effect of VNS on seizure reduction.
- **Visualization**: Outputs include training loss graphs, model predictions, and a 3D visualization of VNS application.

Below are the different graphs generated in this application:
""")

# ================================
# Explanation of Graphs
# ================================
st.header("Explanation of Graphs")

st.markdown("""
1. **Training Loss Graph** (Top-Left)  
   - Displays how the model’s loss decreases over epochs during training.  
   - A lower loss indicates better model performance.

2. **Actual vs. Predicted Seizure Reduction (Rat) (Top-Right)**  
   - Scatter plot comparing actual seizure reduction with the model’s predictions for Rats.  
   - Points near the red dashed line indicate better accuracy.

3. **Actual vs. Predicted Seizure Reduction (Chimpanzee) (Bottom-Left)**  
   - Similar scatter plot for Chimpanzees.  
   - Checks if the model generalizes well across species.

4. **Actual vs. Predicted Seizure Reduction (Human) (Bottom-Right)**  
   - Same analysis but for human data.  
   - Helps in evaluating model reliability for clinical use.

5. **3D Visualization of the VNS Device**  
   - A 3D representation showing an **Arduino-based VNS system** connecting to different species via nerve pathways.
   - Includes electrical pulses traveling to the vagus nerve.
""")

st.write("### Below are the generated graphs:")

# ================================
# Load and Display Precomputed Graphs
# ================================

# Load images
training_loss_img = Image.open("Figure_1.png")
rat_vs_predicted_img = Image.open("Figure_2.png")
chimp_vs_predicted_img = Image.open("Figure_3.png")
human_vs_predicted_img = Image.open("Figure_4.png")

# Display images with updated parameter
st.image(training_loss_img, caption="Training Loss Over Epochs", use_container_width=True)
st.image(rat_vs_predicted_img, caption="Actual vs. Predicted Seizure Reduction (Rat)", use_container_width=True)
st.image(chimp_vs_predicted_img, caption="Actual vs. Predicted Seizure Reduction (Chimpanzee)", use_container_width=True)
st.image(human_vs_predicted_img, caption="Actual vs. Predicted Seizure Reduction (Human)", use_container_width=True)
