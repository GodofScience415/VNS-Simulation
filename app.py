import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pyvista as pv
import tempfile
import sys
import subprocess
import pyvista as pv

# Ensure PyVista runs in off-screen mode
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_OSMESA"] = "true"  # Use OSMesa for headless rendering
os.environ["PYVISTA_VTK_DATA"] = "true"

# Set a fallback DISPLAY variable (needed for some OpenGL alternatives)
os.environ["DISPLAY"] = ":99.0"

# Disable interactive rendering in Streamlit Cloud
pv.global_theme.render_window_off_screen = True

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
# SECTION 1: Synthetic Data Generation
# ================================

species_params = {
    'Rat': {'heart_rate': (300, 500), 'blood_pressure': (80, 120), 'neural_activity': (0.5, 1.5),
            'respiration_rate': (80, 150), 'oxygen_saturation': (90, 100)},
    'Chimpanzee': {'heart_rate': (70, 110), 'blood_pressure': (110, 140), 'neural_activity': (0.8, 1.2),
                   'respiration_rate': (15, 25), 'oxygen_saturation': (95, 100)},
    'Human': {'heart_rate': (60, 100), 'blood_pressure': (110, 140), 'neural_activity': (0.8, 1.2),
              'respiration_rate': (12, 20), 'oxygen_saturation': (95, 100)}
}

device_params = {'amplitude': (0.25, 3.0), 'pulse_width': (130, 500), 'frequency': (20, 50)}
samples_per_species = 5000


def generate_dataset(species_name, samples):
    sp = species_params[species_name]
    amplitude = np.random.uniform(device_params['amplitude'][0], device_params['amplitude'][1], samples)
    pulse_width = np.random.uniform(device_params['pulse_width'][0], device_params['pulse_width'][1], samples)
    frequency = np.random.uniform(device_params['frequency'][0], device_params['frequency'][1], samples)
    heart_rate = np.random.uniform(sp['heart_rate'][0], sp['heart_rate'][1], samples)
    blood_pressure = np.random.uniform(sp['blood_pressure'][0], sp['blood_pressure'][1], samples)
    neural_activity = np.random.uniform(sp['neural_activity'][0], sp['neural_activity'][1], samples)
    respiration_rate = np.random.uniform(sp['respiration_rate'][0], sp['respiration_rate'][1], samples)
    oxygen_saturation = np.random.uniform(sp['oxygen_saturation'][0], sp['oxygen_saturation'][1], samples)

    X = np.stack([amplitude, pulse_width, frequency, heart_rate, blood_pressure, neural_activity,
                  respiration_rate, oxygen_saturation], axis=1)

    base_efficacies = {'Rat': 20, 'Chimpanzee': 40, 'Human': 60}
    base_eff = base_efficacies[species_name]
    device_factor = (amplitude / device_params['amplitude'][1]) * \
                    (pulse_width / device_params['pulse_width'][1]) * \
                    (frequency / device_params['frequency'][1])

    physiology_factor = (1 - (heart_rate - sp['heart_rate'][0]) / (sp['heart_rate'][1] - sp['heart_rate'][0])) * \
                        (1 - (blood_pressure - sp['blood_pressure'][0]) / (sp['blood_pressure'][1] - sp['blood_pressure'][0])) * \
                        neural_activity

    y = base_eff * device_factor * physiology_factor + np.random.normal(0, 5, samples)
    y = np.clip(y, 0, 100)

    return X, y


X_rat, y_rat = generate_dataset('Rat', samples_per_species)
X_chimp, y_chimp = generate_dataset('Chimpanzee', samples_per_species)
X_human, y_human = generate_dataset('Human', samples_per_species)

X_all = np.concatenate([X_rat, X_chimp, X_human], axis=0)
y_all = np.concatenate([y_rat, y_chimp, y_human], axis=0)

shuffle_idx = np.random.permutation(X_all.shape[0])
X_all = X_all[shuffle_idx]
y_all = y_all[shuffle_idx]

X_tensor = torch.tensor(X_all, dtype=torch.float32)
y_tensor = torch.tensor(y_all, dtype=torch.float32).view(-1, 1)

# ================================
# SECTION 2: Train PyTorch Model
# ================================

class VNSNet(nn.Module):
    def __init__(self, input_size):
        super(VNSNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


model = VNSNet(input_size=8)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50
batch_size = 256
num_samples = X_tensor.shape[0]
training_losses = []

for epoch in range(epochs):
    permutation = torch.randperm(num_samples)
    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i + batch_size]
        batch_X = X_tensor[indices]
        batch_y = y_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    avg_loss = epoch_loss / num_samples
    training_losses.append(avg_loss)

# ================================
# Display 4 Graphs in 2x2 Layout
# ================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Creates a 2x2 grid of plots

# Training Loss (Top-Left)
axes[0, 0].plot(range(1, epochs + 1), training_losses, marker='o', label='Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Over Epochs')
axes[0, 0].legend()

# Rat Predictions (Top-Right)
X_species = torch.tensor(X_rat, dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_species).numpy().flatten()
axes[0, 1].scatter(y_rat, predictions, alpha=0.3)
axes[0, 1].set_xlabel('Actual Seizure Reduction (%)')
axes[0, 1].set_ylabel('Predicted Seizure Reduction (%)')
axes[0, 1].set_title('Rat: Actual vs. Predicted')
axes[0, 1].plot([0, 100], [0, 100], 'r--')

# Chimpanzee Predictions (Bottom-Left)
X_species = torch.tensor(X_chimp, dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_species).numpy().flatten()
axes[1, 0].scatter(y_chimp, predictions, alpha=0.3)
axes[1, 0].set_xlabel('Actual Seizure Reduction (%)')
axes[1, 0].set_ylabel('Predicted Seizure Reduction (%)')
axes[1, 0].set_title('Chimpanzee: Actual vs. Predicted')
axes[1, 0].plot([0, 100], [0, 100], 'r--')

# Human Predictions (Bottom-Right)
X_species = torch.tensor(X_human, dtype=torch.float32)
with torch.no_grad():
    predictions = model(X_species).numpy().flatten()
axes[1, 1].scatter(y_human, predictions, alpha=0.3)
axes[1, 1].set_xlabel('Actual Seizure Reduction (%)')
axes[1, 1].set_ylabel('Predicted Seizure Reduction (%)')
axes[1, 1].set_title('Human: Actual vs. Predicted')
axes[1, 1].plot([0, 100], [0, 100], 'r--')

plt.tight_layout()  # Adjusts spacing for better visibility
st.pyplot(fig)  # Display the combined 2x2 layout in Streamlit

# ================================
# SECTION 3: 3D Visualization of VNS Device
# ================================
# Ensure PyVista runs in headless mode
os.environ["PYVISTA_OFF_SCREEN"] = "true"

# Use an off-screen PyVista plotter
plotter = pv.Plotter(off_screen=True)

# Create a 3D model of the Arduino-based VNS device as a cube
arduino = pv.Cube(center=(0, 0, 0), x_length=1.0, y_length=0.5, z_length=0.2)
plotter.add_mesh(arduino, color='gray', opacity=0.9, label="Arduino VNS Device")

# Create 3D models for each species
rat_model = pv.Sphere(center=(3, -2, 0.5), radius=0.5)
chimp_model = pv.Sphere(center=(3, 0, 0.5), radius=0.8)
human_model = pv.Sphere(center=(3, 2, 0.5), radius=1.0)

plotter.add_mesh(rat_model, color='tan', opacity=0.9)
plotter.add_mesh(chimp_model, color='tan', opacity=0.9)
plotter.add_mesh(human_model, color='tan', opacity=0.9)

# Draw nerve pathway connections
rat_line = pv.Line(pointa=(0.5, 0, 0), pointb=(2.5, -2, 0.5), resolution=50)
chimp_line = pv.Line(pointa=(0.5, 0, 0), pointb=(2.5, 0, 0.5), resolution=50)
human_line = pv.Line(pointa=(0.5, 0, 0), pointb=(2.5, 2, 0.5), resolution=50)

plotter.add_mesh(rat_line, color='red', line_width=5)
plotter.add_mesh(chimp_line, color='red', line_width=5)
plotter.add_mesh(human_line, color='red', line_width=5)

# Create a temporary file for the screenshot
temp_dir = tempfile.gettempdir()
screenshot_path = os.path.join(temp_dir, "vns_3d_visualization.png")

# Generate the screenshot and save
plotter.screenshot(screenshot_path)

# Display the saved screenshot in Streamlit
st.image(screenshot_path, caption="3D Visualization of VNS Device", use_column_width=True)