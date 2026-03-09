# AI_For_Predictive_Maintenance
AI-Driven Predictive Maintenance System for Industrial Bearings
Overview
This project provides an end-to-end solution for monitoring industrial bearing health and predicting Remaining Useful Life (RUL). By utilizing Sensor Fusion (Vibration & Temperature) and the Random Forest algorithm, the system identifies mechanical degradation before catastrophic failure occurs.
Key Technical Features
98% Recall Rate: Achieved high sensitivity to failure states by implementing the SMOTE algorithm to balance the training data.
Feature Engineering: Extracted 22 distinct features using Fast Fourier Transform (FFT) for frequency-domain analysis and statistical metrics like Kurtosis and RMS.
Real-Time Dashboard: Integrated a Streamlit web interface (app.py) for live data visualization and RUL countdowns.
Dual-Model Architecture: Uses a Random Forest Classifier for status diagnosis (Healthy/Danger) and a Regressor for RUL estimation.
System Methodology
Data Ingestion: Loads high-frequency vibration data (25.6 kHz) and temperature data from the PRONOSTIA (FEMTO-ST) dataset using create_femto_data.py.
Signal Processing: Segments raw signals and applies FFT to identify specific fault frequencies.
Inference: Trained models are saved as .pkl files and loaded into the dashboard for instantaneous prediction.
Installation & Usage
Clone the repository:
git clone https://github.com/manunaik111/AI_For_Predictive_Maintenance.git
Install dependencies:
pip install -r requirements.txt
Run the Dashboard:
streamlit run src/app.py
Project Structure
src/: Core application logic and Streamlit UI.
models/: Trained .pkl model files for classification and regression.
scripts/: Data preprocessing, training, and visualization scripts (visualize_data.py).
License & Copyright
© 2026 Manu Naik. Distributed under the MIT License. See LICENSE for details.
