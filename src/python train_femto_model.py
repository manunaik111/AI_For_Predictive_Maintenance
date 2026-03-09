import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# --- Configuration ---
# Path to where create_femto_data.py saved the .npy files.
# Using relative path based on your project structure.
PROCESSED_DATA_DIR = r'C:\project file 002\processed data001' 
SCALERS_DIR = r'C:\project file 002\scalers001' 
MODEL_DIR = r'C:\project file 002\models'

# Input Files (from Step 1)
X_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'X_data_femto_fusion.npy')
Y_REG_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_reg_femto_fusion.npy')
Y_CLASS_FILE = os.path.join(PROCESSED_DATA_DIR, 'y_class_femto_fusion.npy')

# Output Files (V9 Models)
MODEL_REG_PATH = os.path.join(MODEL_DIR, 'regressor_v9_femto.pkl')
MODEL_CLASS_PATH = os.path.join(MODEL_DIR, 'classifier_v9_femto.pkl')
SCALER_PATH = os.path.join(SCALERS_DIR, 'scaler_X_v9_femto.pkl')

TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

# --- Main Script ---
print("--- Starting V9 Training (Sensor Fusion) ---")

# 1. Load Data
if not os.path.exists(X_DATA_FILE):
    print(f"Error: Data file not found at {X_DATA_FILE}")
    print("Please run 'create_femto_data.py' first.")
    exit()

print("Loading data...")
try:
    X = np.load(X_DATA_FILE)
    y_reg = np.load(Y_REG_FILE)
    y_class = np.load(Y_CLASS_FILE)
    print(f"Data Loaded. Features shape: {X.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Split Data
# We need to split carefully to ensure we have data for both tasks
X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
    X, y_reg, y_class, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED
)

# 3. Scale Features
print("Scaling features...")
# Initialize and fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Regressor (RUL)
print("Training Regressor (RUL)...")
# Random Forest is robust and handles tabular features well
regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)
regressor.fit(X_train_scaled, y_reg_train)

# Evaluate Regressor
y_pred_reg = regressor.predict(X_test_scaled)
mae = mean_absolute_error(y_reg_test, y_pred_reg)
# Convert "files" to "seconds" (each file is roughly 10 seconds in FEMTO if sampling rate matches acquisition time)
# In FEMTO, files are snapshots. Let's just report MAE in 'files' and 'approx seconds'
print(f"--- Regressor MAE: {mae:.2f} files (Approx {mae * 10:.2f} seconds) ---")

# 5. Train Classifier (Status) with SMOTE
print("Training Classifier (Status)...")
print(f"Original class distribution: Healthy={sum(y_class_train==0)}, Danger={sum(y_class_train==1)}")

print("Applying SMOTE...")
try:
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_res, y_class_train_res = smote.fit_resample(X_train_scaled, y_class_train)
    print(f"Resampled class distribution: Healthy={sum(y_class_train_res==0)}, Danger={sum(y_class_train_res==1)}")
except ValueError as e:
    print(f"SMOTE Error: {e}. (This might happen if classes are too rare). Training on original data.")
    X_train_res, y_class_train_res = X_train_scaled, y_class_train

classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED)
classifier.fit(X_train_res, y_class_train_res)

# Evaluate Classifier
y_pred_class = classifier.predict(X_test_scaled)
print("\n--- Classification Report ---")
print(classification_report(y_class_test, y_pred_class, target_names=['Healthy', 'Danger']))

# 6. Save Everything
print("Saving V9 models...")
# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)

with open(MODEL_REG_PATH, 'wb') as f: pickle.dump(regressor, f)
with open(MODEL_CLASS_PATH, 'wb') as f: pickle.dump(classifier, f)
with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)

print(f"Models saved to: {MODEL_DIR}")
print(f"Scaler saved to: {SCALERS_DIR}")
print("--- Training Complete. ---")