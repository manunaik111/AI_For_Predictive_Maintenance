# ⚙️ AI-Driven Predictive Maintenance System
### Industrial Bearing Health Monitoring & Remaining Useful Life Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**Detect mechanical degradation before catastrophic failure — with 98% recall.**

</div>

---

## 📌 Overview

This project delivers an **end-to-end predictive maintenance pipeline** for industrial bearings. By fusing vibration and temperature sensor data with machine learning, the system identifies early-stage mechanical degradation and forecasts **Remaining Useful Life (RUL)** — enabling proactive maintenance decisions that prevent costly downtime.

---

## ✨ Key Technical Features

| Feature | Detail |
|---|---|
| 🎯 **98% Recall Rate** | SMOTE oversampling used to balance training data and maximize failure-state sensitivity |
| 🔬 **Feature Engineering** | 22 features extracted via FFT (frequency-domain) + statistical metrics (Kurtosis, RMS) |
| 📊 **Real-Time Dashboard** | Streamlit UI (`app.py`) with live sensor visualization and RUL countdown |
| 🤖 **Dual-Model Architecture** | RF Classifier for Healthy/Danger diagnosis + RF Regressor for RUL estimation |

---

## 🏗️ System Architecture

```
Raw Sensor Data (25.6 kHz)
        │
        ▼
┌───────────────────┐
│  Data Ingestion   │  ← create_femto_data.py | PRONOSTIA (FEMTO-ST) Dataset
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Signal Processing │  ← FFT · Fault Frequency Analysis · Segmentation
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│Feature Engineering│  ← 22 Features: RMS, Kurtosis, FFT Peaks, Temperature Stats
└────────┬──────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────┐
│  RF   │ │    RF    │
│Classi-│ │Regressor │
│ fier  │ │ (RUL est)│
└───┬───┘ └────┬─────┘
    └────┬─────┘
         ▼
┌───────────────────┐
│Streamlit Dashboard│  ← Live Status · RUL Countdown · Sensor Plots
└───────────────────┘
```

---

## 📂 Project Structure

```
AI_For_Predictive_Maintenance/
│
├── src/
│   └── app.py                  # Streamlit dashboard (entry point)
│
├── models/
│   ├── classifier.pkl          # Trained RF Classifier (Healthy/Danger)
│   └── regressor.pkl           # Trained RF Regressor (RUL estimation)
│
├── scripts/
│   ├── create_femto_data.py    # Data ingestion & preprocessing
│   └── visualize_data.py       # Exploratory visualization utilities
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/manunaik111/AI_For_Predictive_Maintenance.git
cd AI_For_Predictive_Maintenance
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run src/app.py
```

> 🖥️ The Streamlit dashboard will open in your browser at `http://localhost:8501`

---

## 🔬 Methodology

**1. Data Ingestion**
Loads high-frequency vibration data sampled at **25.6 kHz** alongside temperature readings from the **PRONOSTIA (FEMTO-ST)** benchmark dataset via `create_femto_data.py`.

**2. Signal Processing**
Raw signals are windowed into fixed-length segments. **Fast Fourier Transform (FFT)** is applied to each segment to isolate fault-characteristic frequencies (BPFI, BPFO, BSF, FTF).

**3. Feature Engineering**
22 discriminative features are extracted per window — including RMS, Kurtosis, Crest Factor, spectral centroid, and FFT peak magnitudes — forming the model input vector.

**4. Dual-Model Inference**
- **RF Classifier** → Binary status label: `Healthy` or `Danger`
- **RF Regressor** → Continuous RUL estimate (in minutes/cycles)

Both models are serialized as `.pkl` files and loaded at runtime by the dashboard for low-latency predictions.

**5. Class Imbalance Handling**
**SMOTE** (Synthetic Minority Oversampling Technique) is applied during training to correct the heavy class imbalance between healthy and failure-state samples, achieving a **98% recall** on the minority (failure) class.

---

## 📈 Model Performance

| Metric | Classifier | Regressor |
|---|---|---|
| **Recall (Failure Class)** | **98%** | — |
| **Algorithm** | Random Forest | Random Forest |
| **Balancing Technique** | SMOTE | — |
| **Input Features** | 22 (FFT + Statistical) | 22 (FFT + Statistical) |

---

## 📡 Dataset

This project uses the **PRONOSTIA (FEMTO-ST) Bearing Dataset** — a publicly available industrial benchmark for bearing fault detection and RUL prediction, recorded under accelerated degradation conditions.

> Dataset source: [FEMTO-ST Institute, IEEE PHM 2012 Challenge](https://www.femto-st.fr/en/Research-departments/AS2M/Research-groups/PHM/IEEE-PHM-2012-Data-challenge.php)

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Scikit-learn** — Random Forest, SMOTE (via `imbalanced-learn`)
- **NumPy / SciPy** — Signal processing & FFT
- **Pandas** — Data manipulation
- **Streamlit** — Real-time web dashboard
- **Matplotlib / Seaborn** — Visualization

---

## 📄 License & Copyright

© 2026 **Manu Naik**. Distributed under the **MIT License**.
See [`LICENSE`](./LICENSE) for full terms.

---

<div align="center">

*Built with precision for industrial-grade reliability.*

</div>
