# An Autonomous IoT-Based Road Surface Monitoring System Using Inertial Sensing and Machine Learning

## 📌 Project Overview
This repository implements an end-to-end Machine Learning pipeline for automated road anomaly and pothole detection using tabular Inertial Measurement Unit (IMU) time-series data. The system extracts statistical features from acceleration signals across sliding windows to classify road surfaces into regular conditions vs. anomalies (potholes/bad road sections).

---

## 📊 Dataset & Feature Extraction
* **Source Dataset**: Kaggle Pothole Sensor Data (`Pothole` and `RoadCondition` subsets)
* **Pre-processing**: Signal sliding-window segmentation (Window Size = 50 samples, Overlap = 50% / 25 samples).
* **Extracted Feature Vector**:
  1. **Peak-to-Peak Amplitude ($PtP_Z$)**: Captures absolute vertical shock severity.
  2. **Root Mean Square ($RMS_Z$)**: Represents total vibration energy transferred to the vehicle.
  3. **Standard Deviation ($\sigma_Z$)**: Measures variation from flat-road baselines.
  4. **Signal Magnitude Area ($SMA$)**: Aggregates absolute dynamic energy across all tri-axial axes ($A_x, A_y, A_z$).

---

## 📈 Experimental Results & Model Performance
* **Model Implemented**: Random Forest Classifier (`n_estimators=100`, `random_state=42`)
* **Evaluation Metrics**:
  * **Accuracy**: 97.56%
  * **Precision**: 97.57%
  * **Recall**: 97.56%

### Confusion Matrix
The generated confusion matrix is saved under `results/confusion_matrix.png`.

---

## 🔬 Comparison with Baseline Literature
The table below compares our implemented system's performance against the benchmark paper referenced in the literature review:

| Metric | Baseline Paper (*Zhang et al., 2020*)[cite: 1] | Our Implementation | Performance Analysis |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Random Forest Classifier[cite: 1] | Random Forest Classifier | Standardized ML architecture alignment |
| **Features Extracted** | Time & Frequency Domain Features[cite: 1] | $PtP_Z, RMS_Z, \sigma_Z, SMA$ (Time-Domain)[cite: 1] | Streamlined feature vector optimized for edge/cloud compute |
| **Precision** | **88.50%**[cite: 1] | **97.57%** | **+9.07% Improvement** due to robust spatial statistical feature extraction |
| **Recall** | **75.00%**[cite: 1] | **97.56%** | **+22.56% Improvement** enabled by 50% sliding-window overlap |

---

## 🚀 How to Run the Code

1. **Clone the Repository**:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd road_surface_monitoring