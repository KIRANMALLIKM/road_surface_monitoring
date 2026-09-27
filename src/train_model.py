import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Relative paths pointing to your exact folder structure
DATA_DIR = os.path.join("..", "data", "pothole-sensor-data")
POTHOLE_DIR = os.path.join(DATA_DIR, "Pothole")
ROAD_DIR = os.path.join(DATA_DIR, "RoadCondition")
RESULTS_DIR = os.path.join("..", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_extract_features():
    all_features = []
    all_labels = []

    # 1. Process Pothole trip files
    sensor_files = glob.glob(os.path.join(POTHOLE_DIR, "*_sensors.csv"))
    print(f"Found {len(sensor_files)} sensor trip files in Pothole folder...")
    for s_file in sensor_files:
        df = pd.read_csv(s_file)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Identify vertical/acceleration axis columns
        z_cols = [c for c in df.columns if 'z' in c or 'acc' in c]
        z_axis = df[z_cols[0]].values if z_cols else df.iloc[:, 2].values
        x_axis = df.iloc[:, 0].values
        y_axis = df.iloc[:, 1].values

        # Label: Pothole / Anomaly (1)
        window_size, step = 50, 25
        for start in range(0, len(z_axis) - window_size, step):
            end = start + window_size
            z_win, x_win, y_win = z_axis[start:end], x_axis[start:end], y_axis[start:end]

            # Feature Engineering: PtP, RMS, Std, SMA
            ptp_z = np.ptp(z_win)
            rms_z = np.sqrt(np.mean(z_win**2))
            std_z = np.std(z_win)
            sma = np.sum(np.abs(x_win) + np.abs(y_win) + np.abs(z_win))

            all_features.append([ptp_z, rms_z, std_z, sma])
            all_labels.append(1)

    # 2. Process RoadCondition files
    road_files = glob.glob(os.path.join(ROAD_DIR, "*.csv"))
    print(f"Found {len(road_files)} condition files in RoadCondition folder...")
    
    for r_file in road_files:
        df = pd.read_csv(r_file)
        df.columns = [c.strip().lower() for c in df.columns]
        
        z_cols = [c for c in df.columns if 'z' in c or 'acc' in c]
        z_axis = df[z_cols[0]].values if z_cols else df.iloc[:, 2].values
        x_axis = df.iloc[:, 0].values
        y_axis = df.iloc[:, 1].values

        # Label: Bad Road / Anomaly (1), Good Road (0)
        label = 1 if "bad" in os.path.basename(r_file).lower() else 0

        window_size, step = 50, 25
        for start in range(0, len(z_axis) - window_size, step):
            end = start + window_size
            z_win, x_win, y_win = z_axis[start:end], x_axis[start:end], y_axis[start:end]

            ptp_z = np.ptp(z_win)
            rms_z = np.sqrt(np.mean(z_win**2))
            std_z = np.std(z_win)
            sma = np.sum(np.abs(x_win) + np.abs(y_win) + np.abs(z_win))

            all_features.append([ptp_z, rms_z, std_z, sma])
            all_labels.append(label)

    return np.array(all_features), np.array(all_labels)

# Load data and run Random Forest pipeline
print("Extracting features from sensor CSV files...")
X, y = load_and_extract_features()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')

print("\n================ MODEL RESULTS ================")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))

# Save Confusion Matrix image for repo results
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Smooth', 'Pothole/Anomaly'], yticklabels=['Smooth', 'Pothole/Anomaly'])
plt.title('Confusion Matrix - Road Surface Monitoring')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
print(f"\nSaved confusion matrix plot to results/confusion_matrix.png")
