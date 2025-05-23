import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Feature extraction function
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Dataset directory
DATASET_DIR = 'dataset'
LABELS = {'horn': 1, 'nothorn': 0}

features = []
labels = []

# Load and extract features
for label_name, label_value in LABELS.items():
    folder_path = os.path.join(DATASET_DIR, label_name)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            try:
                mfcc = extract_features(file_path)
                features.append(mfcc)
                labels.append(label_value)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Convert to arrays
X = np.array(features)
y = np.array(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['nothorn', 'horn']))

# Save the model
joblib.dump(model, 'horn_detector_classifier.joblib')
print("Model saved as horn_detector_classifier.joblib")
