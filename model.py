import joblib
import numpy as np
import librosa

model = joblib.load('horn_detector_classifier.joblib')

def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def predict_horn(audio_data, sr):
    features = extract_features(audio_data, sr).reshape(1, -1)
    prediction = model.predict(features)
    return bool(prediction[0])
