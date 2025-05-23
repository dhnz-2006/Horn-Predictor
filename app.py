from flask import Flask, render_template, Response, jsonify
import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
import joblib
import time
from scipy.io import wavfile
import os
from scipy import signal

app = Flask(__name__)

audio_queue = queue.Queue()
model = None
is_recording = False
sample_rate = 44100
duration = 2.0
chunk_size = int(sample_rate * duration)

def load_model():
    global model
    try:
        model_path = 'horn_detector_classifier.joblib'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")
        
        print(f"Attempting to load model from: {os.path.abspath(model_path)}")
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model parameters: {model.get_params()}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def filter_noise(audio_data, sr):
    try:
        audio_float32 = audio_data.astype(np.float32)
        
        pre_emphasis = 0.96
        emphasized_audio = np.append(audio_float32[0], audio_float32[1:] - pre_emphasis * audio_float32[:-1])
        
        nyquist = sr // 2
        low = 100 / nyquist
        high = 2000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_audio = signal.filtfilt(b, a, emphasized_audio)
        
        filtered_audio = librosa.util.normalize(filtered_audio)
        
        return filtered_audio
    except Exception as e:
        print(f"Error in noise filtering: {e}")
        return audio_data

def extract_features(audio_data, sr):
    try:
        audio_float32 = audio_data.astype(np.float32)
        
        filtered_audio = filter_noise(audio_float32, sr)
        
        mfccs = librosa.feature.mfcc(y=filtered_audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    if is_recording:
        audio_queue.put(indata.copy())

def process_audio():
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            features = extract_features(audio_data, sample_rate)
            if features is not None:
                prediction = model.predict([features])[0]
                probability = model.predict_proba([features])[0]
                
                print(f"Prediction: {'Horn' if prediction == 1 else 'Not Horn'} (Confidence: {probability[1]:.2f})")
        time.sleep(0.1)

def start_audio_stream():
    with sd.InputStream(callback=audio_callback,
                       channels=1,
                       samplerate=sample_rate,
                       blocksize=chunk_size):
        print("Audio stream started. Press Ctrl+C to stop.")
        while True:
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording')
def start_recording():
    global is_recording
    is_recording = True
    return jsonify({"status": "Recording started"})

@app.route('/stop_recording')
def stop_recording():
    global is_recording
    is_recording = False
    return jsonify({"status": "Recording stopped"})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    
    try:
        load_model()
        
        audio_thread = threading.Thread(target=process_audio, daemon=True)
        audio_thread.start()
        
        stream_thread = threading.Thread(target=start_audio_stream, daemon=True)
        stream_thread.start()
        
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting application: {e}")