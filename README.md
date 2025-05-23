# 🚗 Real-Time Horn Detection System

This project is a real-time horn sound detection system developed using **Flask**, **Python**, and **Machine Learning**. It listens to audio input from a microphone, extracts meaningful features, filters out noise, and uses a trained ML model to classify whether the sound is a horn or not — all in real time.

## 🧠 Key Features

- 🎧 **Live Microphone Input**: Captures audio using `sounddevice` in 2-second chunks.
- 🔍 **Noise Filtering**: Applies a pre-emphasis filter and bandpass filtering to isolate horn-relevant frequencies.
- 📈 **Feature Extraction**: Uses `librosa` to compute MFCCs (Mel-frequency cepstral coefficients).
- 🤖 **ML Model Integration**: A pre-trained classifier (`joblib`) predicts whether the sound contains a horn.
- 🌐 **Web Interface**: A minimal Flask-based frontend allows starting/stopping recording via `/start_recording` and `/stop_recording` routes.
- 🧵 **Multithreading**: Audio capture and processing run on background threads to ensure non-blocking real-time performance.

## 🛠️ Technologies Used

- `Flask` – for the web server and frontend routes
- `sounddevice` – for real-time audio streaming from the microphone
- `librosa` – for feature extraction (MFCCs)
- `scipy.signal` – for filtering and signal processing
- `joblib` – for loading the trained classifier model
- `threading` and `queue` – for efficient real-time audio processing

## 📦 Folder Structure

<pre><code> 📁 horn-detection-app/ ├── 📁 templates/ │ └── index.html # Web interface template ├── horn_detector_classifier.joblib # Pre-trained ML model ├── app.py # Main Flask application ├── requirements.txt # Python dependencies └── README.md # Project documentation </code></pre>

## 📂 Dataset

This project uses the [Horn Detection Dataset](https://data.mendeley.com/datasets/y5stjsnp8s/2) from Mendeley Data. The dataset consists of labeled audio recordings of vehicle horn sounds and other environmental noises. The audio files are sampled at 44.1 kHz and cover various real-world scenarios for accurate horn detection.

- Dataset Size: Approximately 1000 audio clips  
- Audio Format: WAV (44.1 kHz sampling rate)  
- Labels: Horn and Non-Horn sounds  

## 📊 Model Performance

The machine learning model was trained on this dataset using extracted MFCC features with noise filtering preprocessing. After training and evaluation with an 80-20 train-test split, the model achieved:

**Test Accuracy:** 94%

This result demonstrates the model’s strong ability to distinguish horn sounds from other noises in real-time audio streams.


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- A working microphone

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/horn-detection-app.git
cd horn-detection-app
```
### Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Application
```bash
python app.py

```





