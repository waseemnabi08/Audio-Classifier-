import os
import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow import keras
import random

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
model = keras.models.load_model('audio_classifier_model.h5')
print("Model loaded successfully!")

# Bandpass filter functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

# 2. Preprocessing functions
def extract_mfcc(audio_data, num_mfcc=13, sr=22050, n_fft=1024, hop_length=512):
    """Extract MFCC features from audio data."""
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = np.mean(mfcc, axis=1)
    return mfcc

def segment_audio(audio_data, sr, segment_length= 3, overlap=0.5):
    """Segment audio into overlapping chunks."""
    if len(audio_data) < segment_length * sr:
        padding = (segment_length * sr) - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding))
    segment_samples = int(segment_length * sr)
    step_size = int(segment_samples * (1 - overlap))
    segments = [audio_data[start:start + segment_samples] 
                for start in range(0, len(audio_data) - segment_samples, step_size)]
    return segments

def classify_audio(audio_file):
    """Classify audio into speech, music, or noise."""
    # Load audio
    audio_data, sr = librosa.load(audio_file, sr=22050)

    # Segment audio
    segments = segment_audio(audio_data, sr)

    # Predictions for each segment
    segment_predictions = []
    for segment in segments:
        mfcc = extract_mfcc(segment)
        mfcc = mfcc.reshape(1, -1)  # Reshape to match input shape of the model
        prediction = model.predict(mfcc, verbose=0)  # Get probabilities
        segment_predictions.append(prediction[0])  # Store probabilities for each segment

    # Convert segment predictions into a numpy array
    segment_predictions = np.array(segment_predictions)

    # Calculate the average prediction across all segments
    avg_prediction = np.mean(segment_predictions, axis=0)

    # Class labels
    class_labels = {0: 'music', 1: 'speech', 2: 'noise'}

   
    # Otherwise, use the class with the highest probability
    predicted_class = np.argmax(avg_prediction)
    predicted_label = class_labels[predicted_class]
    predicted_probability = avg_prediction[predicted_class]

    return predicted_label, predicted_probability, avg_prediction

# Route for homepage
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, uploaded_file.filename)
    uploaded_file.save(file_path)

    # Call your classification function
    predicted_label, predicted_probability, _ = classify_audio(file_path)

    return jsonify({
        "predicted_label": predicted_label,
        "probability": float(predicted_probability)
    })

@app.route('/about.html')
def about_html():
    return render_template('about.html')


    
# Run the app
if __name__ == "__main__":
    app.run(debug=True)
