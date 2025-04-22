import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# Use a better speech emotion recognition model
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# Load the pre-trained model
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Emotion labels (from MSP-Podcast dataset)
emotion_labels = ["Angry", "Cheerful", "Excited", "Happy", "Neutral", "Sad", "Tender"]

def detect_emotion(audio_path):
    """Predicts emotion from an audio file."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Preprocess audio
    input_values = processor(y, sampling_rate=sr, return_tensors="pt").input_values

    # Run model inference
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return emotion_labels[predicted_class]

if __name__ == "__main__":
    audio_file = input("Enter the path to your audio file (WAV format): ")
    emotion = detect_emotion(audio_file)
    print(f"Predicted Emotion: {emotion}")
