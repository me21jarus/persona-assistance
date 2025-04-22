from fastapi import FastAPI, File, UploadFile, Body
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import librosa
import numpy as np
import soundfile as sf
from deepface import DeepFace
import cv2
import tempfile
import os
from typing import Dict, Any
from pydantic import BaseModel

app = FastAPI()

# Ensure storage folders exist
VIDEO_STORAGE = "videos"
if not os.path.exists(VIDEO_STORAGE):
    os.makedirs(VIDEO_STORAGE)

# 1️⃣ TEXT EMOTION DETECTION (NLP)
nlp_emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

@app.post("/analyze-text/")
def analyze_text(text: str):
    """Analyze emotion from text input."""
    try:
        results = nlp_emotion(text)
        emotions = {item["label"]: item["score"] for item in results[0]}
        return {"text": text, "emotions": emotions}
    except Exception as e:
        return {"error": str(e)}

# 2️⃣ AUDIO EMOTION DETECTION
model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
emotion_labels = ["Angry", "Cheerful", "Excited", "Happy", "Neutral", "Sad", "Tender"]

@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    """Predict emotion from an audio file."""
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(temp_audio, "wb") as f:
            f.write(file.file.read())

        audio, sr = librosa.load(temp_audio, sr=16000)
        input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values).logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        os.remove(temp_audio)
        return {"emotion": emotion_labels[predicted_class]}
    except Exception as e:
        return {"error": str(e)}

# 3️⃣ VIDEO EMOTION DETECTION
@app.post("/analyze-video/")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze emotions from a video file or image."""
    try:
        temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_file_path, "wb") as f:
            f.write(file.file.read())
        
        # Try to open as video first
        cap = cv2.VideoCapture(temp_file_path)
        if cap.isOpened():
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            if len(frames) > 0:
                middle_frame = frames[len(frames) // 2]
                result = DeepFace.analyze(middle_frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            else:
                emotion = "No Face Detected"
        else:
            # Try to process as an image if can't open as video
            image = cv2.imread(temp_file_path)
            if image is not None:
                result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            else:
                emotion = "Invalid Media Format"

        os.remove(temp_file_path)
        return {"emotion": emotion}
    except Exception as e:
        return {"error": str(e)}

# 4️⃣ COMBINED ANALYSIS ENDPOINT
class CombinedAnalysisRequest(BaseModel):
    text_emotion: Dict[str, float] = None
    audio_emotion: str = None
    video_emotion: str = None

@app.post("/analyze-combined/")
async def analyze_combined(data: CombinedAnalysisRequest):
    """Analyze emotions from combined text, audio, and video results."""
    try:
        # Map emotions from different sources
        # This is where you could implement more sophisticated fusion logic
        
        # Start with a list of detected emotions
        detected_emotions = []
        
        # Add text emotion (highest scoring one)
        if data.text_emotion:
            max_text_emotion = max(data.text_emotion.items(), key=lambda x: x[1])[0]
            detected_emotions.append(max_text_emotion)
        
        # Add audio and video emotions
        if data.audio_emotion:
            detected_emotions.append(data.audio_emotion)
        if data.video_emotion:
            detected_emotions.append(data.video_emotion)
        
        # Simple majority voting 
        if detected_emotions:
            # Count occurrences of each emotion
            emotion_counts = {}
            for emotion in detected_emotions:
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1
            
            # Get the dominant emotion (most frequent)
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            # Confidence is the proportion of sources that agree
            confidence = emotion_counts[dominant_emotion] / len(detected_emotions)
            
            return {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "emotion_distribution": emotion_counts
            }
        else:
            return {"error": "No emotions detected"}
            
    except Exception as e:
        return {"error": str(e)}

# Root endpoint to check if the server is running
@app.get("/")
def read_root():
    return {"message": "FastAPI server is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)