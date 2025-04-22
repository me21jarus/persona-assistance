import streamlit as st
import cv2
import numpy as np
import requests
import tempfile
import time
import os
import sounddevice as sd
import queue
import soundfile as sf
import threading
import speech_recognition as sr
import random

# Backend API URLs
TEXT_API_URL = "http://127.0.0.1:8000/analyze-text/"
AUDIO_API_URL = "http://127.0.0.1:8000/analyze-audio/"
VIDEO_API_URL = "http://127.0.0.1:8000/analyze-video/"
COMBINED_API_URL = "http://127.0.0.1:8000/analyze-combined/"

# Set page config for a wider layout
st.set_page_config(layout="wide", page_title="Persona Assistant", page_icon="🧠")

# Custom CSS for styling
st.markdown("""
<style>
    .welcome-header {
        font-size: 4rem;
        text-align: center;
        margin-top: 20vh;
        color: #1E88E5;
        font-weight: bold;
    }
    .welcome-subheader {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 5vh;
        color: #424242;
    }
    .start-btn {
        display: block;
        margin: 0 auto;
        padding: 12px 30px;
        font-size: 1.2rem;
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .start-btn:hover {
        background-color: #1565C0;
        transform: scale(1.05);
    }
    .end-btn {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1000;
        padding: 8px 16px;
        background-color: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
    }
    .emotion-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .emotion-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #424242;
    }
    .emotion-value {
        font-size: 1.2rem;
        color: #1E88E5;
    }
    .summary-emotion {
        font-size: 3rem;
        text-align: center;
        margin: 30px 0;
        color: #1E88E5;
    }
    .emoji-container {
        font-size: 5rem;
        text-align: center;
        margin: 20px 0;
    }
    .text-input-area {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .big-emoji {
        font-size: 8rem;
        text-align: center;
        margin: 20px 0;
    }
    .small-card {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .small-emotion {
        font-size: 1rem;
        color: #424242;
    }
    .small-emoji {
        font-size: 2rem;
        margin: 5px 0;
    }
    .combo-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 5px;
    }
    .priority-indicator {
        font-size: 0.8rem;
        color: #757575;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'app_state' not in st.session_state:
    st.session_state.app_state = "welcome"  # welcome, recording, summary
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
if 'emotion_results' not in st.session_state:
    st.session_state.emotion_results = {
        'text': None,
        'audio': None,
        'video': None,
        'combined': None,
        'audio_text': None,
        'audio_video': None,
        'text_video': None
    }
if 'frame_result' not in st.session_state:
    st.session_state.frame_result = None
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'all_emotions_history' not in st.session_state:
    st.session_state.all_emotions_history = []

# Emotion to emoji mapping
emotion_emojis = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "surprised": "😲",
    "fearful": "😨",
    "disgusted": "🤢",
    "neutral": "😐",
    "calm": "😌",
    "excited": "🤩",
    "confused": "😕"
}

# Function to start the app
def start_app():
    st.session_state.app_state = "recording"
    st.session_state.recording = True
    st.session_state.text_input = ""
    st.session_state.emotion_results = {
        'text': None,
        'audio': None,
        'video': None,
        'combined': None,
        'audio_text': None,
        'audio_video': None,
        'text_video': None
    }
    st.session_state.emotion_history = []
    st.session_state.all_emotions_history = []

# Function to end recording and show summary
def end_recording():
    st.session_state.recording = False
    st.session_state.app_state = "summary"

# Text input handling
def process_text_input():
    if st.session_state.text_input != st.session_state.text_area and st.session_state.text_area.strip():
        st.session_state.text_input = st.session_state.text_area
        response = requests.post(TEXT_API_URL, params={"text": st.session_state.text_area})
        if response.status_code == 200:
            st.session_state.emotion_results['text'] = response.json().get("emotions")
            return response.json().get("emotions")
    return None

# Audio processing function
def process_audio_stream(audio_data):
    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(temp_audio_path, audio_data, 16000)

    # Send audio for emotion analysis
    with open(temp_audio_path, "rb") as f:
        response = requests.post(AUDIO_API_URL, files={"file": f})

    # Also convert audio to text using speech recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            if text.strip():
                # Process the transcribed text for emotion
                text_response = requests.post(TEXT_API_URL, params={"text": text})
                if text_response.status_code == 200:
                    st.session_state.emotion_results['text'] = text_response.json().get("emotions")
                    # Update text area with transcribed speech
                    st.session_state.text_input += " " + text
                    st.session_state.text_area = st.session_state.text_input
        except:
            pass  # If speech can't be recognized, continue silently

    os.remove(temp_audio_path)

    if response.status_code == 200:
        st.session_state.emotion_results['audio'] = response.json().get("emotion")
        return response.json().get("emotion")
    return None

# Video processing function
def process_video_frame(frame):
    # Create a temporary file for the frame
    temp_frame_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
    cv2.imwrite(temp_frame_path, frame)

    # Send the frame for analysis
    with open(temp_frame_path, "rb") as f:
        response = requests.post(VIDEO_API_URL, files={"file": f})

    os.remove(temp_frame_path)

    if response.status_code == 200:
        emotion = response.json().get("emotion")
        st.session_state.emotion_results['video'] = emotion
        return emotion
    return None

# Get text emotion string (highest scoring emotion)
def get_text_emotion_string():
    if isinstance(st.session_state.emotion_results['text'], dict) and st.session_state.emotion_results['text']:
        return max(st.session_state.emotion_results['text'].items(), key=lambda x: x[1])[0]
    return None

# Weighted emotion detection function with priorities
def analyze_combined_emotions():
    # Define weights for each modality (higher weight = higher priority)
    weights = {'text': 0.5, 'audio': 0.3, 'video': 0.2}
    
    # Get emotions from each modality
    audio_emotion = st.session_state.emotion_results['audio']
    video_emotion = st.session_state.emotion_results['video']
    text_emotion = get_text_emotion_string()
    
    # Store all emotion combinations for display
    if audio_emotion and text_emotion:
        # Audio-Text combination (higher priority to audio)
        audio_text_counts = {audio_emotion: weights['audio'], text_emotion: weights['text']}
        st.session_state.emotion_results['audio_text'] = max(audio_text_counts.items(), key=lambda x: x[1])[0]
    
    if audio_emotion and video_emotion:
        # Audio-Video combination (higher priority to audio)
        audio_video_counts = {audio_emotion: weights['audio'], video_emotion: weights['video']}
        st.session_state.emotion_results['audio_video'] = max(audio_video_counts.items(), key=lambda x: x[1])[0]
    
    if text_emotion and video_emotion:
        # Text-Video combination (higher priority to text)
        text_video_counts = {text_emotion: weights['text'], video_emotion: weights['video']}
        st.session_state.emotion_results['text_video'] = max(text_video_counts.items(), key=lambda x: x[1])[0]
    
    # All three modalities combined
    if audio_emotion and text_emotion and video_emotion:
        # Create a weighted vote system
        combined_counts = {}
        
        # Add audio with highest weight
        if audio_emotion in combined_counts:
            combined_counts[audio_emotion] += weights['audio']
        else:
            combined_counts[audio_emotion] = weights['audio']
        
        # Add text with medium weight
        if text_emotion in combined_counts:
            combined_counts[text_emotion] += weights['text']
        else:
            combined_counts[text_emotion] = weights['text']
        
        # Add video with lowest weight
        if video_emotion in combined_counts:
            combined_counts[video_emotion] += weights['video']
        else:
            combined_counts[video_emotion] = weights['video']
        
        # Get the emotion with the highest weighted score
        combined_emotion = max(combined_counts.items(), key=lambda x: x[1])[0]
        st.session_state.emotion_results['combined'] = combined_emotion
        
        # Add to emotion history for summary
        st.session_state.emotion_history.append(combined_emotion)
        
        # Store the full emotional state for historical tracking
        st.session_state.all_emotions_history.append({
            'audio': audio_emotion,
            'text': text_emotion,
            'video': video_emotion,
            'combined': combined_emotion,
            'timestamp': time.time()
        })
        
        return combined_emotion
    return None

# Main detection loop
def run_detection():
    cap = cv2.VideoCapture(0)
    audio_queue = queue.Queue()

    # Set up audio capture
    def audio_callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    # Start audio stream
    audio_stream = sd.InputStream(callback=audio_callback, 
                                 samplerate=16000, 
                                 channels=1,
                                 blocksize=16000)  # 1-second blocks
    audio_stream.start()

    last_audio_process_time = time.time()
    last_video_process_time = time.time()
    last_combined_analysis_time = time.time()

    try:
        while st.session_state.recording:
            # Process video frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB for display in Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
            # Process video frame for emotion every 1 second
            current_time = time.time()
            if current_time - last_video_process_time >= 1.0:
                video_emotion = process_video_frame(frame)
                if video_emotion:
                    video_emotion_display.markdown(
                        f"""<div class="emotion-card">
                            <div class="emotion-title">Video Emotion:</div>
                            <div class="emotion-value">{video_emotion} {emotion_emojis.get(video_emotion.lower(), '')}</div>
                            <div class="priority-indicator">Priority: Low</div>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                last_video_process_time = current_time
            
            # Process accumulated audio every 3 seconds
            if current_time - last_audio_process_time >= 3.0:
                audio_data = []
                while not audio_queue.empty():
                    audio_data.append(audio_queue.get())
                
                if audio_data:
                    audio_data = np.concatenate(audio_data, axis=0)
                    audio_emotion = process_audio_stream(audio_data)
                    if audio_emotion:
                        audio_emotion_display.markdown(
                            f"""<div class="emotion-card">
                                <div class="emotion-title">Audio Emotion:</div>
                                <div class="emotion-value">{audio_emotion} {emotion_emojis.get(audio_emotion.lower(), '')}</div>
                                <div class="priority-indicator">Priority: Medium</div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                
                last_audio_process_time = current_time
            
            # Process text input
            text_emotions = process_text_input()
            if text_emotions:
                # Find the highest scoring emotion
                max_emotion = max(text_emotions.items(), key=lambda x: x[1])
                text_emotion_display.markdown(
                    f"""<div class="emotion-card">
                        <div class="emotion-title">Text Emotion:</div>
                        <div class="emotion-value">{max_emotion[0]} ({max_emotion[1]:.2f}) {emotion_emojis.get(max_emotion[0].lower(), '')}</div>
                        <div class="priority-indicator">Priority: MeHighdium</div>
                    </div>""", 
                    unsafe_allow_html=True
                )
            
            # Perform combined analysis every 3 seconds
            if current_time - last_combined_analysis_time >= 3.0:
                combined_emotion = analyze_combined_emotions()
                
                # Update all combination displays
                if combined_emotion:
                    # Combined (all three)
                    combined_emotion_display.markdown(
                        f"""<div class="emotion-card">
                            <div class="emotion-title">Overall Emotion:</div>
                            <div class="emotion-value">{combined_emotion} {emotion_emojis.get(combined_emotion.lower(), '')}</div>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                    
                    # Update the combination displays
                    if st.session_state.emotion_results['audio_text']:
                        audio_text_display.markdown(
                            f"""<div class="small-card">
                                <div class="combo-title">Audio + Text</div>
                                <div class="small-emoji">{emotion_emojis.get(st.session_state.emotion_results['audio_text'].lower(), '')}</div>
                                <div class="small-emotion">{st.session_state.emotion_results['audio_text']}</div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    
                    if st.session_state.emotion_results['audio_video']:
                        audio_video_display.markdown(
                            f"""<div class="small-card">
                                <div class="combo-title">Audio + Video</div>
                                <div class="small-emoji">{emotion_emojis.get(st.session_state.emotion_results['audio_video'].lower(), '')}</div>
                                <div class="small-emotion">{st.session_state.emotion_results['audio_video']}</div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    
                    if st.session_state.emotion_results['text_video']:
                        text_video_display.markdown(
                            f"""<div class="small-card">
                                <div class="combo-title">Text + Video</div>
                                <div class="small-emoji">{emotion_emojis.get(st.session_state.emotion_results['text_video'].lower(), '')}</div>
                                <div class="small-emotion">{st.session_state.emotion_results['text_video']}</div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                
                last_combined_analysis_time = current_time
            
            # Brief pause to prevent excessive CPU usage
            time.sleep(0.03)

    finally:
        # Clean up resources
        if 'audio_stream' in locals() and audio_stream.active:
            audio_stream.stop()
            audio_stream.close()
        cap.release()

# Welcome screen
if st.session_state.app_state == "welcome":
    st.markdown('<h1 class="welcome-header">Welcome to Persona Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subheader">Your intelligent emotion detection companion</p>', unsafe_allow_html=True)
    
    # Center the emoji
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="big-emoji">🧠</div>', unsafe_allow_html=True)
    
    # Start button
    if st.button("Start Detection", key="start_button"):
        start_app()

# Recording screen
elif st.session_state.app_state == "recording":
    # End recording button
    st.markdown(
        f"""
        <button class="end-btn" onclick="document.getElementById('end_recording_button').click()">
            End Recording
        </button>
        """,
        unsafe_allow_html=True
    )
    
    # Hidden button that will be clicked by the custom button
    if st.button("End Recording", key="end_recording_button", help="End the recording session"):
        end_recording()
    
    # Create layout with columns (60% for video+text, 40% for emotions)
    col1, col2 = st.columns([6, 4])
    
    with col1:
        # Video feed placeholder
        video_placeholder = st.empty()
        
        # Text input below video
        st.text_area("Type here:", value=st.session_state.text_input, key="text_area", 
                    height=100, help="Type here to analyze text emotions")
    
    with col2:
        st.markdown("### Real-time Emotion Analysis")
        
        # Primary emotion displays
        text_emotion_display = st.empty()
        audio_emotion_display = st.empty()
        video_emotion_display = st.empty()
        combined_emotion_display = st.empty()
        
        # Combination displays in three columns
        st.markdown("### Emotion Combinations")
        combo_col1, combo_col2, combo_col3 = st.columns(3)
        
        with combo_col1:
            audio_text_display = st.empty()
        
        with combo_col2:
            audio_video_display = st.empty()
        
        with combo_col3:
            text_video_display = st.empty()
    
    # Run the detection loop
    if st.session_state.recording:
        run_detection()

# Summary screen
elif st.session_state.app_state == "summary":
    st.markdown('<h1 style="text-align: center;">Emotion Analysis Summary</h1>', unsafe_allow_html=True)
    
    # Determine the most frequent emotion
    if st.session_state.emotion_history:
        emotion_counts = {}
        for emotion in st.session_state.emotion_history:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Display the dominant emotion with emoji
        st.markdown(
            f'<div class="summary-emotion">Dominant Emotion: {dominant_emotion} {emotion_emojis.get(dominant_emotion.lower(), "")}</div>',
            unsafe_allow_html=True
        )
        
        # Display big emoji for the emotion
        st.markdown(
            f'<div class="big-emoji">{emotion_emojis.get(dominant_emotion.lower(), "")}</div>',
            unsafe_allow_html=True
        )
        
        # Display emotion distribution
        st.subheader("Emotion Distribution")
        
        # Calculate percentages
        total = sum(emotion_counts.values())
        emotion_percentages = {k: (v/total)*100 for k, v in emotion_counts.items()}
        
        # Sort emotions by percentage
        sorted_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Display as a horizontal bar chart
        for emotion, percentage in sorted_emotions:
            st.markdown(
                f"""
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 120px;">{emotion} {emotion_emojis.get(emotion.lower(), "")}</div>
                        <div style="flex-grow: 1; background-color: #f0f0f0; border-radius: 5px; height: 25px;">
                            <div style="width: {percentage}%; background-color: #1E88E5; height: 25px; border-radius: 5px; color: white; text-align: right; padding-right: 10px;">
                                {percentage:.1f}%
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Modality contribution analysis
        st.subheader("Modality Contribution Analysis")
        
        # Count times each modality contributed to the final emotion
        modality_matches = {"Audio": 0, "Text": 0, "Video": 0}
        total_readings = len(st.session_state.all_emotions_history)
        
        if total_readings > 0:
            for reading in st.session_state.all_emotions_history:
                if reading['audio'] == reading['combined']:
                    modality_matches["Audio"] += 1
                if reading['text'] == reading['combined']:
                    modality_matches["Text"] += 1
                if reading['video'] == reading['combined']:
                    modality_matches["Video"] += 1
            
            # Convert to percentages
            for key in modality_matches:
                modality_matches[key] = (modality_matches[key] / total_readings) * 100
            
            # Display as horizontal bars
            st.markdown("#### How often each modality matched the final emotion:")
            for modality, percentage in sorted(modality_matches.items(), key=lambda x: x[1], reverse=True):
                st.markdown(
                    f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; align-items: center;">
                            <div style="width: 120px;">{modality}</div>
                            <div style="flex-grow: 1; background-color: #f0f0f0; border-radius: 5px; height: 25px;">
                                <div style="width: {percentage}%; background-color: #4CAF50; height: 25px; border-radius: 5px; color: white; text-align: right; padding-right: 10px;">
                                    {percentage:.1f}%
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Add an insight about which modality most accurately reflected the user's emotions
            most_accurate = max(modality_matches.items(), key=lambda x: x[1])[0]
            st.markdown(f"**Insight:** Your {most_accurate.lower()} was the most consistent indicator of your emotional state during this session.")
        
        # Emotion insights based on dominant emotion
        st.subheader("Emotion Insights")
        
        insights = {
            "happy": "You appeared to be in a positive mood during this session. Your expressions, tone, and words conveyed happiness and contentment.",
            "sad": "You seemed to be experiencing some sadness during this session. Your expressions and tone indicated feelings of melancholy or disappointment.",
            "angry": "You displayed signs of frustration or anger during this session. Your expressions and tone suggested irritation or displeasure.",
            "surprised": "You showed elements of surprise during this session. Your reactions indicated unexpected discoveries or revelations.",
            "fearful": "You exhibited signs of concern or fear during this session. Your expressions suggested worry or apprehension.",
            "disgusted": "You displayed signs of discomfort or disgust during this session. Your expressions indicated aversion or displeasure.",
            "neutral": "You maintained a mostly neutral demeanor during this session. Your expressions and tone were balanced and even.",
            "calm": "You appeared relaxed and at ease during this session. Your expressions and tone conveyed tranquility and composure.",
            "excited": "You showed enthusiasm and excitement during this session. Your expressions and tone were energetic and animated.",
            "confused": "You displayed signs of uncertainty or confusion during this session. Your expressions suggested you were processing complex information."
        }
        
        st.write(insights.get(dominant_emotion.lower(), "Your emotions varied throughout this session."))
        
        # Text summary if text was provided
        if st.session_state.text_input:
            st.subheader("Text Analysis")
            st.write(f"Based on your text input: \"{st.session_state.text_input[:100]}{'...' if len(st.session_state.text_input) > 100 else ''}\"")
            
            if isinstance(st.session_state.emotion_results['text'], dict):
                # Display text emotions as a horizontal bar
                for emotion, score in sorted(st.session_state.emotion_results['text'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 10px;">
                            <div style="display: flex; align-items: center;">
                                <div style="width: 120px;">{emotion} {emotion_emojis.get(emotion.lower(), "")}</div>
                                <div style="flex-grow: 1; background-color: #f0f0f0; border-radius: 5px; height: 25px;">
                                    <div style="width: {score*100}%; background-color: #4CAF50; height: 25px; border-radius: 5px; color: white; text-align: right; padding-right: 10px;">
                                        {score:.2f}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.warning("No emotions were detected during the session. Try recording for a longer period.")
    
    # Restart button
    if st.button("Start New Session", key="restart_button"):
        start_app()

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #666;">
    <p>Persona Assistant - Your Emotion Analysis Companion</p>
    <p style="font-size: 0.8rem;">© 2023 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)