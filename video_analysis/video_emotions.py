import cv2
from deepface import DeepFace

def detect_emotion():
    """Detects emotions from real-time webcam feed."""
    cap = cv2.VideoCapture(0)  # Open webcam (0 = default camera)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze the face emotion
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            
            # Display detected emotion
            cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        except:
            pass

        cv2.imshow("Emotion Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion()
