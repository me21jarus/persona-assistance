from transformers import pipeline

# Load pre-trained DistilBERT model for sentiment analysis
nlp_emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def detect_emotion(text):
    results = nlp_emotion(text)
    return results

if __name__ == "__main__":
    text = input("Enter a sentence: ")
    emotions = detect_emotion(text)
    for emotion in emotions[0]:  # Extract results
        print(f"{emotion['label']}: {emotion['score']:.4f}")
