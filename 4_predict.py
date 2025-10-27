# ========================================
# 🔍 Predict Scam or Real from Audio
# ========================================

import whisper
import pickle
import os
import sys

def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer"""
    try:
        with open('scam_detector_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print("✅ Model and vectorizer loaded successfully!")
        return model, vectorizer
    
    except FileNotFoundError as e:
        print("❌ Error: Model files not found!")
        print("Please run '3_train_model.py' first to train and save the model.")
        sys.exit(1)


def predict_from_audio(audio_path, model, vectorizer, whisper_model=None):
    """
    Predict if an audio file is scam or real
    
    Args:
        audio_path: Path to the audio file
        model: Trained classification model
        vectorizer: Trained TF-IDF vectorizer
        whisper_model: Whisper model (will load if None)
    
    Returns:
        Tuple of (prediction, transcribed_text)
    """
    
    # Load Whisper model if not provided
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
    
    # Transcribe audio
    print(f"\nTranscribing '{audio_path}'...")
    result = whisper_model.transcribe(audio_path)
    transcribed_text = result['text']
    
    print("\n" + "=" * 50)
    print("Transcribed Text:")
    print("=" * 50)
    print(transcribed_text)
    
    # Transform text using TF-IDF
    X_tfidf = vectorizer.transform([transcribed_text])
    
    # Make prediction
    prediction = model.predict(X_tfidf)[0]
    
    # Get decision function score (confidence)
    decision_score = model.decision_function(X_tfidf)[0]
    
    return prediction, transcribed_text, decision_score


def predict_from_text(text, model, vectorizer):
    """
    Predict if a text is scam or real
    
    Args:
        text: Text to classify
        model: Trained classification model
        vectorizer: Trained TF-IDF vectorizer
    
    Returns:
        Tuple of (prediction, decision_score)
    """
    
    # Transform text using TF-IDF
    X_tfidf = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(X_tfidf)[0]
    
    # Get decision function score (confidence)
    decision_score = model.decision_function(X_tfidf)[0]
    
    return prediction, decision_score


def main():
    print("=" * 50)
    print("Scam Detection Prediction")
    print("=" * 50)
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        
        if not os.path.exists(audio_path):
            print(f"❌ Error: File not found - {audio_path}")
            sys.exit(1)
        
        # Make prediction from audio
        whisper_model = whisper.load_model("base")
        prediction, transcribed_text, score = predict_from_audio(
            audio_path, model, vectorizer, whisper_model
        )
        
        # Display result
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        
        if prediction == 1:
            print("🚨 This audio is likely a SCAM")
        else:
            print("✅ This audio is likely REAL")
        
        print(f"Confidence Score: {abs(score):.4f}")
        
    else:
        # Interactive mode
        print("\nNo audio file provided. Choose an option:")
        print("1. Predict from audio file")
        print("2. Predict from text")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            audio_path = input("Enter path to audio file: ").strip()
            
            if not os.path.exists(audio_path):
                print(f"❌ Error: File not found - {audio_path}")
                sys.exit(1)
            
            whisper_model = whisper.load_model("base")
            prediction, transcribed_text, score = predict_from_audio(
                audio_path, model, vectorizer, whisper_model
            )
            
            print("\n" + "=" * 50)
            print("PREDICTION RESULT")
            print("=" * 50)
            
            if prediction == 1:
                print("🚨 This audio is likely a SCAM")
            else:
                print("✅ This audio is likely REAL")
            
            print(f"Confidence Score: {abs(score):.4f}")
            
        elif choice == "2":
            print("\nEnter the text to classify (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            text = "\n".join(lines)
            
            if not text.strip():
                print("❌ Error: No text provided")
                sys.exit(1)
            
            prediction, score = predict_from_text(text, model, vectorizer)
            
            print("\n" + "=" * 50)
            print("PREDICTION RESULT")
            print("=" * 50)
            
            if prediction == 1:
                print("🚨 This text is likely a SCAM")
            else:
                print("✅ This text is likely REAL")
            
            print(f"Confidence Score: {abs(score):.4f}")
        
        else:
            print("Invalid choice")
            sys.exit(1)


if __name__ == "__main__":
    main()
