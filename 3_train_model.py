# ========================================
# 🤖 Train Scam Detection Model
# Using TF-IDF + Linear SVM
# ========================================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def train_model(csv_path, test_size=0.2, random_state=42):
    """
    Train a scam detection model using TF-IDF and Linear SVM
    
    Args:
        csv_path: Path to the dataset CSV file
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    
    print("=" * 50)
    print("Training Scam Detection Model")
    print("=" * 50)
    
    # Load dataset
    print(f"\nLoading dataset from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"Dataset loaded: {len(df_shuffled)} samples")
    print(f"Real (0): {len(df_shuffled[df_shuffled['label']==0])}")
    print(f"Scam (1): {len(df_shuffled[df_shuffled['label']==1])}")
    
    # Split features and labels
    X = df_shuffled['conversation']
    y = df_shuffled['label']
    
    # Train-test split
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # TF-IDF Vectorization
    print("\nPerforming TF-IDF vectorization...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF feature dimensions: {X_train_tfidf.shape[1]}")
    
    # Train Linear SVM
    print("\nTraining Linear SVM model...")
    model_svm = LinearSVC(random_state=random_state, max_iter=2000)
    model_svm.fit(X_train_tfidf, y_train)
    print("✅ Model training completed!")
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = model_svm.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Display results
    print("\n" + "=" * 50)
    print("Model Performance Metrics")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['Real', 'Scam']))
    
    print("=" * 50)
    print("Confusion Matrix")
    print("=" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n[[TN FP]")
    print(" [FN TP]]")
    
    # Save model and vectorizer
    print("\n" + "=" * 50)
    print("Saving Model and Vectorizer")
    print("=" * 50)
    
    with open('scam_detector_model.pkl', 'wb') as f:
        pickle.dump(model_svm, f)
    print("✅ Model saved as 'scam_detector_model.pkl'")
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print("✅ Vectorizer saved as 'tfidf_vectorizer.pkl'")
    
    return model_svm, tfidf_vectorizer


if __name__ == "__main__":
    # Configuration
    CSV_PATH = "transcripts_dataset.csv"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Train the model
    model, vectorizer = train_model(CSV_PATH, TEST_SIZE, RANDOM_STATE)
