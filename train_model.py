
import os
import re
import string
import pandas as pd
import joblib

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Download required English stopwords
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# Project directories
DATA_PATH = "data/SMSSpamCollection"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def preprocess_text(text):
    """Clean and normalize message text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]

    return " ".join(tokens)


def main():
    print("=" * 60)
    print("        SPAM MAIL DETECTOR - MODEL TRAINING")
    print("=" * 60)

    # Load the real SMS Spam Collection dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place SMSSpamCollection inside the data folder."
        )

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    # Basic dataset validation
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]

    df = df.drop_duplicates(subset=["label", "message"])
    df = df.reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both ham and spam labels.")

    print(f"\nTotal messages: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # Convert labels to numbers: ham=0, spam=1
    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].apply(preprocess_text)

    # Remove messages that become empty after preprocessing
    df = df[df["clean_message"].str.strip() != ""].reset_index(drop=True)

    X = df["clean_message"]
    y = df["label_encoded"]

    # Split data before fitting the vectorizer to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Convert text into numerical TF-IDF features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train the Naive Bayes classifier
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_tfidf, y_train)

    # Evaluate on the unseen test set
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )

    # Save the confusion matrix
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Ham", "Actual Spam"],
        columns=["Predicted Ham", "Predicted Spam"]
    )
    matrix_df.to_csv(os.path.join(RESULTS_DIR, "confusion_matrix.csv"))

    # Save evaluation metrics
    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "training_samples": len(X_train),
        "testing_samples": len(X_test)
    }])
    metrics_df.to_csv(
        os.path.join(RESULTS_DIR, "evaluation_metrics.csv"),
        index=False
    )

    # Save the trained model and matching vectorizer
    joblib.dump(model, os.path.join(MODEL_DIR, "spam_classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    print("\nSaved files:")
    print("- models/spam_classifier.pkl")
    print("- models/tfidf_vectorizer.pkl")
    print("- results/confusion_matrix.csv")
    print("- results/evaluation_metrics.csv")

    print("\nModel training completed successfully!")


if __name__ == "__main__":
    main()
