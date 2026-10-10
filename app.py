
import os
import re
import string
import joblib
import streamlit as st

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Paths to trained model files
MODEL_PATH = "models/spam_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    """Apply the same preprocessing used during model training."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]

    return " ".join(tokens)


# Page configuration
st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="🛡️",
    layout="centered"
)

# Load the trained model and vectorizer
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run: python train_model.py"
        )

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "TF-IDF vectorizer not found. Run: python train_model.py"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


st.title("🛡️ Spam Mail Detector")
st.write(
    "An AI/ML application that classifies text messages "
    "as spam or legitimate (ham) using Natural Language Processing."
)

st.divider()

st.subheader("Analyze a message")

message = st.text_area(
    "Enter your message below:",
    placeholder="Example: Congratulations! You have won a free prize. Claim now!",
    height=160
)

if st.button("Check Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        try:
            model, vectorizer = load_artifacts()

            cleaned_message = preprocess_text(message)

            if not cleaned_message.strip():
                st.warning(
                    "This message contains no usable words. "
                    "Please enter a more descriptive message."
                )

            else:
                features = vectorizer.transform([cleaned_message])
                prediction = model.predict(features)[0]

                probabilities = model.predict_proba(features)[0]
                spam_index = list(model.classes_).index(1)
                spam_probability = probabilities[spam_index]

                st.divider()
                st.subheader("Prediction Result")

                if prediction == 1:
                    st.error("🚨 SPAM MESSAGE DETECTED")
                else:
                    st.success("✅ MESSAGE CLASSIFIED AS NOT SPAM")

                st.metric(
                    "Estimated spam probability",
                    f"{spam_probability * 100:.2f}%"
                )

                st.caption(
                    "This probability is the model's estimate, "
                    "not a guarantee. Review suspicious messages carefully."
                )

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"An unexpected error occurred: {error}")

st.divider()

with st.expander("About this project"):
    st.markdown(
        """
        **Machine Learning:** Multinomial Naive Bayes

        **Feature Extraction:** TF-IDF with unigrams and bigrams

        **Text Processing:** Lowercasing, URL and email normalization,
        punctuation removal, and stopword removal

        **Evaluation:** Accuracy, precision, recall, F1-score,
        and confusion matrix

        **Dataset:** UCI SMS Spam Collection
        """
    )

st.caption("Academic Machine Learning Project | Spam Classification")
