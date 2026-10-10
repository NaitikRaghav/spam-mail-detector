# Spam Mail Detector Using Machine Learning

A Machine Learning project that classifies text messages as spam or legitimate (ham) using TF-IDF text vectorization and a trained classification model. It includes model evaluation and an interactive Streamlit application.

## Features

- Text message classification
- Text preprocessing
- TF-IDF feature extraction
- Trained spam classification model
- Evaluation using accuracy, precision, recall, and F1-score
- Confusion matrix and evaluation results
- Interactive Streamlit web application

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK / Scikit-learn text preprocessing utilities
- Joblib
- Streamlit
- Matplotlib and Seaborn

## Project Structure

```text
spam_detector/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── models/
└── results/
```

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/NaitikRaghav/spam-mail-detector.git
   cd spam-mail-detector
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model if the saved model files are not available:

   ```bash
   python train_model.py
   ```

5. Launch the application:

   ```bash
   streamlit run app.py
   ```

## Dataset

The project uses the UCI SMS Spam Collection dataset, containing text messages labelled as spam or ham.

## Model Evaluation

In the recorded training run, the model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 98.54% |
| Spam precision | 1.0000 |
| Spam recall | 0.8828 |
| Spam F1-score | 0.9378 |

These results describe the recorded test run. Performance may vary with different data splits or model settings.

## Limitations

This model was developed for an academic demonstration using labelled SMS messages. It may not correctly classify every message, particularly messages that differ from the training data.

## Author

Naitik Raghav

GitHub: https://github.com/NaitikRaghav

## Purpose

Created as an academic Machine Learning project to practise text preprocessing, feature extraction, classification, model evaluation, and application deployment.
