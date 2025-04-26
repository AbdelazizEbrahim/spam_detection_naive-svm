import pandas as pd
import time
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import re
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('svm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Timer utility
def log_time(message, start_time):
    elapsed = time.time() - start_time
    logger.info(f"{message} took {elapsed:.4f} seconds")
    return elapsed

def preview_file(file_path, num_lines=5):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logger.info(f"Previewing first {num_lines} lines of {file_path}:")
            for i, line in enumerate(f, 1):
                if i > num_lines:
                    break
                logger.info(f"Line {i}: {line.strip()}")
    except Exception as e:
        logger.error(f"Error previewing file: {e}")

def clean_dataset(file_path, output_path='data/cleaned_dataset_svm.csv'):
    logger.info(f"Cleaning dataset: {file_path}")
    start = time.time()
    cleaned_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                line = re.sub(r'\t+', '\t', line)
                line = line.replace('"', '')
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    label, text = parts[0], ' '.join(parts[1:])
                    cleaned_lines.append(f"{label}\t{text}")
                else:
                    logger.warning(f"Skipping malformed line: {line.strip()}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cleaned_lines))
        logger.info(f"Cleaned dataset saved to {output_path}")
        log_time("Dataset cleaning", start)
        return output_path
    except Exception as e:
        logger.error(f"Error cleaning dataset: {e}")
        raise

def load_and_preprocess_data(file_path='data/dataset.csv'):
    start = time.time()
    preview_file(file_path)
    cleaned_file_path = clean_dataset(file_path)
    try:
        df = pd.read_csv(
            cleaned_file_path,
            sep='\t',
            header=None,
            names=['label', 'text'],
            engine='python',
            encoding='utf-8',
            quoting=3
        )
        df['label'] = df['label'].map({'ham': 'not_spam', 'spam': 'spam'})
        df = df.dropna(subset=['label', 'text'])
        df = df[df['label'].isin(['not_spam', 'spam'])]
        logger.info(f"Loaded {len(df)} valid samples")
        logger.info(f"Class distribution: {df['label'].value_counts().to_dict()}")
        log_time("Data loading and preprocessing", start)
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def vectorize_text(X_train, X_test):
    start = time.time()
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=20000,
        ngram_range=(1, 3),
        min_df=2
    )
    X_train_vect = vectorizer.fit_transform(X_train).astype(np.float32)
    X_test_vect = vectorizer.transform(X_test).astype(np.float32)
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    log_time("Text vectorization", start)
    return vectorizer, X_train_vect, X_test_vect

def train_svm(X_train, y_train):
    start = time.time()
    model = LinearSVC(C=1.0, max_iter=1000, dual=False, class_weight='balanced')
    model.fit(X_train, y_train)
    logger.info("SVM model trained successfully")
    log_time("SVM training", start)
    return model

def evaluate_model(model, X_test, y_test):
    start = time.time()
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    logger.info("\n--- SVM Results ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (spam): {report['spam']['precision']:.4f}")
    logger.info(f"Recall (spam): {report['spam']['recall']:.4f}")
    logger.info(f"F1-score (spam): {report['spam']['f1-score']:.4f}")
    log_time("SVM evaluation", start)
    return preds, accuracy

def explain_prediction(model, vectorizer, text):
    start = time.time()
    text_vect = vectorizer.transform([text]).toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    top_features = sorted(
        [(feature_names[i], weights[i]) for i in np.where(text_vect > 0)[0]],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    logger.info(f"Top features for prediction: {top_features}")
    log_time("Feature explanation", start)

def classify_text(model, vectorizer, text):
    start = time.time()
    text_vect = vectorizer.transform([text]).astype(np.float32)
    pred = model.predict(text_vect)[0]
    elapsed = log_time("Single text prediction", start)
    return pred, elapsed

def main():
    logger.info("Starting SVM spam classification pipeline")
    try:
        df = load_and_preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        vectorizer, X_train_vect, X_test_vect = vectorize_text(X_train, X_test)
        
        svm_model = train_svm(X_train_vect, y_train)
        svm_preds, svm_accuracy = evaluate_model(svm_model, X_test_vect, y_test)
        
        # Log top SVM features
        feature_names = vectorizer.get_feature_names_out()
        svm_weights = svm_model.coef_[0]
        top_svm_features = sorted(zip(feature_names, svm_weights), key=lambda x: abs(x[1]), reverse=True)[:10]
        logger.info("Top SVM features: %s", top_svm_features)
        
        logger.info("\n=== Interactive Spam Detector ===")
        while True:
            user_input = input("\nEnter text to classify (or 'exit' to quit):\n> ")
            if user_input.lower() == 'exit':
                logger.info("Exiting interactive mode")
                break
            pred, elapsed = classify_text(svm_model, vectorizer, user_input)
            logger.info(f"Prediction: {pred} (Time: {elapsed:.4f}s)")
            explain_prediction(svm_model, vectorizer, user_input)
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()