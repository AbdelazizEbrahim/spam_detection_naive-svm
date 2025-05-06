import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import re
df = pd.read_csv('tweets_dataset.csv')

def preprocess_text(text):
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'but'}
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

cleaned_texts = []
for text in df['Text']:
    cleaned_text = preprocess_text(str(text))
    cleaned_texts.append(cleaned_text)

df['cleaned_text'] = cleaned_texts

le = LabelEncoder()
df['sentiment_encoded'] = le.fit_transform(df['Label'])

# Basic information
print("=== Dataset Overview ===")
print(f"Shape: {df.shape} (rows, columns)")
print("\nColumn Info:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Missing values
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Class distribution
print("\n=== Sentiment Class Distribution ===")
print(df['Label'].value_counts())
plt.figure(figsize=(8, 5))
sns.countplot(x='Label', data=df, order=['positive', 'negative', 'neutral'])
plt.title('Sentiment Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.show()

# Text statistics
print("\n=== Text Statistics ===")
df['text_length'] = df['Text'].apply(lambda x: len(str(x).split()))
df['word_count'] = df['Text'].apply(lambda x: len(str(x).split()))
print(f"Average Text Length (words): {df['text_length'].mean():.2f}")
print(f"Max Text Length (words): {df['text_length'].max()}")
print(f"Min Text Length (words): {df['text_length'].min()}")

# Visualize text length distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Text Length Distribution (Word Count)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.savefig('text_length_distribution.png')
plt.show()
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

print("=== SVM Evaluation ===")
svm_y_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_y_pred, target_names=le.classes_))
svm_cm = confusion_matrix(y_test, svm_y_pred)
svm_cm_normalized = svm_cm.astype('float') / svm_cm.sum(axis=1)[:, np.newaxis]
print("\nSVM Raw Confusion Matrix (rows=actual, columns=predicted):")
print(pd.DataFrame(svm_cm, index=le.classes_, columns=le.classes_))
print("\nSVM Error Analysis:")
svm_errors = []
for i, actual in enumerate(le.classes_):
    for j, predicted in enumerate(le.classes_):
        if i != j and svm_cm[i, j] > 0:
            svm_errors.append((actual, predicted, svm_cm[i, j]))
for actual, predicted, count in svm_errors:
    print(f"Misclassified {count} '{actual}' as '{predicted}'")
plt.figure(figsize=(10, 8))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, 
            cbar_kws={'label': 'Count'})
plt.title('SVM Confusion Matrix (Raw Counts)')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.savefig('svm_confusion_matrix_raw.png')
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(svm_cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, 
            cbar_kws={'label': 'Proportion'})
plt.title('SVM Normalized Confusion Matrix (Proportions)')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.savefig('svm_confusion_matrix_normalized.png')
plt.show()

print("\n=== k-NN Evaluation ===")
knn_y_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print(f"k-NN Accuracy: {knn_accuracy:.4f}")
print("\nk-NN Classification Report:")
print(classification_report(y_test, knn_y_pred, target_names=le.classes_))
knn_cm = confusion_matrix(y_test, knn_y_pred)
knn_cm_normalized = knn_cm.astype('float') / knn_cm.sum(axis=1)[:, np.newaxis]
print("\nk-NN Raw Confusion Matrix (rows=actual, columns=predicted):")
print(pd.DataFrame(knn_cm, index=le.classes_, columns=le.classes_))
print("\nk-NN Error Analysis:")
knn_errors = []
for i, actual in enumerate(le.classes_):
    for j, predicted in enumerate(le.classes_):
        if i != j and knn_cm[i, j] > 0:
            knn_errors.append((actual, predicted, knn_cm[i, j]))
for actual, predicted, count in knn_errors:
    print(f"Misclassified {count} '{actual}' as '{predicted}'")
plt.figure(figsize=(10, 8))
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, 
            cbar_kws={'label': 'Count'})
plt.title('k-NN Confusion Matrix (Raw Counts)')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.savefig('knn_confusion_matrix_raw.png')
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(knn_cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, 
            cbar_kws={'label': 'Proportion'})
plt.title('k-NN Normalized Confusion Matrix (Proportions)')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.savefig('knn_confusion_matrix_normalized.png')
plt.show()

print("\n=== Model Comparison ===")
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"k-NN Accuracy: {knn_accuracy:.4f}")
print(f"Accuracy Difference (SVM - k-NN): {(svm_accuracy - knn_accuracy):.4f}")

import joblib

joblib.dump(svm, 'svm_model.joblib')
print("SVM model saved to 'svm_model.joblib'")

joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
print("TF-IDF vectorizer saved to 'tfidf_vectorizer.joblib'")

joblib.dump(le, 'label_encoder.joblib')
print("LabelEncoder saved to 'label_encoder.joblib'")
