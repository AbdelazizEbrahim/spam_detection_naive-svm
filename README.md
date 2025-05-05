Spam Classification with Naïve Bayes and SVM 🚀

This repository contains two powerful Python scripts for spam classification using Naïve Bayes and Support Vector Machine (SVM) algorithms. Each script implements a complete pipeline for loading, cleaning, and classifying text data as spam or not_spam, with detailed logging and an interactive mode for real-time predictions.
📑 Table of Contents

Overview
Features
Prerequisites
Installation
Dataset Format
Usage
Example Output
Logs and Artifacts
Contributing
License
Resources

🌟 Overview
This project provides a robust solution for detecting spam messages using two machine learning models:

Naïve Bayes: A probabilistic model for fast and efficient spam detection.
SVM: A discriminative model for high-accuracy spam classification.

The pipeline processes a text dataset (e.g., SMS Spam Collection), cleans it, converts text to numerical features using TF-IDF, trains the models, evaluates performance, and allows users to classify new text interactively. Detailed logs track every step, making it ideal for both learning and production use.
✨ Features

Naïve Bayes Classifier (naive_classifier.py): Fast probabilistic model for spam detection.
SVM Classifier (svm_model.py): Robust discriminative model for improved accuracy.
Data Preprocessing: Cleans and prepares text datasets by removing invalid entries and normalizing data.
TF-IDF Vectorization: Converts text into numerical features with stop-word removal and n-gram support.
Interactive Mode: Classify text in real-time by typing messages in the terminal.
Comprehensive Logging: Saves pipeline progress, metrics, and predictions to naive_bayes.log or svm.log and the console.
Performance Metrics: Reports accuracy, precision, recall, and F1-score for model evaluation.
Feature Analysis: Identifies top features contributing to spam predictions.

🛠️ Prerequisites

Python 3.8+
Libraries:
pandas
numpy
scikit-learn

Dataset: A tab-separated file (data/dataset.csv) with label (ham or spam) and text columns.

📦 Installation

Clone the Repository:
git clone https://github.com/AbdelazizEbrahim/spam_detection_naive-svm.git
cd spam-classification

Install Dependencies:
pip install pandas numpy scikit-learn

Prepare the Dataset:

Place your dataset (e.g., dataset.csv) in the data/ directory.
Ensure it follows the format described below.

📋 Dataset Format
The dataset must be a tab-separated file (data/dataset.csv) with two columns:

Label: ham (not spam) or spam.
Text: The message content.

Example:
ham Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121...

🚀 Usage

Run the Naïve Bayes Classifier:
python naive_classifier.py

Run the SVM Classifier:
python svm_model.py

Interactive Mode:

After running either script, the pipeline will:
Load and clean the dataset.
Train the model and display performance metrics.
Enter interactive mode, prompting you to enter text.

Type a message to classify it as spam or not_spam.
Type exit to quit or press Ctrl+C to interrupt.

📈 Example Output
Below are example interactions based on the SMS Spam Collection dataset (5,574 samples: 4,827 not_spam, 747 spam).
Model Performance

Naïve Bayes:--- Naïve Bayes Results ---
Accuracy: 0.9776
Precision (spam): 0.9921
Recall (spam): 0.8389
F1-score (spam): 0.9091

SVM:--- SVM Results ---
Accuracy: 0.9848
Precision (spam): 0.9521
Recall (spam): 0.9329
F1-score (spam): 0.9424

Interactive Mode Example
Input:
FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv

Naïve Bayes Output:
Prediction: not_spam (Time: 0.0018s)
Top features: [('50', 0.0009), ('send', 0.0008), ('week', 0.0008), ('word', 0.0004), ('freemsg', 0.0003)]

SVM Output:
Prediction: spam (Time: 0.0019s)
Top features: [('50', 1.7381), ('std', 0.9097), ('freemsg', 0.8272), ('hey', -0.7746), ('ok', -0.7111)]

More Examples:

Not Spam:Input: Nah I don't think he goes to usf, he lives around here though
Naïve Bayes: not_spam (Time: 0.0024s)
SVM: not_spam (Time: 0.0016s)

Spam:Input: Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free!...
Naïve Bayes: spam (Time: 0.0020s)
SVM: spam (Time: 0.0020s)

📜 Logs and Artifacts

Logs:
Naïve Bayes: Saved to naive_bayes.log.
SVM: Saved to svm.log.
Includes timestamps, dataset details, metrics, top features, and predictions.

Cleaned Data:
Naïve Bayes: data/cleaned_dataset_nb.csv
SVM: data/cleaned_dataset_svm.csv

Console Output:
Displays real-time progress, model performance, and interactive results.

Example Log Snippet:
2025-05-06 02:02:24,154 - INFO - Loaded 5574 valid samples
2025-05-06 02:02:24,156 - INFO - Class distribution: {'not_spam': 4827, 'spam': 747}
2025-05-06 02:02:24,329 - INFO - Vocabulary size: 9977

🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m "Add YourFeature").
Push to the branch (git push origin feature/YourFeature).
Open a Pull Request.

Please include tests and update the documentation as needed.
📄 License
This project is licensed under the MIT License. The SMS Spam Collection dataset is publicly available for research purposes.
🔗 Resources

SMS Spam Collection Dataset
Scikit-learn Documentation
Naïve Bayes Explanation
SVM Explanation

Built with ❤️ for spam-free communication!
