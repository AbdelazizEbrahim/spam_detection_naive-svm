Spam Classification with Naïve Bayes and SVM

This repository contains two Python scripts for spam classification using Naïve Bayes and Support Vector Machine (SVM) algorithms. Each script implements a complete pipeline for loading, cleaning, and classifying text data as spam or not spam, with detailed logging.

Features





Naïve Bayes Classifier: Probabilistic model for fast spam detection (naive_bayes_classifier.py).



SVM Classifier: Discriminative model for robust spam detection (svm_classifier.py).



Data Preprocessing: Cleans and prepares text datasets for classification.



TF-IDF Vectorization: Converts text into numerical features for model training.



Interactive Mode: Allows users to input text for real-time spam classification.



Logging: Saves detailed logs to files (naive_bayes.log, svm.log) and console.

Prerequisites





Python 3.8+



Required libraries: pandas, numpy, scikit-learn



A dataset in tab-separated format with labels (ham or spam) and text.

Installation





Clone the repository:

git clone https://github.com/your-username/spam-classification.git
cd spam-classification



Install dependencies:

pip install pandas numpy scikit-learn



Place your dataset (e.g., dataset.csv) in the data/ directory.

Usage





Run the Naïve Bayes classifier:

python naive_bayes_classifier.py



Run the SVM classifier:

python svm_classifier.py



Follow the interactive prompt to input text for classification or type exit to quit.

Dataset Format

The dataset should be a tab-separated file (data/dataset.csv) with two columns:





First column: Label (ham for not spam, spam for spam).



Second column: Text content.

Example:

ham	This is a normal message
spam	Win a free prize now!!!

Output





Logs: Saved to naive_bayes.log or svm.log with timestamps, metrics, and predictions.



Cleaned Data: Saved as data/cleaned_dataset_nb.csv or data/cleaned_dataset_svm.csv.



Console: Displays model performance (accuracy, precision, recall, F1-score) and interactive results.

License

This project is licensed under the MIT License.