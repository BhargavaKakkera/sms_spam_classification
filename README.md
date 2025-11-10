# sms_spam_classification

# SMS Spam Classification using ML Models and Neural Networks

This project classifies SMS messages as Spam or Ham (Not Spam) using a combination of Machine Learning models and a Neural Network.

# Overview

The goal is to detect spam messages from text data efficiently.
The workflow includes:

Data preprocessing and cleaning

Feature extraction using TF-IDF for ML models

Tokenization and padding for the Neural Network

Model training, evaluation, and comparison

The deep learning model was trained and evaluated in Google Colab(notebook).

# Features

Text preprocessing (punctuation removal, stopword removal, stemming)

Multiple ML model comparisons — Logistic Regression, Naive Bayes, Random Forest,Linear SVM

Deep Learning model (Feedforward Neural Network) built using tensorflow

Evaluation using Accuracy, Precision, Recall, and F1-score

Saved models and vectorizers for local prediction


# Best models report

| Model                      | Accuracy  | F1-score |
| -------------------------- | --------- | -------- |
| Random Forest              | 99.2%     | 0.967    |
| Neural Network             | 98.5%     | 0.93     |



### 🗂️ Project Structure


'''
sms_spam_classification/
│
├── spam.csv                       # Dataset
│
├── models/                        # Saved models
│   ├── Random_Forest_model.pkl
│   └── Vectorizer.pkl
│
├── notebooks/                     # Jupyter notebook
│   └── spam1.ipynb
│
├── src/                           # Core Python scripts
│   ├── data_transformation.py
│   ├── model_trainer.py
│   └── predict_pipeline.py
│
├── requirements.txt               # Dependencies
└── README.md

'''


