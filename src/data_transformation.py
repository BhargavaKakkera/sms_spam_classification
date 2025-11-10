import os
import pandas as pd
import numpy as np
import joblib
import spacy
import warnings
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

# Suppress spaCy version mismatch warning
warnings.filterwarnings("ignore", message="[W095]")

# Load SpaCy model once
print(" Loading SpaCy model (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def lemmatize_text(text):
    """Lemmatize text and remove stopwords."""
    doc = nlp(str(text))
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("models", "vectorizer.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, data_path):
        print(" Transforming data (SpaCy Lemmatization + TF-IDF)...")

        # 1️. Load dataset
        df = pd.read_csv(data_path, encoding='latin-1')[['v2', 'v1']]
        df.columns = ['message', 'label']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

        # 2️. Lemmatize text
        print(" Cleaning & lemmatizing messages...")
        df['message'] = df['message'].apply(lemmatize_text)

        # 3️. Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'], df['label'], test_size=0.2, random_state=42
        )

        # 4️. TF-IDF vectorization (word + char)
        word_vec = TfidfVectorizer(
            sublinear_tf=True, strip_accents='unicode',
            analyzer='word', token_pattern=r'\w{1,}',
            ngram_range=(1,3), max_features=10000
        )

        char_vec = TfidfVectorizer(
            sublinear_tf=True, strip_accents='unicode',
            analyzer='char', ngram_range=(2,5),
            max_features=10000
        )

        tfidf = FeatureUnion([('word', word_vec), ('char', char_vec)])
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # 5️ Save vectorizer + data
        os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
        joblib.dump(tfidf, self.config.preprocessor_path)
        print(f" Vectorizer saved at: {self.config.preprocessor_path}")

        train_arr = np.c_[X_train_tfidf.toarray(), np.array(y_train)]
        test_arr = np.c_[X_test_tfidf.toarray(), np.array(y_test)]

        joblib.dump(train_arr, "models/train_arr.pkl")
        joblib.dump(test_arr, "models/test_arr.pkl")

        print(" Transformation complete.")
        print("   Train shape:", train_arr.shape)
        print("   Test shape:", test_arr.shape)


#  Script entry point
if __name__ == "__main__":
    transformer = DataTransformation()
    transformer.initiate_data_transformation("spam.csv")
