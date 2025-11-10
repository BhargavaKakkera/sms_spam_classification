import os
import joblib
import numpy as np
import pandas as pd

# Paths
MODEL_PATH = os.path.join("models", "Random_Forest_model.pkl")  
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

# Load model and vectorizer
print(" Loading model and vectorizer...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print(" Loaded successfully.\n")

def predict_spam(messages):
    """Predict spam/ham for list of messages"""
    if isinstance(messages, str):
        messages = [messages]

    # Transform text → features
    X_tfidf = vectorizer.transform(messages)

    # Predict
    preds = model.predict(X_tfidf)
    probs = (
        model.predict_proba(X_tfidf)[:, 1]
        if hasattr(model, "predict_proba")
        else np.zeros(len(preds))
    )

    # Convert to readable output
    df = pd.DataFrame({
        "Message": messages,
        "Prediction": ["SPAM" if p == 1 else "HAM" for p in preds],
        "Confidence": np.round(probs, 3)
    })
    return df


if __name__ == "__main__":
    print("\n SMS Spam Classifier\n")
    print("Enter a custom message to test or press Enter to use default examples.\n")

    user_input = input("Enter message: ").strip()

    if user_input == "":
        print("\n No custom input provided — using default sample messages...\n")
        test_messages = [
            "Congratulations! You've won a $1000 Walmart gift card. Click to claim now!",
            "C@ngratulati@ns You  w@on a $1000  g!ft card. Click to claim now",
            "Hey, are we still meeting at 5 today?",
            "Your account will be blocked. Update your details immediately!",
            "Free entry to win iPhone 15 Pro. Reply YES!"

        ]
    else:
        print(f"\n Using your input: {user_input}\n")
        test_messages = [user_input]

    results = predict_spam(test_messages)
    print("\n Predictions:\n")
    print(results.to_string(index=False))
