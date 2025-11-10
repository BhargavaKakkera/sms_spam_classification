import os
import time
import numpy as np
import pandas as pd
import joblib
import warnings
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"


@dataclass
class ModelTrainerConfig:
    model_dir = os.path.join("models")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        print(" Starting model training")
        overall_start = time.time()

        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            #  Base models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
                "Multinomial NB": MultinomialNB(),
                "Linear SVM": LinearSVC(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42)
            }

            #  Parameter grids (as used in Colab)
            param_grids = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs"]
                },
                "Multinomial NB": {
                    "alpha": [0.01, 0.05, 0.1, 0.5, 1.0]
                },
                "Linear SVM": {
                    "C": [0.01, 0.1, 1, 10],
                    "loss": ["squared_hinge"],
                    "class_weight": ["balanced"],
                    "max_iter": [3000]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "criterion": ["gini", "entropy"]
                }
            }

            results = []
            tuned_models = {}
            best_f1 = 0
            best_model = None
            best_name = None

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            #  Tune & evaluate each model
            for name, model in models.items():
                print(f"\n Tuning {name}...")
                start_time = time.time()

                grid = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=cv,
                    scoring="f1",
                    n_jobs=-1,
                    error_score="raise"
                )
                grid.fit(X_train, y_train)
                best_estimator = grid.best_estimator_
                tuned_models[name] = best_estimator

                y_pred = best_estimator.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-score": f1
                })

                print(f"[RESULT] {name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
                print(f"⏱ Time taken: {(time.time() - start_time):.2f} sec")
                print(f" Best {name} params: {grid.best_params_}")

                if f1 > best_f1:
                    best_f1, best_model, best_name = f1, best_estimator, name

            #  Display results
            df_results = pd.DataFrame(results).sort_values(by="F1-score", ascending=False).reset_index(drop=True)
            print("\n [SUMMARY] Model Performance (sorted by F1):\n", df_results)

            #  Save best model (unique filename per run)
            os.makedirs(self.config.model_dir, exist_ok=True)
            save_path = os.path.join(self.config.model_dir, f"{best_name.replace(' ', '_')}_model.pkl")
            joblib.dump(best_model, save_path)
            print(f"\n Saved best model: {best_name} (F1={best_f1:.4f}) → {save_path}")

            total_time = (time.time() - overall_start) / 60
            print(f"\n [DONE] Total training time: {total_time:.2f} minutes")

        except Exception as e:
            print(" Error during model training:", str(e))


if __name__ == "__main__":
    print("[INFO] Loading transformed data...")
    try:
        train_arr = joblib.load("models/train_arr.pkl")
        test_arr = joblib.load("models/test_arr.pkl")
        print("[INFO] Data loaded successfully.")
        print("Train shape:", train_arr.shape, " Test shape:", test_arr.shape)

        trainer = ModelTrainer()
        trainer.initiate_model_trainer(train_arr, test_arr)

    except Exception as e:
        import traceback
        print(" Model training pipeline failed:")
        traceback.print_exc()
