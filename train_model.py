import pickle
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "predictive_maintenance_model.pkl"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def download_dataset() -> Path:
    """Download and extract the AI4I 2020 dataset if it is not already present."""
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "ai4i2020.csv"

    if csv_path.exists():
        print(f"Dataset found at {csv_path}")
        return csv_path

    zip_path = DATA_DIR / "ai4i2020.zip"
    print("Downloading AI4I 2020 Predictive Maintenance Dataset...")
    urlretrieve(DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    extracted_csv_files = list(DATA_DIR.glob("*.csv"))
    if not extracted_csv_files:
        raise FileNotFoundError("No CSV file was found after extracting the dataset.")

    extracted_csv_files[0].rename(csv_path)
    print(f"Dataset saved to {csv_path}")
    return csv_path


def load_dataset() -> pd.DataFrame:
    csv_path = download_dataset()
    df = pd.read_csv(csv_path)

    missing_columns = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return df


def train_model() -> None:
    df = load_dataset()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    report = classification_report(y_test, predictions, target_names=["No Failure", "Failure"])

    random_forest = pipeline.named_steps["model"]
    feature_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    MODEL_DIR.mkdir(exist_ok=True)
    model_package = {
        "model": pipeline,
        "features": FEATURES,
        "target": TARGET,
        "feature_importance": feature_importance,
        "accuracy": float(np.round(accuracy, 4)),
        "roc_auc": float(np.round(roc_auc, 4)),
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model_package, file)

    print("\nModel training complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
