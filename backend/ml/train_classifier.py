
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BACKEND_DIR / "ml" / "features_cifake.npz"

MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "clip_logreg_detector.pkl"


def load_features():
    """
    Load CLIP embeddings and labels from the compressed features file.

    The file `features_cifake.npz` must contain:
        - 'X': 2D array of CLIP embeddings, shape (N, D)
        - 'y': 1D array of integer labels, shape (N,)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (x, y)
            - x: feature matrix of shape (N, D)
            - y: label vector of shape (N,)
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {FEATURES_PATH}. "
            f"Run extract_features.py first."
        )
    data = np.load(FEATURES_PATH)
    x = data["X"]
    y = data["y"]
    print(f"Loaded features: x = {x.shape}, y = {y.shape}")
    return x, y


def split_data(x: np.ndarray, y: np.ndarray):
    """
    Split the dataset into train, validation, and test sets.

    Splits:
        - 70% train
        - 15% validation
        - 15% test

    Stratified splitting is used to preserve label balance.

    Args:
        x (np.ndarray): Feature matrix, shape (N, D)
        y (np.ndarray): Labels vector, shape (N,)

    Returns:
        Tuple of:
            x_train, x_val, x_test, y_train, y_val, y_test
    """
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_logreg(x_train, y_train, x_val, y_val):
    """
    Train a Logistic Regression classifier on CLIP embeddings.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_val: Validation features.
        y_val: Validation labels.

    Returns:
        LogisticRegression: The trained model.
    """
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs",
    )

    print("Training Logistic Regression...")
    clf.fit(x_train, y_train)

    # Evaluate on validation set
    val_preds = clf.predict(x_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_acc:.4f}")

    return clf


def evaluate_model(clf, x_test, y_test):
    """
    Evaluate the trained classifier on the test set.

    Prints:
        - Test accuracy
        - Precision/Recall/F1 report
        - Confusion matrix

    Args:
        clf: Trained classifier.
        x_test: Test features.
        y_test: Test labels.

    Returns:
        None
    """
    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, preds, target_names=["real", "ai"]))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))


def save_model(clf):
    """
    Save the trained classifier to disk.

    Args:
        clf: Trained Logistic Regression classifier.

    Returns:
        None
    """
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved classifier to {MODEL_PATH}")


def main():
    """
    End-to-end training workflow:

        1. Load CLIP embeddings and labels
        2. Split data into train/val/test
        3. Train a Logistic Regression classifier
        4. Evaluate on the test set
        5. Save trained model for FastAPI inference
    """
    # 1. Load embeddings and labels
    x, y = load_features()

    # 2. Split into train/val/test
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

    # 3. Train classifier
    clf = train_logreg(x_train, y_train, x_val, y_val)

    # 4. Evaluate on test set
    evaluate_model(clf, x_test, y_test)

    # 5. Save the trained model
    save_model(clf)


if __name__ == "__main__":
    main()
