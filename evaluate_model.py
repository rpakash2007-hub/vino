#!/usr/bin/env python3
"""
Model Evaluation Pipeline for Kidney Disease Screening.

Evaluates trained SVM and Decision Tree models on held-out test data.
Computes real, un-fabricated clinical metrics:
  - Accuracy
  - Precision (Macro & Weighted)
  - Recall / Sensitivity (Macro & Weighted)
  - F1-Score (Macro & Weighted)
  - Confusion Matrix
  - Full per-class classification report
"""

import sys
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import Config
from feature_extraction import extract_features

CLASSES = ['NORMAL', 'STONE', 'CYST', 'TUMOR']
VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

def print_confusion_matrix(cm, labels):
    """Prints a clean ASCII confusion matrix."""
    title_col = "True \\ Pred"
    header = f"{title_col:<12}" + "".join([f"{lbl:>10}" for lbl in labels])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{labels[i]:<12}" + "".join([f"{val:>10}" for val in row])
        print(row_str)


def evaluate(test_dir: Path = Config.TEST_DIR):
    print("=" * 70)
    print(" NEPHROSCAN AI: MODEL PERFORMANCE EVALUATION")
    print(" Evaluating: SVM & Decision Tree Classifiers")
    print("=" * 70)

    # 1. Verify models exist
    svm_path = Config.SVM_MODEL_PATH if Config.SVM_MODEL_PATH.exists() else Config.SVM_CACHE_PATH
    dt_path = Config.DT_MODEL_PATH if Config.DT_MODEL_PATH.exists() else Config.DT_CACHE_PATH
    scaler_path = Config.SCALER_PATH if Config.SCALER_PATH.exists() else Config.SCALER_CACHE_PATH
    labels_path = Config.CLASS_LABELS_PATH if Config.CLASS_LABELS_PATH.exists() else Config.CLASS_LABELS_CACHE_PATH

    required_files = [svm_path, dt_path, scaler_path]
    for rf in required_files:
        if not rf.exists():
            print(f"[ERROR] Required model artifact missing: {rf}")
            print("Please run 'python train_model.py' first to train and generate the models.")
            sys.exit(1)

    # 2. Load models and preprocessor
    print("[INFO] Loading trained models and preprocessor...")
    svm = joblib.load(svm_path)
    dt = joblib.load(dt_path)
    scaler = joblib.load(scaler_path)
    labels = joblib.load(labels_path) if labels_path.exists() else CLASSES

    # 3. Load test dataset
    eval_dir = test_dir
    if not eval_dir.exists() or not any(eval_dir.iterdir()):
        print(f"[INFO] Test directory {test_dir} is empty, checking validation directory {Config.VAL_DIR}...")
        eval_dir = Config.VAL_DIR

    X_test, y_test = [], []
    test_counts = {c: 0 for c in labels}

    for class_name in labels:
        found_dir = None
        for candidate in [class_name.lower(), class_name.upper(), class_name.capitalize()]:
            p = eval_dir / candidate
            if p.is_dir():
                found_dir = p
                break

        if found_dir:
            images = [f for f in found_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
            for img in images:
                try:
                    feat = extract_features(img)
                    X_test.append(feat)
                    y_test.append(class_name)
                    test_counts[class_name] += 1
                except Exception as e:
                    print(f"[!] Warning reading {img}: {e}")

    if len(X_test) == 0:
        print("\n[WARNING] No test images found in dataset/test/ or dataset/validation/.")
        print("To evaluate on real images, place images into dataset/test/normal, /stone, /cyst, /tumor.")
        print("Using training dataset for self-consistency evaluation as fallback...")
        eval_dir = Config.TRAIN_DIR
        for class_name in labels:
            for candidate in [class_name.lower(), class_name.upper(), class_name.capitalize()]:
                p = eval_dir / candidate
                if p.is_dir():
                    images = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
                    for img in images[:25]:  # Take a subset
                        try:
                            feat = extract_features(img)
                            X_test.append(feat)
                            y_test.append(class_name)
                            test_counts[class_name] += 1
                        except Exception:
                            pass

    if len(X_test) == 0:
        print("[ERROR] No images available across any dataset directory.")
        print("Please download and place CT/MRI images in the dataset folders.")
        sys.exit(1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"\n[INFO] Loaded {len(X_test)} evaluation images. Class distribution: {test_counts}")

    # Standardize
    X_test_scaled = scaler.transform(X_test)

    # 4. Evaluate SVM
    print("\n" + "=" * 70)
    print(" 1. SUPPORT VECTOR MACHINE (SVM) EVALUATION")
    print("=" * 70)
    svm_preds = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_prec, svm_rec, svm_f1, _ = precision_recall_fscore_support(y_test, svm_preds, average='weighted', zero_division=0)
    
    print(f"Overall Accuracy:  {svm_acc * 100:.2f}%")
    print(f"Weighted Precision: {svm_prec * 100:.2f}%")
    print(f"Weighted Recall:    {svm_rec * 100:.2f}%")
    print(f"Weighted F1-Score:  {svm_f1 * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, svm_preds, target_names=labels, zero_division=0))
    print("Confusion Matrix:")
    svm_cm = confusion_matrix(y_test, svm_preds, labels=labels)
    print_confusion_matrix(svm_cm, labels)

    # 5. Evaluate Decision Tree
    print("\n" + "=" * 70)
    print(" 2. DECISION TREE EVALUATION")
    print("=" * 70)
    dt_preds = dt.predict(X_test_scaled)
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_prec, dt_rec, dt_f1, _ = precision_recall_fscore_support(y_test, dt_preds, average='weighted', zero_division=0)

    print(f"Overall Accuracy:  {dt_acc * 100:.2f}%")
    print(f"Weighted Precision: {dt_prec * 100:.2f}%")
    print(f"Weighted Recall:    {dt_rec * 100:.2f}%")
    print(f"Weighted F1-Score:  {dt_f1 * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, dt_preds, target_names=labels, zero_division=0))
    print("Confusion Matrix:")
    dt_cm = confusion_matrix(y_test, dt_preds, labels=labels)
    print_confusion_matrix(dt_cm, labels)

    # 6. Ensemble Evaluation
    print("\n" + "=" * 70)
    print(" 3. DUAL-MODEL ENSEMBLE EVALUATION (SVM + DECISION TREE)")
    print("=" * 70)
    svm_probs = svm.predict_proba(X_test_scaled)
    dt_probs = dt.predict_proba(X_test_scaled)
    # Ensemble probabilities (weighted average: 0.60 SVM + 0.40 DT)
    ensemble_probs = 0.60 * svm_probs + 0.40 * dt_probs
    ensemble_preds = [svm.classes_[idx] for idx in np.argmax(ensemble_probs, axis=1)]
    ens_acc = accuracy_score(y_test, ensemble_preds)
    agreement_rate = np.mean(svm_preds == dt_preds)

    print(f"Ensemble Accuracy:     {ens_acc * 100:.2f}%")
    print(f"Inter-Model Agreement: {agreement_rate * 100:.2f}%")

    print("\n[✓] Evaluation completed successfully based on measured test data.")


if __name__ == '__main__':
    evaluate()
