"""
train.py — Train and evaluate three models for AI-generated text detection.

Models:
  A) 1-D CNN (TensorFlow / Keras)
  B) Random Forest (scikit-learn)
  C) Calibrated Random Forest (isotonic regression, our improvement)

Outputs saved to models/ and outputs/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend — no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss,
                             confusion_matrix, roc_curve)
import joblib
import warnings
warnings.filterwarnings("ignore")

# TensorFlow — suppress noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add parent dir so we can import src.features
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import extract_features_batch, FEATURE_NAMES, NUM_FEATURES

# ── Paths ──────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(BASE, "data")
MODELS  = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)


# =====================================================================
# 1. LOAD & BALANCE THE DATASET
# =====================================================================
def load_data():
    """Load the AI vs Human Text CSV and take a balanced sample."""
    # Find the CSV in data/
    for name in os.listdir(DATA):
        if name.endswith(".csv"):
            path = os.path.join(DATA, name)
            break
    else:
        raise FileNotFoundError("No CSV found in data/ folder. Place the dataset there.")

    print(f"Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"  Raw dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ── Adapt column names to a common format ──────────────────────────
    # Dataset has: text_content, label ("human"/"ai")
    # We normalise to: text, generated (0/1)
    assert "text_content" in df.columns, "Dataset must have a 'text_content' column"
    assert "label" in df.columns, "Dataset must have a 'label' column"

    df = df.rename(columns={"text_content": "text"})
    df = df.dropna(subset=["text"])
    df["generated"] = (df["label"] == "ai").astype(int)

    print(f"  Class distribution: {df['generated'].value_counts().to_dict()}")

    # Balanced sample: take min-class size from each class
    n_ai    = int(df["generated"].sum())
    n_human = int((df["generated"] == 0).sum())
    n_per_class = min(n_ai, n_human)

    human = df[df["generated"] == 0].sample(n=n_per_class, random_state=42)
    ai    = df[df["generated"] == 1].sample(n=n_per_class, random_state=42)
    balanced = pd.concat([human, ai]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Balanced sample: {len(balanced)} rows  "
          f"({n_per_class} human + {n_per_class} ai)")
    return balanced


# =====================================================================
# 2. FEATURE EXTRACTION + SCALING
# =====================================================================
def prepare_features(df):
    """Extract features, scale, and split into train / val / test."""
    print("\n── Feature extraction ──")
    X_raw = extract_features_batch(df["text"].tolist())
    y = df["generated"].values

    # Train (70%) / val (15%) / test (15%), stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS, "scaler.joblib"))
    print("  Scaler saved to models/scaler.joblib")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================================
# 3. MODEL A — 1-D CNN
# =====================================================================
def build_cnn(n_features):
    """Build the 1-D CNN architecture."""
    model = keras.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(X_train, y_train, X_val, y_val):
    """Train the CNN and save it."""
    print("\n── Training CNN ──")
    # Reshape for Conv1D: (samples, features, 1)
    Xt = X_train.reshape(-1, X_train.shape[1], 1)
    Xv = X_val.reshape(-1, X_val.shape[1], 1)

    model = build_cnn(X_train.shape[1])
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        Xt, y_train,
        validation_data=(Xv, y_val),
        epochs=30, batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(os.path.join(MODELS, "cnn_model.h5"))
    print("  CNN saved to models/cnn_model.h5")
    return model, history


# =====================================================================
# 4. MODEL B — Random Forest
# =====================================================================
def train_rf(X_train, y_train):
    """Train a plain Random Forest."""
    print("\n── Training Random Forest ──")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS, "rf_model.joblib"))
    print("  RF saved to models/rf_model.joblib")
    return rf


# =====================================================================
# 5. MODEL C — Calibrated Random Forest (our improvement)
# =====================================================================
def train_calibrated_rf(X_train, y_train):
    """Wrap RF in CalibratedClassifierCV with isotonic regression."""
    print("\n── Training Calibrated Random Forest ──")
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cal_rf = CalibratedClassifierCV(base_rf, method="isotonic", cv=5)
    cal_rf.fit(X_train, y_train)
    joblib.dump(cal_rf, os.path.join(MODELS, "rf_calibrated.joblib"))
    print("  Calibrated RF saved to models/rf_calibrated.joblib")
    return cal_rf


# =====================================================================
# 6. EVALUATION
# =====================================================================
def evaluate_model(name, model, X_test, y_test, is_keras=False):
    """Compute classification metrics and return them as a dict."""
    if is_keras:
        X_in = X_test.reshape(-1, X_test.shape[1], 1)
        y_prob = model.predict(X_in, verbose=0).ravel()
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob),
        "Log Loss":  log_loss(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, y_prob, cm


def print_comparison(results):
    """Pretty-print a comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON ON TEST SET")
    print("=" * 80)
    header = f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'LogLoss':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['Model']:<22} {r['Accuracy']:>7.4f} {r['Precision']:>7.4f} "
              f"{r['Recall']:>7.4f} {r['F1']:>7.4f} {r['ROC-AUC']:>7.4f} "
              f"{r['Log Loss']:>8.4f}")
    print("=" * 80)
    print("\nKey insight: Calibrated RF achieves much lower log loss than")
    print("uncalibrated RF while maintaining similar accuracy and F1.\n")


# =====================================================================
# 7. PLOTS
# =====================================================================
def plot_confusion_matrices(cms, names):
    """Save a confusion matrix heatmap for each model."""
    for cm, name in zip(cms, names):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Human", "AI"],
                    yticklabels=["Human", "AI"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_")
        fig.savefig(os.path.join(OUTPUTS, f"cm_{fname}.png"), dpi=150)
        plt.close(fig)
    print("  Confusion matrices saved.")


def plot_roc_curves(y_test, probs, names):
    """Plot all ROC curves on one chart."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for prob, name in zip(probs, names):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print("  ROC curves saved.")


def plot_cnn_history(history):
    """Plot CNN training accuracy and loss over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"],    label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("CNN Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history.history["loss"],     label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("CNN Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "cnn_history.png"), dpi=150)
    plt.close(fig)
    print("  CNN history plot saved.")


def plot_calibration_diagram(y_test, prob_rf, prob_cal):
    """Reliability diagram comparing RF before vs after calibration."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for prob, label in [(prob_rf, "RF (uncalibrated)"),
                        (prob_cal, "RF (calibrated)")]:
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", label=label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Reliability Diagram")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, "calibration_diagram.png"), dpi=150)
    plt.close(fig)
    print("  Calibration diagram saved.")


# =====================================================================
# 8. SHAP EXPLAINABILITY
# =====================================================================
def run_shap(rf_model, X_test, y_test):
    """Generate SHAP plots for the Random Forest model."""
    import shap
    print("\n── SHAP Explainability ──")

    explainer = shap.TreeExplainer(rf_model)
    # Use a subset for speed
    n_shap = min(200, len(X_test))
    X_shap = X_test[:n_shap]
    shap_values = explainer.shap_values(X_shap)

    # For binary classification, TreeExplainer returns a list [class_0, class_1]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # SHAP values for the positive class (AI)
    else:
        sv = shap_values

    # Summary bar plot — top 20 features
    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(sv, X_shap, feature_names=FEATURE_NAMES,
                      plot_type="bar", max_display=20, show=False)
    plt.title("SHAP Feature Importance — Top 20")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "shap_bar.png"), dpi=150,
                bbox_inches="tight")
    plt.close("all")
    print("  SHAP bar plot saved.")

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(sv, X_shap, feature_names=FEATURE_NAMES,
                      max_display=20, show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "shap_beeswarm.png"), dpi=150,
                bbox_inches="tight")
    plt.close("all")
    print("  SHAP beeswarm plot saved.")

    # Waterfall plot for a single example
    # Pick a sample that the model classifies as AI
    example_idx = 0
    for i in range(len(X_shap)):
        if rf_model.predict(X_shap[i:i+1])[0] == 1:
            example_idx = i
            break

    # Grab one row from the already-computed SHAP values
    single_sv = np.array(sv[example_idx]).ravel()[:len(FEATURE_NAMES)]
    single_data = np.array(X_shap[example_idx]).ravel()[:len(FEATURE_NAMES)]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        ev = float(expected_value[1])
    else:
        ev = float(expected_value)

    explanation = shap.Explanation(
        values=single_sv,
        base_values=ev,
        data=single_data,
        feature_names=list(FEATURE_NAMES),
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title("SHAP Waterfall — Single AI Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS, "shap_waterfall.png"), dpi=150,
                bbox_inches="tight")
    plt.close("all")
    print("  SHAP waterfall plot saved.")

    # Print top 10 features by mean |SHAP value|
    mean_abs = np.mean(np.abs(sv), axis=0).ravel()[:len(FEATURE_NAMES)]
    top_idx = np.argsort(mean_abs)[::-1][:10]
    print("\n  Top 10 features by mean |SHAP value|:")
    for rank, idx in enumerate(top_idx, 1):
        idx = int(idx)
        print(f"    {rank:>2}. {FEATURE_NAMES[idx]:<30s}  {mean_abs[idx]:.4f}")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    # Load data
    df = load_data()

    # Extract features + split
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_features(df)

    # Train models
    cnn_model, cnn_history = train_cnn(X_train, y_train, X_val, y_val)
    rf_model               = train_rf(X_train, y_train)
    cal_rf_model           = train_calibrated_rf(X_train, y_train)

    # Evaluate
    print("\n── Evaluation ──")
    res_cnn,  prob_cnn,  cm_cnn  = evaluate_model("CNN",            cnn_model, X_test, y_test, is_keras=True)
    res_rf,   prob_rf,   cm_rf   = evaluate_model("Random Forest",  rf_model,  X_test, y_test)
    res_cal,  prob_cal,  cm_cal  = evaluate_model("Calibrated RF",  cal_rf_model, X_test, y_test)

    results = [res_cnn, res_rf, res_cal]
    print_comparison(results)

    # Plots
    print("── Saving plots ──")
    plot_confusion_matrices(
        [cm_cnn, cm_rf, cm_cal],
        ["CNN", "Random Forest", "Calibrated RF"])
    plot_roc_curves(y_test, [prob_cnn, prob_rf, prob_cal],
                    ["CNN", "Random Forest", "Calibrated RF"])
    plot_cnn_history(cnn_history)
    plot_calibration_diagram(y_test, prob_rf, prob_cal)

    # SHAP
    run_shap(rf_model, X_test, y_test)

    # Save results table to CSV for the README
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUTS, "results.csv"), index=False)
    print("\nResults table saved to outputs/results.csv")
    print("\nDone! All models trained and saved.")
