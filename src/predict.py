"""
predict.py — Inference functions for AI-generated text detection.

  predict_document(text)   → overall label, confidence, top features
  predict_sentences(text)  → per-sentence breakdown
"""

import os
import numpy as np
import joblib
import spacy
import shap
import warnings
warnings.filterwarnings("ignore")

from src.features import extract_features, FEATURE_NAMES, NUM_FEATURES

# ── Paths ──────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")

# ── Load models once at import time ────────────────────────────────────
_scaler = joblib.load(os.path.join(MODELS, "scaler.joblib"))
_rf     = joblib.load(os.path.join(MODELS, "rf_model.joblib"))
_cal_rf = joblib.load(os.path.join(MODELS, "rf_calibrated.joblib"))

# SHAP explainer for the plain RF (TreeExplainer is fast)
_explainer = shap.TreeExplainer(_rf)

# spaCy model for sentence splitting
_nlp = spacy.load("en_core_web_sm")


def predict_document(text: str) -> dict:
    """
    Predict whether a full document is AI-generated or human-written.

    Returns:
        {
            "label": "AI-Generated" | "Human-Written",
            "confidence": float (0–100),
            "top_features": [(feature_name, shap_value), ...] (top 5)
        }
    """
    # Extract & scale features
    feats = extract_features(text).reshape(1, -1)
    feats_scaled = _scaler.transform(feats)

    # Predict with calibrated RF for the best-calibrated probability
    prob = _cal_rf.predict_proba(feats_scaled)[0, 1]
    label = "AI-Generated" if prob >= 0.5 else "Human-Written"
    confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100

    # SHAP values for the top contributing features (use plain RF)
    sv = _explainer.shap_values(feats_scaled)
    if isinstance(sv, list):
        sv = sv[1]  # positive-class SHAP values

    sv_flat = sv.ravel()[:NUM_FEATURES]  # clip to 32 features
    top_idx = np.argsort(np.abs(sv_flat))[::-1][:5]
    top_features = [(FEATURE_NAMES[int(i)], float(sv_flat[int(i)])) for i in top_idx]

    return {
        "label": label,
        "confidence": round(confidence, 1),
        "top_features": top_features,
    }


def predict_sentences(text: str) -> list:
    """
    Split text into sentences and classify each one individually.

    Returns a list of dicts:
        {
            "sentence": str,
            "label": "AI-Generated" | "Human-Written" | "Too short to score",
            "probability": float (0–1, probability of AI)
        }
    """
    doc = _nlp(text)
    results = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        # Count real words (not punctuation / spaces)
        word_count = len([t for t in sent if t.is_alpha])

        if word_count < 5:
            results.append({
                "sentence": sentence_text,
                "label": "Too short to score",
                "probability": None,
            })
            continue

        feats = extract_features(sentence_text).reshape(1, -1)
        feats_scaled = _scaler.transform(feats)
        prob = _cal_rf.predict_proba(feats_scaled)[0, 1]
        label = "AI-Generated" if prob >= 0.5 else "Human-Written"

        results.append({
            "sentence": sentence_text,
            "label": label,
            "probability": round(float(prob), 4),
        })

    return results
