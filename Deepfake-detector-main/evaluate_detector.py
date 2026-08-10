"""Evaluate the image detector on a labeled folder.

Expected layout:
    dataset/
      real/      # genuine images
      deepfake/  # manipulated/AI-generated images

Usage:
    python evaluate_detector.py --data dataset

This script never fabricates metrics. It reports only what was actually measured.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Use the HF pipeline directly for evaluation (no Streamlit here).
from transformers import pipeline
import torch

DEFAULT_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(label: str) -> str:
    """Map known model labels to REAL / DEEPFAKE. Keep list consistent with the app.

    Unknown labels remain UNKNOWN.
    """
    l = str(label).strip().lower().replace("_", " ").replace("-", " ")

    fake_keywords = (
        "fake",
        "spoof",
        "deepfake",
        "synthetic",
        "generated",
        "ai generated",
        "manipulated",
        "forged",
    )

    real_keywords = (
        "real",
        "realism",
        "bonafide",
        "genuine",
        "authentic",
        "human",
        "original",
    )

    if any(keyword in l for keyword in fake_keywords):
        return "DEEPFAKE"

    if any(keyword in l for keyword in real_keywords):
        return "REAL"

    return "UNKNOWN"


def score_image(clf, img: Image.Image, invert_labels: bool = False):
    """Score a single PIL image using the HF image-classification pipeline.

    Important: do NOT force a hard resize to 224x224 here. The pipeline's own
    processor will perform the model-appropriate resize / crop / normalization.
    Forcing an extra resize (especially a square, aspect-distorting one) was
    observed to introduce artifacts that can cause real images to be misclassified
    as DEEPFAKE.

    Returns a (real_prob, fake_prob) tuple in [0,1], or None if the model
    returned no usable REAL/DEEPFAKE scores.
    """
    # Only ensure RGB — leave sizing to the HF processor.
    img = img.convert("RGB")

    try:
        results = clf(img, top_k=None)
    except TypeError:
        results = clf(img)

    real = fake = 0.0
    for r in results:
        sem = normalize_label(r.get("label", ""))
        # Optionally flip the semantic meaning if user requested inversion
        # (useful for calibrating models that were shipped with inverted id2label).
        if invert_labels:
            if sem == "REAL":
                sem = "DEEPFAKE"
            elif sem == "DEEPFAKE":
                sem = "REAL"

        if sem == "REAL":
            real += float(r.get("score", 0.0))
        elif sem == "DEEPFAKE":
            fake += float(r.get("score", 0.0))

    total = real + fake
    if total <= 0:
        return None

    real, fake = real / total, fake / total
    return real, fake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder containing real/ and deepfake/")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=0.75, help="Decision threshold for the winning class")
    ap.add_argument("--invert-labels", action="store_true", help="Flip REAL<->DEEPFAKE semantics returned by the model (use only if you calibrated the model)")
    args = ap.parse_args()

    root = Path(args.data)
    real_dir = root / "real"
    fake_dir = root / "deepfake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise SystemExit("Dataset must contain real/ and deepfake/ directories.")

    # Use GPU if available for faster evaluation.
    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("image-classification", model=args.model, device=device)

    y_true, y_pred, y_score = [], [], []
    skipped = 0
    counts = {"real": 0, "deepfake": 0}

    for expected, folder in (("REAL", real_dir), ("DEEPFAKE", fake_dir)):
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(path) as im:
                    score = score_image(clf, im, invert_labels=bool(args.invert_labels))
                if score is None:
                    skipped += 1
                    continue
                real, fake = score
                confidence = max(real, fake)
                if confidence < args.threshold:
                    skipped += 1
                    continue
                pred = "REAL" if real > fake else "DEEPFAKE"
                y_true.append(1 if expected == "DEEPFAKE" else 0)
                y_pred.append(1 if pred == "DEEPFAKE" else 0)
                y_score.append(fake)
                counts[expected.lower()] += 1
            except Exception as exc:
                skipped += 1
                print(f"SKIP {path}: {exc}")

    if not y_true:
        raise SystemExit("No evaluable images. Provide labeled real/deepfake test data.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "model": args.model,
        "threshold": args.threshold,
        "invert_labels": bool(args.invert_labels),
        "evaluated": len(y_true),
        "skipped_or_uncertain": skipped,
        "real_evaluated": counts["real"],
        "deepfake_evaluated": counts["deepfake"],
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None,
        "false_positive_rate": float(cm[0, 1] / max(1, cm[0, 0] + cm[0, 1])),
        "false_negative_rate": float(cm[1, 0] / max(1, cm[1, 0] + cm[1, 1])),
        "confusion_matrix": cm.tolist(),
    }
    print(json.dumps(metrics, indent=2))
    Path("evaluation_results.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
