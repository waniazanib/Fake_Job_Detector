import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features_df

# ── Config ────────────────────────────────────────────────────

DATA_PATH   = Path("data/fake_job_postings.csv")
MODEL_DIR   = Path("models")
RANDOM_SEED = 42
TEST_SIZE   = 0.15

# ── Load data ─────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
_, test = train_test_split(
    df,
    test_size   = TEST_SIZE,
    stratify    = df["fraudulent"],
    random_state= RANDOM_SEED,
)
print(f"Test set: {len(test)} rows — {test['fraudulent'].sum()} fraud\n")

# ── Load model ────────────────────────────────────────────────

xgb       = joblib.load(MODEL_DIR / "xgb_model.joblib")
X_test    = extract_features_df(test.copy())
y_test    = test["fraudulent"].values
probs     = xgb.predict_proba(X_test)[:, 1]

# ── Threshold sweep ───────────────────────────────────────────

print("=" * 65)
print(f"{'Threshold':>10}  {'F1':>6}  {'Precision':>10}  {'Recall':>8} {'Flagged':>8}")
print("=" * 65)

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
best_f1, best_thresh = 0.0, 0.5

for t in thresholds:
    preds   = (probs >= t).astype(int)
    f1      = f1_score(y_test, preds, zero_division=0)
    prec    = precision_score(y_test, preds, zero_division=0)
    rec     = recall_score(y_test, preds, zero_division=0)
    flagged = preds.sum()
    marker  = "  <-- current" if abs(t - joblib.load(MODEL_DIR / "xgb_threshold.joblib")) < 0.01 else ""
    print(f"{t:>10.2f}  {f1:>6.3f}  {prec:>10.3f}  {rec:>8.3f}  {flagged:>8}{marker}")
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print("=" * 65)
print(f"\nBest F1={best_f1:.3f} at threshold={best_thresh:.2f}")

# ── Full metrics at current threshold ─────────────────────────

current_thresh = float(joblib.load(MODEL_DIR / "xgb_threshold.joblib"))
preds          = (probs >= current_thresh).astype(int)

print(f"\nFull metrics at current threshold ({current_thresh}):")
print("-" * 40)
print(f"ROC-AUC   : {roc_auc_score(y_test, probs):.4f}")
print(f"PR-AUC    : {average_precision_score(y_test, probs):.4f}")
print(f"F1        : {f1_score(y_test, preds, zero_division=0):.4f}")
print(f"Precision : {precision_score(y_test, preds, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, preds, zero_division=0):.4f}")
print()
print(classification_report(y_test, preds, target_names=["Legitimate", "Fraud"]))

# ── Confusion matrix ──────────────────────────────────────────

ConfusionMatrixDisplay.from_predictions(
    y_test,
    preds,
    display_labels = ["Legitimate", "Fraud"],
    cmap           = "Blues",
)
plt.title(f"JobGuard — Confusion Matrix (threshold={current_thresh})")
plt.tight_layout()
output_path = MODEL_DIR / "confusion_matrix.png"
plt.savefig(output_path, dpi=150)
plt.show()
print(f"Confusion matrix saved → {output_path}")