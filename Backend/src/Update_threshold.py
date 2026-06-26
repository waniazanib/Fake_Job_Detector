"""
update_threshold.py — update XGBoost decision threshold and sync to HF

Run from backend/ directory:
    python src/update_threshold.py --threshold 0.7

Optional flags:
    --threshold   float   New threshold value (default 0.7)
    --no-upload           Skip HF upload, only save locally
    --repo        str     HF model repo ID (default reads from .env)

Example:
    python src/update_threshold.py --threshold 0.7
    python src/update_threshold.py --threshold 0.65 --no-upload
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features_df

# ── Args ──────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Update JobGuard decision threshold")
parser.add_argument("--threshold", type=float, default=0.7,
                    help="New threshold value between 0 and 1 (default: 0.7)")
parser.add_argument("--no-upload", action="store_true",
                    help="Skip uploading to Hugging Face")
parser.add_argument("--repo", type=str, default=None,
                    help="HF model repo ID e.g. waniazanib/jobguard-models")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────

MODEL_DIR   = Path("models")
DATA_PATH   = Path("data/fake_job_postings.csv")
RANDOM_SEED = 42
TEST_SIZE   = 0.15
NEW_THRESH  = args.threshold

# Load HF repo from .env if not passed as arg
if args.repo is None:
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv(Path(__file__).parent.parent / ".env")
        HF_REPO = os.getenv("HF_MODEL_REPO", "waniazanib/jobguard-models")
    except ImportError:
        HF_REPO = "waniazanib/jobguard-models"
else:
    HF_REPO = args.repo

# ── Validate threshold ────────────────────────────────────────

if not 0.0 < NEW_THRESH < 1.0:
    print(f"Error: threshold must be between 0 and 1, got {NEW_THRESH}")
    sys.exit(1)

# ── Show before/after metrics ─────────────────────────────────

print(f"Loading model and data to preview metrics...")

xgb        = joblib.load(MODEL_DIR / "xgb_model.joblib")
old_thresh = float(joblib.load(MODEL_DIR / "xgb_threshold.joblib"))

df = pd.read_csv(DATA_PATH)
_, test = train_test_split(
    df,
    test_size    = TEST_SIZE,
    stratify     = df["fraudulent"],
    random_state = RANDOM_SEED,
)

X_test = extract_features_df(test.copy())
y_test = test["fraudulent"].values
probs  = xgb.predict_proba(X_test)[:, 1]

def metrics(threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "f1":        round(float(f1_score(y_test, preds, zero_division=0)),        3),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 3),
        "recall":    round(float(recall_score(y_test, preds, zero_division=0)),    3),
        "flagged":   int(preds.sum()),
    }

old = metrics(old_thresh)
new = metrics(NEW_THRESH)

print()
print(f"{'':20} {'Current':>10}  {'New':>10}")
print("-" * 44)
print(f"{'Threshold':20} {old_thresh:>10.2f}  {NEW_THRESH:>10.2f}")
print(f"{'F1':20} {old['f1']:>10.3f}  {new['f1']:>10.3f}")
print(f"{'Precision':20} {old['precision']:>10.3f}  {new['precision']:>10.3f}")
print(f"{'Recall':20} {old['recall']:>10.3f}  {new['recall']:>10.3f}")
print(f"{'Postings flagged':20} {old['flagged']:>10}  {new['flagged']:>10}")
print()

# ── Confirm ───────────────────────────────────────────────────

confirm = input(f"Save threshold={NEW_THRESH} and upload to HF? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted — no changes made.")
    sys.exit(0)

# ── Save locally ──────────────────────────────────────────────

joblib.dump(NEW_THRESH, MODEL_DIR / "xgb_threshold.joblib")
print(f"Saved xgb_threshold.joblib → {NEW_THRESH}")

# ── Upload to HF ──────────────────────────────────────────────

if args.no_upload:
    print("Skipping HF upload (--no-upload flag set).")
    print("Done. Restart your server to apply the new threshold.")
    sys.exit(0)

print(f"\nUploading to {HF_REPO}...")

try:
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj = str(MODEL_DIR / "xgb_threshold.joblib"),
        path_in_repo    = "xgb_threshold.joblib",
        repo_id         = HF_REPO,
        repo_type       = "model",
    )
    print(f"Uploaded xgb_threshold.joblib → {HF_REPO}")
    print("\nDone. Restart your HF Space to apply the new threshold:")
    print(f"  https://huggingface.co/spaces/waniazanib/jobguard/settings")
except ImportError:
    print("huggingface_hub not installed. Install with: pip install huggingface_hub")
    print("Then re-run or upload manually:")
    print(f"  huggingface-cli upload {HF_REPO} models/xgb_threshold.joblib xgb_threshold.joblib --repo-type=model")
except Exception as exc:
    print(f"Upload failed: {exc}")
    print("Upload manually:")
    print(f"  huggingface-cli upload {HF_REPO} models/xgb_threshold.joblib xgb_threshold.joblib --repo-type=model")