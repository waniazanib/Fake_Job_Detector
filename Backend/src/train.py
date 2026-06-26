"""
train.py — JobGuard full training pipeline

Run once to produce:
  models/xgb_model.joblib
  models/xgb_feature_names.joblib
  models/fusion_weights.joblib
  models/distilbert_finetuned/          (HuggingFace SavedModel format)
  models/distilbert_onnx/model.onnx     (ONNX export for fast inference)
  models/distilbert_onnx/tokenizer/     (tokenizer saved alongside ONNX)

Usage:
  cd Backend
  python src/train.py

Expects:
  data/fake_job_postings.csv            (EMSCAD dataset from Kaggle)
  .env                                  (or environment variables set)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from xgboost import XGBClassifier

# Add project root to path so sibling imports work
sys.path.insert(0, str(Path(__file__).parent))

from features import (
    FEATURE_COLUMNS,
    build_bert_input,
    extract_features_df,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jobguard.train")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_PATH = Path(os.getenv("DATA_PATH", "data/fake_job_postings.csv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
DISTILBERT_MODEL = os.getenv("DISTILBERT_MODEL", "distilbert-base-uncased")
FUSION_TEXT_WEIGHT = float(os.getenv("FUSION_TEXT_WEIGHT", "0.55"))
FUSION_STRUCT_WEIGHT = float(os.getenv("FUSION_STRUCT_WEIGHT", "0.45"))

# Training hyperparameters
BERT_MAX_LEN = 512
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LR = 2e-5
BERT_WARMUP_RATIO = 0.1
FOCAL_GAMMA = 2.0        # focal loss focusing parameter
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", DEVICE)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss — down-weights easy negatives so the model focuses
    on hard-to-classify fraudulent examples in the imbalanced dataset.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha is set per-sample based on class weight; gamma defaults to 2.0.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 2), targets: (B,) long
        probs = torch.softmax(logits, dim=-1)
        # probability of the true class
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        p_t = (probs * targets_one_hot).sum(dim=-1)

        # class weights: upweight the minority fraud class
        alpha = torch.ones_like(p_t)
        if self.pos_weight is not None:
            alpha[targets == 1] = self.pos_weight

        focal_weight = alpha * (1.0 - p_t) ** self.gamma
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        loss = focal_weight * ce_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class JobPostingDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_len: int = BERT_MAX_LEN,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    log.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    log.info("Loaded %d rows — %d fraudulent (%.1f%%)",
             len(df),
             df["fraudulent"].sum(),
             df["fraudulent"].mean() * 100)

    # Ensure required columns exist
    required = [
        "title", "description", "requirements", "benefits", "company_profile",
        "location", "salary_range", "employment_type", "required_experience",
        "required_education", "has_company_logo", "has_questions",
        "telecommuting", "fraudulent",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split → train / val / test.
    Stratified on the target to preserve the ~4.5% fraud rate in each split.
    """
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["fraudulent"],
        random_state=RANDOM_SEED,
    )
    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val["fraudulent"],
        random_state=RANDOM_SEED,
    )
    log.info(
        "Split → train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# XGBoost structural branch
# ---------------------------------------------------------------------------

def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: Path,
) -> tuple[XGBClassifier, float]:
    """
    Train XGBoost on engineered structural features.
    Returns (fitted model, val PR-AUC).
    """
    log.info("=== Training XGBoost structural branch ===")

    X_train = extract_features_df(train_df.copy())
    y_train = train_df["fraudulent"].values

    X_val = extract_features_df(val_df.copy())
    y_val = val_df["fraudulent"].values

    # Class imbalance: scale_pos_weight = count(negatives) / count(positives)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info("scale_pos_weight = %.2f  (neg=%d / pos=%d)", scale_pos_weight, neg, pos)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, val_probs)
    roc_auc = roc_auc_score(y_val, val_probs)

    # Threshold tuning — find threshold maximising F1 on validation
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    log.info(
        "XGBoost val  PR-AUC=%.4f  ROC-AUC=%.4f  best-F1=%.4f @ thresh=%.2f",
        pr_auc, roc_auc, best_f1, best_thresh,
    )

    # Save model and feature column order
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "xgb_model.joblib")
    joblib.dump(FEATURE_COLUMNS, model_dir / "xgb_feature_names.joblib")
    joblib.dump(best_thresh, model_dir / "xgb_threshold.joblib")
    log.info("XGBoost model saved → %s", model_dir / "xgb_model.joblib")

    return model, pr_auc


# ---------------------------------------------------------------------------
# DistilBERT text branch
# ---------------------------------------------------------------------------

def train_distilbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: Path,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, float]:
    """
    Fine-tune DistilBERT for binary fraud classification.
    Uses Focal Loss to handle class imbalance.
    Returns (model, tokenizer, val PR-AUC).
    """
    log.info("=== Training DistilBERT text branch ===")

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL)

    def make_texts(df: pd.DataFrame) -> list[str]:
        return [
            build_bert_input(
                row["title"],
                row["description"],
                row["requirements"],
            )
            for _, row in df.iterrows()
        ]

    log.info("Building text inputs for train split (%d rows)…", len(train_df))
    train_texts = make_texts(train_df)
    train_labels = train_df["fraudulent"].tolist()

    log.info("Building text inputs for val split (%d rows)…", len(val_df))
    val_texts = make_texts(val_df)
    val_labels = val_df["fraudulent"].tolist()

    train_dataset = JobPostingDataset(train_texts, train_labels, tokenizer)
    val_dataset = JobPostingDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BERT_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BERT_BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL,
        num_labels=2,
    ).to(DEVICE)

    # Focal loss pos_weight mirrors XGBoost scale_pos_weight
    neg = sum(1 for l in train_labels if l == 0)
    pos = sum(1 for l in train_labels if l == 1)
    focal_pos_weight = neg / pos
    criterion = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=focal_pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BERT_LR, weight_decay=0.01)

    total_steps = len(train_loader) * BERT_EPOCHS
    warmup_steps = int(total_steps * BERT_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_pr_auc = 0.0
    best_model_state: Optional[dict] = None


    early_stopping_patience = 2  # Number of epochs to wait without improvement
    epochs_no_improve = 0
    total_train_steps = len(train_loader)

    for epoch in range(1, BERT_EPOCHS + 1):
        # ---- Training epoch ----
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            # Gradient clipping to stabilise training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # ---- Step-by-Step Percentage Counter ----
            current_percentage = int((step / total_train_steps) * 100)
            print(f"\rEpoch {epoch}/{BERT_EPOCHS} | Progress: {current_percentage}% completed (Step {step}/{total_train_steps})", end="", flush=True)

            if step % 50 == 0:
                log.info(
                    "Epoch %d | step %d/%d | loss=%.4f",
                    epoch, step, len(train_loader), epoch_loss / step,
                )

        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        all_probs: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].numpy().tolist()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels)

        pr_auc = average_precision_score(all_labels, all_probs)
        roc_auc = roc_auc_score(all_labels, all_probs)
        elapsed = time.time() - t0

        log.info(
            "Epoch %d/%d | loss=%.4f | PR-AUC=%.4f | ROC-AUC=%.4f | %.0fs",
            epoch, BERT_EPOCHS, avg_loss, pr_auc, roc_auc, elapsed,
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info("\n  ↑ New best PR-AUC=%.4f — checkpoint saved", best_pr_auc)
            
            # Save hard checkpoint to disk so training can be resumed/recovered
            checkpoint_path = model_dir / "distilbert_checkpoint.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_pr_auc': best_pr_auc,
            }, checkpoint_path)
            log.info("  -> Saved full training checkpoint to %s", checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            log.info("\n  No improvement in PR-AUC for %d epoch(s)", epochs_no_improve)
            if epochs_no_improve >= early_stopping_patience:
                log.info("Early stopping triggered! Halting DistilBERT training.")
                break

    # Restore best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log.info("Restored best checkpoint (PR-AUC=%.4f)", best_pr_auc)

    # Save fine-tuned model in HuggingFace format
    bert_dir = model_dir / "distilbert_finetuned"
    bert_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(bert_dir)
    tokenizer.save_pretrained(bert_dir)
    log.info("DistilBERT saved → %s", bert_dir)

    return model, tokenizer, best_pr_auc


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    model_dir: Path,
) -> None:
    """
    Export fine-tuned DistilBERT to ONNX for fast CPU inference via
    onnxruntime. The tokenizer is saved alongside the ONNX model so
    predict.py only needs the onnx/ directory.
    """
    log.info("=== Exporting DistilBERT → ONNX ===")

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from optimum.exporters.onnx import main_export

        onnx_dir = model_dir / "distilbert_onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        # Export using optimum — handles all the dynamic axes boilerplate
        main_export(
            model_name_or_path=str(model_dir / "distilbert_finetuned"),
            output=onnx_dir,
            task="text-classification",
            framework="pt",
        )

        # Save tokenizer alongside ONNX model
        tokenizer.save_pretrained(onnx_dir)
        log.info("ONNX model exported → %s", onnx_dir)

    except ImportError:
        log.warning(
            "optimum not installed — skipping ONNX export. "
            "Install with: pip install optimum[onnxruntime]"
        )
    except Exception as exc:
        log.error("ONNX export failed: %s — inference will use PyTorch.", exc)


# ---------------------------------------------------------------------------
# Fusion weight tuning
# ---------------------------------------------------------------------------

def tune_fusion_weights(
    xgb_model: XGBClassifier,
    bert_model: AutoModelForSequenceClassification,
    bert_tokenizer: AutoTokenizer,
    val_df: pd.DataFrame,
    model_dir: Path,
) -> tuple[float, float]:
    """
    Grid search over text_weight in [0.3, 0.7] to find the combination
    that maximises PR-AUC on the validation set.
    struct_weight = 1 - text_weight.
    """
    log.info("=== Tuning fusion weights ===")

    # XGBoost structural probs
    X_val_struct = extract_features_df(val_df.copy())
    struct_probs = xgb_model.predict_proba(X_val_struct)[:, 1]

    # DistilBERT text probs
    val_texts = [
        build_bert_input(
            row["title"], row["description"], row["requirements"]
        )
        for _, row in val_df.iterrows()
    ]
    val_labels = val_df["fraudulent"].tolist()

    val_dataset = JobPostingDataset(val_texts, val_labels, bert_tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    bert_model.eval()
    text_probs: list[float] = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            text_probs.extend(probs.cpu().numpy().tolist())

    text_probs_arr = np.array(text_probs)
    y_val = np.array(val_labels)

    best_pr_auc = 0.0
    best_tw = FUSION_TEXT_WEIGHT

    for tw in np.arange(0.30, 0.71, 0.05):
        sw = round(1.0 - tw, 2)
        tw = round(tw, 2)
        fused = tw * text_probs_arr + sw * struct_probs
        pr_auc = average_precision_score(y_val, fused)
        log.info("  text_w=%.2f struct_w=%.2f → PR-AUC=%.4f", tw, sw, pr_auc)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_tw = tw

    best_sw = round(1.0 - best_tw, 2)
    log.info(
        "Best fusion: text_w=%.2f struct_w=%.2f → PR-AUC=%.4f",
        best_tw, best_sw, best_pr_auc,
    )

    joblib.dump(
        {"text_weight": best_tw, "struct_weight": best_sw},
        model_dir / "fusion_weights.joblib",
    )
    return best_tw, best_sw


# ---------------------------------------------------------------------------
# Final evaluation on held-out test set
# ---------------------------------------------------------------------------

def evaluate_on_test(
    xgb_model: XGBClassifier,
    bert_model: AutoModelForSequenceClassification,
    bert_tokenizer: AutoTokenizer,
    test_df: pd.DataFrame,
    text_weight: float,
    struct_weight: float,
    model_dir: Path,
) -> None:
    log.info("=== Final evaluation on test set ===")

    # Structural probs
    X_test_struct = extract_features_df(test_df.copy())
    struct_probs = xgb_model.predict_proba(X_test_struct)[:, 1]

    # Text probs
    test_texts = [
        build_bert_input(
            row["title"], row["description"], row["requirements"]
        )
        for _, row in test_df.iterrows()
    ]
    test_labels = test_df["fraudulent"].tolist()

    test_dataset = JobPostingDataset(test_texts, test_labels, bert_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    bert_model.eval()
    text_probs_list: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            text_probs_list.extend(probs.cpu().numpy().tolist())

    text_probs = np.array(text_probs_list)
    y_test = np.array(test_labels)

    # Load tuned XGB threshold
    xgb_thresh_path = model_dir / "xgb_threshold.joblib"
    threshold = joblib.load(xgb_thresh_path) if xgb_thresh_path.exists() else 0.5

    fused_probs = text_weight * text_probs + struct_weight * struct_probs
    preds = (fused_probs >= threshold).astype(int)

    pr_auc  = average_precision_score(y_test, fused_probs)
    roc_auc = roc_auc_score(y_test, fused_probs)
    f1      = f1_score(y_test, preds, zero_division=0)
    prec    = precision_score(y_test, preds, zero_division=0)
    rec     = recall_score(y_test, preds, zero_division=0)

    results = {
        "pr_auc":    round(float(pr_auc),  4),
        "roc_auc":   round(float(roc_auc), 4),
        "f1":        round(float(f1),       4),
        "precision": round(float(prec),     4),
        "recall":    round(float(rec),      4),
        "threshold": round(float(threshold), 4),
        "text_weight":   text_weight,
        "struct_weight": struct_weight,
        "test_size":     len(test_df),
        "fraud_count":   int(y_test.sum()),
    }

    log.info("━" * 50)
    log.info("TEST SET RESULTS")
    log.info("  PR-AUC    : %.4f  (target ≥ 0.85)", pr_auc)
    log.info("  ROC-AUC   : %.4f  (target ≥ 0.97)", roc_auc)
    log.info("  F1 (fraud): %.4f  (target ≥ 0.90)", f1)
    log.info("  Precision : %.4f  (target ≥ 0.88)", prec)
    log.info("  Recall    : %.4f  (target ≥ 0.92)", rec)
    log.info("━" * 50)

    log.info("\nClassification report:\n%s",
             classification_report(y_test, preds, target_names=["Legitimate", "Fraud"]))

    results_path = model_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Evaluation results saved → %s", results_path)

    # Flag if any target is missed
    targets = [
        ("PR-AUC",  pr_auc,  0.85),
        ("ROC-AUC", roc_auc, 0.97),
        ("F1",      f1,      0.90),
        ("Prec",    prec,    0.88),
        ("Recall",  rec,     0.92),
    ]
    missed = [(name, val, tgt) for name, val, tgt in targets if val < tgt]
    if missed:
        log.warning("Some targets not met — consider more epochs or data augmentation:")
        for name, val, tgt in missed:
            log.warning("  %s: %.4f < %.2f", name, val, tgt)
    else:
        log.info("All evaluation targets met.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("JobGuard training pipeline starting")
    log.info("Data   : %s", DATA_PATH)
    log.info("Models : %s", MODEL_DIR)
    log.info("Device : %s", DEVICE)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download from https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction "
            "and place fake_job_postings.csv in Backend/data/"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1 — Load data
    df = load_data(DATA_PATH)

    # 2 — Stratified split
    train_df, val_df, test_df = split_data(df)

    # 3 — Train XGBoost structural branch
    xgb_model, xgb_pr_auc = train_xgboost(train_df, val_df, MODEL_DIR)

    # 4 — Fine-tune DistilBERT text branch
    bert_model, bert_tokenizer, bert_pr_auc = train_distilbert(
        train_df, val_df, MODEL_DIR
    )

    # 5 — Export DistilBERT to ONNX
    export_onnx(bert_model, bert_tokenizer, MODEL_DIR)

    # 6 — Tune fusion weights on validation set
    text_weight, struct_weight = tune_fusion_weights(
        xgb_model, bert_model, bert_tokenizer, val_df, MODEL_DIR
    )

    # 7 — Final evaluation on held-out test set
    evaluate_on_test(
        xgb_model, bert_model, bert_tokenizer,
        test_df, text_weight, struct_weight,
        MODEL_DIR,
    )

    log.info("Training pipeline complete. Models saved to %s", MODEL_DIR)


if __name__ == "__main__":
    main()