"""
predict.py — JobGuard inference engine

Responsibilities:
  - Load XGBoost + DistilBERT (ONNX preferred, PyTorch fallback) once at startup
  - Run dual-branch inference on a single JobPostingRequest
  - Fuse branch scores using tuned weights
  - Derive label, confidence, and a plain-English summary
  - Return a fully populated AnalyzeResponse (SHAP signals added by explainer.py)

Usage (called from main.py):
    from src.predict import Predictor
    predictor = Predictor()          # loads models
    response  = predictor.predict(request)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from dotenv import load_dotenv

from schemas import (
    AnalyzeResponse,
    ConfidenceLevel,
    FraudLabel,
    JobPostingRequest,
    ShapSignal,
)
from features import (
    build_bert_input,
    extract_features_single,
    feature_dict_to_array,
)

log = logging.getLogger("jobguard.predict")

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

# Score thresholds — must match AppFlow §8 UI states
_LABEL_LEGITIMATE_MAX = 0.35
_LABEL_SUSPICIOUS_MIN = 0.65

# Confidence — measured as disagreement between both branch scores
_CONFIDENCE_HIGH_MAX   = 0.15   # branches agree closely   → HIGH
_CONFIDENCE_MEDIUM_MAX = 0.30   # moderate disagreement    → MEDIUM
                                 # above 0.30               → LOW


# ---------------------------------------------------------------------------
# ONNX runtime (optional — falls back to PyTorch if not installed)
# ---------------------------------------------------------------------------

def _try_import_onnxruntime():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Predictor class — singleton loaded once by FastAPI lifespan
# ---------------------------------------------------------------------------

class Predictor:
    """
    Holds all model artefacts in memory for the lifetime of the server process.
    Call predict(request) for each incoming request — no re-loading per call.
    """

    def __init__(self) -> None:
        self._xgb_model = None
        self._xgb_threshold: float = 0.5
        self._feature_names: list[str] = []
        self._fusion_text_w: float = float(os.getenv("FUSION_TEXT_WEIGHT", "0.55"))
        self._fusion_struct_w: float = float(os.getenv("FUSION_STRUCT_WEIGHT", "0.45"))

        # DistilBERT — either ONNX session or PyTorch model
        self._ort_session = None          # onnxruntime.InferenceSession
        self._bert_model = None           # transformers AutoModelForSequenceClassification
        self._bert_tokenizer = None       # transformers AutoTokenizer
        self._bert_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.xgb_ready: bool = False
        self.bert_ready: bool = False

        self._load_all()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._load_xgb()
        self._load_bert()
        self._load_fusion_weights()

    def _load_xgb(self) -> None:
        xgb_path   = MODEL_DIR / "xgb_model.joblib"
        feat_path  = MODEL_DIR / "xgb_feature_names.joblib"
        thresh_path = MODEL_DIR / "xgb_threshold.joblib"

        if not xgb_path.exists():
            log.error(
                "XGBoost model not found at %s — run src/train.py first.", xgb_path
            )
            return

        self._xgb_model    = joblib.load(xgb_path)
        self._feature_names = joblib.load(feat_path) if feat_path.exists() else []

        if thresh_path.exists():
            self._xgb_threshold = float(joblib.load(thresh_path))

        self.xgb_ready = True
        log.info("XGBoost loaded from %s (threshold=%.2f)", xgb_path, self._xgb_threshold)

    def _load_bert(self) -> None:
        """
        Try ONNX first (faster CPU inference, no PyTorch overhead at request time).
        Fall back to loading the fine-tuned HuggingFace model with PyTorch.
        """
        onnx_dir   = MODEL_DIR / "distilbert_onnx"
        bert_dir   = MODEL_DIR / "distilbert_finetuned"
        ort        = _try_import_onnxruntime()

        onnx_model_path = onnx_dir / "model.onnx"

        if ort is not None and onnx_model_path.exists():
            self._load_bert_onnx(ort, onnx_dir, onnx_model_path)
        elif bert_dir.exists():
            self._load_bert_pytorch(bert_dir)
        else:
            log.error(
                "No DistilBERT model found. Expected ONNX at %s or PyTorch at %s. "
                "Run src/train.py first.",
                onnx_dir, bert_dir,
            )

    def _load_bert_onnx(self, ort, onnx_dir: Path, model_path: Path) -> None:
        from transformers import AutoTokenizer

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 4
        sess_options.intra_op_num_threads = 4

        self._ort_session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        tokenizer_path = onnx_dir / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = onnx_dir  # tokenizer saved directly in onnx_dir
        self._bert_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        self.bert_ready = True
        log.info("DistilBERT loaded via ONNX Runtime from %s", model_path)

    def _load_bert_pytorch(self, bert_dir: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
        self._bert_model = (
            AutoModelForSequenceClassification
            .from_pretrained(str(bert_dir))
            .to(self._bert_device)
            .eval()
        )

        self.bert_ready = True
        log.info(
            "DistilBERT loaded via PyTorch from %s (device=%s)",
            bert_dir, self._bert_device,
        )

    def _load_fusion_weights(self) -> None:
        weights_path = MODEL_DIR / "fusion_weights.joblib"
        if weights_path.exists():
            weights = joblib.load(weights_path)
            self._fusion_text_w   = float(weights.get("text_weight",   self._fusion_text_w))
            self._fusion_struct_w = float(weights.get("struct_weight",  self._fusion_struct_w))
            log.info(
                "Fusion weights loaded: text=%.2f struct=%.2f",
                self._fusion_text_w, self._fusion_struct_w,
            )
        else:
            log.info(
                "fusion_weights.joblib not found — using env defaults: "
                "text=%.2f struct=%.2f",
                self._fusion_text_w, self._fusion_struct_w,
            )

    # ------------------------------------------------------------------
    # Branch inference
    # ------------------------------------------------------------------

    def _run_xgb(self, request: JobPostingRequest) -> float:
        """Return fraud probability [0,1] from the structural branch."""
        if not self.xgb_ready:
            raise RuntimeError("XGBoost model is not loaded.")

        feature_dict = extract_features_single(
            title              = request.title,
            description        = request.description,
            requirements       = request.requirements,
            benefits           = request.benefits,
            company_profile    = request.company_profile,
            location           = request.location,
            salary_range       = request.salary_range,
            employment_type    = request.employment_type,
            required_experience= request.required_experience,
            required_education = request.required_education,
            has_company_logo   = request.has_company_logo,
            has_questions      = request.has_questions,
            telecommuting      = request.telecommuting,
        )

        X = feature_dict_to_array(feature_dict)
        prob = float(self._xgb_model.predict_proba(X)[0, 1])
        return prob, feature_dict

    def _run_bert_onnx(self, text: str) -> float:
        """Return fraud probability via ONNX Runtime — no PyTorch at request time."""
        encoding = self._bert_tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        inputs = {
            "input_ids":      encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        }
        logits = self._ort_session.run(None, inputs)[0]          # shape (1, 2)
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        return float(probs[0, 1])

    def _run_bert_pytorch(self, text: str) -> float:
        """Return fraud probability via PyTorch — fallback when ONNX unavailable."""
        encoding = self._bert_tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._bert_model(
                input_ids      = encoding["input_ids"].to(self._bert_device),
                attention_mask = encoding["attention_mask"].to(self._bert_device),
            )
        probs = torch.softmax(outputs.logits, dim=-1)
        return float(probs[0, 1].cpu().item())

    def _run_bert(self, request: JobPostingRequest) -> float:
        """Route to ONNX or PyTorch depending on what was loaded."""
        if not self.bert_ready:
            raise RuntimeError("DistilBERT model is not loaded.")

        text = build_bert_input(
            request.title,
            request.description,
            request.requirements,
        )

        if self._ort_session is not None:
            return self._run_bert_onnx(text)
        return self._run_bert_pytorch(text)

    # ------------------------------------------------------------------
    # Score → label / confidence / summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_label(score: float) -> FraudLabel:
        if score < _LABEL_LEGITIMATE_MAX:
            return FraudLabel.LEGITIMATE
        if score <= _LABEL_SUSPICIOUS_MIN:
            return FraudLabel.CAUTION
        return FraudLabel.SUSPICIOUS

    @staticmethod
    def _branch_disagreement_to_confidence(
        text_score: float,
        struct_score: float,
    ) -> ConfidenceLevel:
        """
        Confidence is inversely proportional to how much the two branches
        disagree. If BERT says 0.9 fraud but XGBoost says 0.3, we are uncertain.
        """
        disagreement = abs(text_score - struct_score)
        if disagreement <= _CONFIDENCE_HIGH_MAX:
            return ConfidenceLevel.HIGH
        if disagreement <= _CONFIDENCE_MEDIUM_MAX:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @staticmethod
    def _build_summary(
        fraud_score: float,
        label: FraudLabel,
        shap_signals: list[ShapSignal],
    ) -> str:
        """
        Auto-generate a one-sentence plain-English summary from the top
        fraud-direction SHAP signals.  Falls back to a generic sentence
        when no signals are available.
        """
        fraud_signals = [s for s in shap_signals if s.direction.value == "fraud"]

        if label == FraudLabel.LEGITIMATE:
            safe_signals = [s for s in shap_signals if s.direction.value == "safe"]
            if safe_signals:
                top = safe_signals[0].display_name.lower()
                return (
                    f"This posting appears legitimate — key indicators like "
                    f"{top} are consistent with a genuine employer."
                )
            return "This posting shows no strong indicators of fraud."

        if not fraud_signals:
            if label == FraudLabel.SUSPICIOUS:
                return (
                    f"This posting scored {fraud_score:.0%} on our fraud model — "
                    "treat it with caution before sharing personal information."
                )
            return (
                "Some signals are ambiguous — verify the employer before applying."
            )

        # Pull display names of the top fraud signals
        names = [s.display_name.lower() for s in fraud_signals[:3]]

        if len(names) == 1:
            signal_str = names[0]
        elif len(names) == 2:
            signal_str = f"{names[0]} and {names[1]}"
        else:
            signal_str = f"{names[0]}, {names[1]}, and {names[2]}"

        if label == FraudLabel.SUSPICIOUS:
            return (
                f"This posting raises serious concerns — {signal_str} "
                f"{'are' if len(names) > 1 else 'is'} among the strongest "
                "indicators of a fraudulent listing."
            )
        # CAUTION
        return (
            f"This posting has mixed signals — {signal_str} "
            f"{'warrant' if len(names) > 1 else 'warrants'} closer inspection "
            "before you apply."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def models_loaded(self) -> bool:
        return self.xgb_ready and self.bert_ready

    def predict(
        self,
        request: JobPostingRequest,
        shap_signals: Optional[list[ShapSignal]] = None,
    ) -> AnalyzeResponse:
        """
        Run full dual-branch inference and return a populated AnalyzeResponse.

        shap_signals is injected by explainer.py after this call — predict()
        returns an AnalyzeResponse with an empty list if not supplied, and
        main.py replaces it. This keeps the inference and explainability
        concerns cleanly separated.

        Raises RuntimeError if either model is not loaded.
        """
        if not self.xgb_ready:
            raise RuntimeError(
                "XGBoost model is not ready. Run src/train.py to train models."
            )
        if not self.bert_ready:
            raise RuntimeError(
                "DistilBERT model is not ready. Run src/train.py to train models."
            )

        # -- Structural branch --
        struct_score, feature_dict = self._run_xgb(request)

        # -- Text branch --
        text_score = self._run_bert(request)

        # -- Late fusion --
        fraud_score = (
            self._fusion_text_w   * text_score +
            self._fusion_struct_w * struct_score
        )
        fraud_score = float(np.clip(fraud_score, 0.0, 1.0))

        # -- Derived fields --
        label      = self._score_to_label(fraud_score)
        confidence = self._branch_disagreement_to_confidence(text_score, struct_score)
        signals    = shap_signals or []
        summary    = self._build_summary(fraud_score, label, signals)

        return AnalyzeResponse(
            fraud_score  = round(fraud_score,   4),
            label        = label,
            confidence   = confidence,
            text_score   = round(text_score,    4),
            struct_score = round(struct_score,  4),
            shap_signals = signals,
            summary      = summary,
        )

    def get_feature_dict(self, request: JobPostingRequest) -> dict[str, float]:
        """
        Expose the raw feature dict for the current request.
        Called by explainer.py so it can compute SHAP values using the same
        feature values that were passed to XGBoost.
        """
        return extract_features_single(
            title              = request.title,
            description        = request.description,
            requirements       = request.requirements,
            benefits           = request.benefits,
            company_profile    = request.company_profile,
            location           = request.location,
            salary_range       = request.salary_range,
            employment_type    = request.employment_type,
            required_experience= request.required_experience,
            required_education = request.required_education,
            has_company_logo   = request.has_company_logo,
            has_questions      = request.has_questions,
            telecommuting      = request.telecommuting,
        )