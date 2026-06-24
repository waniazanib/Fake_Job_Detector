"""
explainer.py — JobGuard SHAP explainability layer

Responsibilities:
  - Wrap the XGBoost model in a shap.TreeExplainer (computed once at startup)
  - For each inference request, compute SHAP values for the structural features
  - Map raw SHAP values → sorted ShapSignal list (top 5, by absolute impact)
  - Enrich each signal with display_name and direction-aware explanation
    from FEATURE_META in features.py

Why XGBoost SHAP only (not BERT):
  shap.DeepExplainer on DistilBERT is ~10× slower per request and produces
  token-level attributions that are hard to surface cleanly in a UI.
  The structural features are the most interpretable signals anyway — a user
  can immediately act on "salary not listed" or "no company logo" in a way
  they cannot act on "token 'amazing' has high attention weight".

Usage (called from main.py after predictor.predict()):
    from src.explainer import Explainer
    explainer = Explainer(xgb_model)                   # once at startup
    signals   = explainer.explain(feature_dict)        # per request
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import shap

from features import FEATURE_COLUMNS, FEATURE_META
from schemas import ShapDirection, ShapSignal

log = logging.getLogger("jobguard.explainer")

# Number of top signals to return to the frontend
TOP_N = 5

# SHAP value magnitude below which a signal is considered negligible.
# Filters out features that technically contributed but are noise-level.
_MIN_IMPACT_THRESHOLD = 0.005


class Explainer:
    """
    Wraps a fitted XGBClassifier with a shap.TreeExplainer.
    The explainer is initialised once and reused for all requests —
    TreeExplainer initialisation is expensive (~1–2s); inference is fast (~5ms).
    """

    def __init__(self, xgb_model) -> None:
        """
        Parameters
        ----------
        xgb_model : fitted XGBClassifier
            The trained structural-branch model from predict.py.
        """
        log.info("Initialising SHAP TreeExplainer…")
        # model_output="probability" makes SHAP values directly interpretable
        # as contributions to the fraud probability (not log-odds).
        self._explainer = shap.TreeExplainer(
            xgb_model,
            model_output="probability",
            feature_names=FEATURE_COLUMNS,
        )
        log.info("SHAP TreeExplainer ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        feature_dict: dict[str, float],
        base_class: int = 1,
    ) -> list[ShapSignal]:
        """
        Compute SHAP values for a single posting and return the top-N signals
        sorted by absolute impact (descending).

        Parameters
        ----------
        feature_dict : dict[str, float]
            Output of features.extract_features_single() — already in
            FEATURE_COLUMNS order when converted via feature_dict_to_array,
            but here we keep it as a dict for readability.
        base_class : int
            SHAP class index to explain. 1 = fraud probability (default).

        Returns
        -------
        list[ShapSignal]
            Up to TOP_N signals, sorted by |impact| descending.
        """
        # Build ordered numpy row [1 × n_features]
        feature_vector = np.array(
            [feature_dict[col] for col in FEATURE_COLUMNS],
            dtype=np.float32,
        ).reshape(1, -1)

        # Compute SHAP values
        # shap_values shape: [n_classes][n_samples × n_features]
        # We want class 1 (fraud), sample 0
        try:
            shap_values = self._explainer.shap_values(feature_vector)
        except Exception as exc:
            log.error("SHAP computation failed: %s", exc)
            return []

        # Handle both single-output (binary) and multi-output formats
        # TreeExplainer with model_output="probability" on XGBoost binary
        # classification returns either a 2-element list or a 2D array.
        if isinstance(shap_values, list):
            # list of [n_samples × n_features] — one per class
            values_fraud = shap_values[base_class][0]   # shape (n_features,)
        else:
            # single 2D array (n_samples × n_features) for binary classification
            values_fraud = shap_values[0]               # shape (n_features,)

        return self._build_signals(values_fraud, feature_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_signals(
        self,
        shap_values: np.ndarray,
        feature_dict: dict[str, float],
    ) -> list[ShapSignal]:
        """
        Convert raw per-feature SHAP values into sorted ShapSignal objects.

        Direction logic:
          positive SHAP value → pushed score toward fraud
          negative SHAP value → pushed score toward safe / legitimate

        Only features whose |SHAP| exceeds _MIN_IMPACT_THRESHOLD are included.
        """
        signals: list[ShapSignal] = []

        for idx, feature_name in enumerate(FEATURE_COLUMNS):
            raw_shap = float(shap_values[idx])
            abs_impact = abs(raw_shap)

            if abs_impact < _MIN_IMPACT_THRESHOLD:
                continue

            direction = ShapDirection.FRAUD if raw_shap > 0 else ShapDirection.SAFE
            meta = FEATURE_META.get(feature_name)

            if meta is None:
                # Fallback for any feature not in FEATURE_META
                display_name = feature_name.replace("_", " ").title()
                explanation = (
                    f"This feature contributed {abs_impact:.3f} toward "
                    f"{'fraud' if direction == ShapDirection.FRAUD else 'a safe'} prediction."
                )
            else:
                display_name = meta["display_name"]
                explanation = (
                    meta["fraud_explanation"]
                    if direction == ShapDirection.FRAUD
                    else meta["safe_explanation"]
                )

            signals.append(
                ShapSignal(
                    feature      = feature_name,
                    display_name = display_name,
                    value        = round(float(feature_dict[feature_name]), 4),
                    impact       = round(abs_impact, 4),
                    direction    = direction,
                    explanation  = explanation,
                )
            )

        # Sort by absolute impact descending, return top N
        signals.sort(key=lambda s: s.impact, reverse=True)
        return signals[:TOP_N]

    # ------------------------------------------------------------------
    # Diagnostic helper — useful during development / eval
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        feature_dicts: list[dict[str, float]],
        base_class: int = 1,
    ) -> list[list[ShapSignal]]:
        """
        Vectorised SHAP over a batch of feature dicts.
        More efficient than calling explain() in a loop for large eval sets.

        Returns a list of ShapSignal lists, one per input row.
        """
        if not feature_dicts:
            return []

        matrix = np.array(
            [[fd[col] for col in FEATURE_COLUMNS] for fd in feature_dicts],
            dtype=np.float32,
        )   # shape (n_samples, n_features)

        try:
            shap_values = self._explainer.shap_values(matrix)
        except Exception as exc:
            log.error("Batch SHAP computation failed: %s", exc)
            return [[] for _ in feature_dicts]

        if isinstance(shap_values, list):
            values_fraud = shap_values[base_class]   # (n_samples, n_features)
        else:
            values_fraud = shap_values               # (n_samples, n_features)

        return [
            self._build_signals(values_fraud[i], feature_dicts[i])
            for i in range(len(feature_dicts))
        ]

    def global_feature_importance(self) -> list[dict]:
        """
        Return mean absolute SHAP values across the training background
        as a ranked list. Useful for the findings.md report and a global
        importance chart in the UI (future enhancement).

        Note: only available if the TreeExplainer was initialised with a
        background dataset. Returns an empty list otherwise.
        """
        try:
            bg = self._explainer.data
            if bg is None:
                return []
            shap_vals = self._explainer.shap_values(bg)
            if isinstance(shap_vals, list):
                vals = shap_vals[1]
            else:
                vals = shap_vals
            mean_abs = np.abs(vals).mean(axis=0)
            ranked = sorted(
                [
                    {
                        "feature": FEATURE_COLUMNS[i],
                        "display_name": FEATURE_META.get(
                            FEATURE_COLUMNS[i], {}
                        ).get("display_name", FEATURE_COLUMNS[i]),
                        "mean_abs_shap": round(float(mean_abs[i]), 5),
                    }
                    for i in range(len(FEATURE_COLUMNS))
                ],
                key=lambda x: x["mean_abs_shap"],
                reverse=True,
            )
            return ranked
        except Exception as exc:
            log.warning("global_feature_importance failed: %s", exc)
            return []