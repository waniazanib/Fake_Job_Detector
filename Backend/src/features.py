from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature column order — must match exactly between training and inference.
# XGBoost is sensitive to column ordering; this list is the single source
# of truth imported by both train.py and predict.py.
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "has_salary",
    "has_company_logo",
    "has_questions",
    "telecommuting",
    "employment_type_missing",
    "experience_missing",
    "education_missing",
    "company_profile_missing",
    "location_vagueness",
    "description_length",
    "requirements_length",
    "benefit_length",
    "company_profile_length",
    "url_count",
    "all_caps_ratio",
    "exclamation_count",
    "avg_word_length_desc",
    "title_length",
    "benefit_to_requirement_ratio",
    "text_total_length",
]

# ---------------------------------------------------------------------------
# Human-readable display names + explanations for every feature.
# Used by explainer.py to populate ShapSignal.display_name and .explanation.
# ---------------------------------------------------------------------------

FEATURE_META: dict[str, dict[str, str]] = {
    "has_salary": {
        "display_name": "Salary not listed",
        "fraud_explanation": "Legitimate job postings almost always include a salary range.",
        "safe_explanation": "Providing a salary range is a strong signal of a legitimate posting.",
    },
    "has_company_logo": {
        "display_name": "No company logo",
        "fraud_explanation": "Verified employers almost always attach a recognisable company logo.",
        "safe_explanation": "Having a company logo suggests a verified, registered employer.",
    },
    "has_questions": {
        "display_name": "No screening questions",
        "fraud_explanation": "Legitimate employers typically screen applicants with job-specific questions.",
        "safe_explanation": "Screening questions indicate a structured, genuine hiring process.",
    },
    "telecommuting": {
        "display_name": "Remote / telecommuting role",
        "fraud_explanation": "Fully remote roles with vague descriptions are disproportionately fraudulent.",
        "safe_explanation": "Remote roles are common and not inherently suspicious on their own.",
    },
    "employment_type_missing": {
        "display_name": "Employment type not specified",
        "fraud_explanation": "Omitting contract type (full-time, part-time, etc.) is unusual for a real job.",
        "safe_explanation": "Specifying employment type shows a well-structured, genuine posting.",
    },
    "experience_missing": {
        "display_name": "Experience level not specified",
        "fraud_explanation": "Real employers state the experience level they require.",
        "safe_explanation": "Specifying required experience is a mark of a legitimate posting.",
    },
    "education_missing": {
        "display_name": "Education requirement not specified",
        "fraud_explanation": "Most genuine job postings state a minimum education requirement.",
        "safe_explanation": "Stating an education requirement indicates a structured hiring process.",
    },
    "company_profile_missing": {
        "display_name": "No company description",
        "fraud_explanation": "Real employers describe their company — its absence is a strong fraud signal.",
        "safe_explanation": "A detailed company profile supports the legitimacy of the posting.",
    },
    "location_vagueness": {
        "display_name": "Vague or missing location",
        "fraud_explanation": "Fraudulent postings frequently list 'Anywhere' or omit location entirely.",
        "safe_explanation": "A specific location ties the role to a real, verifiable workplace.",
    },
    "description_length": {
        "display_name": "Very short job description",
        "fraud_explanation": "Fraudulent postings tend to be unusually brief to avoid scrutiny.",
        "safe_explanation": "A detailed job description suggests a well-considered, genuine role.",
    },
    "requirements_length": {
        "display_name": "Requirements section length",
        "fraud_explanation": "Sparse requirements sections are common in fake postings.",
        "safe_explanation": "Detailed requirements indicate a real, role-specific hiring need.",
    },
    "benefit_length": {
        "display_name": "Benefits section length",
        "fraud_explanation": "Fraudulent postings often over-promise vague benefits with little detail.",
        "safe_explanation": "A measured, realistic benefits section is typical of genuine employers.",
    },
    "company_profile_length": {
        "display_name": "Company profile length",
        "fraud_explanation": "A very short or empty company profile is a common omission in scam postings.",
        "safe_explanation": "A substantial company profile supports the employer's credibility.",
    },
    "url_count": {
        "display_name": "External links in description",
        "fraud_explanation": "Multiple external URLs are a common phishing and scam signal.",
        "safe_explanation": "Minimal external links suggest a straightforward, non-phishing posting.",
    },
    "all_caps_ratio": {
        "display_name": "Excessive capitalisation",
        "fraud_explanation": "Heavy use of ALL CAPS is a spam and urgency tactic common in scam postings.",
        "safe_explanation": "Normal capitalisation usage is consistent with professional job postings.",
    },
    "exclamation_count": {
        "display_name": "Excessive exclamation marks",
        "fraud_explanation": "Fraudulent postings frequently use exclamation marks to create false urgency.",
        "safe_explanation": "Measured punctuation is typical of professional job descriptions.",
    },
    "avg_word_length_desc": {
        "display_name": "Average word length",
        "fraud_explanation": "Very short average word length can indicate thin, low-effort copy.",
        "safe_explanation": "Normal word length distribution suggests professionally written content.",
    },
    "title_length": {
        "display_name": "Job title length",
        "fraud_explanation": "Extremely short or very long titles can indicate templated scam postings.",
        "safe_explanation": "A concise, specific title is typical of a genuine role.",
    },
    "benefit_to_requirement_ratio": {
        "display_name": "Benefits outweigh requirements",
        "fraud_explanation": "Scam postings often promise lavish benefits while asking for almost nothing.",
        "safe_explanation": "A balanced benefits-to-requirements ratio suggests a realistic role.",
    },
    "text_total_length": {
        "display_name": "Overall posting length",
        "fraud_explanation": "Fraudulent postings are often either extremely short or padded with filler.",
        "safe_explanation": "A substantive overall posting length indicates genuine effort from an employer.",
    },
}


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------

def _safe_str(value: object) -> str:
    """Return a clean string from any input, treating NaN/None as empty."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _url_count(text: str) -> int:
    return len(re.findall(r"https?://", text, re.IGNORECASE))


def _all_caps_ratio(text: str) -> float:
    """Fraction of words that are fully uppercase (length > 1 to skip 'I', 'A')."""
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    caps_words = [w for w in words if len(w) > 1 and w.isupper()]
    return len(caps_words) / len(words)


def _exclamation_count(text: str) -> int:
    return text.count("!")


def _avg_word_length(text: str) -> float:
    if not text:
        return 0.0
    words = [w for w in text.split() if w.isalpha()]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _location_is_vague(location: str) -> int:
    """
    Returns 1 if the location is missing or suspiciously non-specific.
    Patterns: empty, 'anywhere', 'remote' only, single-word generic terms.
    """
    if not location:
        return 1
    loc_lower = location.lower().strip()
    vague_patterns = [
        r"^anywhere$",
        r"^remote$",
        r"^online$",
        r"^worldwide$",
        r"^global$",
        r"^home$",
        r"^work from home$",
        r"^virtual$",
        r"^n/?a$",
    ]
    for pattern in vague_patterns:
        if re.match(pattern, loc_lower):
            return 1
    return 0


# ---------------------------------------------------------------------------
# Core feature extraction — single posting (dict or Pydantic model fields)
# ---------------------------------------------------------------------------

def extract_features_single(
    title: Optional[str],
    description: Optional[str],
    requirements: Optional[str],
    benefits: Optional[str],
    company_profile: Optional[str],
    location: Optional[str],
    salary_range: Optional[str],
    employment_type: Optional[str],
    required_experience: Optional[str],
    required_education: Optional[str],
    has_company_logo: bool,
    has_questions: bool,
    telecommuting: bool,
) -> dict[str, float]:
    """
    Compute all structural features for a single posting.
    Returns a dict keyed by FEATURE_COLUMNS — ready to be passed to XGBoost.

    This function is called at inference time by predict.py.
    It must produce identical output to extract_features_df() for training
    to be valid — any logic change must be mirrored in both.
    """
    t = _safe_str(title)
    d = _safe_str(description)
    r = _safe_str(requirements)
    b = _safe_str(benefits)
    cp = _safe_str(company_profile)
    loc = _safe_str(location)
    sal = _safe_str(salary_range)
    et = _safe_str(employment_type)
    exp = _safe_str(required_experience)
    edu = _safe_str(required_education)

    full_text = " ".join([t, d, r, b, cp])

    desc_len = _word_count(d)
    req_len = _word_count(r)
    ben_len = _word_count(b)
    cp_len = _word_count(cp)

    return {
        "has_salary":                   int(bool(sal)),
        "has_company_logo":             int(has_company_logo),
        "has_questions":                int(has_questions),
        "telecommuting":                int(telecommuting),
        "employment_type_missing":      int(not bool(et)),
        "experience_missing":           int(not bool(exp)),
        "education_missing":            int(not bool(edu)),
        "company_profile_missing":      int(not bool(cp)),
        "location_vagueness":           _location_is_vague(loc),
        "description_length":           desc_len,
        "requirements_length":          req_len,
        "benefit_length":               ben_len,
        "company_profile_length":       cp_len,
        "url_count":                    _url_count(full_text),
        "all_caps_ratio":               round(_all_caps_ratio(d), 4),
        "exclamation_count":            _exclamation_count(full_text),
        "avg_word_length_desc":         round(_avg_word_length(d), 4),
        "title_length":                 _word_count(t),
        "benefit_to_requirement_ratio": round(
            ben_len / req_len if req_len > 0 else float(ben_len > 0), 4
        ),
        "text_total_length":            _word_count(full_text),
    }


# ---------------------------------------------------------------------------
# Batch feature extraction — DataFrame (used by train.py)
# ---------------------------------------------------------------------------

def extract_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised feature extraction over a full DataFrame.
    Input: raw EMSCAD DataFrame with original column names.
    Output: DataFrame with exactly FEATURE_COLUMNS as columns, in order.

    NaN values in the source columns are handled explicitly — never propagated.
    """

    # ---- Fill NaN with empty string for text columns ----
    text_cols = [
        "title", "description", "requirements", "benefits",
        "company_profile", "location", "salary_range",
        "employment_type", "required_experience", "required_education",
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # ---- Boolean columns (may arrive as 0/1 int or bool) ----
    for bool_col in ["has_company_logo", "has_questions", "telecommuting"]:
        if bool_col not in df.columns:
            df[bool_col] = 0
        else:
            df[bool_col] = df[bool_col].fillna(0).astype(int)

    # ---- Derived text features ----
    full_text = (
        df["title"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"] + " " +
        df["company_profile"]
    )

    features = pd.DataFrame(index=df.index)

    features["has_salary"] = df["salary_range"].apply(lambda x: int(bool(x))).astype(int)
    features["has_company_logo"] = df["has_company_logo"]
    features["has_questions"] = df["has_questions"]
    features["telecommuting"] = df["telecommuting"]

    features["employment_type_missing"] = df["employment_type"].apply(lambda x: int(not bool(x)))
    features["experience_missing"] = df["required_experience"].apply(lambda x: int(not bool(x)))
    features["education_missing"] = df["required_education"].apply(lambda x: int(not bool(x)))
    features["company_profile_missing"] = df["company_profile"].apply(lambda x: int(not bool(x)))

    features["location_vagueness"] = df["location"].apply(_location_is_vague)

    features["description_length"] = df["description"].apply(_word_count)
    features["requirements_length"] = df["requirements"].apply(_word_count)
    features["benefit_length"] = df["benefits"].apply(_word_count)
    features["company_profile_length"] = df["company_profile"].apply(_word_count)

    features["url_count"] = full_text.apply(_url_count)
    features["all_caps_ratio"] = df["description"].apply(_all_caps_ratio).round(4)
    features["exclamation_count"] = full_text.apply(_exclamation_count)
    features["avg_word_length_desc"] = df["description"].apply(_avg_word_length).round(4)
    features["title_length"] = df["title"].apply(_word_count)

    def _benefit_req_ratio(row: pd.Series) -> float:
        req = _word_count(row["requirements"])
        ben = _word_count(row["benefits"])
        if req > 0:
            return round(ben / req, 4)
        return float(ben > 0)

    features["benefit_to_requirement_ratio"] = df.apply(_benefit_req_ratio, axis=1)
    features["text_total_length"] = full_text.apply(_word_count)

    # ---- Enforce column order ----
    return features[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Text concatenation for DistilBERT input
# ---------------------------------------------------------------------------

def build_bert_input(
    title: Optional[str],
    description: Optional[str],
    requirements: Optional[str],
) -> str:
    """
    Concatenate the three most signal-dense text fields for DistilBERT.
    Format: '<title> [SEP] <description> [SEP] <requirements>'
    Missing fields are replaced with empty string — the [SEP] tokens remain
    so the tokenizer's segment boundaries stay consistent.

    Truncation to 512 tokens is handled by the tokenizer in predict.py /
    train.py — we do not truncate here to avoid double-cutting.
    """
    parts = [
        _safe_str(title),
        _safe_str(description),
        _safe_str(requirements),
    ]
    return " [SEP] ".join(parts)


# ---------------------------------------------------------------------------
# Feature vector → ordered numpy array (for XGBoost at inference)
# ---------------------------------------------------------------------------

def feature_dict_to_array(feature_dict: dict[str, float]) -> np.ndarray:
    """
    Convert the output of extract_features_single() to a 1×N numpy array
    in the canonical FEATURE_COLUMNS order.
    Raises KeyError if any expected feature is missing.
    """
    vector = [feature_dict[col] for col in FEATURE_COLUMNS]
    return np.array(vector, dtype=np.float32).reshape(1, -1)