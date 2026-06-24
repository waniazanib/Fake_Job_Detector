from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FraudLabel(str, Enum):
    LEGITIMATE = "LEGITIMATE"
    CAUTION = "CAUTION"
    SUSPICIOUS = "SUSPICIOUS"


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ShapDirection(str, Enum):
    FRAUD = "fraud"
    SAFE = "safe"


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class JobPostingRequest(BaseModel):
    """
    Represents a single job posting submitted for fraud analysis.
    All text fields are optional because real-world postings are often
    incomplete — missingness itself is a fraud signal captured in features.py.
    """

    title: Optional[str] = Field(
        default=None,
        max_length=300,
        description="Job title as shown in the posting.",
        examples=["Senior Software Engineer"],
    )
    description: Optional[str] = Field(
        default=None,
        max_length=10_000,
        description="Full job description body.",
        examples=["We are looking for a passionate engineer to join our team..."],
    )
    requirements: Optional[str] = Field(
        default=None,
        max_length=5_000,
        description="Skills and qualifications section.",
        examples=["5+ years Python experience, strong communication skills."],
    )
    benefits: Optional[str] = Field(
        default=None,
        max_length=3_000,
        description="Perks and benefits listed in the posting.",
        examples=["Health insurance, unlimited PTO, remote work."],
    )
    company_profile: Optional[str] = Field(
        default=None,
        max_length=5_000,
        description="Company description or 'About us' section.",
        examples=["Acme Corp is a Fortune 500 company founded in 1985..."],
    )
    location: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Job location string (city, country, or 'Anywhere').",
        examples=["New York, NY", "Anywhere", "Remote"],
    )
    salary_range: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Salary range string, if provided.",
        examples=["80000-100000", "$50/hr", "Competitive"],
    )
    employment_type: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Contract type (Full-time, Part-time, Contract, etc.).",
        examples=["Full-time", "Contract", "Temporary"],
    )
    required_experience: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Experience level required.",
        examples=["Mid-Senior level", "Entry level", "Executive"],
    )
    required_education: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Minimum education level required.",
        examples=["Bachelor's Degree", "High School or equivalent", "Master's Degree"],
    )
    has_company_logo: bool = Field(
        default=False,
        description="Whether the posting includes a company logo.",
    )
    has_questions: bool = Field(
        default=False,
        description="Whether the posting includes screening questions.",
    )
    telecommuting: bool = Field(
        default=False,
        description="Whether the role is listed as telecommuting / remote.",
    )

    @field_validator("title", "description", "requirements", "benefits",
                     "company_profile", "location", "salary_range",
                     "employment_type", "required_experience", "required_education",
                     mode="before")
    @classmethod
    def empty_string_to_none(cls, v: object) -> Optional[str]:
        """Treat blank strings as missing — both are meaningful missingness."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @model_validator(mode="after")
    def at_least_one_text_field(self) -> JobPostingRequest:
        """
        Require at minimum a title or description so the text branch has
        something to work with. Purely structural-only submissions are
        rejected here to prevent degenerate predictions.
        """
        if not self.title and not self.description:
            raise ValueError(
                "At least one of 'title' or 'description' must be provided."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Data Entry Specialist — Work From Home",
                "description": (
                    "AMAZING OPPORTUNITY!! Earn $5000/week from home. "
                    "No experience needed. Send your resume to jobs@quickcash.biz "
                    "and we will contact you within 24 hours. http://apply-now.xyz"
                ),
                "requirements": "Must have internet access. No degree required.",
                "benefits": "Unlimited earnings, flexible hours, full benefits from day one.",
                "company_profile": None,
                "location": "Anywhere",
                "salary_range": None,
                "employment_type": None,
                "required_experience": "Not Applicable",
                "required_education": None,
                "has_company_logo": False,
                "has_questions": False,
                "telecommuting": True,
            }
        }
    }


# ---------------------------------------------------------------------------
# SHAP signal — one entry per top feature
# ---------------------------------------------------------------------------

class ShapSignal(BaseModel):
    """
    A single SHAP feature contribution surfaced to the user.
    'impact' is always positive; 'direction' tells you which way it pushed.
    """

    feature: str = Field(
        description="Internal feature name, e.g. 'has_salary'.",
        examples=["has_salary"],
    )
    display_name: str = Field(
        description="Human-readable label shown in the UI.",
        examples=["Salary not listed"],
    )
    value: float = Field(
        description="Raw feature value for this posting.",
        examples=[0, 1, 45, 0.18],
    )
    impact: float = Field(
        ge=0.0,
        description="Absolute SHAP contribution (always ≥ 0).",
        examples=[0.312],
    )
    direction: ShapDirection = Field(
        description="Whether this feature pushed toward fraud or safe.",
        examples=["fraud"],
    )
    explanation: str = Field(
        description="One-sentence plain-English explanation for the user.",
        examples=["Legitimate job postings almost always include a salary range."],
    )


# ---------------------------------------------------------------------------
# Analysis response
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    """
    Full response returned by POST /api/analyze.
    """

    fraud_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fused fraud probability in [0, 1]. Higher = more suspicious.",
        examples=[0.847],
    )
    label: FraudLabel = Field(
        description=(
            "LEGITIMATE if score < 0.35, CAUTION if 0.35–0.65, "
            "SUSPICIOUS if > 0.65."
        ),
        examples=["SUSPICIOUS"],
    )
    confidence: ConfidenceLevel = Field(
        description=(
            "LOW if both branch scores disagree by > 0.3, "
            "MEDIUM if 0.15–0.3, HIGH if < 0.15."
        ),
        examples=["HIGH"],
    )
    text_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraud probability from the DistilBERT text branch alone.",
        examples=[0.91],
    )
    struct_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraud probability from the XGBoost structural branch alone.",
        examples=[0.76],
    )
    shap_signals: list[ShapSignal] = Field(
        description="Top 5 SHAP feature contributions, sorted by impact descending.",
        max_length=5,
    )
    summary: str = Field(
        description="One auto-generated sentence summarising why the posting scored as it did.",
        examples=[
            "This posting lacks a salary, company profile, and logo — "
            "three of the strongest indicators of a fraudulent listing."
        ],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "fraud_score": 0.847,
                "label": "SUSPICIOUS",
                "confidence": "HIGH",
                "text_score": 0.91,
                "struct_score": 0.76,
                "shap_signals": [
                    {
                        "feature": "has_salary",
                        "display_name": "Salary not listed",
                        "value": 0,
                        "impact": 0.312,
                        "direction": "fraud",
                        "explanation": "Legitimate job postings almost always include a salary range.",
                    },
                    {
                        "feature": "company_profile_missing",
                        "display_name": "No company description",
                        "value": 1,
                        "impact": 0.198,
                        "direction": "fraud",
                        "explanation": "Real employers typically describe their company.",
                    },
                    {
                        "feature": "url_count",
                        "display_name": "External links in description",
                        "value": 4,
                        "impact": 0.154,
                        "direction": "fraud",
                        "explanation": "Multiple external URLs are a common phishing signal.",
                    },
                    {
                        "feature": "has_company_logo",
                        "display_name": "No company logo",
                        "value": 0,
                        "impact": 0.143,
                        "direction": "fraud",
                        "explanation": "Verified employers almost always attach a company logo.",
                    },
                    {
                        "feature": "description_length",
                        "display_name": "Very short description",
                        "value": 45,
                        "impact": 0.091,
                        "direction": "fraud",
                        "explanation": "Fraudulent postings tend to have unusually brief descriptions.",
                    },
                ],
                "summary": (
                    "This posting lacks a salary, company profile, and logo — "
                    "three of the strongest indicators of a fraudulent listing."
                ),
            }
        }
    }


# ---------------------------------------------------------------------------
# Health check response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    models_loaded: bool = Field(
        description="True once both XGBoost and DistilBERT models are in memory.",
    )
    xgb_ready: bool
    bert_ready: bool


# ---------------------------------------------------------------------------
# Error response (used by FastAPI exception handlers)
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str = Field(
        description="Human-readable error message.",
        examples=["At least one of 'title' or 'description' must be provided."],
    )
    code: str = Field(
        description="Machine-readable error code.",
        examples=["VALIDATION_ERROR", "MODEL_NOT_LOADED", "INFERENCE_FAILED"],
    )