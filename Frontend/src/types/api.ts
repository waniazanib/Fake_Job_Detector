// ============================================================
// api.ts — JobGuard frontend type contracts
// Must stay in sync with backend/src/schemas.py
// ============================================================

// ── Enums ────────────────────────────────────────────────────

export type FraudLabel = 'LEGITIMATE' | 'CAUTION' | 'SUSPICIOUS'

export type ConfidenceLevel = 'LOW' | 'MEDIUM' | 'HIGH'

export type ShapDirection = 'fraud' | 'safe'

// ── Request ──────────────────────────────────────────────────

export interface JobPostingRequest {
  title:               string | null
  description:         string | null
  requirements:        string | null
  benefits:            string | null
  company_profile:     string | null
  location:            string | null
  salary_range:        string | null
  employment_type:     string | null
  required_experience: string | null
  required_education:  string | null
  has_company_logo:    boolean
  has_questions:       boolean
  telecommuting:       boolean
}

// ── SHAP signal ───────────────────────────────────────────────

export interface ShapSignal {
  feature:      string         // internal name e.g. 'has_salary'
  display_name: string         // human label e.g. 'Salary not listed'
  value:        number         // raw feature value
  impact:       number         // absolute SHAP contribution ≥ 0
  direction:    ShapDirection  // 'fraud' | 'safe'
  explanation:  string         // one-sentence plain-English reason
}

// ── Response ─────────────────────────────────────────────────

export interface AnalyzeResponse {
  fraud_score:  number          // fused probability [0, 1]
  label:        FraudLabel
  confidence:   ConfidenceLevel
  text_score:   number          // DistilBERT branch probability
  struct_score: number          // XGBoost branch probability
  shap_signals: ShapSignal[]    // top 5, sorted by impact desc
  summary:      string          // auto-generated plain-English sentence
}

// ── Health ────────────────────────────────────────────────────

export interface HealthResponse {
  status:        string
  models_loaded: boolean
  xgb_ready:    boolean
  bert_ready:   boolean
}

// ── API error ─────────────────────────────────────────────────

export interface ApiError {
  detail: string
  code:   string
}

// ── UI state helpers ──────────────────────────────────────────

/** Maps FraudLabel → CSS class suffix used in index.css badge utilities */
export const LABEL_CLASS: Record<FraudLabel, string> = {
  LEGITIMATE: 'badge--legitimate',
  CAUTION:    'badge--caution',
  SUSPICIOUS: 'badge--suspicious',
}

/** Maps FraudLabel → user-facing display string */
export const LABEL_COPY: Record<FraudLabel, string> = {
  LEGITIMATE: 'Likely Legitimate',
  CAUTION:    'Proceed with Caution',
  SUSPICIOUS: 'Likely Fraudulent',
}

/** Maps ConfidenceLevel → display string */
export const CONFIDENCE_COPY: Record<ConfidenceLevel, string> = {
  HIGH:   'High confidence',
  MEDIUM: 'Medium confidence',
  LOW:    'Low confidence — branches disagree',
}

/** Score thresholds — must match backend predict.py constants */
export const SCORE_THRESHOLDS = {
  LEGITIMATE_MAX: 0.35,
  SUSPICIOUS_MIN: 0.65,
} as const

/** Derive FraudLabel from a raw score (client-side, for optimistic UI) */
export function scoreToLabel(score: number): FraudLabel {
  if (score < SCORE_THRESHOLDS.LEGITIMATE_MAX) return 'LEGITIMATE'
  if (score <= SCORE_THRESHOLDS.SUSPICIOUS_MIN) return 'CAUTION'
  return 'SUSPICIOUS'
}

/** Empty request — used to initialise the JobForm */
export const EMPTY_REQUEST: JobPostingRequest = {
  title:               null,
  description:         null,
  requirements:        null,
  benefits:            null,
  company_profile:     null,
  location:            null,
  salary_range:        null,
  employment_type:     null,
  required_experience: null,
  required_education:  null,
  has_company_logo:    false,
  has_questions:       false,
  telecommuting:       false,
}