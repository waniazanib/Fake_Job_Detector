// ============================================================
// analyze.ts — JobGuard API layer
// All HTTP calls go through here. Components never import axios.
// ============================================================

import axios, { AxiosError, AxiosInstance } from 'axios'
import type {
  AnalyzeResponse,
  ApiError,
  HealthResponse,
  JobPostingRequest,
} from '@/types/api'

// ── Axios instance ────────────────────────────────────────────

const client: AxiosInstance = axios.create({
  // In dev, Vite proxies /api → localhost:8000 (vite.config.ts)
  // In prod, set VITE_API_BASE_URL to your deployed backend origin
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '',
  timeout: 60_000,   // BERT inference can take a few seconds on CPU
  headers: {
    'Content-Type': 'application/json',
  },
})

// ── Error normalisation ───────────────────────────────────────

export class JobGuardApiError extends Error {
  public readonly code: string
  public readonly status: number

  constructor(detail: string, code: string, status: number) {
    super(detail)
    this.name    = 'JobGuardApiError'
    this.code    = code
    this.status  = status
  }
}

function normaliseError(err: unknown): never {
  if (err instanceof AxiosError) {
    const data  = err.response?.data as Partial<ApiError> | undefined
    const detail = data?.detail ?? err.message ?? 'An unexpected error occurred.'
    const code   = data?.code   ?? 'UNKNOWN_ERROR'
    const status = err.response?.status ?? 0
    throw new JobGuardApiError(detail, code, status)
  }
  // Non-axios errors (network down, timeout, etc.)
  throw new JobGuardApiError(
    'Could not reach the JobGuard server. Make sure the backend is running.',
    'NETWORK_ERROR',
    0,
  )
}

// ── Request sanitisation ──────────────────────────────────────

/**
 * Convert empty strings → null before sending to the backend.
 * The backend does this too (via Pydantic validator) but doing it
 * on the client avoids sending needless empty-string payloads.
 */
function sanitiseRequest(req: JobPostingRequest): JobPostingRequest {
  const sanitised = { ...req }
  const textFields = [
    'title', 'description', 'requirements', 'benefits',
    'company_profile', 'location', 'salary_range',
    'employment_type', 'required_experience', 'required_education',
  ] as const

  for (const field of textFields) {
    const val = sanitised[field]
    if (typeof val === 'string' && val.trim() === '') {
      sanitised[field] = null
    }
  }

  return sanitised
}

// ── API calls ─────────────────────────────────────────────────

/**
 * POST /api/analyze
 * Runs dual-branch fraud detection on the submitted job posting.
 * Returns a fully populated AnalyzeResponse including SHAP signals.
 *
 * Throws JobGuardApiError on any failure — callers should catch it
 * and display err.message in the UI.
 */
export async function analyzePosting(
  request: JobPostingRequest,
): Promise<AnalyzeResponse> {
  try {
    const { data } = await client.post<AnalyzeResponse>(
      '/api/analyze',
      sanitiseRequest(request),
    )
    return data
  } catch (err) {
    return normaliseError(err)
  }
}

/**
 * GET /api/health
 * Checks whether both ML models are loaded and ready.
 * Called on app mount — if models aren't ready, the UI shows a banner.
 *
 * Returns null on network failure (backend not running) rather than
 * throwing, so the app can degrade gracefully.
 */
export async function fetchHealth(): Promise<HealthResponse | null> {
  try {
    const { data } = await client.get<HealthResponse>('/api/health')
    return data
  } catch {
    return null
  }
}

/**
 * POST /api/train
 * Dev-only — triggers the training pipeline as a background subprocess.
 * Returns immediately with a 202 Accepted; monitor progress in server logs.
 *
 * Throws JobGuardApiError if ALLOW_TRAIN=false on the backend (403).
 */
export async function triggerTraining(): Promise<{ message: string }> {
  try {
    const { data } = await client.post<{ message: string }>('/api/train')
    return data
  } catch (err) {
    return normaliseError(err)
  }
}