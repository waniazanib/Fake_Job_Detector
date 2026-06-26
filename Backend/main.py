"""
main.py — JobGuard FastAPI application

Entry point for the backend server. Handles:
  - Lifespan: model loading at startup, cleanup at shutdown
  - CORS: permits React dev server (localhost:5173) and production origin
  - POST /api/analyze  — dual-branch inference + SHAP explanation
  - GET  /api/health   — readiness probe
  - POST /api/train    — dev-only training trigger (gated by ALLOW_TRAIN env flag)
  - Global exception handlers for clean JSON error responses

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Path setup — allows `from src.X import Y` to work when running from
# the backend/ directory as well as from backend/src/
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
SRC_DIR  = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from src.schemas  import AnalyzeResponse, ErrorResponse, HealthResponse, JobPostingRequest
from src.predict  import Predictor
from src.explainer import Explainer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jobguard.main")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(BASE_DIR / ".env")

ALLOW_TRAIN    = os.getenv("ALLOW_TRAIN", "false").lower() == "true"
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# ---------------------------------------------------------------------------
# Application state — holds singletons alive for process lifetime
# ---------------------------------------------------------------------------

class _AppState:
    predictor: Predictor | None = None
    explainer: Explainer | None = None


state = _AppState()


# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models once at startup; release references at shutdown."""
    log.info("━" * 55)
    log.info("JobGuard API starting up")
    log.info("━" * 55)

    t0 = time.perf_counter()

    try:
        state.predictor = Predictor()
    except Exception as exc:
        log.error("Failed to initialise Predictor: %s", exc)
        state.predictor = None

    if state.predictor is not None and state.predictor.xgb_ready:
        try:
            # Explainer needs the fitted XGBoost model directly
            state.explainer = Explainer(state.predictor._xgb_model)
        except Exception as exc:
            log.error("Failed to initialise Explainer: %s", exc)
            state.explainer = None
    else:
        log.warning(
            "XGBoost not ready — Explainer skipped. "
            "Run src/train.py to generate models."
        )

    elapsed = time.perf_counter() - t0
    log.info(
        "Startup complete in %.2fs — XGBoost=%s  BERT=%s  SHAP=%s",
        elapsed,
        "✓" if (state.predictor and state.predictor.xgb_ready)  else "✗",
        "✓" if (state.predictor and state.predictor.bert_ready)  else "✗",
        "✓" if state.explainer else "✗",
    )

    yield  # Server is live — handle requests

    log.info("JobGuard API shutting down — releasing model references")
    state.predictor = None
    state.explainer = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "JobGuard API",
    description = (
        "Dual-branch fake job posting detector. "
        "Combines DistilBERT text analysis with XGBoost structural feature scoring, "
        "fused via weighted late fusion. SHAP explanations on every prediction."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins     = [FRONTEND_ORIGIN, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials = True,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code = exc.status_code,
        content     = ErrorResponse(
            detail = exc.detail,
            code   = _status_to_code(exc.status_code),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = ErrorResponse(
            detail = "An unexpected server error occurred.",
            code   = "INTERNAL_ERROR",
        ).model_dump(),
    )


def _status_to_code(status_code: int) -> str:
    return {
        400: "BAD_REQUEST",
        422: "VALIDATION_ERROR",
        503: "MODEL_NOT_LOADED",
        500: "INFERENCE_FAILED",
    }.get(status_code, "ERROR")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    response_model = HealthResponse,
    summary        = "Readiness probe",
    tags           = ["System"],
)
async def health() -> HealthResponse:
    """
    Returns the readiness status of both ML models.
    Frontend polls this on mount to show a loading banner
    if models are still being loaded.
    """
    xgb_ready  = bool(state.predictor and state.predictor.xgb_ready)
    bert_ready = bool(state.predictor and state.predictor.bert_ready)

    return HealthResponse(
        status        = "ok",
        models_loaded = xgb_ready and bert_ready,
        xgb_ready     = xgb_ready,
        bert_ready    = bert_ready,
    )


@app.post(
    "/api/analyze",
    
    response_model  = AnalyzeResponse,
    summary         = "Analyse a job posting for fraud signals",
    tags            = ["Inference"],
    responses       = {
        503: {"model": ErrorResponse, "description": "Models not yet loaded"},
        422: {"model": ErrorResponse, "description": "Request validation failed"},
        500: {"model": ErrorResponse, "description": "Inference error"},
    },
)
async def analyze(request: JobPostingRequest) -> AnalyzeResponse:
    """
    Runs dual-branch fraud detection on the submitted job posting.

    **Pipeline (per request):**
    1. XGBoost structural branch — engineered features → fraud probability
    2. DistilBERT text branch — title + description + requirements → fraud probability
    3. Late fusion — weighted average of both branch scores
    4. SHAP — top-5 feature contributions from the structural branch
    5. Label + confidence + plain-English summary derived from fused score

    Returns a fully populated `AnalyzeResponse`.
    """
    if state.predictor is None or not state.predictor.models_loaded:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                "Models are not loaded. "
                "Run src/train.py to train and save models, then restart the server."
            ),
        )

    t0 = time.perf_counter()

    try:
        # Step 1 + 2 + 3 — inference (no SHAP signals yet)
        result = state.predictor.predict(request, shap_signals=None)

        # Step 4 — SHAP explanation (structural branch only)
        shap_signals = []
        if state.explainer is not None:
            try:
                feature_dict = state.predictor.get_feature_dict(request)
                shap_signals = state.explainer.explain(feature_dict)
            except Exception as shap_exc:
                # SHAP failure must never block the inference response
                log.warning("SHAP explanation failed (non-fatal): %s", shap_exc)

        # Step 5 — re-build response with SHAP signals so summary can reference them
        result = state.predictor.predict(request, shap_signals=shap_signals)

    except RuntimeError as exc:
        log.error("Inference RuntimeError: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = str(exc),
        )
    except Exception as exc:
        log.exception("Inference failed for request: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Inference failed — check server logs for details.",
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "analyze → label=%s score=%.3f text=%.3f struct=%.3f shap=%d signals  %.0fms",
        result.label.value,
        result.fraud_score,
        result.text_score,
        result.struct_score,
        len(result.shap_signals),
        elapsed_ms,
    )

    return result


@app.post(
    "/api/train",
    summary  = "Trigger training pipeline (dev only)",
    tags     = ["System"],
    responses= {
        403: {"model": ErrorResponse, "description": "Training disabled in this environment"},
        500: {"model": ErrorResponse, "description": "Training process failed"},
    },
)
async def trigger_train() -> JSONResponse:
    """
    Launches `src/train.py` as a subprocess.
    Only available when `ALLOW_TRAIN=true` in the environment.
    Intended for development use only — disable in production.

    Training runs asynchronously; this endpoint returns immediately.
    Monitor progress via server logs.
    """
    if not ALLOW_TRAIN:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Training is disabled. Set ALLOW_TRAIN=true in .env to enable.",
        )

    train_script = SRC_DIR / "train.py"
    if not train_script.exists():
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Training script not found at {train_script}.",
        )

    try:
        subprocess.Popen(
            [sys.executable, str(train_script)],
            cwd    = str(BASE_DIR),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
        )
        log.info("Training subprocess launched: %s", train_script)
        return JSONResponse(
            status_code = status.HTTP_202_ACCEPTED,
            content     = {
                "message": "Training started. Models will be available after the process completes.",
                "script":  str(train_script),
            },
        )
    except Exception as exc:
        log.exception("Failed to launch training subprocess: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Failed to start training: {exc}",
        )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"detail": "Frontend not built."}, status_code=404)
    
    @app.get("/", include_in_schema=False)
    async def serve_root():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"detail": "Frontend not built."}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        log_level = "info",
    )