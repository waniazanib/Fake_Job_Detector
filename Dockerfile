# ============================================================
# JobGuard — Dockerfile for Hugging Face Spaces
# Models downloaded from HF Model Hub at build time
#
# Before building, upload models once:
#   huggingface-cli upload <your-username>/jobguard-models Backend/models . --repo-type=model
#
# Set HF_TOKEN as a Space secret in HF Spaces settings
# (Settings → Repository secrets → New secret → HF_TOKEN)
# ============================================================

# ── Stage 1: Build React Frontend ────────────────────────────

FROM node:20-slim AS Frontend-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/Frontend

COPY Frontend/package.json Frontend/package-lock.json* ./
RUN npm install 2>&1 || (cat /root/.npm/_logs/*.log && exit 1)

COPY Frontend/ .
RUN npm run build || (cat /app/Frontend/node_modules/.bin/tsc 2>/dev/null; exit 1)
# Output: /app/Frontend/dist


# ── Stage 2: Python Backend ───────────────────────────────────

FROM python:3.11-slim AS Backend

# HF Spaces requires non-root user uid 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────
COPY requirements.txt .

# CPU-only torch — saves ~1.5GB vs CUDA build

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

# spaCy language model
RUN python -m spacy download en_core_web_sm

# ── Copy Backend source ───────────────────────────────────────
COPY Backend/main.py  ./main.py
COPY Backend/src/     ./src/

# ── Download models from HF Hub at build time ─────────────────
# HF_TOKEN is passed as a build secret from Space settings
# repo_id must match where you uploaded with huggingface-cli

COPY Backend/download_models.py ./download_models.py
RUN python download_models.py

# ── Copy built Frontend ───────────────────────────────────────
COPY --from=Frontend-builder /app/Frontend/dist ./static/

# ── Environment ───────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models \
    ALLOW_TRAIN=false \
    Frontend_ORIGIN=* \
    HF_MODEL_REPO=waniazanib/Job_Checking_Model

# HF Spaces only exposes 7860
EXPOSE 7860

RUN chown -R appuser:appuser /app
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]