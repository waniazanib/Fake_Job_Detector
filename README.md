---
title: JobGuard
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# JobGuard — Fake Job Posting Detector

An AI-powered web application that analyses job postings for fraud signals using a dual-branch machine learning architecture. Paste any job listing and receive an instant fraud probability score with explainable feature contributions.

---

## Features

- **Dual-branch inference** — DistilBERT analyses job description language while XGBoost scores structural signals (salary, location, company logo, links). Both scores are fused via weighted late fusion
- **SHAP explainability** — every prediction surfaces the top 5 feature contributions with plain-English explanations, not just a score
- **Animated result UI** — SVG half-circle gauge, animated branch breakdown bars, and a horizontal SHAP impact chart
- **Confidence scoring** — flags when the two branches strongly disagree, indicating ambiguous postings
- **Model-ready architecture** — ONNX export for fast CPU inference, PyTorch fallback if ONNX is unavailable

---

## Tech Stack

**Backend**
| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Text branch | DistilBERT (`distilbert-base-uncased`) fine-tuned via HuggingFace Transformers |
| Structural branch | XGBoost with 20 engineered features |
| Explainability | SHAP TreeExplainer |
| Inference | ONNX Runtime (PyTorch fallback) |

**Frontend**
| Layer | Technology |
|---|---|
| Framework | React 18 + TypeScript |
| Build | Vite |
| Animations | Framer Motion |
| Charts | Recharts |
| HTTP | Axios |

**Training**
- Dataset: [EMSCAD — Employment Scam Aegean Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) (17,880 postings, ~4.5% fraudulent)

---

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- Trained model files in `Backend/models/` (see Training section below)

### Backend

```bash
cd Backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Frontend

```bash
cd frontend
npm install
```

---

## Training

Download `fake_job_postings.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) and place it in `Backend/data/`.

**Option A — Local (CPU, ~4 hours)**
```bash
cd Backend
python src/train.py
```

**Option B — Google Colab (GPU, ~35 minutes, recommended)**

1. Upload `colab_train.py` and `Backend/src/features.py` to Colab
2. Upload `fake_job_postings.csv` to the Colab files tab
3. Run the notebook — models save to Google Drive automatically
4. Download the generated `jobguard_models.zip` and extract into `Backend/models/`

---

## Usage

```bash
# Start Backend (from Backend/)
uvicorn main:app --reload --port 8000

# Start frontend (from frontend/)
npm run dev
```

Open `http://localhost:5173`, paste a job posting, and click **Analyse posting**.

The API is also available directly at `http://localhost:8000/docs` (Swagger UI).

---

## Environment Variables

### `Backend/.env`
| Variable | Default | Description |
|---|---|---|
| `ALLOW_TRAIN` | `true` | Enables `POST /api/train` endpoint |
| `MODEL_DIR` | `./models` | Path to saved model artefacts |
| `DATA_PATH` | `./data/fake_job_postings.csv` | Training data path |
| `DISTILBERT_MODEL` | `distilbert-base-uncased` | HuggingFace model ID |
| `FUSION_TEXT_WEIGHT` | `0.55` | DistilBERT branch weight in fusion |
| `FUSION_STRUCT_WEIGHT` | `0.45` | XGBoost branch weight in fusion |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Allowed CORS origin |

### `frontend/.env`
| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend base URL |

---

## Folder Structure

```
jobguard/
├── Backend/
│   ├── data/                         # EMSCAD dataset (not committed)
│   ├── models/                       # Trained model artefacts (not committed)
│   │   ├── xgb_model.joblib
│   │   ├── xgb_threshold.joblib
│   │   ├── fusion_weights.joblib
│   │   ├── distilbert_finetuned/
│   │   └── distilbert_onnx/
│   ├── src/
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── features.py               # Feature engineering (train + inference)
│   │   ├── train.py                  # Local training pipeline
│   │   ├── predict.py                # Dual-branch inference engine
│   │   └── explainer.py              # SHAP integration
│   ├── main.py                       # FastAPI application
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   └── src/
│       ├── api/analyze.ts            # Axios API layer
│       ├── types/api.ts              # TypeScript contracts
│       └── components/
│           ├── Header/
│           ├── JobForm/              # Input form with progressive disclosure
│           ├── ScoreGauge/           # Animated SVG half-circle gauge
│           ├── ShapChart/            # SHAP horizontal bar chart + signal cards
│           └── ResultPanel/          # Composed result view
│
├── colab_train.py                    # Colab-optimised training script
└── AppFlow.md                        # Architecture and build reference
```

---

## API Reference

### `POST /api/analyze`
Accepts a job posting and returns a fraud analysis.

**Request body** — all text fields optional except at least one of `title` or `description`:
```json
{
  "title": "Data Entry Specialist",
  "description": "Earn $5000/week from home...",
  "has_company_logo": false,
  "has_questions": false,
  "telecommuting": true
}
```

**Response:**
```json
{
  "fraud_score": 0.847,
  "label": "SUSPICIOUS",
  "confidence": "HIGH",
  "text_score": 0.91,
  "struct_score": 0.76,
  "summary": "This posting lacks a salary, company profile, and logo — three of the strongest indicators of a fraudulent listing.",
  "shap_signals": [...]
}
```

### `GET /api/health`
Returns model readiness status.

---

## Future Improvements

- **Browser extension** — analyse job postings on LinkedIn and Indeed without leaving the page
- **URL input** — scrape and analyse a posting directly from a job board URL
- **Multilingual support** — extend the text branch to Urdu/English code-switched postings using XLM-RoBERTa
- **Feedback loop** — let users flag incorrect predictions to build a correction dataset for retraining
- **Global importance view** — expose mean absolute SHAP values across the training set as a dashboard panel

---

## Dataset

This project uses the [EMSCAD dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) by Shivam Bansal, published under the CC0 Public Domain license.
