# JobGuard — Fake Job Posting Detector
## AppFlow.md — Project Context & Build Reference

---`

## 1. Problem Statement

Fraudulent job postings scam millions of job seekers annually — harvesting resumes, stealing identities, and extorting application fees. This tool lets a user paste a raw job posting (or fill in structured fields) and instantly receives a fraud probability score with a plain-English breakdown of the signals that triggered it.

---

## 2. Tech Stack

### Backend
| Layer | Technology |
|---|---|
| API framework | FastAPI (Python 3.11) |
| ML — text branch | `distilbert-base-uncased` fine-tuned via HuggingFace Transformers |
| ML — structural branch | XGBoost (`xgboost>=2.0`) |
| ML — fusion | Late fusion: weighted average of both branch probabilities |
| Explainability | SHAP (`shap>=0.44`) — TreeExplainer for XGBoost |
| NLP preprocessing | spaCy (`en_core_web_sm`) |
| Model serialization | `joblib` (XGBoost), `safetensors` (DistilBERT via HuggingFace) |
| ONNX export | `optimum[onnxruntime]` — DistilBERT → ONNX for fast inference |
| Data handling | pandas, numpy, scikit-learn |
| Server | Uvicorn |

### Frontend
| Layer | Technology |
|---|---|
| Framework | React 18 + TypeScript |
| Build tool | Vite |
| HTTP client | Axios |
| Animations | Framer Motion |
| Charts | Recharts (SHAP bar chart) |
| Styling | CSS Modules |
| Icons | Lucide React |

---

## 3. Dataset

**EMSCAD — Employment Scam Aegean Dataset**
- Source: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- File: `fake_job_postings.csv`
- Size: 17,880 rows, ~800 fraudulent (≈4.5% positive rate)
- Place at: `backend/data/fake_job_postings.csv`

**Key columns used:**
| Column | Used in |
|---|---|
| `title` | Text branch (concat input) |
| `company_profile` | Text branch |
| `description` | Text branch (primary signal) |
| `requirements` | Text branch |
| `benefits` | Text branch |
| `employment_type` | Structural branch |
| `required_experience` | Structural branch |
| `required_education` | Structural branch |
| `salary_range` | Structural branch |
| `has_company_logo` | Structural branch |
| `has_questions` | Structural branch |
| `telecommuting` | Structural branch |
| `location` | Structural branch (vagueness score derived) |
| `fraudulent` | Target label (0 = real, 1 = fake) |

---

## 4. Engineered Features (Structural Branch)

These are computed at preprocessing time and at inference time:

| Feature name | Type | Logic |
|---|---|---|
| `has_salary` | int (0/1) | `salary_range` is not null/empty |
| `has_company_logo` | int (0/1) | Direct field |
| `has_questions` | int (0/1) | Direct field |
| `telecommuting` | int (0/1) | Direct field |
| `employment_type_missing` | int (0/1) | `employment_type` is null |
| `experience_missing` | int (0/1) | `required_experience` is null |
| `education_missing` | int (0/1) | `required_education` is null |
| `description_length` | int | `len(description.split())` |
| `requirements_length` | int | `len(requirements.split())` |
| `url_count` | int | count of `http` occurrences in full text |
| `all_caps_ratio` | float | uppercase words / total words in description |
| `location_vagueness` | int (0/1) | location is null OR contains "anywhere"/"remote" only |
| `benefit_length` | int | `len(benefits.split())` |
| `company_profile_missing` | int (0/1) | `company_profile` is null |

---

## 5. Model Architecture

```
Raw job posting
      │
      ▼
  Text cleaning (spaCy)
      │
  ┌───┴────────────────────┐
  │                        │
  ▼                        ▼
DistilBERT              XGBoost
(text branch)       (structural branch)
[CLS] → prob_text   features → prob_struct
  │                        │
  └───────────┬────────────┘
              ▼
    Late fusion (weighted avg)
    fraud_score = 0.55 * prob_text + 0.45 * prob_struct
              │
              ▼
    SHAP values (structural branch)
    → top 5 feature contributions
              │
              ▼
    Response: { score, label, confidence, shap_signals }
```

**Training strategy:**
- Class imbalance: `scale_pos_weight = len(real) / len(fraud)` in XGBoost; Focal Loss for DistilBERT
- DistilBERT input: `[CLS] title [SEP] description [SEP] requirements [SEP]` (truncated to 512 tokens)
- XGBoost: 500 estimators, max_depth=6, learning_rate=0.05, eval_metric=aucpr
- Fusion weights tuned on validation set

---

## 6. API Endpoints

### `POST /api/analyze`
**Request body:**
```json
{
  "title": "Software Engineer",
  "description": "We are hiring...",
  "requirements": "5 years experience...",
  "benefits": "Health insurance...",
  "company_profile": "We are a startup...",
  "location": "New York, NY",
  "salary_range": "80000-100000",
  "employment_type": "Full-time",
  "required_experience": "Mid-Senior level",
  "required_education": "Bachelor's Degree",
  "has_company_logo": true,
  "has_questions": false,
  "telecommuting": false
}
```
**Response:**
```json
{
  "fraud_score": 0.847,
  "label": "SUSPICIOUS",
  "confidence": "HIGH",
  "shap_signals": [
    { "feature": "has_salary", "value": 0, "impact": 0.312, "direction": "fraud" },
    { "feature": "company_profile_missing", "value": 1, "impact": 0.198, "direction": "fraud" },
    { "feature": "url_count", "value": 4, "impact": 0.154, "direction": "fraud" },
    { "feature": "has_company_logo", "value": 0, "impact": 0.143, "direction": "fraud" },
    { "feature": "description_length", "value": 45, "impact": 0.091, "direction": "fraud" }
  ],
  "text_score": 0.91,
  "struct_score": 0.76
}
```

### `GET /api/health`
Returns `{ "status": "ok", "models_loaded": true }`

### `POST /api/train`
Triggers training pipeline (dev only, protected by env flag)

---

## 7. Folder Structure

```
jobguard/
├── backend/
│   ├── data/
│   │   └── fake_job_postings.csv          ← download from Kaggle
│   ├── models/
│   │   ├── xgb_model.joblib               ← saved after training
│   │   ├── xgb_feature_names.joblib       ← feature column order
│   │   └── distilbert_onnx/               ← ONNX export folder
│   │       ├── model.onnx
│   │       └── tokenizer/
│   ├── src/
│   │   ├── train.py                       ← full training pipeline
│   │   ├── features.py                    ← feature engineering
│   │   ├── predict.py                     ← inference logic
│   │   ├── explainer.py                   ← SHAP integration
│   │   └── schemas.py                     ← Pydantic models
│   ├── main.py                            ← FastAPI app entry point
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── JobForm/
│   │   │   │   ├── JobForm.tsx
│   │   │   │   └── JobForm.module.css
│   │   │   ├── ResultPanel/
│   │   │   │   ├── ResultPanel.tsx
│   │   │   │   └── ResultPanel.module.css
│   │   │   ├── ShapChart/
│   │   │   │   ├── ShapChart.tsx
│   │   │   │   └── ShapChart.module.css
│   │   │   ├── ScoreGauge/
│   │   │   │   ├── ScoreGauge.tsx
│   │   │   │   └── ScoreGauge.module.css
│   │   │   └── Header/
│   │   │       ├── Header.tsx
│   │   │       └── Header.module.css
│   │   ├── types/
│   │   │   └── api.ts
│   │   ├── api/
│   │   │   └── analyze.ts
│   │   ├── App.tsx
│   │   ├── App.module.css
│   │   └── main.tsx
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── package.json
│
├── AppFlow.md                             ← this file
└── README.md
```

---

## 8. UI Design System

### Color Palette (provided)
```css
/* CSS HEX */
--molten-lava:    #780000ff;   /* deep danger red — fraud alerts, critical badges */
--brick-red:      #c1121fff;   /* primary accent — buttons, highlights, score bar */
--papaya-whip:    #fdf0d5ff;   /* warm off-white — page background */
--deep-space-blue:  #003049ff;   /* dark navy — header bg, text on light bg */
--steel-blue:     #669bbcff;   /* medium blue — safe indicators, secondary elements */
```

### Typography
- Display / headings: `'IBM Plex Serif'` (loaded via Google Fonts) — gives a journalistic, investigative tone fitting the fraud-detection subject
- Body / UI: `'IBM Plex Sans'` — clean, technical, pairs perfectly with the serif
- Monospace (scores, feature names): `'IBM Plex Mono'`

### Design Signature
The **score gauge** is a half-circle arc rendered in SVG — fills from steel-blue (safe) through brick-red to molten-lava (fraud) as the score rises. No generic progress bars. The arc is the centerpiece of the result panel and the thing the eye goes to first.

### Key UI States
1. **Empty** — form ready, hero copy visible, no result panel
2. **Loading** — skeleton pulse on result panel, spinner in submit button
3. **Safe** (score < 0.35) — steel-blue accent, "Likely Legitimate" badge
4. **Caution** (0.35–0.65) — amber accent, "Uncertain" badge
5. **Suspicious** (> 0.65) — brick-red/molten-lava accent, "Suspicious" badge

---

## 9. Environment Variables

### Backend `.env`
```
ALLOW_TRAIN=true          # set false in production
MODEL_DIR=./models
DATA_PATH=./data/fake_job_postings.csv
DISTILBERT_MODEL=distilbert-base-uncased
FUSION_TEXT_WEIGHT=0.55
FUSION_STRUCT_WEIGHT=0.45
```

---

## 10. Evaluation Targets

| Metric | Target |
|---|---|
| F1 (fraud class) | ≥ 0.90 |
| Precision (fraud) | ≥ 0.88 |
| Recall (fraud) | ≥ 0.92 |
| ROC-AUC | ≥ 0.97 |
| PR-AUC | ≥ 0.85 |

Primary metric during training: **PR-AUC** (better than ROC-AUC under heavy class imbalance).

---