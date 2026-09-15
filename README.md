# AquaSense — Water Quality Risk Predictor

**PS-3Y-06 · Tech Utsav 2026 · SDG 6 — Clean Water & Sanitation**

Classifies water samples into risk categories (**Safe / Moderate Risk / High Risk / Unsafe**)
from 9 physicochemical readings, cross-checked against a separately trained ML potability
model and WHO/EPA guideline ranges. Includes a CSV batch-upload dashboard (risk distribution,
parameter breakdown, WQI trend).

## Architecture

```
data/water_potability.csv   Kaggle "Water Potability" dataset (3,276 rows, 9 features)
train.py                    Trains the models, saves model_core.joblib + model_tabpfn.joblib
wqi.py                      Deterministic Water Quality Index formula + risk-category bands
who_ranges.py                WHO/EPA guideline ranges, for the rule-based cross-check
risk.py                     Potability-confidence -> risk-band helper
api.py                      FastAPI backend serving predictions (loads the trained models)
frontend/                   Next.js + React dashboard (calls api.py)
```

**Two models, on purpose:**
- `model_core.joblib` — a `HistGradientBoostingClassifier` fallback + the WQI regressor. Pure
  scikit-learn, small, always loadable.
- `model_tabpfn.joblib` — TabPFN, the current state-of-the-art model for small tabular data
  (Hollmann et al., *Nature* 2025), used as the primary model when available.

`api.py` loads `model_core.joblib` unconditionally, then tries `model_tabpfn.joblib` and
**silently falls back to the HistGB model** if TabPFN's weights/package aren't available
(e.g. a memory-constrained free-tier host). Check which one is live via `/metrics` →
`active_model`.

## Results

| Task | Metric | Why the gap |
|---|---|---|
| Potability (binary, real label) | 68.8% accuracy · 0.723 ROC-AUC | Raw features have ~0 correlation with this label (checked directly) — this is close to the ceiling; see Methodology on the site. |
| WQI (regression, derived from the same features) | R² 0.985 · 93.3% category accuracy | WQI is a formula *of* these features, so a model can legitimately reconstruct it. |

Five modeling approaches were benchmarked honestly (Random Forest, tuned HistGB, SMOTE+
stacking ensemble, TabPFN) rather than chasing an inflated number — full comparison table is
on the site's Methodology section.

## Setup

Requires Python 3.10+ and Node.js 18+.

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Train the models (first run needs a free token from https://ux.priorlabs.ai to
#    download TabPFN's weights; cached locally afterward, no token needed again)
TABPFN_TOKEN=your_token_here python train.py
#    (Windows PowerShell: $env:TABPFN_TOKEN="your_token_here"; python train.py)

# 3. Start the backend
uvicorn api:app --port 8000

# 4. In a separate terminal, start the frontend
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The frontend calls the backend at `http://localhost:8000`
(hardcoded in `frontend/app/page.tsx` — update the `API` constant if you deploy the backend
elsewhere).

To try the dashboard's CSV upload without your own data, use `frontend/public/sample.csv`
(60 rows sampled from the training set).

## Deploying (e.g. Render free tier)

TabPFN's ~200MB weights + PyTorch runtime may not fit in a 512MB free-tier instance. That's
fine — the API falls back to the HistGB model automatically (see Architecture above) rather
than crashing. To force the lightweight path everywhere, just don't run `train.py`'s TabPFN
step / don't ship `model_tabpfn.joblib` with the deploy.

Deploy `api.py` as a Python web service (`uvicorn api:app --host 0.0.0.0 --port $PORT`) and
`frontend/` as a Node web service (`npm run build && npm start`), then update the `API`
constant in `frontend/app/page.tsx` to the backend's deployed URL.
