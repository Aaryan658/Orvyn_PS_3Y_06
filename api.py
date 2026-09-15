"""FastAPI wrapper around the trained models, for the Next.js frontend to call.

Loads model_core.joblib (fallback HistGB model + WQI regressor -- pure sklearn, always
loadable) unconditionally, then tries model_tabpfn.joblib and falls back to the HistGB model
if TabPFN's weights/package aren't available in this environment (e.g. Render free tier).
"""
import io

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from risk import risk_band
from who_ranges import who_flags
from wqi import compute_wqi, wqi_category

core = joblib.load("model_core.joblib")
columns, feature_importance, wqi_model, metrics = (
    core["columns"], core["feature_importance"], core["wqi_model"], core["metrics"],
)

try:
    tabpfn_bundle = joblib.load("model_tabpfn.joblib")
    model, threshold, active_model = tabpfn_bundle["model"], tabpfn_bundle["threshold"], "TabPFN"
except Exception as e:
    model, threshold, active_model = core["fallback_model"], core["fallback_threshold"], "HistGradientBoosting (fallback)"
    print(f"TabPFN unavailable ({e}); serving predictions with the fallback model instead.")

print(f"Serving predictions with: {active_model}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Sample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


@app.get("/metrics")
def get_metrics():
    return {**metrics, "feature_importance": feature_importance, "active_model": active_model}


@app.post("/predict")
def predict(sample: Sample):
    sample_dict = sample.model_dump()
    values = [sample_dict[c] for c in columns]
    proba = float(model.predict_proba([values])[0, 1])
    label = "Potable" if proba >= threshold else "Not Potable"
    true_wqi = compute_wqi(sample_dict)
    ml_wqi = float(wqi_model.predict([values])[0])
    return {
        "prediction": label,
        "confidence": proba,
        "risk_band": risk_band(proba),
        "who_flags": who_flags(sample_dict),
        "wqi_computed": true_wqi,
        "wqi_computed_category": wqi_category(true_wqi),
        "wqi_predicted": ml_wqi,
        "wqi_predicted_category": wqi_category(ml_wqi),
        "active_model": active_model,
    }


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """Dashboard data source: upload a CSV with the 9 parameter columns, get per-row risk
    categories plus aggregate stats for the distribution/trend/parameter charts."""
    df = pd.read_csv(io.BytesIO(await file.read()))
    df = df[columns].fillna(df[columns].median(numeric_only=True))

    proba_all = model.predict_proba(df.values)[:, 1]
    wqi_all = df.apply(lambda r: compute_wqi(r.to_dict()), axis=1)

    rows = [
        {
            "row": i,
            "wqi": float(wqi_all.iloc[i]),
            "risk_category": wqi_category(float(wqi_all.iloc[i])),
            "potability": "Potable" if proba_all[i] >= threshold else "Not Potable",
            "confidence": float(proba_all[i]),
        }
        for i in range(len(df))
    ]

    category_counts: dict[str, int] = {}
    for r in rows:
        category_counts[r["risk_category"]] = category_counts.get(r["risk_category"], 0) + 1

    parameter_stats = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
        for c in columns
    }

    return {"rows": rows, "total": len(rows), "category_counts": category_counts, "parameter_stats": parameter_stats}
