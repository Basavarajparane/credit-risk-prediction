from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

art = joblib.load("model_artifacts.joblib")
model = art["model"]
scaler = art["scaler"]
num_cols, cat_cols = art["num_cols"], art["cat_cols"]
num_medians, cat_modes = art["num_medians"], art["cat_modes"]
model_columns = art["model_columns"]

THRESHOLD = 0.5

app = FastAPI(title="Credit Risk API")


class Applicant(BaseModel):
    features: dict[str, Any]


def prepare(features: dict) -> pd.DataFrame:
    # one row with all raw columns, in the training order
    row = pd.DataFrame([features]).reindex(columns=num_cols + cat_cols)

    # fill missing values with TRAINING medians and modes
    row[num_cols] = row[num_cols].apply(pd.to_numeric, errors="coerce")
    row[num_cols] = row[num_cols].fillna(num_medians).astype(float)
    row[cat_cols] = row[cat_cols].fillna(cat_modes)

    # encode (no drop_first for one row), then match training columns
    row = pd.get_dummies(row, columns=cat_cols, dtype=int)
    row = row.reindex(columns=model_columns, fill_value=0)

    # scale
    return pd.DataFrame(scaler.transform(row), columns=model_columns)


@app.get("/")
def home():
    return {"message": "Credit Risk API is running. Go to /docs to try it."}


@app.post("/predict")
def predict(applicant: Applicant):
    X_new = prepare(applicant.features)
    prob = float(model.predict_proba(X_new)[0, 1])
    return {
        "default_probability": round(prob, 4),
        "risk": "high" if prob >= THRESHOLD else "low",
    }