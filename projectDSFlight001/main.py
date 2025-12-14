"""
main_deploy.py - FastAPI app for Flight Delay Prediction (REAL XGBoost inference)

Expected project layout (recommended):
.
├─ main_deploy.py
├─ templates/
│   └─ index.html
├─ static/
│   ├─ style.css
│   └─ riyadhair.jpg
└─ models/
    ├─ xgb_model.json          <- your TRAINED model (required)
    └─ best_xgb_params.json    <- tuned params (optional; used for /health info)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import json
import os

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# -----------------------------
# App + paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"

INDEX_CANDIDATES = [
    TEMPLATES_DIR / "index.html",
    BASE_DIR / "index.html",
]

MODEL_CANDIDATES = [
    MODELS_DIR / "xgb_model.json",
    BASE_DIR / "xgb_model.json",
]

PARAMS_CANDIDATES = [
    MODELS_DIR / "best_xgb_params.json",
    BASE_DIR / "best_xgb_params.json",
]

app = FastAPI(
    title="Flight Delay Prediction",
    description="Predict flight delays using a trained XGBoost model",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory if it exists. (Your CSS references /static/riyadhair.jpg)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------
# Schemas
# -----------------------------
class PredictionInput(BaseModel):
    Airline: str
    DepTime: str  # YYYY-MM-DD
    DestStateName: str


class PredictionResponse(BaseModel):
    status: str
    airline: str
    departure_date: str
    destination: str
    delay: float
    is_delayed_over_15: bool
    confidence: str
    recommendation: str


# -----------------------------
# Encodings + simple feature engineering
# NOTE: These MUST match what you used during training.
# -----------------------------
AIRLINE_ENCODING = {
    "Endeavor Air Inc.": 1,
    "Mesa Airlines Inc.": 2,
    "Delta Air Lines Inc.": 3,
    "Alaska Airlines Inc.": 4,
    "SkyWest Airlines Inc.": 5,
    "Southwest Airlines Co.": 6,
    "ExpressJet Airlines Inc.": 7,
    "American Airlines Inc.": 8,
    "United Air Lines Inc.": 9,
    "JetBlue Airways": 10,
    "Comair Inc.": 11,
    "Cape Air": 12,
    "Envoy Air": 13,
    "Republic Airlines": 14,
    "Frontier Airlines Inc.": 15,
    "Spirit Air Lines": 16,
    "Empire Airlines Inc.": 17,
    "GoJet Airlines, LLC d/b/a United Express": 18,
    "Allegiant Air": 19,
    "Horizon Air": 20,
    "Virgin America": 21,
    "Air Wisconsin Airlines Corp": 22,
    "Trans States Airlines": 23,
    "Hawaiian Airlines Inc.": 24,
    "Compass Airlines": 25,
    "Commutair Aka Champlain Enterprises, Inc.": 26,
    "Peninsula Airways Inc.": 27,
}

STATE_ENCODING = {
    "California": 1,
    "Texas": 2,
    "New York": 3,
    "Florida": 4,
    "Illinois": 5,
    "Ohio": 6,
    "Georgia": 7,
    "North Carolina": 8,
    "Michigan": 9,
    "Pennsylvania": 10,
    "Arizona": 11,
    "Colorado": 12,
    "Nevada": 13,
    "Washington": 14,
    "Massachusetts": 15,
    "Virginia": 16,
    "New Jersey": 17,
    "Minnesota": 18,
    "Oregon": 19,
    "Indiana": 20,
    "Tennessee": 21,
    "Missouri": 22,
    "Maryland": 23,
    "Wisconsin": 24,
    "Utah": 25,
    "Connecticut": 26,
    "South Carolina": 27,
    "Alabama": 28,
    "Louisiana": 29,
    "Kentucky": 30,
    "Oklahoma": 31,
    "Arkansas": 32,
    "Nebraska": 33,
    "New Mexico": 34,
    "Iowa": 35,
    "North Dakota": 36,
    "South Dakota": 37,
    "Puerto Rico": 38,
    "Wyoming": 39,
    "West Virginia": 40,
    "Vermont": 41,
    "U.S. Virgin Islands": 42,
}

STATE_DISTANCES = {
    "California": 2500,
    "Texas": 1500,
    "New York": 800,
    "Florida": 1200,
    "Illinois": 1000,
    "Ohio": 600,
    "Georgia": 700,
    "North Carolina": 500,
    "Michigan": 650,
    "Pennsylvania": 400,
    "Arizona": 2000,
    "Colorado": 1800,
    "Nevada": 2300,
    "Washington": 2700,
    "Massachusetts": 300,
    "Virginia": 200,
    "New Jersey": 150,
    "Minnesota": 1200,
    "Oregon": 2600,
    "Indiana": 700,
    "Tennessee": 800,
    "Missouri": 1000,
    "Maryland": 100,
    "Wisconsin": 900,
    "Utah": 2100,
    "Connecticut": 250,
    "South Carolina": 600,
    "Alabama": 900,
    "Louisiana": 1300,
    "Kentucky": 600,
    "Oklahoma": 1400,
    "Arkansas": 1100,
    "Nebraska": 1300,
    "New Mexico": 1800,
    "Iowa": 1000,
    "North Dakota": 1500,
    "South Dakota": 1400,
    "Puerto Rico": 2300,
    "Wyoming": 1900,
    "West Virginia": 400,
    "Vermont": 350,
    "U.S. Virgin Islands": 2400,
}

AIRLINE_BASELINES = {
    "Endeavor Air Inc.": 12.5,
    "Mesa Airlines Inc.": 14.2,
    "Delta Air Lines Inc.": 8.2,
    "Alaska Airlines Inc.": 7.5,
    "SkyWest Airlines Inc.": 9.8,
    "Southwest Airlines Co.": 14.7,
    "ExpressJet Airlines Inc.": 16.3,
    "American Airlines Inc.": 11.3,
    "United Air Lines Inc.": 10.8,
    "JetBlue Airways": 13.2,
    "Comair Inc.": 15.1,
    "Cape Air": 8.9,
    "Envoy Air": 12.8,
    "Republic Airlines": 11.5,
    "Frontier Airlines Inc.": 16.3,
    "Spirit Air Lines": 18.5,
    "Empire Airlines Inc.": 9.5,
    "GoJet Airlines, LLC d/b/a United Express": 13.8,
    "Allegiant Air": 15.7,
    "Horizon Air": 8.3,
    "Virgin America": 9.1,
    "Air Wisconsin Airlines Corp": 12.1,
    "Trans States Airlines": 14.5,
    "Hawaiian Airlines Inc.": 6.8,
    "Compass Airlines": 10.5,
    "Commutair Aka Champlain Enterprises, Inc.": 11.9,
    "Peninsula Airways Inc.": 7.2,
}

FEATURE_COLUMNS = [
    "Airline_encoded",
    "DestState_encoded",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Distance",
    "Airline_Baseline",
    "Seasonal_Factor",
    "Weekend_Factor",
    "Holiday_Factor",
]


def extract_features(airline: str, dest_state: str, date_str: str) -> Dict[str, float]:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    month = date_obj.month
    day = date_obj.day
    weekday = date_obj.weekday() + 1  # Monday=1

    airline_encoded = AIRLINE_ENCODING.get(airline, 14)
    state_encoded = STATE_ENCODING.get(dest_state, 14)
    distance = STATE_DISTANCES.get(dest_state, 1000)

    if month in (12, 1, 2):
        seasonal_factor = 1.4
    elif month in (6, 7, 8):
        seasonal_factor = 1.3
    elif month in (3, 4, 5):
        seasonal_factor = 0.9
    else:
        seasonal_factor = 1.0

    weekend_factor = 1.1 if weekday in (6, 7) else 1.0
    holiday_factor = 1.2 if month in (11, 12) else 1.0

    return {
        "Airline_encoded": float(airline_encoded),
        "DestState_encoded": float(state_encoded),
        "Month": float(month),
        "DayofMonth": float(day),
        "DayOfWeek": float(weekday),
        "Distance": float(distance),
        "Airline_Baseline": float(AIRLINE_BASELINES.get(airline, 10.0)),
        "Seasonal_Factor": float(seasonal_factor),
        "Weekend_Factor": float(weekend_factor),
        "Holiday_Factor": float(holiday_factor),
    }


Predictor = Union[xgb.XGBRegressor, xgb.Booster]


def _first_existing(paths: list[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def load_trained_model() -> Tuple[Predictor, Path]:
    model_path = _first_existing(MODEL_CANDIDATES)
    if not model_path:
        raise FileNotFoundError(
            "Could not find xgb_model.json. Put it in ./models/xgb_model.json (recommended) "
            "or next to main_deploy.py."
        )

    try:
        reg = xgb.XGBRegressor()
        reg.load_model(str(model_path))
        return reg, model_path
    except Exception:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        return booster, model_path


def load_best_params() -> Dict[str, Any]:
    params_path = _first_existing(PARAMS_CANDIDATES)
    if not params_path:
        return {}
    try:
        return json.loads(params_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


MODEL, MODEL_PATH = load_trained_model()
BEST_PARAMS = load_best_params()


def predict_delay_minutes(airline: str, dest_state: str, date_str: str) -> float:
    feats = extract_features(airline, dest_state, date_str)
    X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    if isinstance(MODEL, xgb.Booster):
        dmat = xgb.DMatrix(X)
        pred = float(MODEL.predict(dmat)[0])
    else:
        pred = float(MODEL.predict(X)[0])

    return max(0.0, pred)


@app.get("/", response_class=HTMLResponse)
async def home(_: Request):
    index_path = _first_existing(INDEX_CANDIDATES)
    if not index_path:
        raise HTTPException(
            status_code=500,
            detail="index.html not found. Put it in ./templates/index.html (recommended) or next to main_deploy.py.",
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"), status_code=200)


@app.post("/predict_delay", response_model=PredictionResponse)
async def predict_delay(
    Airline: str = Form(...),
    DepTime: str = Form(...),
    DestStateName: str = Form(...),
):
    try:
        datetime.strptime(DepTime, "%Y-%m-%d")
    except ValueError:
        return JSONResponse(status_code=400, content={"status": "error", "error": "DepTime must be YYYY-MM-DD"})

    try:
        delay = round(predict_delay_minutes(Airline, DestStateName, DepTime), 1)
        is_delayed_over_15 = delay > 15.0

        if delay < 5:
            confidence = "very high"
        elif delay < 10:
            confidence = "high"
        elif delay < 20:
            confidence = "medium"
        else:
            confidence = "low"

        if is_delayed_over_15:
            if delay > 30:
                recommendation = "Arrive at least 3 hours early and check flight status"
            elif delay > 20:
                recommendation = "Arrive 2.5 hours early and monitor updates"
            else:
                recommendation = "Arrive 2 hours early"
        else:
            recommendation = "Arrive 1–1.5 hours before departure"

        return JSONResponse(
            content={
                "status": "success",
                "airline": Airline,
                "departure_date": DepTime,
                "destination": DestStateName,
                "delay": delay,
                "is_delayed_over_15": is_delayed_over_15,
                "confidence": confidence,
                "recommendation": recommendation,
            }
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "error": str(e), "message": "Prediction failed"})


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_api(input_data: PredictionInput):
    delay = round(predict_delay_minutes(input_data.Airline, input_data.DestStateName, input_data.DepTime), 1)
    return {
        "status": "success",
        "airline": input_data.Airline,
        "departure_date": input_data.DepTime,
        "destination": input_data.DestStateName,
        "delay": delay,
        "is_delayed_over_15": delay > 15.0,
        "confidence": "high" if delay < 10 else "medium",
        "recommendation": "Arrive early" if delay > 15.0 else "On schedule",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "model_type": "Booster" if isinstance(MODEL, xgb.Booster) else "XGBRegressor",
        "best_params_loaded": bool(BEST_PARAMS),
        "best_params": BEST_PARAMS,
        "feature_columns": FEATURE_COLUMNS,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
