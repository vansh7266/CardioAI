"""
main.py  —  CardioAI FastAPI Backend
──────────────────────────────────────
Start the server:
    uvicorn main:app --reload --port 8000

Then open:  http://localhost:8000
Swagger UI: http://localhost:8000/docs      ← auto-generated API docs (free with FastAPI!)
ReDoc UI:   http://localhost:8000/redoc

Install dependencies first:
    pip install fastapi uvicorn[standard] python-multipart scikit-learn pandas numpy
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Auto-build model artefacts if any are missing
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_FILES = [
    "knn_model.pkl",
    "scaler.pkl",
    "le_sex.pkl",
    "le_angina.pkl",
    "feature_columns.pkl",
    "categorical_cols.pkl",
]

if not all(os.path.exists(f) for f in REQUIRED_FILES):
    print("[STARTUP] Model artefacts missing — running model_pipeline.py ...")
    try:
        from model_pipeline import build_and_save_pipeline
        if not build_and_save_pipeline():
            raise RuntimeError("build_and_save_pipeline() failed. "
                               "Ensure heart.csv is in the same folder.")
    except Exception as e:
        print(f"[STARTUP ERROR] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Load all saved model artefacts
# ─────────────────────────────────────────────────────────────────────────────
def _load(filename: str):
    """Load a pickle file and return the object, or None on failure."""
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load '{filename}': {e}")
        return None

model           = _load("knn_model.pkl")
scaler          = _load("scaler.pkl")
le_sex          = _load("le_sex.pkl")
le_angina       = _load("le_angina.pkl")
feature_columns = _load("feature_columns.pkl")

if model:
    print("[OK] All model artefacts loaded successfully.")
else:
    print("[WARNING] Model not loaded — /predict_risk will return 503.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Pydantic input schema (auto-validates every incoming request)
# ─────────────────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    """
    Defines the shape and rules for the JSON body sent by the frontend.
    FastAPI automatically rejects requests that don't match — no manual
    if/else validation needed.
    """

    Age: int              = Field(..., ge=1,   le=120,  description="Age in years")
    Sex: str              = Field(...,          description="'M' or 'F'")
    ChestPainType: str    = Field(...,          description="ATA | NAP | ASY | TA")
    RestingBP: float      = Field(..., ge=80,  le=220,  description="mm Hg")
    Cholesterol: float    = Field(..., ge=0,   le=600,  description="mg/dl")
    FastingBS: int        = Field(..., ge=0,   le=1,    description="0 or 1")
    RestingECG: str       = Field(...,          description="Normal | ST | LVH")
    MaxHR: int            = Field(..., ge=60,  le=220,  description="BPM")
    ExerciseAngina: str   = Field(...,          description="'Y' or 'N'")
    Oldpeak: float        = Field(..., ge=0.0, le=7.0,  description="ST depression")
    ST_Slope: str         = Field(...,          description="Up | Flat | Down")

    # Field-level validators — checked automatically before the endpoint runs
    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        if v not in ("M", "F"):
            raise ValueError("Sex must be 'M' or 'F'")
        return v

    @field_validator("ChestPainType")
    @classmethod
    def validate_cpt(cls, v: str) -> str:
        if v not in ("ATA", "NAP", "ASY", "TA"):
            raise ValueError("ChestPainType must be ATA | NAP | ASY | TA")
        return v

    @field_validator("RestingECG")
    @classmethod
    def validate_ecg(cls, v: str) -> str:
        if v not in ("Normal", "ST", "LVH"):
            raise ValueError("RestingECG must be Normal | ST | LVH")
        return v

    @field_validator("ExerciseAngina")
    @classmethod
    def validate_angina(cls, v: str) -> str:
        if v not in ("Y", "N"):
            raise ValueError("ExerciseAngina must be 'Y' or 'N'")
        return v

    @field_validator("ST_Slope")
    @classmethod
    def validate_slope(cls, v: str) -> str:
        if v not in ("Up", "Flat", "Down"):
            raise ValueError("ST_Slope must be Up | Flat | Down")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Output schema
# ─────────────────────────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    prediction:  int    # 0 = Low Risk, 1 = High Risk
    probability: str    # e.g. "78.5"  (percentage as string, kept for frontend compat)
    label:       str    # "Low Risk" or "High Risk"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Create FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "CardioAI — Heart Risk Predictor",
    description = "KNN-based cardiovascular disease risk assessment API.",
    version     = "2.0.0",
)

# CORS — allow any origin during development so the frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten to your domain in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Serve the frontend HTML file
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    """Serve index.html at the root URL so one server handles everything."""
    html_path = Path("index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(html_path))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Prediction endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict_risk", response_model=PredictionOutput)
async def predict_risk(patient: PatientInput) -> PredictionOutput:
    """
    POST /predict_risk
    ──────────────────
    Accepts a JSON body matching PatientInput, runs the KNN model,
    returns prediction (0 / 1) + confidence percentage.

    FastAPI validates the input automatically using the Pydantic schema above.
    If any field is wrong or missing, FastAPI returns a 422 error before this
    function even runs.
    """

    # Guard — model not loaded
    if model is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Check server logs."
        )

    try:
        # ── 7a. Label-encode binary categoricals ──────────────────────────────
        sex_enc    = int(le_sex.transform([patient.Sex])[0])
        angina_enc = int(le_angina.transform([patient.ExerciseAngina])[0])

        # ── 7b. Build a single-row DataFrame ──────────────────────────────────
        row = {
            "Age"           : [patient.Age],
            "Sex"           : [sex_enc],
            "ChestPainType" : [patient.ChestPainType],
            "RestingBP"     : [patient.RestingBP],
            "Cholesterol"   : [patient.Cholesterol],
            "FastingBS"     : [patient.FastingBS],
            "RestingECG"    : [patient.RestingECG],
            "MaxHR"         : [patient.MaxHR],
            "ExerciseAngina": [angina_enc],
            "Oldpeak"       : [patient.Oldpeak],
            "ST_Slope"      : [patient.ST_Slope],
        }
        input_df = pd.DataFrame(row)

        # ── 7c. One-hot encode multi-class categoricals ───────────────────────
        input_df = pd.get_dummies(
            input_df,
            columns=["ChestPainType", "RestingECG", "ST_Slope"],
            drop_first=True,
        )

        # ── 7d. Align columns to training feature set ─────────────────────────
        #   Some OHE columns may not appear in a single-row input (e.g. if the
        #   user chose "ASY" but the training data has columns for ATA, NAP, TA).
        #   We create a zero-filled frame with ALL training columns, then fill in
        #   the ones that are present.
        final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
        for col in input_df.columns:
            if col in final_input.columns:
                final_input[col] = input_df[col].values

        # ── 7e. Scale using the same scaler fitted during training ─────────────
        scaled = scaler.transform(final_input)

        # ── 7f. Predict class and probability ─────────────────────────────────
        prediction  = int(model.predict(scaled)[0])          # 0 or 1
        proba_array = model.predict_proba(scaled)[0]          # [P(0), P(1)]
        confidence  = float(proba_array[prediction]) * 100   # probability of the predicted class

        return PredictionOutput(
            prediction  = prediction,
            probability = f"{confidence:.1f}",
            label       = "High Risk" if prediction == 1 else "Low Risk",
        )

    except Exception as e:
        # Any unexpected error → 400 with details
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Health check endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Quick check — useful for deployment platforms to verify the app is alive."""
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Entry-point (run with: python main.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port  = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FASTAPI_DEBUG", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
