
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

def _load(filename: str):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load '{filename}': {e}")
        return N
model = _load("knn_model.pkl")
scaler = _load("scaler.pkl")
le_sex = _load("le_sex.pkl")
le_angina = _load("le_angina.pkl")
feature_columns = _load("feature_columns.pkl")

if model:
    print("[OK] All model artefacts loaded successfully.")
else:
    print("[WARNING] Model not loaded — /predict_risk will return 503.")



class PatientInput(BaseModel):


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


class PredictionOutput(BaseModel):
    prediction:  int  
    probability: str  
    label:       str    



app = FastAPI(
    title       = "CardioAI — Heart Risk Predictor",
    description = "KNN-based cardiovascular disease risk assessment API.",
    version     = "2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)



@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    """Serve index.html at the root URL so one server handles everything."""
    html_path = Path("index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(html_path))


@app.post("/predict_risk", response_model=PredictionOutput)
async def predict_risk(patient: PatientInput) -> PredictionOutput:

    if model is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded. Check server logs."
        )

    try:
        sex_enc    = int(le_sex.transform([patient.Sex])[0])
        angina_enc = int(le_angina.transform([patient.ExerciseAngina])[0])

   
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

       
        input_df = pd.get_dummies(
            input_df,
            columns=["ChestPainType", "RestingECG", "ST_Slope"],
            drop_first=True,
        )


        final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
        for col in input_df.columns:
            if col in final_input.columns:
                final_input[col] = input_df[col].values

    
        scaled = scaler.transform(final_input)

     
        prediction  = int(model.predict(scaled)[0])   
        proba_array = model.predict_proba(scaled)[0]        
        confidence  = float(proba_array[prediction]) * 100   

        return PredictionOutput(
            prediction  = prediction,
            probability = f"{confidence:.1f}",
            label       = "High Risk" if prediction == 1 else "Low Risk",
        )

    except Exception as e:
       
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    """Quick check — useful for deployment platforms to verify the app is alive."""
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
    }


if __name__ == "__main__":
    import uvicorn
    port  = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("FASTAPI_DEBUG", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=debug)
