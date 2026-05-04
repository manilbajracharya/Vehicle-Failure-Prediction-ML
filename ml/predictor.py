"""
main.py - Vehicle Failure Prediction API (Fixed for Render)
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path (Important for Render)
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import with error handling for better debugging
try:
    from schemas import (
        VehicleRequest,
        PredictionResponse,
        HealthResponse,
        ErrorResponse,
    )
    from predictor import predict
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current directory: {BASE_DIR}")
    print(f"Python Path: {sys.path}")
    raise

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vehicle Failure Prediction API",
    description="Predicts the probability of major vehicle repair within next 12 months.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "version": "1.0.0", "model": "VehicleFailureNN-v1"}


# ── POST Predict ───────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_post(vehicle: VehicleRequest):
    try:
        result = predict(vehicle.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET Predict ────────────────────────────────────────────────────────────────
@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_get(
    make: str = Query(...),
    model: str = Query(...),
    year: int = Query(...),
    mileage_km: int = Query(...),
    engine_type: str = Query(...),
    service_frequency: int = Query(...),
    recall_count: int = Query(...),
    region: str = Query(...),
    driving_style: str = Query(...),
):
    try:
        vehicle = VehicleRequest(
            make=make,
            model=model,
            year=year,
            mileage_km=mileage_km,
            engine_type=engine_type,
            service_frequency=service_frequency,
            recall_count=recall_count,
            region=region,
            driving_style=driving_style,
        )
        result = predict(vehicle.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Vehicle Failure Prediction API", "docs": "/docs"}


# ── Local Run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
