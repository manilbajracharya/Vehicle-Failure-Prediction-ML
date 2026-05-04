"""
main.py
-------
Vehicle Failure Prediction API
"""

import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    VehicleRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
)
from predictor import predict

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vehicle Failure Prediction API",
    description=(
        "Predicts the probability of a major vehicle repair within the next 12 months "
        "using a trained neural network."
    ),
    version="1.0.0",
    contact={"name": "Vehicle ML Team"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns API and model status."""
    return {"status": "ok", "version": "1.0.0", "model": "VehicleFailureNN-v1"}


# ── POST /predict ──────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={422: {"model": ErrorResponse}},
)
def predict_post(vehicle: VehicleRequest):
    """Predict using JSON body."""
    try:
        result = predict(vehicle.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /predict ───────────────────────────────────────────────────────────────
@app.get(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={422: {"model": ErrorResponse}},
)
def predict_get(
    make: str = Query(..., description="Vehicle make"),
    model: str = Query(..., description="Vehicle model"),
    year: int = Query(..., ge=1990, le=2026),
    mileage_km: int = Query(..., ge=0, le=999999),
    engine_type: str = Query(..., description="Gasoline | Hybrid | Electric | Diesel"),
    service_frequency: int = Query(..., ge=1, le=24),
    recall_count: int = Query(..., ge=0, le=20),
    region: str = Query(..., description="Ontario | Quebec | BC | Alberta | Prairies"),
    driving_style: str = Query(..., description="City | Highway | Mixed"),
):
    """Predict using query parameters."""
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
    return JSONResponse(
        {
            "message": "Vehicle Failure Prediction API",
            "docs": "/docs",
            "health": "/health",
        }
    )


# ── Run locally ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))  # Important for Render
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Only works in development
    )
