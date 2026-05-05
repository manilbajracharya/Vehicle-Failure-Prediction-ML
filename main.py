"""
main.py
-------
Vehicle Failure Prediction API
"""

import os
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas import (
    VehicleRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
)
from ml.predictor import predict


# ── Pydantic Models ───────────────────────────────────────────────────────────
class VINRequest(BaseModel):
    vin: str


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vehicle Failure Prediction API",
    description=(
        "Predicts the probability of a major vehicle repair within the next 12 months "
        "using a trained neural network. Supports VIN lookup."
    ),
    version="1.1.0",
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
    return {"status": "ok", "version": "1.1.0", "model": "VehicleFailureNN-v1"}


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


# ── POST /predict/vin ─────────────────────────────────────────────────────────
@app.post(
    "/predict/vin",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict using VIN number",
)
async def predict_by_vin(request: VINRequest):
    """
    Decode VIN using carapi.app and return failure prediction.
    """
    vin = request.vin.strip().upper()

    if len(vin) != 17:
        raise HTTPException(
            status_code=422, detail="VIN must be exactly 17 characters long."
        )

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(f"https://carapi.app/api/vin/{vin}")

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode VIN. carapi.app returned status {response.status_code}",
            )

        car_data = response.json()

        # Prepare vehicle data for prediction
        vehicle_dict = {
            "make": car_data.get("make", "Unknown"),
            "model": car_data.get("model", "Unknown"),
            "year": car_data.get("year", 2020),
            "mileage_km": 85000,  # Default value - can be improved later
            "engine_type": car_data.get("engine_type", "Gasoline"),
            "service_frequency": 6,
            "recall_count": 0,
            "region": "Ontario",
            "driving_style": "Mixed",
        }

        # Get prediction
        result = predict(vehicle_dict)

        # Add VIN information to response
        result["vin"] = vin
        result["decoded_vehicle"] = {
            "make": car_data.get("make"),
            "model": car_data.get("model"),
            "year": car_data.get("year"),
            "trim": car_data.get("trim"),
            "engine": car_data.get("engine"),
        }

        return result

    except httpx.RequestError:
        raise HTTPException(
            status_code=503, detail="VIN decoding service is currently unavailable."
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return JSONResponse(
        {
            "message": "Vehicle Failure Prediction API",
            "docs": "/docs",
            "health": "/health",
            "vin_endpoint": "/predict/vin",
        }
    )


# ── Run locally ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
