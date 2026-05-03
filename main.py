"""
main.py
-------
Vehicle Failure Prediction API
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    VehicleRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse,
)
from ml.predictor import predict

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vehicle Failure Prediction API",
    description=(
        "Predicts the probability of a major vehicle repair within the next 12 months "
        "using a trained neural network. Supports both GET (query params) and POST (JSON body)."
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


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
def health_check():
    """Returns API and model status."""
    return {"status": "ok", "version": "1.0.0", "model": "VehicleFailureNN-v1"}


# ── POST /predict ──────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict vehicle failure (POST)",
    responses={422: {"model": ErrorResponse, "description": "Validation error"}},
)
def predict_post(vehicle: VehicleRequest):
    """
    Submit a single vehicle as a **JSON body** and receive a failure
    probability, risk level, and maintenance recommendations.

    **Example body:**
    ```json
    {
      "make": "Toyota",
      "model": "Camry",
      "year": 2018,
      "mileage_km": 168000,
      "engine_type": "Hybrid",
      "service_frequency": 6,
      "recall_count": 2,
      "region": "Ontario",
      "driving_style": "Mixed"
    }
    ```
    """
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
    summary="Predict vehicle failure (GET)",
    responses={422: {"model": ErrorResponse, "description": "Validation error"}},
)
def predict_get(
    make: str = Query(
        ..., description="Vehicle make", examples={"Toyota": {"value": "Toyota"}}
    ),
    model: str = Query(
        ..., description="Vehicle model", examples={"Camry": {"value": "Camry"}}
    ),
    year: int = Query(..., ge=1990, le=2026, description="Model year"),
    mileage_km: int = Query(
        ..., ge=0, le=999_999, description="Odometer reading in km"
    ),
    engine_type: str = Query(..., description="Gasoline | Hybrid | Electric | Diesel"),
    service_frequency: int = Query(
        ..., ge=1, le=24, description="Months between services"
    ),
    recall_count: int = Query(..., ge=0, le=20, description="Number of recalls"),
    region: str = Query(..., description="Ontario | Quebec | BC | Alberta | Prairies"),
    driving_style: str = Query(..., description="City | Highway | Mixed"),
):
    """
    Pass all vehicle fields as **query parameters**.

    Useful for quick browser testing or simple HTTP GET integrations.

    Example:
    ```
    GET /predict?make=Toyota&model=Camry&year=2018&mileage_km=168000
                &engine_type=Hybrid&service_frequency=6&recall_count=2
                &region=Ontario&driving_style=Mixed
    ```
    """
    # Reuse the Pydantic model for validation
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
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        result = predict(vehicle.model_dump())
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Root redirect to docs ──────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({"message": "Vehicle Failure Prediction API", "docs": "/docs"})
