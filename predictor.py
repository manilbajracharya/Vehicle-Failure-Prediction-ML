"""
ml/predictor.py
---------------
Pure prediction logic.  No FastAPI dependency — easy to unit-test.
"""

import pandas as pd
from datetime import datetime, timedelta
from ml.model import (
    nn_model,
    scaler,
    engineer_features,
    FEATURE_COLS,
    CURRENT_YEAR,
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _risk_label(prob: float) -> str:
    if prob > 0.65:
        return "High"
    if prob > 0.35:
        return "Medium"
    return "Low"


def _build_recommendations(
    age: int, mileage: int, recall_count: int, region: str, engine_type: str
) -> list[dict]:
    recs = []

    if mileage > 150_000:
        recs.append(
            {
                "task": "Full Synthetic Oil Change + Filters",
                "timeline": "Every 8,000 km or 3 months",
                "priority": "High",
            }
        )
    else:
        recs.append(
            {
                "task": "Oil Change + Filters",
                "timeline": "Every 10,000 km or 6 months",
                "priority": "Medium",
            }
        )

    if age >= 7 or mileage > 120_000:
        recs.append(
            {
                "task": "Brake Inspection & Pads",
                "timeline": "Within 3 months",
                "priority": "High",
            }
        )

    if age >= 9 or mileage > 160_000:
        recs.append(
            {
                "task": "Transmission Fluid + Filter",
                "timeline": "Within 6 months",
                "priority": "High",
            }
        )
        recs.append(
            {
                "task": "Spark Plugs & Ignition Coils",
                "timeline": "Within 8 months",
                "priority": "Medium",
            }
        )

    if recall_count >= 1:
        recs.append(
            {
                "task": "Check & Fix Open Recalls",
                "timeline": "Within 1 month",
                "priority": "Critical",
            }
        )

    if region in ["Ontario", "Quebec", "Prairies", "Alberta"]:
        recs.append(
            {
                "task": "Winter Tire & Battery Test",
                "timeline": "Before October 2026",
                "priority": "High",
            }
        )

    if engine_type == "Hybrid":
        recs.append(
            {
                "task": "Hybrid System Health Check",
                "timeline": "Within 6 months",
                "priority": "Medium",
            }
        )

    if mileage > 200_000:
        recs.append(
            {
                "task": "Suspension, Bushings & Steering",
                "timeline": "Within 4 months",
                "priority": "High",
            }
        )

    return recs[:8]


# ── Main prediction entry-point ────────────────────────────────────────────────
def predict(vehicle: dict) -> dict:
    """
    Parameters
    ----------
    vehicle : dict — raw vehicle fields from the API request

    Returns
    -------
    dict — fully structured JSON-serialisable prediction result
    """
    df = pd.DataFrame([vehicle])
    df = engineer_features(df)

    input_scaled = scaler.transform(df[FEATURE_COLS].values)
    prob = float(nn_model.predict(input_scaled, verbose=0)[0][0])

    age = int(df["vehicle_age"].values[0])
    mileage = vehicle["mileage_km"]

    recommendations = _build_recommendations(
        age=age,
        mileage=mileage,
        recall_count=vehicle.get("recall_count", 0),
        region=vehicle.get("region", ""),
        engine_type=vehicle.get("engine_type", ""),
    )

    next_service_date = (datetime.now() + timedelta(days=75)).strftime("%Y-%m-%d")

    return {
        "vehicle": {
            "year": vehicle["year"],
            "make": vehicle["make"],
            "model": vehicle["model"],
        },
        "vehicle_age_years": age,
        "mileage_km": mileage,
        "region": vehicle.get("region"),
        "failure_probability": round(prob, 4),
        "failure_probability_pct": f"{prob:.1%}",
        "risk_level": _risk_label(prob),
        "maintenance_recommendations": recommendations,
        "suggested_next_service_date": next_service_date,
    }
