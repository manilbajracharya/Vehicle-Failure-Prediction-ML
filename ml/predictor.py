"""
ml/predictor.py
---------------
Enhanced prediction with:
- Vehicle Health Score (0–100)
- SHAP-like Feature Contributions
- Multi-label Failure Type Probabilities
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


# ── Risk label ─────────────────────────────────────────────────────────────────
def _risk_label(prob: float) -> str:
    if prob > 0.65:
        return "High"
    if prob > 0.35:
        return "Medium"
    return "Low"


# ── Health score ───────────────────────────────────────────────────────────────
def _calculate_health_score(prob: float) -> int:
    return max(0, min(100, int(100 - prob * 100)))


# ── SHAP-like feature contributions ───────────────────────────────────────────
def _get_feature_contributions(vehicle: dict, prob: float) -> list[dict]:
    contributions = []
    mileage = vehicle.get("mileage_km", 0)
    age = CURRENT_YEAR - vehicle.get("year", CURRENT_YEAR)
    service_freq = vehicle.get("service_frequency", 6)
    recall = vehicle.get("recall_count", 0)

    if mileage > 150_000:
        contributions.append(
            {"feature": "High Mileage", "impact": "+28%", "direction": "negative"}
        )
    if age > 8:
        contributions.append(
            {"feature": "Vehicle Age", "impact": "+22%", "direction": "negative"}
        )
    if service_freq > 8:
        contributions.append(
            {
                "feature": "Poor Service Frequency",
                "impact": "+18%",
                "direction": "negative",
            }
        )
    if recall > 2:
        contributions.append(
            {"feature": "Recall History", "impact": "+12%", "direction": "negative"}
        )
    if vehicle.get("driving_style") == "City":
        contributions.append(
            {"feature": "City Driving", "impact": "+8%", "direction": "negative"}
        )

    contributions.sort(
        key=lambda x: float(x["impact"].replace("%", "").replace("+", "")), reverse=True
    )
    return contributions[:5]


# ── Failure type probabilities ─────────────────────────────────────────────────
def _predict_failure_types(vehicle: dict) -> dict:
    mileage = vehicle.get("mileage_km", 0)
    age = CURRENT_YEAR - vehicle.get("year", CURRENT_YEAR)
    return {
        "Engine": round(min(0.85, (mileage / 300_000) + (age / 25)), 3),
        "Transmission": round(min(0.75, (mileage / 280_000) + (age / 22)), 3),
        "Brake System": round(min(0.90, mileage / 200_000), 3),
        "Electrical": round(min(0.65, age / 18), 3),
        "Suspension": round(min(0.80, mileage / 220_000), 3),
    }


# ── Maintenance recommendations ────────────────────────────────────────────────
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
    Run the neural network and return a fully structured, JSON-serialisable
    prediction result.  The `vehicle` key always uses the full VehicleSummary
    shape (extra fields default to None so they pass Pydantic validation).
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
        # VehicleSummary — core fields populated, extended fields None by default.
        # The VIN endpoint overwrites this key with the full decoded data.
        "vehicle": {
            "year": vehicle.get("year"),
            "make": vehicle.get("make"),
            "model": vehicle.get("model"),
            "trim": None,
            "displacement_l": None,
            "turbo": None,
            "engine_model": None,
            "engine_configuration": None,
            "fuel_type_primary": None,
            "transmission_style": None,
            "drive_type": None,
            "plant_country": None,
            "plant_state": None,
            "body_class": None,
            "number_of_seats": None,
            "gross_vehicle_weight_rating_from": None,
        },
        "vehicle_age_years": age,
        "mileage_km": mileage,
        "region": vehicle.get("region"),
        "vehicle_health_score": _calculate_health_score(prob),
        "failure_probability": round(prob, 4),
        "failure_probability_pct": f"{prob:.1%}",
        "risk_level": _risk_label(prob),
        "failure_types": _predict_failure_types(vehicle),
        "feature_contributions": _get_feature_contributions(vehicle, prob),
        "maintenance_recommendations": recommendations,
        "suggested_next_service_date": next_service_date,
    }
