"""
ml/predictor.py
Enhanced Version with:
- Vehicle Health Score (0-100)
- Feature Contributions (SHAP-like)
- Multi-label Failure Type Prediction
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


def _calculate_health_score(prob: float) -> int:
    """Convert failure probability to Vehicle Health Score (0-100)"""
    score = int(100 - (prob * 100))
    return max(0, min(100, score))


def _get_feature_contributions(vehicle: dict, prob: float) -> list[dict]:
    """Simulate SHAP-like feature contributions"""
    contributions = []

    mileage = vehicle.get("mileage_km", 0)
    age = CURRENT_YEAR - vehicle.get("year", CURRENT_YEAR)
    service_freq = vehicle.get("service_frequency", 6)
    recall = vehicle.get("recall_count", 0)

    if mileage > 150000:
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

    contributions.sort(key=lambda x: float(x["impact"].strip("%")), reverse=True)
    return contributions[:5]


def _predict_failure_types(vehicle: dict) -> dict:
    """Predict probability for each major failure type"""
    mileage = vehicle.get("mileage_km", 0)
    age = CURRENT_YEAR - vehicle.get("year", CURRENT_YEAR)

    return {
        "Engine": round(min(0.85, (mileage / 300000) + (age / 25)), 3),
        "Transmission": round(min(0.75, (mileage / 280000) + (age / 22)), 3),
        "Brake System": round(min(0.9, mileage / 200000), 3),
        "Electrical": round(min(0.65, age / 18), 3),
        "Suspension": round(min(0.8, mileage / 220000), 3),
    }


def _build_recommendations(
    age: int, mileage: int, recall_count: int, region: str, engine_type: str
) -> list[dict]:
    """Your original recommendation logic"""
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


# ── Main Prediction Function ─────────────────────────────────────────────────
def predict(vehicle: dict) -> dict:
    """
    Enhanced prediction with Health Score, Feature Importance & Failure Types
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
        # New Enhanced Fields
        "vehicle_health_score": _calculate_health_score(prob),
        "failure_probability": round(prob, 4),
        "failure_probability_pct": f"{prob:.1%}",
        "risk_level": _risk_label(prob),
        "failure_types": _predict_failure_types(vehicle),
        "feature_contributions": _get_feature_contributions(vehicle, prob),
        "maintenance_recommendations": recommendations,
        "suggested_next_service_date": next_service_date,
    }
