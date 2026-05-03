"""
schemas.py
----------
Pydantic v2 models for request validation and response serialisation.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ── Allowed enum values (keep in sync with ml/model.py) ───────────────────────
VALID_MAKES          = {"Toyota", "Honda", "Mazda", "Hyundai", "Ford", "Chevrolet", "BMW", "Mercedes"}
VALID_ENGINE_TYPES   = {"Gasoline", "Hybrid", "Electric", "Diesel"}
VALID_DRIVING_STYLES = {"City", "Highway", "Mixed"}
VALID_REGIONS        = {"Ontario", "Quebec", "BC", "Alberta", "Prairies"}


# ── Request ────────────────────────────────────────────────────────────────────
class VehicleRequest(BaseModel):
    make: str = Field(..., examples=["Toyota"],   description="Vehicle manufacturer")
    model: str = Field(..., examples=["Camry"],   description="Vehicle model name")
    year: int  = Field(..., ge=1990, le=2026,     description="Model year (1990–2026)")
    mileage_km: int = Field(..., ge=0, le=999_999, description="Odometer reading in kilometres")
    engine_type: str = Field(..., examples=["Gasoline"],  description="One of: Gasoline, Hybrid, Electric, Diesel")
    service_frequency: int = Field(..., ge=1, le=24,      description="Months between routine services")
    recall_count: int      = Field(..., ge=0, le=20,      description="Number of open/past recalls")
    region: str = Field(..., examples=["Ontario"],        description="Canadian province/region")
    driving_style: str = Field(..., examples=["Mixed"],   description="One of: City, Highway, Mixed")

    @field_validator("make")
    @classmethod
    def validate_make(cls, v: str) -> str:
        if v not in VALID_MAKES:
            raise ValueError(f"make must be one of {sorted(VALID_MAKES)}")
        return v

    @field_validator("engine_type")
    @classmethod
    def validate_engine_type(cls, v: str) -> str:
        if v not in VALID_ENGINE_TYPES:
            raise ValueError(f"engine_type must be one of {sorted(VALID_ENGINE_TYPES)}")
        return v

    @field_validator("driving_style")
    @classmethod
    def validate_driving_style(cls, v: str) -> str:
        if v not in VALID_DRIVING_STYLES:
            raise ValueError(f"driving_style must be one of {sorted(VALID_DRIVING_STYLES)}")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        if v not in VALID_REGIONS:
            raise ValueError(f"region must be one of {sorted(VALID_REGIONS)}")
        return v


# ── Response ───────────────────────────────────────────────────────────────────
class VehicleSummary(BaseModel):
    year:  int
    make:  str
    model: str


class MaintenanceItem(BaseModel):
    task:     str
    timeline: str
    priority: Literal["Critical", "High", "Medium", "Low"]


class PredictionResponse(BaseModel):
    vehicle:                    VehicleSummary
    vehicle_age_years:          int
    mileage_km:                 int
    region:                     str
    failure_probability:        float = Field(..., description="Raw probability 0–1")
    failure_probability_pct:    str   = Field(..., description="Human-readable percentage")
    risk_level:                 Literal["High", "Medium", "Low"]
    maintenance_recommendations: list[MaintenanceItem]
    suggested_next_service_date: str  = Field(..., description="ISO date YYYY-MM-DD")


class HealthResponse(BaseModel):
    status:  str
    version: str
    model:   str


class ErrorResponse(BaseModel):
    detail: str
