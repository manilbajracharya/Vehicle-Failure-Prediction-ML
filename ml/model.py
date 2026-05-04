"""
ml/model.py
-----------
Trains the vehicle-failure neural network once at import time and
exposes the artefacts (model, scaler, category lists, maps) that
the prediction logic needs.
"""

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── Category lists (used in both training and inference) ───────────────────────
MAKES = ["Toyota", "Honda", "Mazda", "Hyundai", "Ford", "Chevrolet", "BMW", "Mercedes"]
ENGINE_TYPES = ["Gasoline", "Hybrid", "Electric", "Diesel"]
DRIVING_STYLES = ["City", "Highway", "Mixed"]
REGIONS = ["Ontario", "Quebec", "BC", "Alberta", "Prairies"]

RELIABILITY_MAP = {
    "Toyota": 0.30,
    "Honda": 0.35,
    "Mazda": 0.40,
    "Hyundai": 0.55,
    "Ford": 0.70,
    "Chevrolet": 0.75,
    "BMW": 0.85,
    "Mercedes": 0.90,
}
WINTER_MAP = {
    "Ontario": 0.90,
    "Quebec": 1.10,
    "BC": 0.40,
    "Alberta": 0.85,
    "Prairies": 1.00,
}

CURRENT_YEAR = 2026

FEATURE_COLS = [
    "vehicle_age",
    "mileage_km",
    "mileage_per_year",
    "high_mileage",
    "old_vehicle",
    "service_frequency",
    "recall_count",
    "winter_exposure",
    "reliability_score",
    "make_encoded",
    "engine_type_encoded",
    "driving_style_encoded",
    "region_encoded",
]


# ── Feature engineering (shared between training & inference) ─────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vehicle_age"] = CURRENT_YEAR - df["year"]
    df["mileage_per_year"] = df["mileage_km"] / (df["vehicle_age"] + 1)
    df["high_mileage"] = (df["mileage_km"] > 200_000).astype(int)
    df["old_vehicle"] = (df["vehicle_age"] > 10).astype(int)
    df["winter_exposure"] = df["region"].map(WINTER_MAP).fillna(0.70)
    df["reliability_score"] = df["make"].map(RELIABILITY_MAP).fillna(0.60)

    df["make_encoded"] = df["make"].apply(
        lambda x: MAKES.index(x) if x in MAKES else -1
    )
    df["engine_type_encoded"] = df["engine_type"].apply(
        lambda x: ENGINE_TYPES.index(x) if x in ENGINE_TYPES else -1
    )
    df["driving_style_encoded"] = df["driving_style"].apply(
        lambda x: DRIVING_STYLES.index(x) if x in DRIVING_STYLES else -1
    )
    df["region_encoded"] = df["region"].apply(
        lambda x: REGIONS.index(x) if x in REGIONS else -1
    )

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df


# ── Synthetic training data ────────────────────────────────────────────────────
def _generate_data(n: int = 5000) -> pd.DataFrame:
    raw = {
        "make": np.random.choice(MAKES, n),
        "year": np.random.randint(2005, 2024, n),
        "mileage_km": np.random.randint(5_000, 300_000, n),
        "engine_type": np.random.choice(ENGINE_TYPES, n),
        "service_frequency": np.random.randint(1, 12, n),
        "recall_count": np.random.randint(0, 6, n),
        "region": np.random.choice(REGIONS, n),
        "driving_style": np.random.choice(DRIVING_STYLES, n),
    }
    df = pd.DataFrame(raw)
    df = engineer_features(df)

    failure_prob = (
        df["reliability_score"] * 0.25
        + (df["vehicle_age"] / 20) * 0.20
        + (df["mileage_km"] / 300_000) * 0.25
        + (df["recall_count"] / 5) * 0.15
        + (1 / (df["service_frequency"] + 1)) * 0.10
        + df["winter_exposure"] * 0.05
        + np.random.normal(0, 0.05, n)
    )
    df["failure"] = (failure_prob > 0.50).astype(int)
    return df


# ── Model architecture ─────────────────────────────────────────────────────────
def _build_model(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,))
    x = Dense(128, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inputs=inp, outputs=out)


# ── Train ──────────────────────────────────────────────────────────────────────
print("[model] Generating synthetic data and training …")

_df = _generate_data(5000)
_X = _df[FEATURE_COLS].values
_y = _df["failure"].values

_X_train, _X_test, _y_train, _y_test = train_test_split(
    _X, _y, test_size=0.2, random_state=42
)

scaler: StandardScaler = StandardScaler()
_X_train_s = scaler.fit_transform(_X_train)
_X_test_s = scaler.transform(_X_test)

nn_model: Model = _build_model(len(FEATURE_COLS))
nn_model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
nn_model.fit(
    _X_train_s, _y_train, validation_split=0.1, epochs=30, batch_size=64, verbose=0
)

_loss, _acc = nn_model.evaluate(_X_test_s, _y_test, verbose=0)
print(f"[model] Training complete — test accuracy: {_acc:.4f}")
