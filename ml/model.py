"""
ml/model.py -  Vehicle Failure Predictor
"""

import warnings

warnings.simplefilter("ignore")

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
CURRENT_YEAR = 2026
VERSION = "2.1"

MAKES = ["Toyota", "Honda", "Mazda", "Hyundai", "Ford", "Chevrolet", "BMW", "Mercedes"]
ENGINE_TYPES = ["Gasoline", "Hybrid", "Electric", "Diesel"]
DRIVING_STYLES = ["City", "Highway", "Mixed"]
REGIONS = ["Ontario", "Quebec", "BC", "Alberta", "Prairies"]

RELIABILITY_MAP = {
    "Toyota": 0.28,
    "Honda": 0.32,
    "Mazda": 0.38,
    "Hyundai": 0.52,
    "Ford": 0.68,
    "Chevrolet": 0.72,
    "BMW": 0.82,
    "Mercedes": 0.88,
}

WINTER_MAP = {
    "Ontario": 0.92,
    "Quebec": 1.15,
    "BC": 0.38,
    "Alberta": 0.88,
    "Prairies": 1.05,
}

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
    "mileage_age_interaction",
    "service_recall_ratio",
]

# Paths
MODEL_PATH = f"ml/vehicle_nn_model_v{VERSION}.keras"
SCALER_PATH = "ml/scaler_v2.pkl"
METADATA_PATH = "ml/model_metadata.pkl"


# ── Advanced Feature Engineering ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["vehicle_age"] = CURRENT_YEAR - df["year"]
    df["mileage_per_year"] = df["mileage_km"] / (df["vehicle_age"] + 1)
    df["high_mileage"] = (df["mileage_km"] > 180_000).astype(int)
    df["old_vehicle"] = (df["vehicle_age"] > 9).astype(int)

    df["winter_exposure"] = df["region"].map(WINTER_MAP).fillna(0.75)
    df["reliability_score"] = df["make"].map(RELIABILITY_MAP).fillna(0.60)

    # Interaction features (very important for performance)
    df["mileage_age_interaction"] = df["mileage_km"] * df["vehicle_age"]
    df["service_recall_ratio"] = df["recall_count"] / (df["service_frequency"] + 1)

    # Encodings
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


# ── Improved Synthetic Data ───────────────────────────────────────────────────
def _generate_data(n: int = 12000) -> pd.DataFrame:
    np.random.seed(42)
    raw = {
        "make": np.random.choice(MAKES, n),
        "year": np.random.randint(2000, 2026, n),
        "mileage_km": np.random.randint(2_000, 420_000, n),
        "engine_type": np.random.choice(ENGINE_TYPES, n),
        "service_frequency": np.random.randint(1, 18, n),
        "recall_count": np.random.randint(0, 8, n),
        "region": np.random.choice(REGIONS, n),
        "driving_style": np.random.choice(DRIVING_STYLES, n),
    }
    df = pd.DataFrame(raw)
    df = engineer_features(df)

    # Much more realistic failure probability
    base_prob = (
        df["reliability_score"] * 0.22
        + (df["vehicle_age"] / 18) * 0.25
        + (df["mileage_km"] / 280_000) * 0.28
        + (df["recall_count"] / 4) * 0.12
        + (1 / (df["service_frequency"] + 1)) * 0.08
        + df["winter_exposure"] * 0.05
    )

    df["failure"] = (base_prob + np.random.normal(0, 0.07, n) > 0.48).astype(int)
    return df


# ── Advanced Model with Residual Connections ────────────────────────────────
def _build_advanced_model(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,))

    x1 = Dense(256, activation="relu")(inp)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)

    x2 = Dense(128, activation="relu")(x1)
    x2 = BatchNormalization()(x2)

    # Residual connection
    x3 = Dense(128, activation="relu")(x2)
    x3 = Add()([x2, x3])  # Skip connection
    x3 = Dropout(0.25)(x3)

    x4 = Dense(64, activation="relu")(x3)
    x4 = Dropout(0.2)(x4)

    out = Dense(1, activation="sigmoid")(x4)

    model = Model(inputs=inp, outputs=out)
    return model


# ── Training Function ───────────────────────────────────────────────────────
def train_and_save():
    print(f"[model v{VERSION}] Generating advanced synthetic data...")
    df = _generate_data(12000)

    X = df[FEATURE_COLS].values
    y = df["failure"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = _build_advanced_model(len(FEATURE_COLS))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    print("[model] Training advanced neural network...")
    model.fit(
        X_train_s,
        y_train,
        validation_split=0.15,
        epochs=80,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Save everything
    os.makedirs("ml", exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    loss, acc, auc = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"[model v{VERSION}] Training Complete!")
    print(f"   Test Accuracy : {acc:.4f}")
    print(f"   Test AUC      : {auc:.4f}")

    # Save metadata
    metadata = {
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "test_acc": acc,
        "test_auc": auc,
    }
    joblib.dump(metadata, METADATA_PATH)

    return model, scaler


# ── Load or Train ─────────────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print(f"[model v{VERSION}] Loading pre-trained model...")
    nn_model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    nn_model, scaler = train_and_save()

print(f"[model v{VERSION}] Ready for predictions ✅")
