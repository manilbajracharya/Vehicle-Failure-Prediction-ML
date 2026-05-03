# Vehicle Failure Prediction API

FastAPI service that predicts the probability of a major vehicle repair within
the next 12 months, using a TensorFlow neural network trained on synthetic
Canadian vehicle data.

---

## Project Structure

```
vehicle_api/
├── main.py                              # FastAPI app + routes
├── schemas.py                           # Pydantic request/response models
├── requirements.txt
├── VehicleFailureAPI.postman_collection.json
└── ml/
    ├── __init__.py
    ├── model.py                         # Data generation + model training
    └── predictor.py                     # Prediction logic
```

---

## Setup

```bash
# 1. Create & activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The model trains automatically on startup (~10–20 s).  
Interactive docs available at **http://localhost:8000/docs**

---

## Endpoints

### GET /health
```
GET http://localhost:8000/health
```
Returns service status.

---

### POST /predict
Send vehicle data as a JSON body.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Toyota",
    "model": "Camry",
    "year": 2018,
    "mileage_km": 168000,
    "engine_type": "Hybrid",
    "service_frequency": 6,
    "recall_count": 2,
    "region": "Ontario",
    "driving_style": "Mixed"
  }'
```

---

### GET /predict
Pass all fields as query parameters.

```
GET http://localhost:8000/predict?make=Toyota&model=Camry&year=2018
    &mileage_km=168000&engine_type=Hybrid&service_frequency=6
    &recall_count=2&region=Ontario&driving_style=Mixed
```

---

## Request Fields

| Field              | Type   | Allowed Values                                           |
|--------------------|--------|----------------------------------------------------------|
| `make`             | string | Toyota, Honda, Mazda, Hyundai, Ford, Chevrolet, BMW, Mercedes |
| `model`            | string | Any string                                               |
| `year`             | int    | 1990 – 2026                                              |
| `mileage_km`       | int    | 0 – 999,999                                              |
| `engine_type`      | string | Gasoline, Hybrid, Electric, Diesel                       |
| `service_frequency`| int    | 1 – 24 (months between services)                         |
| `recall_count`     | int    | 0 – 20                                                   |
| `region`           | string | Ontario, Quebec, BC, Alberta, Prairies                   |
| `driving_style`    | string | City, Highway, Mixed                                     |

---

## Response

```json
{
  "vehicle": { "year": 2018, "make": "Toyota", "model": "Camry" },
  "vehicle_age_years": 8,
  "mileage_km": 168000,
  "region": "Ontario",
  "failure_probability": 0.5421,
  "failure_probability_pct": "54.2%",
  "risk_level": "Medium",
  "maintenance_recommendations": [
    {
      "task": "Full Synthetic Oil Change + Filters",
      "timeline": "Every 8,000 km or 3 months",
      "priority": "High"
    },
    ...
  ],
  "suggested_next_service_date": "2026-07-17"
}
```

**risk_level** thresholds:
- 🔴 **High** — probability > 65 %
- 🟠 **Medium** — probability 35 %–65 %
- 🟢 **Low** — probability < 35 %

---

## Postman

1. Open Postman → **Import**
2. Select `VehicleFailureAPI.postman_collection.json`
3. Set the `base_url` variable to your server URL (default `http://localhost:8000`)
4. Run any request from the collection
