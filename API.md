# AgriNexus Prototype API

Run the server from `/backend`:

```bash
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn main:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`. OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`. All mock farm routes use `farm-001`.

## Existing service routes

### `GET /`

Purpose: confirm the API is running. Request: none.

Example response: `{"message":"FastAPI is running!"}`

### `GET /health`

Purpose: health check. Request: none.

Example response: `{"status":"healthy"}`

## Farm API

### `GET /api/farm`

Purpose: list mock farms. Request: none.

Example response:

```json
[{"farm_id":"farm-001","location":"Nashik, Maharashtra, India","crop":"Tomato","sowing_date":"2026-06-15","soil_type":"Loamy"}]
```

### `GET /api/farm/{farm_id}`

Purpose: get a mock farm. Request parameter: path `farm_id` (use `farm-001`).

Example response: `{"farm_id":"farm-001","location":"Nashik, Maharashtra, India","crop":"Tomato","sowing_date":"2026-06-15","soil_type":"Loamy"}`

## Farm Intelligence API

### `GET /api/farm/{farm_id}/intelligence`

Purpose: return mock satellite, weather, and soil indicators. Request parameter: path `farm_id`.

Example response:

```json
{"satellite":{"NDVI":0.72,"EVI":0.48,"SAVI":0.61,"NDMI":0.3,"cloud":12.0},"weather":{"temperature":29.4,"humidity":67.0,"rainfall":4.2,"wind":11.0,"forecast":"Light rain expected in the next 24 hours."},"soil":{"pH":6.7,"NPK":"Medium N, High P, Medium K","organic_carbon":0.72,"soil_moisture":34.0,"soil_health_score":78}}
```

## AI Advisory API

### `GET /api/farm/{farm_id}/advisory`

Purpose: return Gemini Flash risks and regenerative recommendations for the farm signals. The same signals are cached for 12 hours. `source` is `gemini`, `cache`, `demo`, `quota`, or `unavailable`. Request parameter: path `farm_id`.

Example response:

```json
{"farm_health":78,"water_stress":"Moderate","heat_stress":"Low","disease_risk":"Medium","yield_risk":"Low","recommendation":"Irrigate early morning for 25 minutes within the next 48 hours.","reason":"Soil moisture is adequate today, but forecast conditions may increase water demand.","priority":"High","recommendations":[{"category":"Irrigation","text":"Irrigate early morning for 25 minutes within the next 48 hours.","priority":"High"}],"feature_importance":{"NDVI":34,"soil_moisture":29,"rainfall":21,"temperature":16},"source":"cache"}
```

## Crop Doctor API

### `POST /api/crop-doctor`

Purpose: accept a leaf image and diagnose it with Gemini Flash vision. Identical image bytes are cached for 7 days. Request body: `multipart/form-data`, with required `leaf_image` image file (8 MB max). `confidence` is an integer from 0 to 100.

Example response:

```json
{"disease":"Early blight","confidence":91,"severity":"Moderate","symptoms":["Dark concentric spots","Yellowing lower leaves"],"next_action":"Remove severely affected leaves and consult local guidance before treatment.","source":"gemini"}
```

## BRICS API

### `GET /api/brics/models`

Purpose: list mock model registry records for India, Brazil, China, Russia, and South Africa. Request: none.

Example response: `[{"country":"India","crop":"Tomato","model":"AgriNexus Advisory","version":"0.1","status":"Active"}]`

### `GET /api/brics/schema`

Purpose: return basic mock `BRICS-AGRI-SCHEMA` fields. Request: none.

Example response: `{"schema_name":"BRICS-AGRI-SCHEMA","version":"0.1","fields":["farm_id","location","crop","sowing_date","soil_type","satellite","weather","soil","advisory"]}`
