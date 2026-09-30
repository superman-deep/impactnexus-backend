from datetime import date

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gemini_client import build_advisory, diagnose_leaf, gemini_configured, logger

app = FastAPI(title="AgriNexus Prototype API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"message": "FastAPI is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

class Farm(BaseModel):
    farm_id: str
    location: str
    crop: str
    sowing_date: date
    soil_type: str

class SatelliteIntelligence(BaseModel):
    ndvi: float = Field(serialization_alias="NDVI")
    evi: float = Field(serialization_alias="EVI")
    savi: float = Field(serialization_alias="SAVI")
    ndmi: float = Field(serialization_alias="NDMI")
    cloud: float

class WeatherIntelligence(BaseModel):
    temperature: float
    humidity: float
    rainfall: float
    wind: float
    forecast: str

class SoilIntelligence(BaseModel):
    ph: float = Field(serialization_alias="pH")
    npk: str = Field(serialization_alias="NPK")
    organic_carbon: float
    soil_moisture: float
    soil_health_score: int

class FarmIntelligence(BaseModel):
    satellite: SatelliteIntelligence
    weather: WeatherIntelligence
    soil: SoilIntelligence

class Recommendation(BaseModel):
    category: str
    text: str
    priority: str

class Advisory(BaseModel):
    farm_health: int
    water_stress: str
    heat_stress: str
    disease_risk: str
    yield_risk: str
    recommendation: str
    reason: str
    priority: str
    recommendations: list[Recommendation]
    feature_importance: dict[str, float]
    source: str

class CropDiagnosis(BaseModel):
    disease: str
    confidence: int
    severity: str
    symptoms: list[str]
    next_action: str
    source: str

class BricsModel(BaseModel):
    country: str
    crop: str
    model: str
    version: str
    status: str

class BricsSchema(BaseModel):
    schema_name: str
    version: str
    fields: list[str]

FARMS = {"farm-001": Farm(farm_id="farm-001", location="Nashik, Maharashtra, India", crop="Tomato", sowing_date=date(2026, 6, 15), soil_type="Loamy")}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
logger.info("gemini configured=%s", gemini_configured())

def get_farm_or_404(farm_id: str) -> Farm:
    farm = FARMS.get(farm_id)
    if farm is None:
        raise HTTPException(status_code=404, detail="Farm not found")
    return farm

@app.get("/api/farm", response_model=list[Farm])
def list_farms():
    return list(FARMS.values())

@app.get("/api/farm/{farm_id}", response_model=Farm)
def get_farm(farm_id: str):
    return get_farm_or_404(farm_id)

def farm_intelligence() -> FarmIntelligence:
    return FarmIntelligence(
        satellite=SatelliteIntelligence(ndvi=0.72, evi=0.48, savi=0.61, ndmi=0.30, cloud=12.0),
        weather=WeatherIntelligence(temperature=29.4, humidity=67.0, rainfall=4.2, wind=11.0, forecast="Light rain expected in the next 24 hours."),
        soil=SoilIntelligence(ph=6.7, npk="Medium N, High P, Medium K", organic_carbon=0.72, soil_moisture=34.0, soil_health_score=78),
    )

def advisory_signals(farm: Farm) -> dict:
    intelligence = farm_intelligence().model_dump(by_alias=True)
    return {
        "farm_id": farm.farm_id,
        "location": farm.location,
        "crop": farm.crop,
        "sowing_date": farm.sowing_date.isoformat(),
        "soil_type": farm.soil_type,
        "satellite": intelligence["satellite"],
        "weather": intelligence["weather"],
        "soil": intelligence["soil"],
    }

@app.get("/api/farm/{farm_id}/intelligence", response_model=FarmIntelligence)
def get_farm_intelligence(farm_id: str):
    get_farm_or_404(farm_id)
    return farm_intelligence()

@app.get("/api/farm/{farm_id}/advisory", response_model=Advisory)
async def get_advisory(farm_id: str):
    farm = get_farm_or_404(farm_id)
    payload = await build_advisory(advisory_signals(farm))
    return Advisory.model_validate(payload)

@app.post("/api/crop-doctor", response_model=CropDiagnosis)
async def crop_doctor(leaf_image: UploadFile = File(...)):
    if not leaf_image.content_type or not leaf_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="leaf_image must be an image file")
    raw = await leaf_image.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="leaf_image must be 8 MB or smaller")
    try:
        payload = await diagnose_leaf(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="leaf_image must be a readable image file") from None
    return CropDiagnosis.model_validate(payload)

@app.get("/api/brics/models", response_model=list[BricsModel])
def get_brics_models():
    return [
        BricsModel(country="India", crop="Tomato", model="AgriNexus Advisory", version="0.1", status="Active"),
        BricsModel(country="Brazil", crop="Soybean", model="Tropical Crop Monitor", version="0.1", status="Active"),
        BricsModel(country="China", crop="Rice", model="Smart Irrigation Advisor", version="0.1", status="Prototype"),
        BricsModel(country="Russia", crop="Wheat", model="Climate Crop Risk", version="0.1", status="Active"),
        BricsModel(country="South Africa", crop="Maize", model="Soil Health Monitor", version="0.1", status="Prototype"),
    ]

@app.get("/api/brics/schema", response_model=BricsSchema)
def get_brics_schema():
    return BricsSchema(schema_name="BRICS-AGRI-SCHEMA", version="0.1", fields=["farm_id", "location", "crop", "sowing_date", "soil_type", "satellite", "weather", "soil", "advisory"])
