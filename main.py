from datetime import date

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

class Advisory(BaseModel):
    water_stress: str
    heat_stress: str
    disease_risk: str
    yield_risk: str
    recommendations: list[str]
    feature_importance: dict[str, float]

class CropDiagnosis(BaseModel):
    disease: str
    confidence: float
    severity: str
    symptoms: list[str]
    next_action: str

class BricsModel(BaseModel):
    country: str
    model_name: str
    version: str
    purpose: str
    status: str

class BricsSchema(BaseModel):
    schema_name: str
    version: str
    fields: list[str]

FARMS = {"farm-001": Farm(farm_id="farm-001", location="mumbai, Maharashtra, India", crop="Tomato", sowing_date=date(2026, 6, 15), soil_type="Loamy")}

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

@app.get("/api/farm/{farm_id}/intelligence", response_model=FarmIntelligence)
def get_farm_intelligence(farm_id: str):
    get_farm_or_404(farm_id)
    return FarmIntelligence(
        satellite=SatelliteIntelligence(ndvi=0.72, evi=0.48, savi=0.61, ndmi=0.30, cloud=12.0),
        weather=WeatherIntelligence(temperature=29.4, humidity=67.0, rainfall=4.2, wind=11.0, forecast="Light rain expected in the next 24 hours."),
        soil=SoilIntelligence(ph=6.7, npk="Medium N, High P, Medium K", organic_carbon=0.72, soil_moisture=34.0, soil_health_score=78),
    )

@app.get("/api/farm/{farm_id}/advisory", response_model=Advisory)
def get_advisory(farm_id: str):
    get_farm_or_404(farm_id)
    return Advisory(
        water_stress="Moderate", heat_stress="Low", disease_risk="Medium", yield_risk="Low",
        recommendations=["Irrigate early morning for 25 minutes within the next 48 hours.", "Inspect lower leaves for early blight symptoms."],
        feature_importance={"NDVI": 0.34, "soil_moisture": 0.29, "rainfall": 0.21, "temperature": 0.16},
    )

@app.post("/api/crop-doctor", response_model=CropDiagnosis)
async def crop_doctor(leaf_image: UploadFile = File(...)):
    """Accept a prototype leaf upload; image content is not analysed yet."""
    if not leaf_image.content_type or not leaf_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="leaf_image must be an image file")
    return CropDiagnosis(disease="Early blight", confidence=0.91, severity="Moderate", symptoms=["Dark concentric spots", "Yellowing lower leaves"], next_action="Remove severely affected leaves and consult local guidance before treatment.")

@app.get("/api/brics/models", response_model=list[BricsModel])
def get_brics_models():
    return [
        BricsModel(country="India", model_name="AgriNexus Advisory", version="0.1", purpose="Farm risk advisory", status="mock"),
        BricsModel(country="Brazil", model_name="Tropical Crop Monitor", version="0.1", purpose="Crop monitoring", status="mock"),
        BricsModel(country="China", model_name="Smart Irrigation Advisor", version="0.1", purpose="Irrigation advice", status="mock"),
        BricsModel(country="Russia", model_name="Climate Crop Risk", version="0.1", purpose="Weather risk assessment", status="mock"),
        BricsModel(country="South Africa", model_name="Soil Health Monitor", version="0.1", purpose="Soil assessment", status="mock"),
    ]

@app.get("/api/brics/schema", response_model=BricsSchema)
def get_brics_schema():
    return BricsSchema(schema_name="BRICS-AGRI-SCHEMA", version="0.1", fields=["farm_id", "location", "crop", "sowing_date", "soil_type", "satellite", "weather", "soil", "advisory"])
