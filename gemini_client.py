"""Gemini advisory and leaf diagnosis with a disk cache and a daily call cap.

Only this module talks to Google. Health, farm, intelligence, and BRICS routes
must not import a generate call from here. The API key is read from the
environment and is never logged.
"""

import hashlib
import io
import json
import logging
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent / ".env")

logger = logging.getLogger("agrinexus")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False

ADVISORY_TTL_SECONDS = 12 * 60 * 60
DOCTOR_TTL_SECONDS = 7 * 24 * 60 * 60
FAILURE_COOLDOWN_SECONDS = 10 * 60
MAX_OUTPUT_TOKENS = 768
IMAGE_EDGE_PX = 768
_lock = threading.Lock()
_client = None

Generate = Callable[[str, list, type[BaseModel]], Awaitable[object]]


class RecommendationDraft(BaseModel):
    category: str
    text: str
    priority: str


class AdvisoryDraft(BaseModel):
    farm_health: int
    water_stress: str
    heat_stress: str
    disease_risk: str
    yield_risk: str
    recommendation: str
    reason: str
    priority: str
    recommendations: list[RecommendationDraft]
    ndvi_importance: int
    soil_moisture_importance: int
    rainfall_importance: int
    temperature_importance: int


class DiagnosisDraft(BaseModel):
    is_crop_leaf: bool
    disease: str
    confidence: int
    severity: str
    symptoms: list[str]
    next_action: str


ADVISORY_SYSTEM = (
    "You advise small farmers from the JSON signals in the user message only. "
    "Recommend regenerative practices: cover crops, residue retention, mulching, "
    "reduced tillage, and nutrient timing. Do not prescribe pesticides or doses. "
    "Risk labels must be Low, Moderate, or High. Priorities must be Low, Medium, or High. "
    "Return two or three short actions. "
    "Say that this is decision support, not a treatment prescription."
)

DIAGNOSIS_SYSTEM = (
    "You inspect one crop-leaf photo. If it is not a crop leaf, set is_crop_leaf to false "
    "and do not invent a disease. confidence is an integer from 0 to 100. "
    "severity is Low, Moderate, or High. "
    "next_action must tell the farmer to confirm locally before any treatment. "
    "Do not prescribe pesticides or doses."
)

DEMO_ADVISORY = {
    "farm_health": 78,
    "water_stress": "Moderate",
    "heat_stress": "Low",
    "disease_risk": "Medium",
    "yield_risk": "Low",
    "recommendation": "Irrigate early morning for 25 minutes within the next 48 hours.",
    "reason": "Soil moisture is adequate today, but forecast conditions may increase water demand. This is decision support, not a treatment prescription.",
    "priority": "High",
    "recommendations": [
        {"category": "Irrigation", "text": "Irrigate early morning for 25 minutes within the next 48 hours.", "priority": "High"},
        {"category": "Crop monitoring", "text": "Inspect lower leaves for early blight symptoms.", "priority": "Medium"},
    ],
    "feature_importance": {"NDVI": 34, "soil_moisture": 29, "rainfall": 21, "temperature": 16},
}

DEMO_DIAGNOSIS = {
    "disease": "Early blight",
    "confidence": 91,
    "severity": "Moderate",
    "symptoms": ["Dark concentric spots", "Yellowing lower leaves"],
    "next_action": "Remove severely affected leaves and consult local guidance before treatment.",
}


def gemini_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


DEFAULT_MODEL = "gemini-3.8-flash"


def model_name() -> str:
    configured = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if "pro" in configured.lower():
        logger.warning("refusing Gemini Pro; using %s", DEFAULT_MODEL)
        return DEFAULT_MODEL
    return configured


def daily_budget() -> int:
    raw = os.getenv("GEMINI_DAILY_CALL_BUDGET", "30")
    try:
        return max(0, int(raw))
    except ValueError:
        return 30


def cache_dir() -> Path:
    configured = os.getenv("GEMINI_CACHE_DIR")
    path = Path(configured) if configured else Path(__file__).resolve().parent / ".cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_importance(ndvi: float, soil_moisture: float, rainfall: float, temperature: float) -> dict[str, int]:
    raw = {
        "NDVI": max(0.0, float(ndvi)),
        "soil_moisture": max(0.0, float(soil_moisture)),
        "rainfall": max(0.0, float(rainfall)),
        "temperature": max(0.0, float(temperature)),
    }
    total = sum(raw.values())
    if total <= 0:
        return dict(DEMO_ADVISORY["feature_importance"])
    scaled = {key: int(round(value / total * 100)) for key, value in raw.items()}
    largest = max(scaled, key=scaled.get)
    scaled[largest] += 100 - sum(scaled.values())
    return scaled


def prepare_jpeg(raw: bytes) -> bytes:
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("unreadable image") from exc
    image = image.convert("RGB")
    image.thumbnail((IMAGE_EDGE_PX, IMAGE_EDGE_PX))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=70, optimize=True)
    return output.getvalue()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_cache(name: str) -> dict | None:
    path = cache_dir() / name
    if not path.exists():
        return None
    try:
        body = json.loads(path.read_text())
        if time.time() - float(body["stored_at"]) > float(body["ttl"]):
            return None
        payload = dict(body["payload"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("cache read failed name=%s", name)
        return None
    if payload.get("source") == "gemini":
        payload["source"] = "cache"
    return payload


def _write_cache(name: str, payload: dict, ttl_seconds: int) -> None:
    path = cache_dir() / name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stored_at": time.time(), "ttl": ttl_seconds, "payload": payload}))
    temporary.replace(path)


def _reserve_call() -> bool:
    budget = daily_budget()
    path = cache_dir() / f"budget-{date.today().isoformat()}.json"
    with _lock:
        count = 0
        if path.exists():
            try:
                count = int(json.loads(path.read_text()).get("count", 0))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                count = 0
        if count >= budget:
            return False
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"count": count + 1}))
        temporary.replace(path)
        return True


def _with_source(payload: dict, source: str) -> dict:
    body = dict(payload)
    body["source"] = source
    return body


def _risk(value: str, default: str) -> str:
    labels = {"low": "Low", "moderate": "Moderate", "medium": "Medium", "high": "High"}
    return labels.get(str(value).strip().lower(), default)


def _priority(value: str) -> str:
    labels = {"low": "Low", "medium": "Medium", "high": "High"}
    return labels.get(str(value).strip().lower(), "Medium")


def _advisory_from_draft(draft: AdvisoryDraft) -> dict:
    recommendations = [
        {"category": item.category.strip() or "Farm action", "text": item.text.strip(), "priority": _priority(item.priority)}
        for item in draft.recommendations
        if item.text.strip()
    ][:3]
    if not recommendations:
        recommendations = DEMO_ADVISORY["recommendations"]
    return {
        "farm_health": min(100, max(0, int(draft.farm_health))),
        "water_stress": _risk(draft.water_stress, "Moderate"),
        "heat_stress": _risk(draft.heat_stress, "Low"),
        "disease_risk": _risk(draft.disease_risk, "Medium"),
        "yield_risk": _risk(draft.yield_risk, "Low"),
        "recommendation": draft.recommendation.strip() or recommendations[0]["text"],
        "reason": draft.reason.strip() or DEMO_ADVISORY["reason"],
        "priority": _priority(draft.priority),
        "recommendations": recommendations,
        "feature_importance": normalize_importance(
            draft.ndvi_importance,
            draft.soil_moisture_importance,
            draft.rainfall_importance,
            draft.temperature_importance,
        ),
        "source": "gemini",
    }


def _diagnosis_from_draft(draft: DiagnosisDraft) -> dict:
    if not draft.is_crop_leaf:
        return {
            "disease": "Not a crop leaf",
            "confidence": 0,
            "severity": "Low",
            "symptoms": ["The photo does not show a clear crop leaf."],
            "next_action": "Upload a close, well-lit photo of the affected leaf.",
            "source": "gemini",
        }
    symptoms = [item.strip() for item in draft.symptoms if item.strip()][:4]
    if not symptoms:
        symptoms = ["Visible leaf symptoms need a closer look."]
    return {
        "disease": draft.disease.strip() or "Unidentified leaf symptom",
        "confidence": min(100, max(0, int(draft.confidence))),
        "severity": _risk(draft.severity, "Moderate"),
        "symptoms": symptoms,
        "next_action": draft.next_action.strip() or "Confirm this locally before any treatment.",
        "source": "gemini",
    }


def _client_or_none():
    global _client
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=20_000, retry_options=types.HttpRetryOptions(attempts=1)),
        )
    return _client


def _generation_config(schema: type[BaseModel], system: str):
    from google.genai import types

    config = dict(
        system_instruction=system,
        temperature=0.2,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=schema,
    )
    if "2.5" in model_name():
        config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    else:
        config["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    config["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    return types.GenerateContentConfig(**config)


def _parsed(response: object, schema: type[BaseModel]) -> BaseModel:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    if isinstance(parsed, dict):
        return schema.model_validate(parsed)
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("empty gemini response")
    return schema.model_validate_json(text)


def _failure_source(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "")
    if code == 429 or "RESOURCE_EXHAUSTED" in status or "429" in status:
        return "quota"
    return "unavailable"


async def _generate(system: str, contents: list, schema: type[BaseModel]) -> BaseModel:
    client = _client_or_none()
    if client is None:
        raise RuntimeError("missing_key")
    response = await client.aio.models.generate_content(
        model=model_name(),
        contents=contents,
        config=_generation_config(schema, system),
    )
    return _parsed(response, schema)


async def build_advisory(signals: dict, generate: Generate | None = None) -> dict:
    encoded = json.dumps(signals, sort_keys=True, default=str).encode()
    cache_name = f"advisory-{_digest(encoded)}.json"
    cached = _read_cache(cache_name)
    if cached is not None:
        logger.info("gemini advisory cache_hit")
        return cached
    if not gemini_configured() and generate is None:
        logger.info("gemini advisory skipped reason=missing_key")
        return _with_source(DEMO_ADVISORY, "demo")
    try:
        if not _reserve_call():
            logger.info("gemini advisory skipped reason=budget")
            return _with_source(DEMO_ADVISORY, "quota")
    except OSError:
        logger.warning("gemini advisory skipped reason=budget_storage")
        return _with_source(DEMO_ADVISORY, "quota")

    logger.info("gemini advisory call")
    caller = generate or _generate
    try:
        draft = await caller(ADVISORY_SYSTEM, [json.dumps(signals)], AdvisoryDraft)
        if not isinstance(draft, AdvisoryDraft):
            draft = AdvisoryDraft.model_validate(draft)
        payload = _advisory_from_draft(draft)
    except Exception as exc:
        source = _failure_source(exc)
        logger.warning("gemini advisory failed code=%s status=%s source=%s", getattr(exc, "code", None), getattr(exc, "status", None), source)
        payload = _with_source(DEMO_ADVISORY, source)
        _remember_failure(cache_name, payload)
        return payload
    _remember(cache_name, payload, ADVISORY_TTL_SECONDS)
    return payload


async def diagnose_leaf(raw: bytes, generate: Generate | None = None) -> dict:
    cache_name = f"doctor-{_digest(raw)}.json"
    cached = _read_cache(cache_name)
    if cached is not None:
        logger.info("gemini doctor cache_hit")
        return cached
    jpeg = prepare_jpeg(raw)
    if not gemini_configured() and generate is None:
        logger.info("gemini doctor skipped reason=missing_key")
        return _with_source(DEMO_DIAGNOSIS, "demo")
    try:
        if not _reserve_call():
            logger.info("gemini doctor skipped reason=budget")
            return _with_source(DEMO_DIAGNOSIS, "quota")
    except OSError:
        logger.warning("gemini doctor skipped reason=budget_storage")
        return _with_source(DEMO_DIAGNOSIS, "quota")

    logger.info("gemini doctor call")
    from google.genai import types

    contents = ["Identify the likely condition in this crop leaf photo.", types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")]
    caller = generate or _generate
    try:
        draft = await caller(DIAGNOSIS_SYSTEM, contents, DiagnosisDraft)
        if not isinstance(draft, DiagnosisDraft):
            draft = DiagnosisDraft.model_validate(draft)
        payload = _diagnosis_from_draft(draft)
    except Exception as exc:
        source = _failure_source(exc)
        logger.warning("gemini doctor failed code=%s status=%s source=%s", getattr(exc, "code", None), getattr(exc, "status", None), source)
        payload = _with_source(DEMO_DIAGNOSIS, source)
        _remember_failure(cache_name, payload)
        return payload
    _remember(cache_name, payload, DOCTOR_TTL_SECONDS)
    return payload


def _remember(name: str, payload: dict, ttl_seconds: int) -> None:
    try:
        _write_cache(name, payload, ttl_seconds)
    except OSError:
        logger.warning("gemini cache_write_failed name=%s", name)


def _remember_failure(name: str, payload: dict) -> None:
    _remember(name, payload, FAILURE_COOLDOWN_SECONDS)
