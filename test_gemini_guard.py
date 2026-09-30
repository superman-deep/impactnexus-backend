import io
import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import gemini_client
from gemini_client import AdvisoryDraft, RecommendationDraft, build_advisory, diagnose_leaf, model_name, normalize_importance, prepare_jpeg


def _signals(farm_id: str) -> dict:
    return {"farm_id": farm_id, "crop": "Tomato", "satellite": {"NDVI": 0.72}}


def _draft() -> AdvisoryDraft:
    return AdvisoryDraft(
        farm_health=81,
        water_stress="High",
        heat_stress="Low",
        disease_risk="Moderate",
        yield_risk="Low",
        recommendation="Keep residue on the soil after harvest.",
        reason="Cover and residue reduce moisture loss. This is decision support, not a treatment prescription.",
        priority="High",
        recommendations=[RecommendationDraft(category="Soil cover", text="Keep residue on the soil after harvest.", priority="High")],
        ndvi_importance=40,
        soil_moisture_importance=30,
        rainfall_importance=20,
        temperature_importance=10,
    )


class GeminiGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {"GEMINI_CACHE_DIR": self.tmp.name, "GEMINI_DAILY_CALL_BUDGET": "30", "GEMINI_MODEL": "gemini-2.5-flash"},
            clear=False,
        )
        self.env.start()

    async def asyncTearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_importance_sums_to_100(self):
        self.assertEqual(sum(normalize_importance(1, 1, 1, 1).values()), 100)
        self.assertEqual(sum(normalize_importance(0, 0, 0, 5).values()), 100)

    def test_prepare_jpeg_shrinks_image(self):
        image = Image.new("RGB", (1200, 800), (20, 120, 40))
        raw = io.BytesIO()
        image.save(raw, format="PNG")
        jpeg = prepare_jpeg(raw.getvalue())
        opened = Image.open(io.BytesIO(jpeg))
        self.assertEqual(opened.format, "JPEG")
        self.assertLessEqual(max(opened.size), 768)

    def test_pro_model_is_refused(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-pro"}):
            self.assertEqual(model_name(), "gemini-3.8-flash")

    async def test_advisory_cache_skips_second_call(self):
        calls = {"count": 0}

        async def generate(system, contents, schema):
            calls["count"] += 1
            return _draft()

        first = await build_advisory(_signals("farm-001"), generate)
        second = await build_advisory(_signals("farm-001"), generate)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(first["source"], "gemini")
        self.assertEqual(second["source"], "cache")
        self.assertEqual(second["recommendation"], first["recommendation"])
        self.assertEqual(sum(second["feature_importance"].values()), 100)

    async def test_budget_stops_further_calls(self):
        calls = {"count": 0}

        async def generate(system, contents, schema):
            calls["count"] += 1
            return _draft()

        with patch.dict(os.environ, {"GEMINI_DAILY_CALL_BUDGET": "1"}):
            first = await build_advisory(_signals("farm-a"), generate)
            second = await build_advisory(_signals("farm-b"), generate)
        self.assertEqual(first["source"], "gemini")
        self.assertEqual(second["source"], "quota")
        self.assertEqual(calls["count"], 1)

    async def test_missing_key_does_not_call_gemini(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), patch.object(gemini_client, "_generate", side_effect=AssertionError("network")):
            payload = await build_advisory(_signals("farm-001"), None)
        self.assertEqual(payload["source"], "demo")

    async def test_failed_call_is_not_retried_immediately(self):
        calls = {"count": 0}

        async def generate(system, contents, schema):
            calls["count"] += 1
            raise RuntimeError("upstream")

        first = await build_advisory(_signals("farm-err"), generate)
        second = await build_advisory(_signals("farm-err"), generate)
        self.assertEqual(calls["count"], 1)
        self.assertEqual(first["source"], "unavailable")
        self.assertEqual(second["source"], "unavailable")

    async def test_unreadable_image_does_not_call_gemini(self):
        async def generate(system, contents, schema):
            raise AssertionError("generate should not run")

        with self.assertRaises(ValueError):
            await diagnose_leaf(b"not-an-image", generate)


if __name__ == "__main__":
    unittest.main()
