

import json
import os
from typing import Any, Dict, Optional

ETHICS_DISCLAIMER = "AI-estimated external vehicle damage (not a medical diagnosis)."

VISION_PROMPT = """You are an emergency response AI assessing visible vehicle
damage from a crash scene evidence photo. Evaluate the VISIBLE EXTERNAL
VEHICLE DAMAGE and return ONLY a raw JSON object with this exact structure:

{
  "damage_level": "none" | "minor" | "moderate" | "severe" | "critical",
  "indicators_observed": ["list of visible indicators e.g. crushed front bumper, shattered windshield, deployed airbag, fluid leakage, structural deformation"],
  "confidence": float between 0.0 and 1.0,
  "reasoning": "short plain-language explanation of observed visual vehicle damage"
}

Classification guidelines for damage_level:
- none: No visible collision impact or minor cosmetic scuff.
- minor: Dented fender, scratched paint, broken headlight.
- moderate: Dented doors, crumpled bumper, minor engine hood bend.
- severe: Crushed engine bay, shattered windshield, deployed airbags, major body deformation.
- critical: Rollover deformation, structural cabin crush, heavy metal tearing.

Return ONLY the JSON object. Do not include markdown code blocks.
"""


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _assess_with_gemini(photo_path: str, api_key: str) -> Dict[str, Any]:
    """
    Sends the image to Gemini using the google-genai SDK (same one
    llm_engine.py uses), trying a few candidate models in sequence since a
    single model's free-tier quota can be exhausted while others work.
    """
    from google import genai
    from PIL import Image

    client = genai.Client(api_key=api_key)
    img = Image.open(photo_path)

    candidate_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"]
    last_error = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name, contents=[VISION_PROMPT, img],
            )
            content = _strip_markdown_fence(response.text)
            return json.loads(content)
        except Exception as e:
            last_error = e
            continue
    raise last_error if last_error else RuntimeError("all candidate Gemini models failed")


def _assess_with_openai(photo_path: str, api_key: str) -> Dict[str, Any]:
    """Sends image to OpenAI GPT-4o vision model (used only if OPENAI_API_KEY
    is set — this project's primary configured provider is Gemini)."""
    import base64
    import openai

    client = openai.OpenAI(api_key=api_key)
    with open(photo_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
            ],
        }],
        temperature=0.2,
        max_tokens=300,
    )
    content = _strip_markdown_fence(response.choices[0].message.content)
    return json.loads(content)


def assess_vehicle_damage(photo_path: str) -> Dict[str, Any]:
    """
    Assesses visible vehicle damage from a photo using available vision LLM
    APIs, with a safe fallback if no key is present or every call fails.
    Gemini is tried first (matches this project's configured key), OpenAI
    second (only relevant if OPENAI_API_KEY happens to also be set).
    """
    if not photo_path or not os.path.exists(photo_path):
        return {
            "damage_level": "unknown",
            "indicators_observed": [],
            "confidence": 0.0,
            "reasoning": "No valid photo file provided for visual damage assessment.",
            "disclaimer": ETHICS_DISCLAIMER,
            "_source": "no_photo",
        }

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if gemini_key:
        try:
            result = _assess_with_gemini(photo_path, gemini_key)
            result["disclaimer"] = ETHICS_DISCLAIMER
            result["_source"] = "gemini"
            return result
        except Exception as e:
            print(f"[Damage Assessment Warning] Gemini Vision API failed: {e}")

    if openai_key:
        try:
            result = _assess_with_openai(photo_path, openai_key)
            result["disclaimer"] = ETHICS_DISCLAIMER
            result["_source"] = "openai"
            return result
        except Exception as e:
            print(f"[Damage Assessment Warning] OpenAI Vision API failed: {e}")

    return {
        "damage_level": "unknown",
        "indicators_observed": ["Visual AI API unavailable or not configured"],
        "confidence": 0.0,
        "reasoning": "Vision API unavailable; severity determined solely by tracking-based speed/class signals.",
        "disclaimer": ETHICS_DISCLAIMER,
        "_source": "fallback",
    }


# ---------------------------------------------------------------------------
# Severity integration — lets the Response Planner combine this module's
# output with its own speed/class-based severity estimate, ESCALATE ONLY.
# ---------------------------------------------------------------------------
_DAMAGE_TO_SEVERITY = {
    "none": "moderate",
    "minor": "moderate",
    "moderate": "moderate",
    "severe": "severe",
    "critical": "critical",
    "unknown": None,  # never escalates on an unknown/failed assessment
}
_SEVERITY_RANK = {"moderate": 1, "severe": 2, "critical": 3}


def damage_to_severity_level(damage_level: str) -> Optional[str]:
    """Maps this module's 5-level damage scale onto the Response Planner's
    3-level severity scale (moderate/severe/critical). Returns None if the
    damage level shouldn't influence severity at all (e.g. "unknown")."""
    return _DAMAGE_TO_SEVERITY.get(damage_level)


def combine_severity(existing_severity: str, damage_level: str) -> str:
    """
    Returns the HIGHER of the existing (speed/class-based) severity and the
    damage-implied severity — never lower. A severe-looking photo can raise
    the response level; a mild-looking photo never downgrades a severity
    already justified by other signals (e.g. a pedestrian was involved).
    """
    damage_severity = damage_to_severity_level(damage_level)
    if damage_severity is None:
        return existing_severity
    existing_rank = _SEVERITY_RANK.get(existing_severity, 1)
    damage_rank = _SEVERITY_RANK.get(damage_severity, 1)
    return existing_severity if existing_rank >= damage_rank else damage_severity


# ---------------------------------------------------------------------------
# Standalone self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing damage_assessment.py...")
    test_img = "latest_frame.jpg"
    if os.path.exists(test_img):
        res = assess_vehicle_damage(test_img)
        print(json.dumps(res, indent=2))
        print("\nCombined severity example (existing='moderate'):",
              combine_severity("moderate", res["damage_level"]))
    else:
        print("No test image found — testing fallback path with a missing file instead.")
        res = assess_vehicle_damage("does_not_exist.jpg")
        print(json.dumps(res, indent=2))
