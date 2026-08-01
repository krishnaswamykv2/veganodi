

import json
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

REPORT_KEYS = ["dispatcher_report", "traffic_control_report", "police_report", "citizen_alert"]

SYSTEM_PROMPT = """You are the Veganodi Communication Engine. You will be given a
structured incident packet (JSON) describing a confirmed vehicle/pedestrian
collision detected by an AI monitoring system. Your ONLY job is to write four
short reports from this data, for four different audiences. You must NOT add,
invent, or guess any fact not present in the given data — severity, hospital
name, location, object classes, and ETA are all fixed and must be used exactly
as given, never altered.

The packet may include a "tracking_notes" object with boolean fields
ghost_tracked_object_a, ghost_tracked_object_b, near_blind_spot. These describe
the AI TRACKING METHOD (e.g. whether an object's position was predicted after
briefly leaving camera view), NOT the collision itself. Only mention this in
the police_report, only if at least one field is true, and only as a brief,
factual note (e.g. "one party's position was tracked via predictive
extrapolation after briefly leaving camera view"). Never mention tracking_notes
in the other three reports, and never mention it at all if every field is false.

The packet may also include a "damage_assessment" object with damage_level,
indicators_observed, and a disclaimer. If damage_level is present and NOT
"unknown", briefly mention the visible vehicle damage in dispatcher_report
(helps responders anticipate what they'll find) and police_report (factual
record). Use ONLY the indicators_observed list given — never invent damage
details not present in the data. ALWAYS use the exact word "estimated" or
"visible" when referencing damage (e.g. "visible front-end damage observed"),
and NEVER phrase it as a medical/injury assessment of the people involved —
this module assesses vehicle damage only, never human injury. If damage_level
is "unknown" or missing, do not mention damage assessment at all.

Write exactly these four reports, each 2-4 sentences:

1. dispatcher_report: concise, actionable, for an emergency dispatcher deciding
   what resources to send. Include severity, location, object classes involved,
   and the recommended hospital with ETA.
2. traffic_control_report: focused on what signal/lane action traffic control
   should take near the incident location.
3. police_report: factual incident record — timestamp, severity, parties
   involved, location, note that evidence photo/video exists, plus the
   tracking_notes disclosure described above if relevant.
4. citizen_alert: plain language, no technical jargon, no personal data, calm
   and clear — tell nearby the public there's an incident and emergency
   services are responding.

Respond ONLY with valid JSON in exactly this shape, nothing else:
{"dispatcher_report": "...", "traffic_control_report": "...", "police_report": "...", "citizen_alert": "..."}
"""


def _fallback_reports(incident):
    """Hardcoded, template-based reports built from real incident data —
    used if the LLM API isn't configured or fails. Never invents facts;
    just less naturally phrased than the LLM version."""
    severity = incident["severity"]
    classes = " and ".join(incident["object_classes"])
    hospital = incident["recommended_hospital"]
    lat, lon = incident.get("location", {}).get("lat"), incident.get("location", {}).get("lon")
    timestamp = incident.get("timestamp", "unknown time")

    tracking_notes = incident.get("tracking_notes", {})
    tracking_disclosure = ""
    if tracking_notes.get("ghost_tracked_object_a") or tracking_notes.get("ghost_tracked_object_b"):
        tracking_disclosure = " One party's position was tracked via predictive extrapolation after briefly leaving camera view."
    elif tracking_notes.get("near_blind_spot"):
        tracking_disclosure = " One party was near the edge of the camera's field of view at time of recording."

    damage = incident.get("damage_assessment") or {}
    damage_disclosure = ""
    if damage.get("damage_level") and damage["damage_level"] != "unknown":
        indicators = ", ".join(damage.get("indicators_observed", [])) or "visible external damage"
        damage_disclosure = f" Estimated vehicle damage: {damage['damage_level']} ({indicators})."

    return {
        "dispatcher_report": (
            f"{severity.upper()} severity incident involving {classes}. "
            f"Recommended hospital: {hospital['name']} (ETA {hospital['eta_min']} min, "
            f"trauma_center={hospital['trauma_center']}, icu={hospital['icu']}). "
            f"Location: {lat}, {lon}. Immediate dispatch recommended.{damage_disclosure}"
        ),
        "traffic_control_report": (
            f"Incident detected at monitored location ({lat}, {lon}). "
            f"Recommend holding cross-traffic and clearing a corridor for "
            f"emergency vehicle access. Severity: {severity}."
        ),
        "police_report": (
            f"Incident recorded at {timestamp}. Severity: {severity}. "
            f"Parties involved: {classes}. Location: {lat}, {lon}. "
            f"Evidence photo captured and retained for review."
            f"{tracking_disclosure}{damage_disclosure}"
        ),
        "citizen_alert": (
            f"An incident has been detected near your area. Emergency services "
            f"have been notified and are responding. Please avoid the area and "
            f"follow instructions from personnel if present."
        ),
    }


def generate_incident_reports(incident):
    
    if not GEMINI_API_KEY:
        reports = _fallback_reports(incident)
        reports["_source"] = "fallback_template_no_api_key"
        return reports

    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        incident_json = json.dumps(incident, default=str)
        prompt = f"{SYSTEM_PROMPT}\n\nIncident packet:\n{incident_json}"

        
        candidate_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"]
        last_error = None

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                parsed = json.loads(text.strip())
                if not all(k in parsed for k in REPORT_KEYS):
                    raise ValueError("LLM response missing one or more required report keys")
                parsed["_source"] = f"llm ({model_name})"
                return parsed
            except Exception as e:
                last_error = e
                continue  # try the next model

        raise last_error if last_error else RuntimeError("all candidate models failed")

    except Exception as e:
        reports = _fallback_reports(incident)
        reports["_source"] = f"fallback_template_error: {e}"
        return reports



if __name__ == "__main__":
    fake_incident = {
        "severity": "critical",
        "object_classes": ["person", "car"],
        "recommended_hospital": {
            "name": "BHEL Hospital, Tiruchirappalli", "eta_min": 3.2,
            "trauma_center": True, "icu": True,
        },
        "location": {"lat": 10.7597, "lon": 78.8147},
        "timestamp": "2026-08-01T10:15:00",
    }
    reports = generate_incident_reports(fake_incident)
    print(f"Source: {reports.pop('_source')}\n")
    for key in REPORT_KEYS:
        print(f"--- {key} ---")
        print(reports[key])
        print()
