

import json
import math
import os
import time

STATE_FILE = "incident_state.json"


AMBULANCE_DRIVERS = [
    {"id": "AMB-01", "name": "R. Suresh Kumar", "phone": "+91-90000-11111",
     "vehicle_no": "TN 45 AM 1023", "lat": 10.7610, "lon": 78.8130},
    {"id": "AMB-02", "name": "K. Manikandan", "phone": "+91-90000-22222",
     "vehicle_no": "TN 45 AM 2045", "lat": 10.7560, "lon": 78.8005},
    {"id": "AMB-03", "name": "P. Elumalai", "phone": "+91-90000-33333",
     "vehicle_no": "TN 45 AM 3067", "lat": 10.8010, "lon": 78.6900},
]

SIM_TRAVEL_SECONDS = 45

WORKFLOW_STEPS = [
    "pending_ambulance_accept",
    "ambulance_accepted",
    "en_route_to_scene",
    "arrived_at_scene",
    "en_route_to_hospital",
    "pending_hospital_accept",
    "hospital_accepted",
    "arrived_at_hospital",
]
STEP_LABELS = {
    "pending_ambulance_accept": "Waiting for driver to accept",
    "ambulance_accepted": "Driver accepted — preparing to depart",
    "en_route_to_scene": "En route to incident location",
    "arrived_at_scene": "Arrived at scene",
    "en_route_to_hospital": "Transporting patient to hospital",
    "pending_hospital_accept": "Waiting for hospital to accept patient",
    "hospital_accepted": "Hospital accepted incoming patient",
    "arrived_at_hospital": "Arrived at hospital",
}


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _assign_nearest_driver(lat, lon, state):
    """Nearest driver not currently mid-job in another active incident."""
    busy_ids = {
        v.get("ambulance", {}).get("id")
        for v in state.values()
        if v.get("workflow_status") != "arrived_at_hospital"
    }
    candidates = [d for d in AMBULANCE_DRIVERS if d["id"] not in busy_ids] or AMBULANCE_DRIVERS
    return min(candidates, key=lambda d: _haversine_km(lat, lon, d["lat"], d["lon"]))


def sync_new_incidents(incidents):
    """
    Call every dashboard refresh: registers any incident from
    incidents_log.jsonl not yet tracked, auto-assigning the nearest driver.
    """
    state = load_state()
    changed = False
    for incident in incidents:
        iid = incident["incident_id"]
        if iid not in state:
            lat, lon = incident["location"]["lat"], incident["location"]["lon"]
            driver = _assign_nearest_driver(lat, lon, state)
            state[iid] = {
                "incident_id": iid,
                "severity": incident["severity"],
                "object_classes": incident["object_classes"],
                "location": incident["location"],
                "recommended_hospital": incident["recommended_hospital"],
                "photo_path": incident.get("photo_path"),
                "timestamp": incident["timestamp"],
                "ambulance": driver,
                "workflow_status": "pending_ambulance_accept",
                "history": [{"status": "pending_ambulance_accept", "at": time.time()}],
            }
            changed = True
    if changed:
        save_state(state)
    return state


def advance_status(incident_id, new_status):
    state = load_state()
    if incident_id in state:
        state[incident_id]["workflow_status"] = new_status
        state[incident_id].setdefault("history", []).append({"status": new_status, "at": time.time()})
        save_state(state)
    return state


def current_ambulance_position(record):
   
    driver = record["ambulance"]
    status = record["workflow_status"]
    loc = record["location"]
    hospital = record["recommended_hospital"]

    def _elapsed_since(target_status):
        for h in reversed(record.get("history", [])):
            if h["status"] == target_status:
                return time.time() - h["at"]
        return 0

    def _lerp(a, b, t):
        return a + (b - a) * max(0.0, min(1.0, t))

    if status == "en_route_to_scene":
        t = _elapsed_since("en_route_to_scene") / SIM_TRAVEL_SECONDS
        return _lerp(driver["lat"], loc["lat"], t), _lerp(driver["lon"], loc["lon"], t)
    if status in ("arrived_at_scene", "pending_hospital_accept"):
        return loc["lat"], loc["lon"]
    if status == "en_route_to_hospital":
        t = _elapsed_since("en_route_to_hospital") / SIM_TRAVEL_SECONDS
        h_lat = hospital.get("lat", loc["lat"])
        h_lon = hospital.get("lon", loc["lon"])
        return _lerp(loc["lat"], h_lat, t), _lerp(loc["lon"], h_lon, t)
    if status in ("hospital_accepted", "arrived_at_hospital"):
        return hospital.get("lat", loc["lat"]), hospital.get("lon", loc["lon"])
    return driver["lat"], driver["lon"]  # pending_ambulance_accept / ambulance_accepted
