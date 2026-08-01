

import argparse
import itertools
import json
import os
import shutil
import time
from datetime import datetime

from motion_forecast import TrackManager
from risk_engine import scene_understanding, fuzzy_risk, StateManager, CLASS_WEIGHT
from collision_confidence import CollisionConfidenceEstimator
from response_planner import ResponsePlanner
from emergency_call import place_emergency_call
from llm_engine import generate_incident_reports
from damage_assessment import assess_vehicle_damage



CAMERA_LAT = 10.7597
CAMERA_LON = 78.8147
EVIDENCE_DIR = "incident_evidence"

EMERGENCY_CONTACT_NUMBER = "+919000000000"
ACTIVITY_LOG_FILE = "activity_log.jsonl"


def log_activity(event_type, message, incident_id=None):
   
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event_type": event_type,   # e.g. "collision_confirmed", "dispatch",
                                     # "emergency_call", "llm_reports"
        "message": message,
        "incident_id": incident_id,
    }
    try:
        with open(ACTIVITY_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # never let logging itself break the pipeline


def capture_evidence_photo(incident_id):
    
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    src = "latest_frame.jpg"
    if not os.path.exists(src):
        return None
    dest = os.path.join(EVIDENCE_DIR, f"incident_{incident_id}.jpg")
    shutil.copyfile(src, dest)
    return dest


def dispatch_to_medical_department(incident, photo_path, lat, lon):
   
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    packet = {
        "location": {"lat": lat, "lon": lon, "maps_link": maps_link},
        "photo_evidence": photo_path or "(no frame captured)",
        "severity": incident["severity"],
        "recommended_hospital": incident["recommended_hospital"]["name"],
        "hospital_eta_min": incident["recommended_hospital"]["eta_min"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print("    ---- SIMULATED DISPATCH TO MEDICAL DEPARTMENT ----")
    print(f"    Location: {lat}, {lon}  ->  {maps_link}")
    print(f"    Photo evidence: {packet['photo_evidence']}")
    print(f"    Severity: {packet['severity']}  |  "
          f"Recommended: {packet['recommended_hospital']} "
          f"(ETA {packet['hospital_eta_min']} min)")
    print("    (This is a simulated packet — wire this function to a real")
    print("     SMS/email/hospital API, e.g. smtplib or Twilio, to make it live)")
    print("    ---------------------------------------------------\n")
    return packet


def handle_confirmed_collision(planner, meta_a, meta_b, features, cce_votes,
                                incident_lat=None, incident_lon=None):
    
    lat = incident_lat if incident_lat is not None else CAMERA_LAT
    lon = incident_lon if incident_lon is not None else CAMERA_LON

    incident_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    photo_path = capture_evidence_photo(incident_id)
    damage_result = assess_vehicle_damage(photo_path)
    log_activity(
        "damage_assessment",
        f"Damage level: {damage_result['damage_level']} "
        f"(source: {damage_result.get('_source', 'unknown')})",
        incident_id,
    )

    peak_speed = features.get("relative_speed", 0.0)
    incident = planner.plan(
        incident_lat=lat, incident_lon=lon,
        class_a=meta_a.get("class_name", "person"),
        class_b=meta_b.get("class_name", "person"),
        peak_relative_speed=peak_speed,
        cce_votes=cce_votes,
        damage_assessment=damage_result,
    )

    print("\n    ========== RESPONSE PLANNER ==========")
    print(f"    Severity: {incident['severity']}")
    print(f"    Recommended hospital: {incident['recommended_hospital']['name']} "
          f"(ETA {incident['recommended_hospital']['eta_min']} min, "
          f"trauma_center={incident['recommended_hospital']['trauma_center']}, "
          f"icu={incident['recommended_hospital']['icu']})")
    nearest = incident['nearest_hospital_by_distance_only']
    if nearest['name'] != incident['recommended_hospital']['name']:
        print(f"    (Nearest-by-distance-only would have been: {nearest['name']}, "
              f"ETA {nearest['eta_min']} min, trauma_center={nearest['trauma_center']} "
              f"— capability-aware routing chose better)")
    print("    =======================================\n")
    log_activity(
        "response_planner",
        f"Severity {incident['severity'].upper()} — recommended "
        f"{incident['recommended_hospital']['name']} "
        f"(ETA {incident['recommended_hospital']['eta_min']} min)",
        incident_id,
    )

    dispatch_packet = dispatch_to_medical_department(incident, photo_path, lat, lon)
    log_activity(
        "dispatch",
        f"Simulated dispatch sent to medical department — location {lat}, {lon}",
        incident_id,
    )

    
    call_result = place_emergency_call(EMERGENCY_CONTACT_NUMBER, incident)
    call_mode = call_result.get("mode", "unknown")
    if call_mode == "live":
        log_activity("emergency_call", f"Live call placed to {EMERGENCY_CONTACT_NUMBER}", incident_id)
    elif call_mode == "dry_run":
        log_activity("emergency_call", f"[MOCK] Automated voice call triggered (Twilio not configured)", incident_id)
    else:
        log_activity("emergency_call", f"Call attempt failed or errored: {call_result.get('error', 'unknown')}", incident_id)

    
    tracking_notes = {
        "ghost_tracked_object_a": bool(features.get("is_ghost_a", False)),
        "ghost_tracked_object_b": bool(features.get("is_ghost_b", False)),
        "near_blind_spot": features.get("blind_spot_factor", 1.0) > 1.0,
    }

    
    incident_for_reports = dict(incident)
    incident_for_reports["timestamp"] = dispatch_packet["timestamp"]
    incident_for_reports["location"] = {"lat": lat, "lon": lon}
    incident_for_reports["tracking_notes"] = tracking_notes
    reports = generate_incident_reports(incident_for_reports)
    report_source = reports.pop("_source", "unknown")
    print("    ========== LLM COMMUNICATION ENGINE ==========")
    print(f"    (source: {report_source})")
    for key, text in reports.items():
        print(f"    [{key}] {text}")
    print("    ================================================\n")
    log_activity(
        "llm_reports",
        f"Generated 4 role reports (source: {report_source})",
        incident_id,
    )

    
    log_entry = {
        "incident_id": incident_id,
        "timestamp": dispatch_packet["timestamp"],
        "severity": incident["severity"],
        "object_classes": incident["object_classes"],
        "recommended_hospital": incident["recommended_hospital"],
        "nearest_hospital_by_distance_only": incident["nearest_hospital_by_distance_only"],
        "location": {"lat": lat, "lon": lon},
        "photo_path": photo_path,
        "reports": reports,
        "reports_source": report_source,
        "tracking_notes": tracking_notes,
        "damage_assessment": damage_result,
    }
    with open("incidents_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return incident


def process_frame(track_manager, state_manager, cce, bbox_lookup, detections, timestamp, critical_latch, critical_streak, planner=None):
    
    LATCH_SECONDS = 1.5
    CRITICAL_STREAK_REQUIRED = 3   # consecutive critical readings needed before
                                    # we trust it — filters out single-frame
                                    # jitter, especially at close range where
                                    # TTC is naturally noisy
    MIN_BBOX_AREA = 2500       # px^2 (e.g. 50x50) — smaller boxes are treated
                               # as too distant/unreliable to score for risk;
                               # tune this against your camera's resolution
    MAX_SIZE_RATIO = 6.0       # if one box is this many times larger than the
                               # other, they're very unlikely to be at the same
                               # real-world depth (a fix for the no-depth-sensor
                               # limitation of a single monocular camera)

    def bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    frame_status = {"timestamp": timestamp, "frame_id": detections[0].get("frame_id") if detections else None, "pairs": []}

    track_manager.update(detections, timestamp)
    active_ids = track_manager.active_ids()

    for oid in active_ids:
        if oid in bbox_lookup:
            track = track_manager.get(oid)
            vx, vy = track.velocity()
            cce.update_history(oid, timestamp, bbox_lookup[oid], vx, vy)

    def latest_bbox(oid):
       
        if oid in bbox_lookup:
            return bbox_lookup[oid]
        hist = cce.tracker.get(oid)
        return hist[-1]["bbox"] if hist else None

    for id_a, id_b in itertools.combinations(active_ids, 2):
        
        bbox_a_latest = latest_bbox(id_a)
        bbox_b_latest = latest_bbox(id_b)
        if bbox_a_latest and bbox_b_latest:
            area_a = bbox_area(bbox_a_latest)
            area_b = bbox_area(bbox_b_latest)
            if area_a < MIN_BBOX_AREA or area_b < MIN_BBOX_AREA:
                continue
            size_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1.0)
            if size_ratio > MAX_SIZE_RATIO:
                continue

        track_a = track_manager.get_view(id_a, timestamp)
        track_b = track_manager.get_view(id_b, timestamp)
        meta_a = track_manager.get_meta(id_a)
        meta_b = track_manager.get_meta(id_b)

        features = scene_understanding(track_a, track_b)

        class_weight_avg = (
            CLASS_WEIGHT.get(meta_a.get("class_name"), 0.6)
            + CLASS_WEIGHT.get(meta_b.get("class_name"), 0.6)
        ) / 2.0
        confidence_avg = (
            meta_a.get("confidence", 0.8) + meta_b.get("confidence", 0.8)
        ) / 2.0

        risk = fuzzy_risk(features, class_weight_avg, confidence_avg)
        tier = state_manager.route(id_a, id_b, meta_a, meta_b, features, risk)

        frame_status["pairs"].append({
            "object_a": id_a, "object_b": id_b,
            "class_a": meta_a.get("class_name"), "class_b": meta_b.get("class_name"),
            "risk": round(risk, 3), "tier": tier, "ttc": round(features["ttc"], 2),
        })

    
        pair_key = (id_a, id_b)
        if tier == "critical":
            critical_streak[pair_key] = critical_streak.get(pair_key, 0) + 1
        else:
            critical_streak[pair_key] = 0

        sustained_critical = critical_streak.get(pair_key, 0) >= CRITICAL_STREAK_REQUIRED

        if sustained_critical:
            critical_latch[pair_key] = timestamp + LATCH_SECONDS

        still_in_window = critical_latch.get(pair_key, -1) >= timestamp
        if still_in_window and id_a in bbox_lookup and id_b in bbox_lookup:
            confirmed, votes, signals = cce.evaluate(
                id_a, id_b, bbox_lookup[id_a], bbox_lookup[id_b], features
            )
            print(f"    -> Collision Confidence Estimator: {votes}/5 signals "
                  f"{'CONFIRMED' if confirmed else '(not yet confirmed)'} {signals}")
            if confirmed:
                print(f"    >>> COLLISION CONFIRMED between {id_a} and {id_b} <<<")
                log_activity(
                    "collision_confirmed",
                    f"Collision confirmed between {meta_a.get('class_name','object')} {id_a} "
                    f"and {meta_b.get('class_name','object')} {id_b} ({votes}/5 signals)",
                )
                del critical_latch[pair_key]  # stop re-confirming the same event
                if planner is not None:
                    handle_confirmed_collision(planner, meta_a, meta_b, features, votes)

    
    try:
        with open("status.json", "w") as f:
            json.dump(frame_status, f)
    except OSError:
        pass  


def run_live(jsonl_path, poll_interval=0.05):
    """Tails a growing JSONL file written by perception.py's --out flag."""
    track_manager = TrackManager()
    state_manager = StateManager()
    cce = CollisionConfidenceEstimator()
    planner = ResponsePlanner()
    critical_latch = {}
    critical_streak = {}

    print(f"Watching {jsonl_path} for new detections... (Ctrl+C to stop)")
    print("(Press Ctrl+C then run with --force-collision if you need a manual demo trigger)")
    with open(jsonl_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_interval)
                continue
            record = json.loads(line)
            bbox_lookup = {record["object_id"]: record["bbox"]}
            process_frame(
                track_manager, state_manager, cce, bbox_lookup,
                detections=[record], timestamp=record["timestamp"],
                critical_latch=critical_latch, critical_streak=critical_streak,
                planner=planner,
            )


def run_demo():
   
    track_manager = TrackManager()
    state_manager = StateManager(ledger_path="near_miss_ledger_demo.csv")
    cce = CollisionConfidenceEstimator()
    planner = ResponsePlanner()
    critical_latch = {}
    critical_streak = {}

    print("Running synthetic demo: person and car converging on a crossing point,")
    print("continuing through to a simulated collision.\n")

    dt = 0.1
    person_x, person_y = 50.0, 300.0
    car_x, car_y = 350.0, 300.0
    person_vx, car_vx = 60.0, -60.0
    collided = False
    frames_since_collision = None

    for i in range(70):
        t = round(i * dt, 2)

        if not collided and person_x + 15 >= car_x - 15:
            collided = True
            frames_since_collision = 0
            person_vx, car_vx = -20.0, 20.0  # brief recoil
        elif collided:
            frames_since_collision += 1
            decay = max(0.0, 1.0 - frames_since_collision / 4.0)
            person_vx, car_vx = -20.0 * decay, 20.0 * decay
        else:
            person_x += person_vx * dt
            car_x += car_vx * dt

        person_bbox = [person_x - 40, person_y - 60, person_x + 40, person_y + 60]
        car_bbox = [car_x - 60, car_y - 40, car_x + 60, car_y + 40]

        detections = [
            {"object_id": 1, "class_name": "person", "confidence": 0.9,
             "center": [person_x, person_y]},
            {"object_id": 2, "class_name": "car", "confidence": 0.95,
             "center": [car_x, car_y]},
        ]
        bbox_lookup = {1: person_bbox, 2: car_bbox}

        print(f"t={t:4.1f}s  person_x={person_x:6.1f}  car_x={car_x:6.1f}")
        process_frame(track_manager, state_manager, cce, bbox_lookup, detections, t,
                       critical_latch, critical_streak, planner=planner)

    print("\nDemo complete. Check near_miss_ledger_demo.csv for logged near-misses.")


def run_force_collision():
    
    planner = ResponsePlanner()
    print("Manually triggering a confirmed-collision scenario (demo safety net)...\n")
    meta_a = {"class_name": "person", "confidence": 0.9}
    meta_b = {"class_name": "car", "confidence": 0.9}
    features = {"relative_speed": 220.0}
    handle_confirmed_collision(planner, meta_a, meta_b, features, cce_votes=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veganodi core intelligence")
    parser.add_argument("--demo", action="store_true", help="run synthetic self-test, no camera needed")
    parser.add_argument("--jsonl", default=None, help="path to detections.jsonl produced by perception.py")
    parser.add_argument("--force-collision", action="store_true",
                         help="manually trigger the Response Planner for a demo safety net")
    args = parser.parse_args()

    if args.force_collision:
        run_force_collision()
    elif args.demo:
        run_demo()
    elif args.jsonl:
        run_live(args.jsonl)
    else:
        print("Specify --demo to test without a camera, or --jsonl <file> for live mode.")
