

import csv
import json
import os
import time
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import ambulance_fleet
import response_planner
from core_intelligence import handle_confirmed_collision, ResponsePlanner

app = FastAPI(title="PreCrash Emergency Intelligence API", version="2.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AdvanceStatusRequest(BaseModel):
    incident_id: str
    new_status: str


class ForceCollisionRequest(BaseModel):
    class_a: Optional[str] = "person"
    class_b: Optional[str] = "car"
    speed: Optional[float] = 220.0


def reset_demo_state():
    """Resets all demo state files so each server run starts fresh."""
  
    try:
        with open("incidents_log.jsonl", "w") as f:
            f.write("")
    except Exception:
        pass

  
    try:
        ambulance_fleet.save_state({})
    except Exception:
        pass

   
    try:
        with open("status.json", "w") as f:
            json.dump({"pairs": [], "timestamp": time.time()}, f)
    except Exception:
        pass

    if os.path.exists("incident_evidence"):
        for fname in os.listdir("incident_evidence"):
            try:
                os.remove(os.path.join("incident_evidence", fname))
            except Exception:
                pass
    return {"status": "reset_complete", "timestamp": time.time()}


@app.on_event("startup")
def startup_event():
    reset_demo_state()


@app.get("/api/health")
def health_check():
    return {"status": "online", "timestamp": time.time()}


@app.post("/api/reset-demo")
def api_reset_demo():
    return reset_demo_state()


from calibration_utils import load_calibration

@app.get("/api/system-telemetry")
def get_system_telemetry():
    matrix, calib_meta = load_calibration()
    is_calibrated = matrix is not None
    return {
        "camera_node": {
            "id": "CAM-NIT-TRICHY-04",
            "location": "NIT Trichy Main Junction",
            "status": "Online",
            "calibration": "Calibrated (Homography Matrix Active)" if is_calibrated else "Uncalibrated (Pixel Estimates)",
            "unit": "meters" if is_calibrated else "pixels",
            "fps": 30.0,
            "resolution": "1920x1080",
            "latency_ms": 14.2
        },
        "perception_engine": {
            "model": "YOLOv8n + ByteTrack",
            "status": "Active",
            "conf_thresh": 0.35
        },
        "risk_engine": {
            "mode": "Fuzzy Triangular Sugeno",
            "status": "Active (Homography Ground-Plane)" if is_calibrated else "Active (Pixel Projection)",
            "ttc_horizon_s": 4.0
        },
        "traffic_signal_system": {
            "status": "Priority Override Standby",
            "mode": "Smart Corridor Auto-Clear",
            "active_corridor": "NH-67 Eastbound"
        },
        "lora_mesh": {
            "status": "Connected",
            "nodes_active": 12,
            "signal_strength_dbm": -68
        }
    }


@app.get("/api/status")
def get_status():
    if not os.path.exists("status.json"):
        return {"pairs": [], "timestamp": time.time()}
    try:
        with open("status.json", "r") as f:
            return json.load(f)
    except Exception:
        return {"pairs": [], "timestamp": time.time()}


@app.get("/api/ledger")
def get_ledger(limit: int = 50):
    if not os.path.exists("near_miss_ledger.csv"):
        return []
    rows = []
    try:
        with open("near_miss_ledger.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["risk_score"] = float(row.get("risk_score", 0))
                    row["ttc_s"] = float(row.get("ttc_s", 0))
                    row["distance_px"] = float(row.get("distance_px", 0))
                    row["timestamp"] = float(row.get("timestamp", 0))
                    if "midpoint_x" in row and row["midpoint_x"]:
                        row["midpoint_x"] = float(row["midpoint_x"])
                    if "midpoint_y" in row and row["midpoint_y"]:
                        row["midpoint_y"] = float(row["midpoint_y"])
                    if "midpoint_lat" in row and row["midpoint_lat"]:
                        row["midpoint_lat"] = float(row["midpoint_lat"])
                    if "midpoint_lon" in row and row["midpoint_lon"]:
                        row["midpoint_lon"] = float(row["midpoint_lon"])
                except ValueError:
                    pass
                rows.append(row)
        return rows[-limit:][::-1]
    except Exception as e:
        return []


@app.get("/api/risk-heatmap")
def get_risk_heatmap():
    ledger = get_ledger(limit=100)
    points = []
    for r in ledger:
        px = float(r.get("midpoint_x", 320.0) or 320.0)
        py = float(r.get("midpoint_y", 240.0) or 240.0)
        lat = float(r.get("midpoint_lat", 10.7597) or 10.7597)
        lon = float(r.get("midpoint_lon", 78.8147) or 78.8147)
        risk = float(r.get("risk_score", 0.5) or 0.5)
        tier = r.get("tier", "elevated")
        points.append({
            "x": px,
            "y": py,
            "lat": lat,
            "lon": lon,
            "risk": risk,
            "tier": tier
        })
    return {"points": points, "count": len(points)}


@app.get("/api/incidents")
def get_incidents():
    if not os.path.exists("incidents_log.jsonl"):
        return []
    incidents = []
    try:
        with open("incidents_log.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        incidents.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return incidents
    except Exception:
        return []


@app.get("/api/activity-log")
def get_activity_log(limit: int = 100):
    if not os.path.exists("activity_log.jsonl"):
        return []
    entries = []
    try:
        with open("activity_log.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-limit:][::-1]
    except Exception:
        return []



@app.get("/api/incident-state")
def get_incident_state():
    incidents = get_incidents()
    state = ambulance_fleet.sync_new_incidents(incidents)
   
    response_data = {}
    for iid, record in state.items():
        rec_copy = dict(record)
        amb_lat, amb_lon = ambulance_fleet.current_ambulance_position(record)
        rec_copy["current_ambulance_pos"] = {"lat": amb_lat, "lon": amb_lon}
        response_data[iid] = rec_copy
    return response_data


@app.post("/api/advance-status")
def advance_incident_status(req: AdvanceStatusRequest):
    if req.new_status not in ambulance_fleet.WORKFLOW_STEPS:
        raise HTTPException(status_code=400, detail=f"Invalid status: {req.new_status}")
    state = ambulance_fleet.advance_status(req.incident_id, req.new_status)
    if req.incident_id not in state:
        raise HTTPException(status_code=404, detail="Incident not found")
    record = state[req.incident_id]
    amb_lat, amb_lon = ambulance_fleet.current_ambulance_position(record)
    record["current_ambulance_pos"] = {"lat": amb_lat, "lon": amb_lon}
    return record


@app.post("/api/force-collision")
def force_collision(req: ForceCollisionRequest = Body(default=ForceCollisionRequest())):
    planner = ResponsePlanner()
    meta_a = {"class_name": req.class_a or "person", "confidence": 0.9}
    meta_b = {"class_name": req.class_b or "car", "confidence": 0.9}
    features = {"relative_speed": req.speed or 220.0}
    
    incident = handle_confirmed_collision(planner, meta_a, meta_b, features, cce_votes=4)
    
    
    incidents = get_incidents()
    already_logged = any(i.get("incident_id") == incident.get("incident_id") for i in incidents)
    if not already_logged:
        log_entry = {
            "incident_id": incident.get("incident_id", f"DEMO_{int(time.time())}"),
            "severity": incident["severity"],
            "object_classes": incident["object_classes"],
            "location": {"lat": 10.7597, "lon": 78.8147},
            "recommended_hospital": incident["recommended_hospital"],
            "nearest_hospital_by_distance_only": incident["nearest_hospital_by_distance_only"],
            "all_ranked_candidates": incident.get("all_ranked_candidates", []),
            "photo_path": os.path.join("incident_evidence", f"incident_{incident.get('incident_id', '')}.jpg"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open("incidents_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    
    updated_incidents = get_incidents()
    state = ambulance_fleet.sync_new_incidents(updated_incidents)
    return {"message": "Collision scenario confirmed and dispatched", "incident": incident, "state": state}


from fastapi.responses import FileResponse, Response, StreamingResponse


def _mjpeg_frame_generator():
    """Generates continuous MJPEG multipart stream from latest_frame.jpg for zero-stutter browser video rendering."""
    while True:
        if os.path.exists("latest_frame.jpg"):
            try:
                with open("latest_frame.jpg", "rb") as f:
                    jpeg_bytes = f.read()
                if jpeg_bytes:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n'
                    )
            except Exception:
                pass
        time.sleep(0.04) 


@app.get("/api/live-stream")
def live_stream():
    return StreamingResponse(
        _mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/latest-frame")
def get_latest_frame():
    if os.path.exists("latest_frame.jpg"):
        return FileResponse(
            "latest_frame.jpg",
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
        )
    raise HTTPException(status_code=404, detail="No camera frame available yet")


@app.get("/api/evidence/{filename}")
def get_evidence_photo(filename: str):
    safe_path = os.path.join("incident_evidence", os.path.basename(filename))
    if os.path.exists(safe_path):
        return FileResponse(safe_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Evidence photo not found")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    role: str
    message: str
    history: Optional[List[ChatMessage]] = []


def _build_grounded_context(role: str) -> tuple[str, str]:
    incidents = get_incidents()
    state = ambulance_fleet.load_state()
    status_data = get_status()
    ledger_data = get_ledger(limit=10)
    telemetry = get_system_telemetry()
    
    context_lines = []
    context_lines.append(f"System Telemetry: Camera={telemetry['camera_node']['status']}, Perception={telemetry['perception_engine']['status']}, Signals={telemetry['traffic_signal_system']['status']}")
    context_lines.append(f"Active Monitored Risk Pairs Count: {len(status_data.get('pairs', []))}")
    if status_data.get('pairs'):
        for p in status_data['pairs'][:3]:
            context_lines.append(f"  - Pair: {p.get('class_a')} #{p.get('object_a')} <-> {p.get('class_b')} #{p.get('object_b')}, Risk={p.get('risk')}, Tier={p.get('tier')}, TTC={p.get('ttc')}s")
            
    context_lines.append(f"Logged Near-Miss Count: {len(ledger_data)}")
    context_lines.append(f"Confirmed Incidents Count: {len(incidents)}")
    
    if incidents:
        latest_inc = incidents[-1]
        context_lines.append(f"Latest Confirmed Incident ID: {latest_inc.get('incident_id')}")
        context_lines.append(f"  - Severity: {latest_inc.get('severity')}")
        context_lines.append(f"  - Objects Involved: {', '.join(latest_inc.get('object_classes', []))}")
        context_lines.append(f"  - Location: Lat {latest_inc.get('location', {}).get('lat')}, Lon {latest_inc.get('location', {}).get('lon')}")
        rec = latest_inc.get("recommended_hospital", {})
        context_lines.append(f"  - Recommended Hospital: {rec.get('name')} (ETA {rec.get('eta_min')} min, Trauma={rec.get('trauma_center')}, ICU={rec.get('icu')})")
        nearest = latest_inc.get("nearest_hospital_by_distance_only", {})
        context_lines.append(f"  - Nearest Bypassed Hospital (by distance): {nearest.get('name')} (ETA {nearest.get('eta_min')} min, Trauma={nearest.get('trauma_center')})")
        dmg = latest_inc.get("damage_assessment", {})
        context_lines.append(f"  - Visual Damage Level: {dmg.get('damage_level', 'unknown')}, Indicators={dmg.get('indicators_observed', [])}")
        context_lines.append(f"  - Collision Confidence Signals: {latest_inc.get('collision_confidence_votes', 4)} of 5 signals confirmed (Relative speed high, BBox overlap, Velocity collapse, Post-impact stillness)")

    if state:
        for iid, s_rec in state.items():
            amb = s_rec.get("ambulance", {})
            context_lines.append(f"Workflow State for Incident {iid}: Status={s_rec.get('workflow_status')}")
            if role != "citizen":
                context_lines.append(f"  - Assigned Driver: {amb.get('name')}, Vehicle={amb.get('vehicle_no')}, Phone={amb.get('phone')}")

    data_context_str = "\n".join(context_lines)

    if role == "police":
        sys_prompt = f"You are the Police Tactical AI Assistant for PreCrash Police Command.\nONLY answer using facts in the REAL BACKEND TELEMETRY CONTEXT below. Do NOT invent facts.\nIf asked something not in the context, say 'I don't have that information in the current telemetry logs.'\nKeep responses SHORT (2 to 4 sentences).\n\nREAL BACKEND CONTEXT:\n{data_context_str}"
    elif role == "ambulance":
        sys_prompt = f"You are the EMS Driver Dispatch AI Assistant for PreCrash Ambulance Fleet.\nONLY answer using facts in the REAL BACKEND TELEMETRY CONTEXT below. Do NOT invent facts.\nIf asked something not in the context, say 'I don't have that information in the current dispatch logs.'\nKeep responses SHORT (2 to 4 sentences).\n\nREAL BACKEND CONTEXT:\n{data_context_str}"
    elif role == "hospital":
        sys_prompt = f"You are the Clinical Emergency Triage AI Assistant for PreCrash Hospital Network.\nONLY answer using facts in the REAL BACKEND TELEMETRY CONTEXT below. Do NOT invent facts. Clearly explain capability matching (Trauma/ICU requirement vs plain distance).\nIf asked something not in the context, say 'I don't have that information in the current triage logs.'\nKeep responses SHORT (2 to 4 sentences).\n\nREAL BACKEND CONTEXT:\n{data_context_str}"
    elif role == "citizen":
        sys_prompt = f"You are the Public Safety AI Assistant for PreCrash Citizen Emergency Advisory.\nONLY use plain language. Do NOT use technical jargon or internal algorithms.\nSTRICT PRIVACY: NEVER disclose driver names, phone numbers, or private personnel data.\nONLY answer using facts in the REAL BACKEND CONTEXT below. Do NOT invent facts.\nIf asked something not in the context, say 'I don't have public information regarding that query.'\nKeep responses SHORT (2 to 3 sentences).\n\nREAL BACKEND CONTEXT:\n{data_context_str}"
    else:
        sys_prompt = f"You are the Mission Control AI Assistant for PreCrash Global Command Center.\nONLY answer using facts in the REAL BACKEND TELEMETRY CONTEXT below. Do NOT invent facts.\nIf asked something not in the context, say 'I don't have that information in the current telemetry logs.'\nKeep responses SHORT (2 to 4 sentences).\n\nREAL BACKEND CONTEXT:\n{data_context_str}"

    return sys_prompt, data_context_str


def _fallback_grounded_response(role: str, user_msg: str) -> str:
    """Smart grounded fallback assistant using real context data directly."""
    incidents = get_incidents()
    state = ambulance_fleet.load_state()
    latest_inc = incidents[-1] if incidents else None
    msg_lower = user_msg.lower()

    if role == "police":
        if "why" in msg_lower or "signal" in msg_lower or "cce" in msg_lower:
            return "Collision was confirmed by 4-of-5 fused signals: High Relative Speed, Bounding Box Overlap, Velocity Collapse, and Post-Impact Stillness."
        if "what happened" in msg_lower or "incident" in msg_lower or "fact" in msg_lower:
            if latest_inc:
                return f"Incident #{latest_inc.get('incident_id')} was confirmed at {latest_inc.get('timestamp')} involving {', '.join(latest_inc.get('object_classes', []))}. Severity is {latest_inc.get('severity').upper()} with {latest_inc.get('collision_confidence_votes', 4)}/5 collision signals verified."
            return "No confirmed collision incidents are currently logged in the police dispatch queue."
        if "evidence" in msg_lower or "photo" in msg_lower:
            if latest_inc and latest_inc.get("photo_path"):
                return f"Evidence snapshot recorded at {latest_inc.get('photo_path')}. Photo file is stored in incident_evidence directory."
            return "No evidence photo is logged for the current active query."

    elif role == "ambulance":
        if "where" in msg_lower or "route" in msg_lower or "go" in msg_lower:
            if latest_inc:
                rec = latest_inc.get("recommended_hospital", {})
                return f"Transport target is {rec.get('name')} (ETA {rec.get('eta_min')} min, {rec.get('distance_km')} km). Scene coordinates: Lat {latest_inc.get('location', {}).get('lat')}, Lon {latest_inc.get('location', {}).get('lon')}."
            return "No active dispatch trip route currently assigned."
        if "hospital" in msg_lower or "severity" in msg_lower:
            if latest_inc:
                rec = latest_inc.get("recommended_hospital", {})
                return f"Patient severity is {latest_inc.get('severity').upper()}. Target hospital: {rec.get('name')} (Trauma Unit: {'YES' if rec.get('trauma_center') else 'NO'})."
            return "No active hospital assignment logged."
        if "status" in msg_lower or "job" in msg_lower:
            if state:
                first_st = list(state.values())[0]
                return f"Current job status: {first_st.get('workflow_status').upper().replace('_', ' ')}. Driver: {first_st.get('ambulance', {}).get('name')} ({first_st.get('ambulance', {}).get('vehicle_no')})."
            return "Ambulance fleet status: On Standby."

    elif role == "hospital":
        if "why" in msg_lower or "recommend" in msg_lower or "closer" in msg_lower or "best" in msg_lower:
            if latest_inc:
                rec = latest_inc.get("recommended_hospital", {})
                nearest = latest_inc.get("nearest_hospital_by_distance_only", {})
                if rec.get("name") != nearest.get("name"):
                    return f"{rec.get('name')} was recommended because severity is {latest_inc.get('severity').upper()}, requiring Trauma Unit & ICU capabilities. Geographical nearest hospital ({nearest.get('name')}) lacks Level-1 Trauma facilities."
                return f"{rec.get('name')} meets all required Trauma Unit and ICU capabilities for this {latest_inc.get('severity').upper()} incident."
            return "No active patient transfer recommendation logged."
        if "severity" in msg_lower or "patient" in msg_lower or "incoming" in msg_lower:
            if latest_inc:
                rec = latest_inc.get("recommended_hospital", {})
                return f"Incoming patient severity is {latest_inc.get('severity').upper()} (ETA {rec.get('eta_min')} mins). Objects involved: {', '.join(latest_inc.get('object_classes', []))}."
            return "No incoming emergency patient currently assigned."

    elif role == "citizen":
        if "driver" in msg_lower or "phone" in msg_lower or "number" in msg_lower or "name" in msg_lower:
            return "I cannot disclose private personnel data or driver contact numbers. Please check public traffic advisories for safety info."
        if "safe" in msg_lower or "happen" in msg_lower or "near me" in msg_lower or "traffic" in msg_lower:
            if latest_inc:
                return f"Public Safety Advisory: An emergency incident is active near NIT Trichy main junction. Emergency vehicles are in transit; please clear the left lane on NH-67."
            return "Intersection Status: Traffic flow is clear and normal under standard monitoring."

    else: # command
        if "summary" in msg_lower or "status" in msg_lower or "system" in msg_lower:
            return f"System Telemetry: Camera Node #04 Online (30 FPS), Perception AI Active. {len(incidents)} confirmed collisions logged."

    return "I don't have that information in the current backend logs."


@app.post("/api/chat")
def chat_with_assistant(req: ChatRequest):
    sys_prompt, data_context_str = _build_grounded_context(req.role)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            messages = [{"role": "system", "content": sys_prompt}]
            for h in req.history[-4:]:
                messages.append({"role": h.role, "content": h.content})
            messages.append({"role": "user", "content": req.message})
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.2,
                max_tokens=200
            )
            ans = response.choices[0].message.content.strip()
            return {"reply": ans, "role": req.role}
        except Exception as e:
            print(f"[Chatbot Warning] OpenAI API call failed: {e}")

    
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            full_prompt = f"{sys_prompt}\n\nUser Question: {req.message}"
            response = model.generate_content(full_prompt)
            ans = response.text.strip()
            return {"reply": ans, "role": req.role}
        except Exception as e:
            print(f"[Chatbot Warning] Gemini API call failed: {e}")

    
    fallback_reply = _fallback_grounded_response(req.role, req.message)
    return {"reply": fallback_reply, "role": req.role}


@app.get("/api/hospitals")
def get_hospitals():
    return response_planner.HOSPITALS


@app.get("/api/drivers")
def get_drivers():
    return ambulance_fleet.AMBULANCE_DRIVERS



from fastapi.staticfiles import StaticFiles
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
