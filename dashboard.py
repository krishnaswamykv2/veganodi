

import json
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ambulance_fleet import (
    sync_new_incidents, advance_status, current_ambulance_position,
    STEP_LABELS,
)

st.set_page_config(page_title="Veganodi - Mission Control", layout="wide", page_icon=":vertical_traffic_light:")


st.markdown("""
<style>
    .stApp { background-color: #0a0e14; }
    * { font-family: 'Consolas', 'Courier New', monospace; }
    h1 { color: #00e5ff !important; letter-spacing: 2px; }
    h2, h3 { color: #00b8d4 !important; }
    .card {
        background: #0f1620; border: 1px solid #1a3040; border-radius: 8px;
        padding: 16px; margin-bottom: 12px;
    }
    .metric-value { font-size: 26px; color: #00e5ff; font-weight: bold; }
    .metric-label { font-size: 11px; color: #6b8a99; text-transform: uppercase; letter-spacing: 1px; }
    .badge { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
    .badge-safe { background: #14331c; color: #4caf50; }
    .badge-elevated { background: #332a10; color: #ffb300; }
    .badge-critical { background: #330f0a; color: #ff3d00; }
    .badge-pending { background: #2a2510; color: #ffca28; }
    .badge-progress { background: #0d2233; color: #29b6f6; }
    .badge-done { background: #0e2b16; color: #66bb6a; }
    div[data-testid="stDataFrame"] { border: 1px solid #1a3040; }
    button { font-family: 'Consolas', 'Courier New', monospace !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ⬢ PRECRASH — MISSION CONTROL")
st.caption("Predictive collision intelligence · KAVAL IoT / Circuit Crunch — Interconnected Response Network")



def load_status():
    if not os.path.exists("status.json"):
        return None
    try:
        with open("status.json") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_ledger():
    if not os.path.exists("near_miss_ledger.csv"):
        return pd.DataFrame()
    try:
        return pd.read_csv("near_miss_ledger.csv")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_incidents():
    if not os.path.exists("incidents_log.jsonl"):
        return []
    out = []
    with open("incidents_log.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def badge(text, kind):
    return f"<span class='badge badge-{kind}'>{text}</span>"


def workflow_kind(status):
    if status in ("pending_ambulance_accept", "pending_hospital_accept"):
        return "pending"
    if status == "arrived_at_hospital":
        return "done"
    return "progress"


status = load_status()
ledger = load_ledger()
incidents = load_incidents()
incident_state = sync_new_incidents(incidents) if incidents else {}

tab_cmd, tab_police, tab_ambulance, tab_hospital, tab_citizen = st.tabs(
    ["\U0001F6F0 COMMAND CENTER", "\U0001F693 POLICE", "\U0001F691 AMBULANCE", "\U0001F3E5 HOSPITAL", "\U0001F4E3 CITIZEN ALERT"]
)


with tab_cmd:
    col_feed, col_status = st.columns([3, 2])

    with col_feed:
        st.markdown("### LIVE FEED")
        if os.path.exists("latest_frame.jpg"):
            st.image("latest_frame.jpg", use_container_width=True)
        else:
            st.info("Waiting for perception.py to start writing frames...")

    with col_status:
        st.markdown("### CURRENT RISK STATUS")
        if status and status.get("pairs"):
            for pair in status["pairs"]:
                kind = pair["tier"]
                st.markdown(
                    f"**{pair['class_a']} {pair['object_a']} <-> {pair['class_b']} {pair['object_b']}**  \n"
                    f"{badge(pair['tier'].upper(), kind)} risk={pair['risk']:.2f}, ttc={pair['ttc']:.2f}s",
                    unsafe_allow_html=True,
                )
                st.progress(min(pair["risk"], 1.0))
        else:
            st.info("No active object pairs being scored right now.")

    st.markdown("---")
    st.markdown("### EVALUATION METRICS")
    m1, m2, m3, m4 = st.columns(4)
    near_miss_count = len(ledger) if not ledger.empty else 0
    critical_count = len(ledger[ledger["tier"] == "critical"]) if not ledger.empty and "tier" in ledger.columns else 0
    confirmed_count = len(incidents)
    avg_ttc = round(ledger["ttc_s"].mean(), 2) if not ledger.empty and "ttc_s" in ledger.columns else "-"
    for col, label, value in [
        (m1, "Near-Misses Logged", near_miss_count), (m2, "Critical Escalations", critical_count),
        (m3, "Confirmed Collisions", confirmed_count), (m4, "Avg TTC at Escalation", f"{avg_ttc}s" if avg_ttc != "-" else "-"),
    ]:
        with col:
            st.markdown(f"<div class='card' style='text-align:center'><div class='metric-value'>{value}</div>"
                         f"<div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    col_ledger, col_map = st.columns([3, 2])
    with col_ledger:
        st.markdown("### NEAR-MISS LEDGER (most recent)")
        st.dataframe(ledger.tail(15).iloc[::-1], use_container_width=True, height=280) if not ledger.empty \
            else st.info("No near-misses logged yet.")

    with col_map:
        st.markdown("### LIVE NETWORK MAP")
        if incident_state:
            fig = go.Figure()
            for rec in incident_state.values():
                loc = rec["location"]
                amb_lat, amb_lon = current_ambulance_position(rec)
                hosp = rec["recommended_hospital"]
                fig.add_trace(go.Scattermapbox(lat=[loc["lat"]], lon=[loc["lon"]], mode="markers",
                                                marker=dict(size=14, color="red"), name="Incident",
                                                text=[f"Incident {rec['incident_id']}"]))
                fig.add_trace(go.Scattermapbox(lat=[hosp["lat"]], lon=[hosp["lon"]], mode="markers",
                                                marker=dict(size=14, color="lime"), name="Hospital",
                                                text=[hosp["name"]]))
                fig.add_trace(go.Scattermapbox(lat=[amb_lat], lon=[amb_lon], mode="markers",
                                                marker=dict(size=12, color="cyan"), name="Ambulance",
                                                text=[rec["ambulance"]["name"]]))
            fig.update_layout(
                mapbox=dict(style="open-street-map", zoom=11,
                            center=dict(lat=list(incident_state.values())[0]["location"]["lat"],
                                        lon=list(incident_state.values())[0]["location"]["lon"])),
                paper_bgcolor="#0a0e14", margin=dict(l=0, r=0, t=0, b=0), height=280, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No incidents yet — map activates on first confirmed collision.")


with tab_police:
    st.markdown("### INCIDENT RECORDS — LAW ENFORCEMENT VIEW")
    if not incident_state:
        st.info("No incidents recorded.")
    for rec in reversed(list(incident_state.values())):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                if rec.get("photo_path") and os.path.exists(rec["photo_path"]):
                    st.image(rec["photo_path"], use_container_width=True, caption="Evidence photo")
                else:
                    st.warning("No evidence photo on file.")
            with c2:
                st.markdown(f"**Incident ID:** {rec['incident_id']}")
                st.markdown(f"**Timestamp:** {rec['timestamp']}")
                st.markdown(f"**Severity:** {rec['severity'].upper()}")
                st.markdown(f"**Parties involved:** {', '.join(rec['object_classes'])}")
                lat, lon = rec["location"]["lat"], rec["location"]["lon"]
                st.markdown(f"**Location:** {lat}, {lon} — "
                             f"[View on Maps](https://www.google.com/maps?q={lat},{lon})")
                st.markdown(f"**Status:** {STEP_LABELS.get(rec['workflow_status'], rec['workflow_status'])}")
            st.markdown("</div>", unsafe_allow_html=True)


with tab_ambulance:
    st.markdown("### AMBULANCE DISPATCH — DRIVER VIEW")
    st.caption("Fictional demo fleet — simulates a ride-hailing-style accept/en-route/arrived workflow.")
    if not incident_state:
        st.info("No active dispatch requests.")
    for rec in reversed(list(incident_state.values())):
        if rec["workflow_status"] == "arrived_at_hospital":
            continue  # completed jobs don't need driver action anymore
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            drv = rec["ambulance"]
            st.markdown(f"**Incident {rec['incident_id']}** — {rec['severity'].upper()} "
                        f"{badge(STEP_LABELS.get(rec['workflow_status'], ''), workflow_kind(rec['workflow_status']))}",
                        unsafe_allow_html=True)
            st.markdown(f"Driver: **{drv['name']}** · {drv['phone']} · Vehicle {drv['vehicle_no']}")
            st.markdown(f"Pickup location: {rec['location']['lat']}, {rec['location']['lon']}")
            st.markdown(f"Destination: **{rec['recommended_hospital']['name']}** "
                        f"(ETA {rec['recommended_hospital']['eta_min']} min)")

            status_now = rec["workflow_status"]
            b1, b2, b3, b4 = st.columns(4)
            if status_now == "pending_ambulance_accept":
                if b1.button("Accept Ride", key=f"acc_{rec['incident_id']}"):
                    advance_status(rec["incident_id"], "ambulance_accepted"); st.rerun()
            elif status_now == "ambulance_accepted":
                if b1.button("Depart to Scene", key=f"dep_{rec['incident_id']}"):
                    advance_status(rec["incident_id"], "en_route_to_scene"); st.rerun()
            elif status_now == "en_route_to_scene":
                if b1.button("Arrived at Scene", key=f"arr_{rec['incident_id']}"):
                    advance_status(rec["incident_id"], "arrived_at_scene"); st.rerun()
            elif status_now == "arrived_at_scene":
                if b1.button("Depart to Hospital", key=f"deph_{rec['incident_id']}"):
                    advance_status(rec["incident_id"], "en_route_to_hospital"); st.rerun()
            elif status_now == "en_route_to_hospital":
                if b1.button("Request Hospital Accept", key=f"req_{rec['incident_id']}"):
                    advance_status(rec["incident_id"], "pending_hospital_accept"); st.rerun()
            elif status_now == "pending_hospital_accept":
                st.info("Waiting for hospital to accept patient...")
            elif status_now == "hospital_accepted":
                if b1.button("Arrived at Hospital", key=f"fin_{rec['incident_id']}"):
                    advance_status(rec["incident_id"], "arrived_at_hospital"); st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


with tab_hospital:
    st.markdown("### HOSPITAL PORTAL — INCOMING PATIENTS")
    st.caption("Recommended by capability-aware matching (trauma unit / ICU), not distance alone.")
    if not incident_state:
        st.info("No incoming patients.")
    for rec in reversed(list(incident_state.values())):
        if rec["workflow_status"] not in ("pending_hospital_accept", "hospital_accepted", "arrived_at_hospital"):
            continue
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            hosp = rec["recommended_hospital"]
            st.markdown(f"**Incident {rec['incident_id']}** to **{hosp['name']}** "
                        f"{badge(STEP_LABELS.get(rec['workflow_status'], ''), workflow_kind(rec['workflow_status']))}",
                        unsafe_allow_html=True)
            st.markdown(f"Severity: **{rec['severity'].upper()}** · Parties: {', '.join(rec['object_classes'])}")
            st.markdown(f"Trauma center: {hosp['trauma_center']} · ICU: {hosp['icu']} · ETA: {hosp['eta_min']} min")
            st.markdown(f"Ambulance: {rec['ambulance']['name']} ({rec['ambulance']['vehicle_no']})")
            if rec["workflow_status"] == "pending_hospital_accept":
                c1, c2 = st.columns(2)
                if c1.button("Accept Patient", key=f"hacc_{rec['incident_id']}"):
                    advance_status(rec["incident_id"], "hospital_accepted"); st.rerun()
                if c2.button("Divert", key=f"hdiv_{rec['incident_id']}"):
                    st.warning("Diversion would trigger Response Planner re-routing to next-best hospital "
                               "in a full implementation.")
            st.markdown("</div>", unsafe_allow_html=True)


with tab_citizen:
    st.markdown("### PUBLIC SAFETY ALERTS")
    st.caption("Plain-language notices — no technical details, no personal data.")
    if not incident_state:
        st.info("No active alerts in this area.")
    for rec in reversed(list(incident_state.values())):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**Safety Alert — {rec['timestamp']}**")
            st.markdown(
                f"An incident has been detected near your area. Emergency services "
                f"have been notified and are {STEP_LABELS.get(rec['workflow_status'], 'responding').lower()}. "
                f"Please avoid the area and follow instructions from traffic personnel if present."
            )
            st.markdown("</div>", unsafe_allow_html=True)


time.sleep(1)
st.rerun()
