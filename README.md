# Veganodi

**Confidence-Gated Golden-Hour Response Intelligence**

Built in 36 hours for the NIT Trichy AI Hackathon (Team H26SCT03, Chennai Institute of Technology).

---

## The problem we're actually solving

Survival after a road crash comes down to the "golden hour" — but right now, that clock doesn't start until a bystander notices, decides it's serious enough, finds the right number, and describes where they are. Every one of those steps costs minutes. Nobody's watching for the crash itself, in real time, the second it happens.

Veganodi watches. It tracks road users through an ordinary camera, scores risk before anything goes wrong, and — only once multiple independent signals agree a collision has actually occurred — starts the emergency chain on its own. Ambulance, police, and the hospital all get notified in seconds, not whenever someone remembers to call.

## What it does, in plain terms

1. **Watches** a live camera feed and tracks every person and vehicle in frame
2. **Predicts** where they're heading and how risky the interaction looks, continuously
3. **Confirms** an actual collision only when at least 4 out of 5 independent signals agree — not on a single shaky trigger
4. **Responds** automatically: matches the right hospital (not just the nearest one), generates role-specific reports for dispatcher/police/hospital/citizens, and can place an automated emergency call

We call this a **two-tier authority model**: a continuous "advisory" risk score that never touches the real world on its own, and a discrete "action" gate that's the only thing allowed to trigger a real dispatch. That split is deliberate — it's what keeps the system sensitive enough to be useful and strict enough not to cry wolf.

## How it's built

| Stage | What's doing the work |
|---|---|
| Detection | YOLOv8n (pretrained on COCO — no custom training needed) |
| Tracking | ByteTrack |
| Motion forecasting | A hand-written constant-velocity Kalman filter |
| Real-world distance | Ground-plane homography calibration (per-camera, one-time setup) |
| Risk scoring | A hand-written fuzzy inference engine (TTC, closing speed, heading) |
| Collision confirmation | 4-of-5 signal consensus — speed, overlap, velocity collapse, stillness, direction change |
| Hospital matching | Capability-aware routing (trauma/ICU requirement vs. plain distance) |
| Reports & chat | Gemini, used for report generation, vision-based damage assessment, and grounded chatbots |
| Dashboard | React frontend + FastAPI backend, live map, role-based portals |

## Getting started

You'll need Python 3.10+, Node.js, and a webcam (or a video file).

```bash
git clone https://github.com/<your-username>/veganodi.git
cd veganodi
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**Run everything with one command:**
```bash
python start_all.py --mode demo    # synthetic data, no camera needed
python start_all.py --mode live    # real camera, full pipeline
```

Or run each piece by hand, if you want to see what each one is doing:
```bash
# Terminal 1
python perception.py --source 0 --out detections.jsonl --show

# Terminal 2
python core_intelligence.py --jsonl detections.jsonl

# Terminal 3
python -m uvicorn server:app --host 0.0.0.0 --port 8000

# Terminal 4
cd frontend && npm run dev
```

Open `http://localhost:5173`.

## Where things are

```
veganodi/
├── perception.py            # camera → YOLOv8 → ByteTrack
├── motion_forecast.py       # Kalman filter, ghost tracking
├── risk_engine.py           # fuzzy risk scoring, blind-spot handling
├── collision_confidence.py  # 4-of-5 signal fusion
├── response_planner.py      # hospital matching, severity logic
├── damage_assessment.py     # vision-based vehicle damage estimate
├── llm_engine.py            # role-specific report generation
├── emergency_call.py        # automated voice call (Twilio)
├── ambulance_fleet.py       # simulated dispatch workflow
├── calibration_tool.py      # ground-plane calibration
├── core_intelligence.py     # ties it all together
├── server.py                # FastAPI backend for the website
├── frontend/                # React dashboard
└── start_all.py             # one-command launcher
```

## Honest notes — things worth knowing before you dig in

We'd rather tell you this upfront than have you find it out the hard way:

- **The ambulance driver names, phone numbers, and vehicle numbers are entirely fictional** — a demo fleet, not real people. A real deployment would plug into an actual EMS/108 network.
- **The emergency call defaults to a dry-run/simulated mode** unless you set up Twilio credentials yourself — it prints exactly what it would have said, without placing a real call, so nothing breaks if you haven't configured it.
- **Hospital coordinates are approximate**, based on locality rather than exact GPS pins — verify against a map before relying on them for anything real.
- **No custom dataset was trained.** YOLOv8n's pretrained COCO weights already cover the object classes we need, so we didn't fine-tune anything — we did validate detection quality against a real accident-footage dataset before trusting it live.
- **Risk thresholds were tuned empirically** against our own test footage, not validated against historical crash-outcome data yet. That's real, honest future work, not a finished claim.

## Where this is headed

Multi-camera fusion, PTZ-steerable cameras, V2X vehicle communication, live ambulance routing around traffic, and — most importantly — actually validating our thresholds against real crash data instead of just what looked right on our own test runs.

## Team

 · Monishwar B · Krishnaswamy KV (Team Lead) . Manoranjan VS
Chennai Institute of Technology

---

*Built for the NIT Trichy 36-Hour AI Hackathon. This is a hackathon prototype, not a certified or deployed emergency-response system.*
