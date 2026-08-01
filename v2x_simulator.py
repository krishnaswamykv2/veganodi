

import math
import random
import time


class SimulatedBSM:
    """
    A synthetic Basic Safety Message — the standard V2X broadcast format.
    Real BSMs include many more fields (SAE J2735); we model the subset
    relevant to our risk pipeline: position, speed, heading.
    """

    def __init__(self, vehicle_id, x, y, speed_px_s, heading_rad, timestamp):
        self.vehicle_id = vehicle_id
        self.x = x
        self.y = y
        self.vx = speed_px_s * math.cos(heading_rad)
        self.vy = speed_px_s * math.sin(heading_rad)
        self.timestamp = timestamp

    def as_detection_record(self, frame_id=0):
        """
        Converts this BSM into the SAME detection-record shape perception.py
        produces, so it can be merged into the existing pipeline without any
        special-case handling downstream — a V2X-reported vehicle looks just
        like a camera-detected one to the Motion Forecasting / Risk Engine,
        except its class_name and confidence make clear it came from radio,
        not vision.
        """
        
        half_w, half_h = 60, 40
        return {
            "frame_id": frame_id,
            "timestamp": self.timestamp,
            "object_id": self.vehicle_id,
            "class_name": "v2x_vehicle",
            "confidence": 1.0,  
            "bbox": [self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h],
            "center": [self.x, self.y],
            "source": "v2x",  
        }


class V2XSimulator:
    """
    Simulates one or more V2X-equipped vehicles broadcasting their position
    on a fixed schedule, independent of whatever the camera can or can't
    see. Call .step(timestamp) each frame to get the current BSMs.
    """

    def __init__(self):
        self.vehicles = {}  

    def add_vehicle(self, vehicle_id, start_x, start_y, speed_px_s, heading_rad):
        """Register a simulated V2X-equipped vehicle with a straight-line path."""
        self.vehicles[vehicle_id] = {
            "x": start_x, "y": start_y,
            "speed": speed_px_s, "heading": heading_rad,
        }

    def step(self, timestamp, dt=0.1):
        """Advances every simulated vehicle and returns their current BSMs."""
        bsms = []
        for vid, v in self.vehicles.items():
            v["x"] += v["speed"] * math.cos(v["heading"]) * dt
            v["y"] += v["speed"] * math.sin(v["heading"]) * dt
            bsms.append(SimulatedBSM(vid, v["x"], v["y"], v["speed"], v["heading"], timestamp))
        return bsms


if __name__ == "__main__":
    from motion_forecast import TrackManager
    from risk_engine import scene_understanding, fuzzy_risk, StateManager, CLASS_WEIGHT

    print("Scenario: a pedestrian is visible to the camera crossing at a")
    print("blind corner. A car is approaching FAST from behind a building —")
    print("completely outside the camera's field of view — but it is")
    print("V2X-equipped and broadcasting its position.\n")

    track_manager = TrackManager()
    state_manager = StateManager(ledger_path="near_miss_ledger_v2x_demo.csv")
    v2x = V2XSimulator()

   
    v2x.add_vehicle("v2x_car_1", start_x=500.0, start_y=300.0,
                     speed_px_s=180.0, heading_rad=math.pi) 

    dt = 0.1
    for i in range(30):
        t = round(i * dt, 2)

       
        pedestrian_detection = {
            "object_id": 1, "class_name": "person", "confidence": 0.9,
            "center": [50.0 + t * 40, 300.0],
        }

       
        v2x_bsms = v2x.step(t, dt)
        v2x_detections = [bsm.as_detection_record() for bsm in v2x_bsms]

       
        all_detections = [pedestrian_detection] + v2x_detections
        track_manager.update(all_detections, t)
        active_ids = track_manager.active_ids()

        if len(active_ids) >= 2:
            id_a, id_b = active_ids[0], active_ids[1]
            track_a = track_manager.get_view(id_a, t)
            track_b = track_manager.get_view(id_b, t)
            meta_a = track_manager.get_meta(id_a)
            meta_b = track_manager.get_meta(id_b)

            features = scene_understanding(track_a, track_b)
            class_weight_avg = (
                CLASS_WEIGHT.get(meta_a.get("class_name"), 0.7)
                + CLASS_WEIGHT.get(meta_b.get("class_name"), 0.7)
            ) / 2.0
            confidence_avg = (meta_a.get("confidence", 0.8) + meta_b.get("confidence", 0.8)) / 2.0
            risk = fuzzy_risk(features, class_weight_avg, confidence_avg)
            tier = state_manager.route(id_a, id_b, meta_a, meta_b, features, risk)

            print(f"t={t:4.1f}s  pedestrian_x={pedestrian_detection['center'][0]:6.1f}  "
                  f"v2x_car_x={v2x_detections[0]['center'][0]:7.1f}  "
                  f"risk={risk:.2f}  tier={tier}  "
                  f"ttc={features['ttc']:.2f}s  "
                  f"(car is INVISIBLE to camera, tracked only via V2X)")

    print("\nDemo complete. This risk escalation would have been IMPOSSIBLE")
    print("with camera-only perception — the car never entered the frame.")
