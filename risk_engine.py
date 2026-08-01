

import csv
import math
import os
import time


MAX_TTC_HORIZON_S = 4.0       
MAX_EXPECTED_SPEED = 400.0    
NEAR_MISS_THRESHOLD = 0.35    
ELEVATED_THRESHOLD = 0.35
CRITICAL_THRESHOLD = 0.50    


CLASS_WEIGHT = {
    "person": 1.0,
    "bicycle": 0.85,
    "motorcycle": 0.85,
    "car": 0.6,
    "bus": 0.55,
    "truck": 0.55,
}


FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
EDGE_MARGIN_RATIO = 0.15   
BLIND_SPOT_BOOST_HEADING = 1.25  
BLIND_SPOT_BOOST_NEAR = 1.10     


def blind_spot_boost(x, y, vx, vy):
   
    margin_x = FRAME_WIDTH * EDGE_MARGIN_RATIO
    margin_y = FRAME_HEIGHT * EDGE_MARGIN_RATIO

    near_left = x < margin_x
    near_right = x > FRAME_WIDTH - margin_x
    near_top = y < margin_y
    near_bottom = y > FRAME_HEIGHT - margin_y

    if not (near_left or near_right or near_top or near_bottom):
        return 1.0

    heading_toward_edge = (
        (near_left and vx < 0) or (near_right and vx > 0)
        or (near_top and vy < 0) or (near_bottom and vy > 0)
    )
    return BLIND_SPOT_BOOST_HEADING if heading_toward_edge else BLIND_SPOT_BOOST_NEAR


def scene_understanding(track_a, track_b):
    """
    Given two KalmanTrack objects, compute the raw interaction features.
    Returns a dict of features, or None if the pair is not on a closing path.
    """
    xa, ya = track_a.position()
    xb, yb = track_b.position()
    vxa, vya = track_a.velocity()
    vxb, vyb = track_b.velocity()

   
    def _clamp(v, limit=MAX_EXPECTED_SPEED):
        return max(-limit, min(limit, v))

    vxa, vya = _clamp(vxa), _clamp(vya)
    vxb, vyb = _clamp(vxb), _clamp(vyb)

   
    rpx, rpy = xb - xa, yb - ya
    rvx, rvy = vxb - vxa, vyb - vya

    distance = math.hypot(rpx, rpy)
    relative_speed = math.hypot(rvx, rvy)

    
    closing_dot = rpx * rvx + rpy * rvy
    if relative_speed > 1e-3 and closing_dot < 0:
        ttc = -closing_dot / (relative_speed ** 2)
        ttc = min(ttc, MAX_TTC_HORIZON_S)
    else:
        ttc = MAX_TTC_HORIZON_S  # not closing -> treat as "far away in time"

    
    speed_a = math.hypot(vxa, vya)
    speed_b = math.hypot(vxb, vyb)
    if speed_a > 1e-3 and speed_b > 1e-3:
        cos_theta = (vxa * vxb + vya * vyb) / (speed_a * speed_b)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        heading_diff_rad = math.acos(cos_theta)
    else:
        heading_diff_rad = 0.0  # one or both objects stationary/just-appeared

   
    boost_a = blind_spot_boost(xa, ya, vxa, vya)
    boost_b = blind_spot_boost(xb, yb, vxb, vyb)
    blind_spot_factor = max(boost_a, boost_b)

    return {
        "distance": distance,
        "relative_speed": relative_speed,
        "ttc": ttc,
        "heading_diff_rad": heading_diff_rad,
        "closing": closing_dot < 0,
        "blind_spot_factor": blind_spot_factor,
        "is_ghost_a": getattr(track_a, "is_ghost", False),
        "is_ghost_b": getattr(track_b, "is_ghost", False),
    }



def _tri(x, a, b, c):
    """Triangular membership function. Peaks (membership=1) at b."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _membership_low_med_high(x):
    """x expected in [0,1]. Returns (low, medium, high) membership degrees."""
    low = _tri(x, -0.01, 0.0, 0.55)
    med = _tri(x, 0.15, 0.5, 0.85)
    high = _tri(x, 0.45, 1.0, 1.01)
    return low, med, high



_RULE_TABLE = {
    ("low", "low"): 0.10,
    ("low", "med"): 0.25,
    ("low", "high"): 0.40,
    ("med", "low"): 0.30,
    ("med", "med"): 0.55,
    ("med", "high"): 0.75,
    ("high", "low"): 0.55,
    ("high", "med"): 0.80,
    ("high", "high"): 0.95,
}
_LEVELS = ["low", "med", "high"]


def fuzzy_risk(features, class_weight_avg, confidence_avg):
    """
    Combines TTC and relative speed through a fuzzy rule base (9 rules),
    then scales the result by heading divergence and object-class weight.
    Returns a risk score in [0, 1].
    """
   
    ttc_risk_input = 1.0 - min(features["ttc"] / MAX_TTC_HORIZON_S, 1.0)
    speed_risk_input = min(features["relative_speed"] / MAX_EXPECTED_SPEED, 1.0)

    ttc_mem = dict(zip(_LEVELS, _membership_low_med_high(ttc_risk_input)))
    speed_mem = dict(zip(_LEVELS, _membership_low_med_high(speed_risk_input)))

    
    numerator = 0.0
    denominator = 0.0
    for ttc_level in _LEVELS:
        for speed_level in _LEVELS:
            firing_strength = min(ttc_mem[ttc_level], speed_mem[speed_level])
            if firing_strength <= 0:
                continue
            consequent = _RULE_TABLE[(ttc_level, speed_level)]
            numerator += firing_strength * consequent
            denominator += firing_strength

    base_risk = numerator / denominator if denominator > 0 else 0.0

   
    divergence = (1.0 - math.cos(features["heading_diff_rad"])) / 2.0
    heading_factor = 0.5 + 0.5 * divergence

    
    closing_factor = 1.0 if features["closing"] else 0.3

    risk = base_risk * heading_factor * closing_factor * class_weight_avg
    
    risk *= (0.7 + 0.3 * confidence_avg)

   
    risk *= features.get("blind_spot_factor", 1.0)

    return max(0.0, min(1.0, risk))



class StateManager:
    """
    Routes each scored interaction into a tier and logs near-misses.
    safe        -> risk < ELEVATED_THRESHOLD           : ignored
    elevated    -> ELEVATED_THRESHOLD <= risk < CRITICAL_THRESHOLD : logged to ledger
    critical    -> risk >= CRITICAL_THRESHOLD           : logged + escalated
    """

    def __init__(self, ledger_path="near_miss_ledger.csv"):
        self.ledger_path = ledger_path
        self._ensure_ledger_header()

    def _ensure_ledger_header(self):
        if not os.path.exists(self.ledger_path):
            with open(self.ledger_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "object_a", "object_b", "class_a", "class_b",
                    "risk_score", "ttc_s", "distance_px", "tier",
                ])

    def tier_for(self, risk_score):
        if risk_score >= CRITICAL_THRESHOLD:
            return "critical"
        if risk_score >= ELEVATED_THRESHOLD:
            return "elevated"
        return "safe"

    def route(self, obj_a_id, obj_b_id, meta_a, meta_b, features, risk_score):
        tier = self.tier_for(risk_score)

        if risk_score >= NEAR_MISS_THRESHOLD:
            with open(self.ledger_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    round(time.time(), 3), obj_a_id, obj_b_id,
                    meta_a.get("class_name", "?"), meta_b.get("class_name", "?"),
                    round(risk_score, 3), round(features["ttc"], 2),
                    round(features["distance"], 1), tier,
                ])

        if tier == "critical":
            # This is the handoff point to the Collision Confidence Estimator.
            print(f"[CRITICAL] objects {obj_a_id}<->{obj_b_id} "
                  f"({meta_a.get('class_name')}<->{meta_b.get('class_name')}) "
                  f"risk={risk_score:.2f} ttc={features['ttc']:.2f}s "
                  f"-> escalate to Collision Confidence Estimator")
        elif tier == "elevated":
            print(f"[elevated] objects {obj_a_id}<->{obj_b_id} "
                  f"risk={risk_score:.2f} ttc={features['ttc']:.2f}s -> logged")

        return tier
