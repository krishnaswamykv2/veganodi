

import math
from collections import deque

from risk_engine import MAX_EXPECTED_SPEED

HISTORY_LEN = 10
STILLNESS_SPEED_THRESH = 15.0   # px/s — below this counts as "stopped"
COLLAPSE_RATIO = 0.35           # speed must drop below 35% of its earlier value
DIRECTION_CHANGE_DEG = 100.0    # heading swing beyond this counts as abrupt
IOU_THRESH = 0.02               # any meaningful bbox overlap
SPEED_HIGH_FACTOR = 0.5         # fraction of MAX_EXPECTED_SPEED counted as "high"
CONFIRM_VOTES = 4               # need at least 4 of 5 signals


def bbox_iou(a, b):
   
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class HistoryTracker:
    

    def __init__(self):
        self.history = {}

    def update(self, object_id, t, bbox, vx, vy):
        if object_id not in self.history:
            self.history[object_id] = deque(maxlen=HISTORY_LEN)
        speed = math.hypot(vx, vy)
        self.history[object_id].append(
            {"t": t, "bbox": bbox, "speed": speed, "vx": vx, "vy": vy}
        )

    def get(self, object_id):
        return self.history.get(object_id, deque())


def _velocity_collapsed(hist):
    if len(hist) < 4:
        return False
    early_speed = hist[0]["speed"]
    recent_speed = hist[-1]["speed"]
    return early_speed > 20 and recent_speed < early_speed * COLLAPSE_RATIO


def _post_impact_still(hist):
    if len(hist) < 3:
        return False
    recent = list(hist)[-3:]
    return all(h["speed"] < STILLNESS_SPEED_THRESH for h in recent)


def _peak_speed(hist):
    
    if not hist:
        return 0.0
    return max(h["speed"] for h in hist)


def _large_direction_change(hist):
    if len(hist) < 4:
        return False
    v_old = (hist[0]["vx"], hist[0]["vy"])
    v_new = (hist[-1]["vx"], hist[-1]["vy"])
    mag_old, mag_new = math.hypot(*v_old), math.hypot(*v_new)
    if mag_old < 1e-3 or mag_new < 1e-3:
        return False
    cos_t = (v_old[0] * v_new[0] + v_old[1] * v_new[1]) / (mag_old * mag_new)
    cos_t = max(-1.0, min(1.0, cos_t))
    angle = math.degrees(math.acos(cos_t))
    return angle > DIRECTION_CHANGE_DEG


class CollisionConfidenceEstimator:
    def __init__(self):
        self.tracker = HistoryTracker()

    def update_history(self, object_id, t, bbox, vx, vy):
        self.tracker.update(object_id, t, bbox, vx, vy)

    def evaluate(self, id_a, id_b, bbox_a, bbox_b, features):
       
        hist_a = self.tracker.get(id_a)
        hist_b = self.tracker.get(id_b)

        peak_speed = max(_peak_speed(hist_a), _peak_speed(hist_b))
        signals = {
            "relative_speed_high": (
                features["relative_speed"] > SPEED_HIGH_FACTOR * MAX_EXPECTED_SPEED
                or peak_speed > SPEED_HIGH_FACTOR * MAX_EXPECTED_SPEED
            ),
            "bbox_overlap": bbox_iou(bbox_a, bbox_b) > IOU_THRESH,
            "velocity_collapse": _velocity_collapsed(hist_a) or _velocity_collapsed(hist_b),
            "post_impact_stillness": _post_impact_still(hist_a) or _post_impact_still(hist_b),
            "large_direction_change": _large_direction_change(hist_a) or _large_direction_change(hist_b),
        }
        vote_count = sum(signals.values())
        confirmed = vote_count >= CONFIRM_VOTES
        return confirmed, vote_count, signals


if __name__ == "__main__":
    cce = CollisionConfidenceEstimator()

    print("Simulating approach, then a sudden stop + overlap (a 'collision')...\n")

    t = 0.0
    dt = 0.1
   
    ax, ay, avx, avy = 50.0, 300.0, 150.0, 0.0
    bx, by, bvx, bvy = 350.0, 300.0, -150.0, 0.0
    stopped = False
    frames_since_stop = None

    for step in range(30):
        t = round(step * dt, 2)
        if not stopped and ax >= bx - 10:
            
            stopped = True
            frames_since_stop = 0
            avx, bvx = -40.0, 40.0
        elif stopped:
            frames_since_stop += 1
            decay = max(0.0, 1.0 - frames_since_stop / 4.0)
            avx, bvx = -40.0 * decay, 40.0 * decay
        else:
            ax += avx * dt
            bx += bvx * dt

        bbox_a = [ax - 15, ay - 15, ax + 15, ay + 15]
        bbox_b = [bx - 15, by - 15, bx + 15, by + 15]

        cce.update_history("A", t, bbox_a, avx, avy)
        cce.update_history("B", t, bbox_b, bvx, bvy)

        features = {"relative_speed": abs(avx - bvx) + abs(avy - bvy) + 1.0}
        confirmed, votes, signals = cce.evaluate("A", "B", bbox_a, bbox_b, features)

        print(f"t={t:4.1f}s  ax={ax:6.1f} bx={bx:6.1f}  votes={votes}/5  "
              f"confirmed={confirmed}  signals={signals}")

        if confirmed:
            print("\n>>> Collision CONFIRMED by 4-of-5 signal fusion. <<<")
            break
