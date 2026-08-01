

import numpy as np


class KalmanTrack:
    

    def __init__(self, x, y, t):
        self.state = np.array([x, y, 0.0, 0.0])  # initial velocity unknown -> 0
        
        self.P = np.diag([10.0, 10.0, 1000.0, 1000.0])
        self.last_t = t

        
        self.R = np.diag([5.0, 5.0])
        
        self.Q_base = 5.0

        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

    def _predict(self, dt):
        if dt <= 0:
            return
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Q = np.diag([dt, dt, dt, dt]) * self.Q_base
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update(self, x, y, t):
        
        dt = t - self.last_t
        self._predict(dt)
        self.last_t = t

        z = np.array([x, y])
        y_resid = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y_resid
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def position(self):
        return self.state[0], self.state[1]

    def velocity(self):
        return self.state[2], self.state[3]

    def forecast(self, horizon_s):
        
        x, y, vx, vy = self.state
        return x + vx * horizon_s, y + vy * horizon_s


class GhostAdapter:
    

    def __init__(self, track, current_time):
        elapsed = max(0.0, current_time - track.last_t)
        self._x, self._y = track.forecast(elapsed)
        self._vx, self._vy = track.velocity()
        self.is_ghost = elapsed > 0.15  # >150ms since a real detection = extrapolated
        self.ghost_age_s = elapsed

    def position(self):
        return self._x, self._y

    def velocity(self):
        return self._vx, self._vy


class TrackManager:
    

    def __init__(self, max_missed_s=4.0):
        # max_missed_s raised from 1.0 -> 4.0: this is the "ghost tracking"
        # window — how long we keep extrapolating an object's position after
        # it stops being detected (left the frame, brief occlusion) before
        # dropping the track entirely.
        self.tracks = {}          # object_id -> KalmanTrack
        self.last_seen = {}       # object_id -> last timestamp seen
        self.meta = {}            # object_id -> {"class_name":..., "confidence":...}
        self.max_missed_s = max_missed_s

    def update(self, detections, timestamp):
        
        seen_ids = set()
        for det in detections:
            oid = det["object_id"]
            cx, cy = det["center"]
            seen_ids.add(oid)

            if oid not in self.tracks:
                self.tracks[oid] = KalmanTrack(cx, cy, timestamp)
            else:
                self.tracks[oid].update(cx, cy, timestamp)

            self.last_seen[oid] = timestamp
            self.meta[oid] = {
                "class_name": det["class_name"],
                "confidence": det["confidence"],
            }

        
        stale = [oid for oid, t in self.last_seen.items()
                 if timestamp - t > self.max_missed_s]
        for oid in stale:
            self.tracks.pop(oid, None)
            self.last_seen.pop(oid, None)
            self.meta.pop(oid, None)

        return seen_ids

    def active_ids(self):
        return list(self.tracks.keys())

    def get(self, object_id):
        
        return self.tracks.get(object_id)

    def get_view(self, object_id, current_time):
        
        track = self.tracks.get(object_id)
        return GhostAdapter(track, current_time) if track else None

    def get_meta(self, object_id):
        return self.meta.get(object_id, {})
