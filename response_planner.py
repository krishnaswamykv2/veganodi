

import math


HOSPITALS = [
    {"name": "BHEL Hospital, Tiruchirappalli", "lat": 10.7550, "lon": 78.8020,
     "trauma_center": True, "icu": True},
    {"name": "Apollo Hospitals, Trichy (Royal Road)", "lat": 10.8020, "lon": 78.6890,
     "trauma_center": True, "icu": True},
    {"name": "Kauvery Hospital, Cantonment", "lat": 10.8050, "lon": 78.6870,
     "trauma_center": True, "icu": True},
    {"name": "K.A.P. Viswanatham Govt Medical College Hospital", "lat": 10.8060, "lon": 78.6850,
     "trauma_center": True, "icu": True},
    {"name": "Mahatma Gandhi Memorial Govt Hospital (Trichy GH)", "lat": 10.8075, "lon": 78.6870,
     "trauma_center": True, "icu": True},
    {"name": "Chennai Medical College Hospital & Research Centre (CMCHRC)", "lat": 10.8300, "lon": 78.7000,
     "trauma_center": True, "icu": True},
    {"name": "Royal Care Super Speciality Hospital, Trichy", "lat": 10.8100, "lon": 78.6950,
     "trauma_center": True, "icu": True},
    {"name": "Sugam Hospital, Trichy", "lat": 10.7950, "lon": 78.7050,
     "trauma_center": False, "icu": True},
    {"name": "Retna Global Hospital, Tennur", "lat": 10.8010, "lon": 78.6920,
     "trauma_center": False, "icu": True},
    {"name": "Vinayaga Mission Hospital, Trichy", "lat": 10.8150, "lon": 78.6800,
     "trauma_center": False, "icu": False},
    {"name": "SRM Medical College Hospital, Irungalur", "lat": 10.9020, "lon": 78.8100,
     "trauma_center": True, "icu": True},
    {"name": "Dhanalakshmi Srinivasan Medical College Hospital, Perambalur", "lat": 11.2340, "lon": 78.8800,
     "trauma_center": True, "icu": True},
]


AVG_CITY_SPEED_KMPH = 28.0  


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def eta_minutes(distance_km, avg_speed_kmph=AVG_CITY_SPEED_KMPH):
    return round((distance_km / avg_speed_kmph) * 60, 1)



VULNERABLE_CLASSES = {"person", "bicycle", "motorcycle"}


def estimate_severity(class_a, class_b, peak_relative_speed, cce_votes):
    
    involves_vulnerable = class_a in VULNERABLE_CLASSES or class_b in VULNERABLE_CLASSES
    high_speed = peak_relative_speed > 150.0  # px/s, tune against real footage
    strong_confirmation = cce_votes >= 5

    if involves_vulnerable and (high_speed or strong_confirmation):
        return "critical"
    if involves_vulnerable or high_speed:
        return "severe"
    return "moderate"


SEVERITY_REQUIREMENTS = {
    "critical": {"trauma_center": True, "icu": True},
    "severe": {"trauma_center": True, "icu": False},
    "moderate": {"trauma_center": False, "icu": False},
}



class ResponsePlanner:
    def __init__(self, hospitals=None):
        self.hospitals = hospitals or HOSPITALS

    def recommend_hospital(self, incident_lat, incident_lon, severity):
        
        requirement = SEVERITY_REQUIREMENTS[severity]

        def meets_requirement(h):
            if requirement["trauma_center"] and not h["trauma_center"]:
                return False
            if requirement["icu"] and not h["icu"]:
                return False
            return True

        candidates = [h for h in self.hospitals if meets_requirement(h)]
        fallback_used = False
        if not candidates:
           
            candidates = sorted(
                self.hospitals,
                key=lambda h: (h["trauma_center"], h["icu"]),
                reverse=True,
            )[:1]
            fallback_used = True

        ranked = []
        for h in candidates:
            dist = haversine_km(incident_lat, incident_lon, h["lat"], h["lon"])
            ranked.append({
                "name": h["name"],
                "lat": h["lat"], "lon": h["lon"],
                "distance_km": round(dist, 2),
                "eta_min": eta_minutes(dist),
                "trauma_center": h["trauma_center"],
                "icu": h["icu"],
            })
        ranked.sort(key=lambda x: x["eta_min"])

        return ranked, fallback_used

    def plan(self, incident_lat, incident_lon, class_a, class_b,
             peak_relative_speed, cce_votes, damage_assessment=None):
       
        severity = estimate_severity(class_a, class_b, peak_relative_speed, cce_votes)

        if damage_assessment and damage_assessment.get("damage_level"):
            from damage_assessment import combine_severity
            severity = combine_severity(severity, damage_assessment["damage_level"])

        ranked_hospitals, fallback_used = self.recommend_hospital(
            incident_lat, incident_lon, severity
        )
        best = ranked_hospitals[0] if ranked_hospitals else None

        
        nearest_any = min(
            self.hospitals,
            key=lambda h: haversine_km(incident_lat, incident_lon, h["lat"], h["lon"]),
        )
        nearest_dist = haversine_km(incident_lat, incident_lon, nearest_any["lat"], nearest_any["lon"])

        return {
            "severity": severity,
            "damage_assessment": damage_assessment,
            "object_classes": [class_a, class_b],
            "collision_confidence_votes": cce_votes,
            "recommended_hospital": best,
            "all_ranked_candidates": ranked_hospitals,
            "capability_fallback_used": fallback_used,
            "nearest_hospital_by_distance_only": {
                "name": nearest_any["name"],
                "lat": nearest_any["lat"], "lon": nearest_any["lon"],
                "distance_km": round(nearest_dist, 2),
                "eta_min": eta_minutes(nearest_dist),
                "trauma_center": nearest_any["trauma_center"],
                "icu": nearest_any["icu"],
            },
        }



if __name__ == "__main__":
    planner = ResponsePlanner()

    print("Scenario 1: pedestrian hit at high speed (should demand trauma+ICU)\n")
    incident = planner.plan(
        incident_lat=10.7590, incident_lon=78.7080,
        class_a="person", class_b="car",
        peak_relative_speed=220.0, cce_votes=4,
    )
    for k, v in incident.items():
        print(f"  {k}: {v}")

    print("\nScenario 2: minor low-speed car-car bump (should NOT need trauma center)\n")
    incident2 = planner.plan(
        incident_lat=10.7590, incident_lon=78.7080,
        class_a="car", class_b="car",
        peak_relative_speed=60.0, cce_votes=4,
    )
    for k, v in incident2.items():
        print(f"  {k}: {v}")

    print("\n--- 'Nearest != best' check ---")
    print(f"Scenario 1 recommended: {incident['recommended_hospital']['name']} "
          f"({incident['recommended_hospital']['eta_min']} min)")
    print(f"Scenario 1 nearest-by-distance-only would have been: "
          f"{incident['nearest_hospital_by_distance_only']['name']} "
          f"({incident['nearest_hospital_by_distance_only']['eta_min']} min, "
          f"trauma_center={incident['nearest_hospital_by_distance_only']['trauma_center']})")
