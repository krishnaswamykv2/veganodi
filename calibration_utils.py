
import json
import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

DEFAULT_CALIBRATION_FILE = "camera_calibration.json"


def compute_homography(pixel_points: List[List[float]], world_points_m: List[List[float]]) -> Optional[np.ndarray]:
    
    if len(pixel_points) < 4 or len(world_points_m) < 4:
        raise ValueError("Homography requires at least 4 point correspondences.")

    pts_src = np.float32(pixel_points).reshape(-1, 1, 2)
    pts_dst = np.float32(world_points_m).reshape(-1, 1, 2)

    matrix, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    return matrix


def pixel_to_world(px: float, py: float, homography_matrix: np.ndarray) -> Tuple[float, float]:
    
    pts = np.float32([[[px, py]]])
    transformed = cv2.perspectiveTransform(pts, homography_matrix)
    x_m = float(transformed[0][0][0])
    y_m = float(transformed[0][0][1])
    return round(x_m, 3), round(y_m, 3)


def world_to_pixel(x_m: float, y_m: float, homography_matrix: np.ndarray) -> Tuple[float, float]:
    
    inv_matrix = np.linalg.inv(homography_matrix)
    pts = np.float32([[[x_m, y_m]]])
    transformed = cv2.perspectiveTransform(pts, inv_matrix)
    px = float(transformed[0][0][0])
    py = float(transformed[0][0][1])
    return round(px, 1), round(py, 1)


def save_calibration(
    homography_matrix: np.ndarray,
    pixel_points: List[List[float]],
    world_points_m: List[List[float]],
    filepath: str = DEFAULT_CALIBRATION_FILE,
    camera_id: str = "CAM-NIT-TRICHY-04"
) -> Dict[str, Any]:
   
    data = {
        "camera_id": camera_id,
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        "unit": "meters",
        "homography_matrix": homography_matrix.tolist(),
        "reference_points_pixel": pixel_points,
        "reference_points_world_m": world_points_m,
        "coordinate_convention": {
            "origin": "Bottom-left reference point on ground plane (0,0)",
            "x_axis": "Horizontal distance across road in meters (left to right)",
            "y_axis": "Longitudinal depth distance down corridor in meters (away from camera)",
            "ground_anchor": "Bottom-center of bounding box [cx, y2]"
        }
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_calibration(filepath: str = DEFAULT_CALIBRATION_FILE) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    
    if not os.path.exists(filepath):
        return None, None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        matrix_list = data.get("homography_matrix")
        if not matrix_list:
            return None, None
        matrix = np.array(matrix_list, dtype=np.float64)
        return matrix, data
    except Exception as e:
        print(f"[Calibration Warning] Failed to load {filepath}: {e}")
        return None, None
