

import argparse
import os
import sys
import cv2
import numpy as np
from calibration_utils import compute_homography, save_calibration, pixel_to_world


DEFAULT_PIXEL_POINTS = [
    [100.0, 450.0],  
    [540.0, 450.0],  
    [480.0, 200.0],  
    [160.0, 200.0],  
]

DEFAULT_WORLD_POINTS_M = [
    [0.0, 0.0],   
    [6.0, 0.0],    
    [6.0, 15.0],   
    [0.0, 15.0],  
]


def run_preset_calibration(filepath="camera_calibration.json"):
   
    print("Generating preset ground-plane homography calibration...")
    matrix = compute_homography(DEFAULT_PIXEL_POINTS, DEFAULT_WORLD_POINTS_M)
    saved_data = save_calibration(
        matrix, DEFAULT_PIXEL_POINTS, DEFAULT_WORLD_POINTS_M, filepath=filepath
    )
    print(f"\nCalibration saved successfully to `{filepath}`!")
    print(f"Matrix:\n{matrix}")

   
    px, py = 320.0, 325.0
    wx, wy = pixel_to_world(px, py, matrix)
    print(f"\nVerification Test Point: Pixel ({px}, {py}) -> Real-world Ground ({wx}m, {wy}m)")
    return saved_data


def run_interactive_calibration(image_path=None, source="0", filepath="camera_calibration.json"):
    
    frame = None
    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
    elif os.path.exists("latest_frame.jpg"):
        frame = cv2.imread("latest_frame.jpg")
    else:
        cap_source = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(cap_source)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            print("Failed to capture frame from camera. Falling back to preset calibration.")
            return run_preset_calibration(filepath)

    clicked_pixel_points = []
    clone = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixel_points) < 4:
            clicked_pixel_points.append([float(x), float(y)])
            cv2.circle(clone, (x, y), 6, (0, 229, 255), -1)
            cv2.putText(
                clone, f"P{len(clicked_pixel_points)}: ({x},{y})", (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1
            )
            cv2.imshow("Veganodi - Click 4 Ground Points", clone)

    cv2.namedWindow("Veganodi - Click 4 Ground Points")
    cv2.setMouseCallback("Veganodi - Click 4 Ground Points", on_click)

    print("\n============================================================")
    print("INSTRUCTIONS FOR CAMERA GROUND-PLANE CALIBRATION:")
    print("1. Click 4 points on the ground plane in order:")
    print("   Point 1: Bottom-Left ground point")
    print("   Point 2: Bottom-Right ground point")
    print("   Point 3: Top-Right ground point")
    print("   Point 4: Top-Left ground point")
    print("2. Press 'q' or ESC when finished, or 'r' to reset points.")
    print("============================================================\n")

    while True:
        cv2.imshow("Veganodi - Click 4 Ground Points", clone)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            clicked_pixel_points.clear()
            clone = frame.copy()
        elif key in (27, ord('q')) or len(clicked_pixel_points) == 4:
            break

    cv2.destroyAllWindows()

    if len(clicked_pixel_points) < 4:
        print("Fewer than 4 points clicked. Using preset calibration points.")
        return run_preset_calibration(filepath)

    print("\nPoints captured in pixels:", clicked_pixel_points)
    print("\nEnter real-world positions in meters for each clicked point:")

    world_points = []
    for i, pt in enumerate(clicked_pixel_points, 1):
        print(f"Point {i} pixel {pt}:")
        try:
            xm_str = input(f"  Real-world X (meters, default {DEFAULT_WORLD_POINTS_M[i-1][0]}): ").strip()
            ym_str = input(f"  Real-world Y (meters, default {DEFAULT_WORLD_POINTS_M[i-1][1]}): ").strip()
            xm = float(xm_str) if xm_str else DEFAULT_WORLD_POINTS_M[i-1][0]
            ym = float(ym_str) if ym_str else DEFAULT_WORLD_POINTS_M[i-1][1]
        except Exception:
            xm, ym = DEFAULT_WORLD_POINTS_M[i-1]
        world_points.append([xm, ym])

    matrix = compute_homography(clicked_pixel_points, world_points)
    saved_data = save_calibration(
        matrix, clicked_pixel_points, world_points, filepath=filepath
    )
    print(f"\nCalibration complete and saved to `{filepath}`!")
    return saved_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veganodi Ground-Plane Calibration Tool")
    parser.add_argument("--preset", action="store_true", help="Generate preset demo calibration without GUI")
    parser.add_argument("--image", default=None, help="Path to reference frame image")
    parser.add_argument("--source", default="0", help="Camera index or video file")
    parser.add_argument("--out", default="camera_calibration.json", help="Output JSON calibration file")
    args = parser.parse_args()

    if args.preset:
        run_preset_calibration(filepath=args.out)
    else:
        run_interactive_calibration(image_path=args.image, source=args.source, filepath=args.out)
