

import argparse
import json
import os
import time

import cv2
from ultralytics import YOLO


RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def run(source, model_path="yolov8n.pt", show=False, jsonl_out=None, conf=0.35):
    model = YOLO(model_path)

    cap_source = int(source) if str(source).isdigit() else source
    start_time = time.time()
    frame_id = 0

    out_fh = open(jsonl_out, "w") if jsonl_out else None

   
    results_stream = model.track(
        source=cap_source,
        conf=conf,
        classes=list(RELEVANT_CLASSES.keys()),
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False,
    )

    last_frame_time = time.time()
    for result in results_stream:
        frame_id += 1
        curr_time = time.time()
        fps = 1.0 / (curr_time - last_frame_time) if curr_time > last_frame_time else 30.0
        last_frame_time = curr_time
        timestamp = curr_time - start_time
        frame_records = []

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()

            for obj_id, cls_idx, cf, (x1, y1, x2, y2) in zip(ids, clss, confs, xyxy):
                record = {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 3),
                    "fps": round(fps, 1),
                    "object_id": obj_id,
                    "class_name": RELEVANT_CLASSES.get(cls_idx, "unknown"),
                    "confidence": round(cf, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                }
                frame_records.append(record)

                if out_fh:
                    out_fh.write(json.dumps(record) + "\n")
                    out_fh.flush()

        if frame_records:
            print(json.dumps({"frame_id": frame_id, "fps": round(fps, 1), "objects": frame_records}))

        
        annotated = result.plot()
        display_frame = cv2.resize(annotated, (480, 270))
        tmp_path = "latest_frame_tmp.jpg"
        cv2.imwrite(tmp_path, display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        try:
            os.replace(tmp_path, "latest_frame.jpg")
        except Exception:
            pass

        if show:
            cv2.imshow("Veganodi - Perception", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if out_fh:
        out_fh.close()
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veganodi perception module")
    parser.add_argument("--source", default="0", help="0 for webcam, or path/URL to video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights")
    parser.add_argument("--show", action="store_true", help="show live annotated feed")
    parser.add_argument("--out", default=None, help="optional path to write JSONL log")
    parser.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    args = parser.parse_args()

    run(args.source, model_path=args.model, show=args.show, jsonl_out=args.out, conf=args.conf)
