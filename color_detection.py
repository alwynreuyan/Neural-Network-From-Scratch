"""
color_detection.py — Module 2: Real-Time Color Detection
=========================================================
WSC2026_C1 / WSA2026_TP55_**_M2

Requirements:
- YOLO model detects target objects
- HSV-based color classification extracts primary color of each detected object
- Bounding box + color name label rendered on live frame
- Robust to ambient lighting variations

Setup:
- Place your trained YOLO model as  best.pt  in this folder
- Model detects the objects in the dataset (boxes, items, etc.)
- Color is determined by HSV analysis of the detected ROI

Run:
  python color_detection.py
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "best.pt"      # your trained YOLOv8/v11 weights
CAMERA_INDEX = 0
CONF_THRESH  = 0.45
DEVICE       = "cuda"
# ─────────────────────────────────────────────────────────────────────────────

# HSV color ranges — (lower, upper, color_name, BGR_draw_color)
# Robust ranges tuned for varied lighting
HSV_COLORS = [
    # Red appears in two hue ranges (wraps 0/180)
    (np.array([0,   110,  60]),  np.array([10,  255, 255]), "Red",    (0,   0,   255)),
    (np.array([168, 110,  60]),  np.array([180, 255, 255]), "Red",    (0,   0,   255)),
    # Orange
    (np.array([11,  100,  60]),  np.array([22,  255, 255]), "Orange", (0,   128, 255)),
    # Yellow
    (np.array([23,  100,  60]),  np.array([34,  255, 255]), "Yellow", (0,   220, 220)),
    # Green
    (np.array([35,  60,   40]),  np.array([85,  255, 255]), "Green",  (0,   200,  0)),
    # Cyan
    (np.array([86,  80,   40]),  np.array([95,  255, 255]), "Cyan",   (255, 200,  0)),
    # Blue
    (np.array([96,  100,  40]),  np.array([130, 255, 255]), "Blue",   (255,  50,  0)),
    # Purple / Violet
    (np.array([131, 50,   40]),  np.array([155, 255, 255]), "Purple", (180,  0,  180)),
    # White — low saturation, high value
    (np.array([0,   0,   200]),  np.array([180,  40, 255]), "White",  (220, 220, 220)),
    # Black — low value
    (np.array([0,   0,    0]),   np.array([180,  255, 50]), "Black",  (50,   50,  50)),
]


def classify_roi_color(roi_bgr):
    """
    Classify the dominant color in a BGR ROI using HSV thresholding.
    Returns (color_name, bgr_draw_color).
    Applies CLAHE on V channel before classification for lighting robustness.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "Unknown", (128, 128, 128)

    # Resize to fixed size for consistent processing
    roi = cv2.resize(roi_bgr, (64, 64))

    # CLAHE on V channel → lighting robustness
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

    best_name  = "Unknown"
    best_color = (128, 128, 128)
    best_count = 0

    for (*bounds, name, draw_color) in HSV_COLORS:
        lower, upper = bounds
        mask  = cv2.inRange(hsv, lower, upper)
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_name  = name
            best_color = draw_color

    return best_name, best_color


# ── MAIN ──────────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
model.to(DEVICE)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

prev_time = 0

print("Running — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = model.predict(
        source=frame,
        conf=CONF_THRESH,
        device=DEVICE,
        verbose=False
    )[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf     = float(box.conf[0])
        obj_name = model.names[int(box.cls[0])]

        # Clamp ROI to frame bounds
        rx1, ry1 = max(0, x1), max(0, y1)
        rx2, ry2 = min(w, x2), min(h, y2)
        roi = frame[ry1:ry2, rx1:rx2]

        color_name, draw_color = classify_roi_color(roi)

        label = f"{obj_name} | {color_name}  {conf:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), draw_color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-5))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (w - 140, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    cv2.imshow("M2 - Color Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
