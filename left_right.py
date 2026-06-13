"""
left_right.py — Module 1: Real-Time Left-Right Hand Detection
==============================================================
WSC2026_C1 / WSA2026_TP55_**_M1

Requirements:
- YOLOv8 or YOLOv11 model trained on Left Hand / Right Hand classes
- Detects Left Hand → label "Left Hand" + confidence
- Detects Right Hand → label "Right Hand" + confidence
- Both hands detected → independent bounding boxes for each

Setup:
- Place your trained model as  best.pt  in this folder
- Your data.yaml must have:  names: ['Left Hand', 'Right Hand']
  (or whatever names your dataset uses — update LEFT_CLS / RIGHT_CLS below)

Run:
  python left_right.py
"""

import cv2
import time
from ultralytics import YOLO

# ── CONFIG — update these to match your trained model ─────────────────────────
MODEL_PATH  = "best.pt"       # your trained YOLOv8/v11 weights
CAMERA_INDEX = 0              # 0 = default webcam
CONF_THRESH  = 0.50
DEVICE       = "cuda"         # "cpu" if no GPU

# Class names as defined in your data.yaml — adjust if different
LEFT_CLS  = "Left Hand"
RIGHT_CLS = "Right Hand"

# Bounding box colors (BGR)
COLOR_LEFT  = (0, 255, 0)     # green  for Left Hand
COLOR_RIGHT = (0, 0, 255)     # red    for Right Hand
COLOR_BOTH  = (0, 200, 255)   # yellow for Both (label only)
# ─────────────────────────────────────────────────────────────────────────────

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

    detected = []   # list of (class_name, conf, x1, y1, x2, y2)

    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detected.append((cls_name, conf, x1, y1, x2, y2))

    # Determine state
    names_found = [d[0] for d in detected]
    has_left    = LEFT_CLS  in names_found
    has_right   = RIGHT_CLS in names_found
    both        = has_left and has_right

    # Draw boxes
    for (cls_name, conf, x1, y1, x2, y2) in detected:
        if cls_name == LEFT_CLS:
            color = COLOR_LEFT
        elif cls_name == RIGHT_CLS:
            color = COLOR_RIGHT
        else:
            color = (200, 200, 200)

        label = f"{cls_name}  {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Status banner
    if both:
        status_text  = "Both Hands Detected"
        status_color = COLOR_BOTH
    elif has_left:
        status_text  = "Left Hand Detected"
        status_color = COLOR_LEFT
    elif has_right:
        status_text  = "Right Hand Detected"
        status_color = COLOR_RIGHT
    else:
        status_text  = "No Hand Detected"
        status_color = (120, 120, 120)

    cv2.putText(frame, status_text, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    # FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time + 1e-5))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps}", (w - 140, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    cv2.imshow("M1 - Left / Right Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
