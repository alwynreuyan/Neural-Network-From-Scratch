import cv2
import numpy as np
import math
from ultralytics import YOLO

# Windows Audio System Imports (Pycaw)
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def initialize_system_audio():
    """Initializes and hooks into the native operating system volume controller."""
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_control = cast(interface, POINTER(IAudioEndpointVolume))
        # Get volume range: returns a tuple like (min_dB, max_dB, step)
        vol_range = volume_control.GetVolumeRange()
        return volume_control, vol_range[0], vol_range[1]
    except Exception as e:
        print(f"Audio Initialization Error: {e}")
        return None, -65.25, 0.0

def main():
    # 1. Initialize OS Audio Utilities
    volume_control, min_db, max_db = initialize_system_audio()

    # 2. Load WorldSkills AI Model & Force GPU Execution (Criterion A1 Validation)
    # Note: Use your fine-tuned hand-pose model weight path here
    try:
        model = YOLO('yolov11n-pose.pt').to('cuda') 
        print("M1 Check: YOLOv8/YOLOv11 model loads successfully on GPU/CUDA.")
    except Exception:
        print("CUDA Fallback: Loading model on CPU (Verify environment drivers!).")
        model = YOLO('yolov11n-pose.pt')

    # 3. Initialize Camera Feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Active target tracking indices based on MediaPipe/YOLO Hand Joint mappings
    THUMB_TIP_ID = 4
    INDEX_TIP_ID = 8
    PINKY_TIP_ID = 20
    
    # Toggle to pass Criterion A1 'pinky tip live change test'
    use_pinky_as_active = False 
    
    current_vol_percent = 50
    smooth_vol_bar = 400

    print("Pipeline online. Press 'p' to swap tracking to Pinky. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Camera feed dropped.")
            break

        # Mirror frame for intuitive left/right human interaction
        frame = cv2.flip(frame, 1)
        
        # 4. Run Live Model Inference
        # We track keypoints (poses) for objects in the stream
        results = model(frame, stream=True, verbose=False)

        for result in results:
            # Check if any skeletal keypoint poses were detected in the frame
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                
                # Extract all coordinate tensors [num_detected_hands, 21 landmarks, x/y dimensions]
                keypoints_array = result.keypoints.xy.cpu().numpy()
                boxes = result.boxes.cpu().numpy()
                
                num_hands = len(keypoints_array)

                # Criterion A2: Loop through all detected hands simultaneously
                for i in range(num_hands):
                    hand_landmarks = keypoints_array[i]
                    
                    # Ensure all 21 localized landmark joints are populated and valid
                    if len(hand_landmarks) < 21 or np.all(hand_landmarks == 0):
                        continue

                    # Fetch bounding box coordinates for rendering
                    bbox = boxes[i].xyxy[0].astype(int)
                    conf_score = boxes[i].conf[0]

                    # Extract class ID or determine handedness label programmatically
                    # For this template, we classify based on box orientation or dataset index
                    class_id = int(boxes[i].cls[0]) if boxes[i].cls is not None else 0
                    hand_label = "Right Hand" if class_id == 0 else "Left Hand"

                    # Draw Independent Bounding Box & Label with Confidence (Criterion A1/A2)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{hand_label} {conf_score:.2f}", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw all 21 skeletal connection points on the active hand matrix
                    for kp in hand_landmarks:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

                    # 5. Extract Feature Target Anchors
                    thumb_tip = hand_landmarks[THUMB_TIP_ID]
                    active_tip_id = PINKY_TIP_ID if use_pinky_as_active else INDEX_TIP_ID
                    active_tip = hand_landmarks[active_tip_id]

                    x1, y1 = int(thumb_tip[0]), int(thumb_tip[1])
                    x2, y2 = int(active_tip[0]), int(active_tip[1])

                    # Render tracking reference endpoints
                    cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
                    cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
                    
                    # Draw Distance Line Between Tips (Criterion A1 requirement)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # 6. Calculate Math Geometry & Linear Volumetric Mapping
                    distance = math.hypot(x2 - x1, y2 - y1)
                    
                    # Calibration constraints (Adjust 30 and 250 pixels based on camera distance)
                    min_detection_dist = 30
                    max_detection_dist = 220
                    
                    # Map distance to percentage (0 - 100)
                    current_vol_percent = np.interp(distance, [min_detection_dist, max_detection_dist], [0, 100])
                    smooth_vol_bar = np.interp(distance, [min_detection_dist, max_detection_dist], [400, 150])
                    current_vol_percent = int(np.clip(current_vol_percent, 0, 100))

                    # 7. Pipe Values Directly into Hardware OS Master Mixer (Criterion A2)
                    if volume_control is not None:
                        # Interp to native decibel scale
                        target_db = np.interp(current_vol_percent, [0, 100], [min_db, max_db])
                        volume_control.SetMasterVolumeLevel(target_db, None)

        # 8. Render Continuous Responsive Graphical UI Assets (Criterion A1)
        # Graphical Tracking Bar background
        cv2.rectangle(frame, (50, 150), (85, 400), (100, 100, 100), 3)
        # Dynamic green inner level indicator fill
        cv2.rectangle(frame, (50, int(smooth_vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
        # Label overlays
        cv2.putText(frame, f"{current_vol_percent}%", (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        mode_text = "Active Tracker: PINKY" if use_pinky_as_active else "Active Tracker: INDEX"
        cv2.putText(frame, mode_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show Output Windows Frame
        cv2.imshow("WorldSkills Region X - AI Module 1 Run", frame)

        # Handle Live Key Interventions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Graceful exit execution
            break
        elif key == ord('p'): # Live switch swap test to pinky tip
            use_pinky_as_active = not use_pinky_as_active

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
