import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

active_fingertip = 8

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, c = frame.shape

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        thumb_tip = hand.landmark[4]
        finger_tip = hand.landmark[active_fingertip]

        x1 = int(thumb_tip.x * w)
        y1 = int(thumb_tip.y * h)

        x2 = int(finger_tip.x * w)
        y2 = int(finger_tip.y * h)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.circle(frame, (cx, cy), 8, (0, 255, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [20, 250], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol, None)

        vol_percent = np.interp(length, [20, 250], [0, 100])

        vol_bar = np.interp(length, [20, 250], [400, 150])

        cv2.rectangle(
            frame,
            (50, 150),
            (85, 400),
            (255, 255, 255),
            3
        )

        cv2.rectangle(
            frame,
            (50, int(vol_bar)),
            (85, 400),
            (0, 255, 0),
            cv2.FILLED
        )

        cv2.putText(
            frame,
            f'{int(vol_percent)}%',
            (40, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        fingertip_name = "INDEX" if active_fingertip == 8 else "PINKY"

        cv2.putText(
            frame,
            f'Finger: {fingertip_name}',
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "I = Index Finger",
        (10, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "P = Pinky Finger",
        (10, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Day 1 - Hand Gesture Volume Control", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('i'):
        active_fingertip = 8

    elif key == ord('p'):
        active_fingertip = 20

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()