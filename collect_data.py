
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_FILE = "data.csv"
MODEL_PATH = "hand_landmarker.task"

# MediaPipe new API
BaseOptions = python.BaseOptions

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

# Extract landmarks
def extract_landmarks(hand_landmarks):

    points = []

    for landmark in hand_landmarks:
        points.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    points = np.array(points).reshape(21, 3)

    # Normalize relative to wrist
    points = points - points[0]

    scale = np.max(
        np.linalg.norm(points[:, :2], axis=1)
    )

    if scale > 0:
        points = points / scale

    return points.flatten()


# CSV columns
columns = (
    [f"x{i}" for i in range(21)] +
    [f"y{i}" for i in range(21)] +
    [f"z{i}" for i in range(21)] +
    ["label"]
)

if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=columns).to_csv(
        DATA_FILE, index=False
    )

cap = cv2.VideoCapture(0)

current_label = None
sample_count = 0
frame_timestamp = 0

with vision.HandLandmarker.create_from_options(
    options
) as detector:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Cannot access webcam")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        frame_timestamp += 33

        results = detector.detect_for_video(
            mp_image, frame_timestamp
        )

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]

            # Draw landmarks
            for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame, (x, y), 5,
                    (0, 255, 0), -1
                )

            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]

            for start, end in connections:

                x1 = int(hand[start].x * frame.shape[1])
                y1 = int(hand[start].y * frame.shape[0])

                x2 = int(hand[end].x * frame.shape[1])
                y2 = int(hand[end].y * frame.shape[0])

                cv2.line(
                    frame, (x1, y1), (x2, y2),
                    (255, 0, 0), 2
                )

            # Collect data when a letter is selected
            if current_label is not None:

                features = extract_landmarks(hand)

                row = list(features) + [current_label]

                pd.DataFrame(
                    [row], columns=columns
                ).to_csv(
                    DATA_FILE,
                    mode="a",
                    header=False,
                    index=False
                )

                sample_count += 1

        cv2.putText(
            frame,
            f"Label: {current_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 0), 2
        )

        cv2.putText(
            frame,
            "Press A-Z to collect | Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        cv2.imshow(
            "Sign Language Data Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif 65 <= key <= 90:
            current_label = chr(key)
            print("Collecting:", current_label)

        elif 97 <= key <= 122:
            current_label = chr(key - 32)
            print("Collecting:", current_label)

cap.release()
cv2.destroyAllWindows()

print("Data collection completed!")
