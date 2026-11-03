
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

print("Loading sign language model...")

model = tf.keras.models.load_model(
    "sign_language_model.keras"
)

print("Model loaded successfully!")

# =====================================
# 2. ALPHABET LABELS
# =====================================

labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]

# =====================================
# 3. MEDIAPIPE
# =====================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# 4. SETTINGS
# =====================================

CONFIDENCE_THRESHOLD = 0.40

last_predicted_letter = "Show your hand"
last_confidence = 0.0

# =====================================
# 5. WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Webcam started!")
print("Show a sign. Press Q to quit.")

# =====================================
# 6. MAIN LOOP
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam frame.")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    # =================================
    # 7. HAND DETECTION
    # =================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        x_coordinates = [
            lm.x for lm in hand_landmarks.landmark
        ]

        y_coordinates = [
            lm.y for lm in hand_landmarks.landmark
        ]

        x_min = int(min(x_coordinates) * w)
        x_max = int(max(x_coordinates) * w)

        y_min = int(min(y_coordinates) * h)
        y_max = int(max(y_coordinates) * h)

        # Add padding
        padding = 40

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)

        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        hand_roi = frame[y_min:y_max, x_min:x_max]

        if hand_roi.size > 0:

            # =================================
            # 8. PREPROCESS HAND
            # =================================

            gray = cv2.cvtColor(
                hand_roi,
                cv2.COLOR_BGR2GRAY
            )

            # Maintain aspect ratio and center in
            # a square image instead of stretching

            roi_h, roi_w = gray.shape

            square_size = max(roi_h, roi_w)

            square_image = np.zeros(
                (square_size, square_size),
                dtype=np.uint8
            )

            y_offset = (square_size - roi_h) // 2
            x_offset = (square_size - roi_w) // 2

            square_image[
                y_offset:y_offset + roi_h,
                x_offset:x_offset + roi_w
            ] = gray

            resized = cv2.resize(
                square_image,
                (28, 28)
            )

            normalized = resized.astype(
                "float32"
            ) / 255.0

            input_image = normalized.reshape(
                1, 28, 28, 1
            )

            # =================================
            # 9. INSTANT PREDICTION
            # =================================

            prediction = model(
                input_image,
                training=False
            ).numpy()[0]

            predicted_index = int(
                np.argmax(prediction)
            )

            confidence = float(
                prediction[predicted_index]
            )

            if predicted_index < len(labels):

                if confidence >= CONFIDENCE_THRESHOLD:

                    last_predicted_letter = labels[
                        predicted_index
                    ]

                    last_confidence = confidence

            # Bounding box
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            if predicted_index < len(labels):

                current_letter = labels[predicted_index]

            else:

                current_letter = "Unknown"

            cv2.putText(
            frame,
            f"Confidence: {last_confidence * 100:.1f}%",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # =====================================
    # 10. DISPLAY RESULT
    # =====================================

    # Black background panel for readability
    cv2.rectangle(
        frame,
        (10, 10),
        (450, 155),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "SIGN LANGUAGE RECOGNITION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Prediction: " + last_predicted_letter,
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Confidence: {last_confidence * 100:.1f}%",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Sign Language Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================================
# 11. RELEASE
# =====================================

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Webcam closed.")
