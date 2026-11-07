
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
import os

app = Flask(__name__)

# 1. Load trained model
model_path = os.path.join(
    app.root_path, "sign_language_model.keras"
)
model = tf.keras.models.load_model(model_path)

# 2. Alphabet labels
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]

# 3. Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

CONFIDENCE_THRESHOLD = 0.40
stop_event = threading.Event()
current_prediction = "Show your hand"
current_confidence = 0.0


# 4. Homepage
@app.route("/")
def home():
    return render_template("index.html")


# 5. Webcam video generator
def generate_frames():
    global current_prediction, current_confidence
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        cap.release()
        return
    
    last_letter = "Show your hand"
    last_confidence = 0.0

    try:
        while not stop_event.is_set():

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            # 6. Detect hand
            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                x_coords = [
                    lm.x for lm in hand_landmarks.landmark
                ]

                y_coords = [
                    lm.y for lm in hand_landmarks.landmark
                ]

                x_min = max(0, int(min(x_coords) * w) - 40)
                x_max = min(w, int(max(x_coords) * w) + 40)
                y_min = max(0, int(min(y_coords) * h) - 40)
                y_max = min(h, int(max(y_coords) * h) + 40)

                hand_roi = frame[y_min:y_max, x_min:x_max]

                if hand_roi.size > 0:

                    # 7. Preprocess hand image
                    gray = cv2.cvtColor(
                        hand_roi, cv2.COLOR_BGR2GRAY
                    )

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
                        square_image, (28, 28)
                    )

                    normalized = resized.astype(
                        "float32"
                    ) / 255.0

                    input_image = normalized.reshape(
                        1, 28, 28, 1
                    )

                    # 8. Predict alphabet
                    prediction = model(
                        input_image, training=False
                    ).numpy()[0]

                    predicted_index = int(
                        np.argmax(prediction)
                    )

                    confidence = float(
                        prediction[predicted_index]
                    )

                    if (
                        predicted_index < len(labels)
                        and confidence >= CONFIDENCE_THRESHOLD
                    ):
                        last_letter = labels[predicted_index]
                        last_confidence = confidence
                        current_prediction = last_letter
                        current_confidence = last_confidence        

                    cv2.rectangle(
                        frame,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2
                    )

            # 9. Display prediction
            cv2.rectangle(
                frame, (10, 10), (450, 110),
                (30, 30, 30), -1
            )

            cv2.putText(
                frame,
                "Prediction: " + last_letter,
                (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {last_confidence * 100:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # 10. Convert frame for browser
            ret, buffer = cv2.imencode(
                ".jpg", frame
            )

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    finally:
        cap.release()


# 11. Start webcam stream
@app.route("/video_feed")
def video_feed():

    stop_event.clear()

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# 12. Stop webcam
@app.route("/stop_camera", methods=["POST"])
def stop_camera():

    stop_event.set()

    return jsonify({
        "success": True,
        "message": "Webcam stopped"
    })

@app.route("/get_prediction")
def get_prediction():
    return jsonify({
        "letter": current_prediction,
        "confidence": current_confidence
    })

# 13. Run Flask
if __name__ == "__main__":

    print("Starting Sign Language Dashboard...")

    app.run(
        debug=True,
        threaded=True,
        use_reloader=False
    )
