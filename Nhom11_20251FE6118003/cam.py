import os

os.chdir(os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# --- 1. Load mo hinh CNN ---
model = load_model('best_model.keras')
labels = ['0', '1', '2', '3', '4', '5']

# --- 2. Cau hinh Mediapipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 3. Mo webcam ---
cap = cv2.VideoCapture(0)

# --- 4. Lam muot ---
smooth_factor = 0.2
prev_box = None
pred_history = deque(maxlen=5)


# --- 6. Tien xu ly ---
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img


def smooth_label(pred_history):
    if not pred_history:
        return None
    counts = np.bincount(pred_history)
    return np.argmax(counts)


def smooth_box(prev, current, alpha=0.2):
    if prev is None:
        return current
    x_min = int(prev[0] * (1 - alpha) + current[0] * alpha)
    y_min = int(prev[1] * (1 - alpha) + current[1] * alpha)
    x_max = int(prev[2] * (1 - alpha) + current[2] * alpha)
    y_max = int(prev[3] * (1 - alpha) + current[3] * alpha)
    return (x_min, y_min, x_max, y_max)


# --- 7. Vong lap chinh ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            # === LẤY BOUNDING BOX TỪ LANDMARK ===
            x_coords = [lm.x * w for lm in hand.landmark]
            y_coords = [lm.y * h for lm in hand.landmark]

            x_min = int(max(0, min(x_coords)))
            x_max = int(min(w, max(x_coords)))
            y_min = int(max(0, min(y_coords)))
            y_max = int(min(h, max(y_coords)))

            # === TẠO ROI VUÔNG (SQUARE ROI) ===
            box_w = x_max - x_min
            box_h = y_max - y_min
            side = int(max(box_w, box_h) * 1.4)  # scale 1.4 lần

            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, cx + side // 2)
            y2 = min(h, cy + side // 2)

            # === SMOOTH KHUNG ===
            smoothed_box = smooth_box(prev_box, (x1, y1, x2, y2), smooth_factor)
            prev_box = smoothed_box
            x1, y1, x2, y2 = smoothed_box

            # === CẮT ROI ===
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                input_img = preprocess_frame(roi)
                pred = model.predict(input_img, verbose=0)
                pred_idx = np.argmax(pred)
                pred_history.append(pred_idx)
                pred_label = smooth_label(pred_history)
                conf = np.max(pred)

                text = f"{labels[pred_label]} ({conf * 100:.1f}%)"
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + 150, y1), (0, 0, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # === VE BOUNDING BOX ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    else:
        prev_box = None
        pred_history.clear()
        cv2.putText(frame, "Khong thay ban tay", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Recognition (Dynamic ROI)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
