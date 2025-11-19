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
model = load_model('cnn_model.keras')
labels = ['0', '1', '2', '3', '4', '5']

# --- 2. Cau hinh Mediapipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,          # False: video stream, True: anh static
    max_num_hands=1,                  # So tay toi da phat hien
    min_detection_confidence=0.5,     # Do tin cay toi thieu de phat hien tay
    min_tracking_confidence=0.5       # Do tin cay toi thieu khi theo doi tay da phat hien
)

# --- 3. Mo webcam ---
cap = cv2.VideoCapture(1)

# --- 4. Tham so lam muot khung va nhan ---
smooth_factor = 0.2                  # He so lam muot khung (0-1)
prev_box = None                       # Luu khung truoc do
pred_history = deque(maxlen=5)        # Luu lich su nhan de lam muot nhan

# --- 5. Kich thuoc ROI co dinh ---
ROI_SIZE = 270                        # Khung vuong quanh tam tay (pixel)

# --- 6. Ham tien xu ly anh truoc khi predict ---
def preprocess_frame(frame):
    """
    Chuyen anh sang grayscale, resize ve (128,128),
    chuan hoa gia tri pixel [0,1] va them batch + channel
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(gray, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1,128,128,1)
    return img

# --- 7. Ham lam muot nhan ---
def smooth_label(pred_history):
    """
    Trung binh nhan trong cac khung gan nhat
    """
    if not pred_history:
        return None
    counts = np.bincount(pred_history)
    return np.argmax(counts)

# --- 8. Ham lam muot khung ---
def smooth_box(prev, current, alpha=0.2):
    """
    Lam muot toa do khung ROI de khung di chuyen mem mai
    """
    if prev is None:
        return current
    x_min = int(prev[0] * (1 - alpha) + current[0] * alpha)
    y_min = int(prev[1] * (1 - alpha) + current[1] * alpha)
    x_max = int(prev[2] * (1 - alpha) + current[2] * alpha)
    y_max = int(prev[3] * (1 - alpha) + current[3] * alpha)
    return (x_min, y_min, x_max, y_max)

# --- 9. Vong lap chinh ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Tinh tam tay trung binh
            cx = int(np.mean([lm.x for lm in hand_landmarks.landmark]) * w)
            cy = int(np.mean([lm.y for lm in hand_landmarks.landmark]) * h)

            # Tao khung ROI co dinh quanh tam tay
            x1 = max(0, cx - ROI_SIZE // 2)
            y1 = max(0, cy - ROI_SIZE // 2)
            x2 = min(w, cx + ROI_SIZE // 2)
            y2 = min(h, cy + ROI_SIZE // 2)

            # Lam muot khung
            smoothed_box = smooth_box(prev_box, (x1, y1, x2, y2), smooth_factor)
            prev_box = smoothed_box
            x1, y1, x2, y2 = smoothed_box

            # Cat ROI
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                input_img = preprocess_frame(roi)
                prediction = model.predict(input_img, verbose=0)
                pred_index = np.argmax(prediction)
                pred_history.append(pred_index)
                pred_label = smooth_label(pred_history)
                confidence = np.max(prediction)

                # Hien thi nhan va do tin cay
                text = f"{labels[pred_label]} ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + 120, y1), (0, 0, 0), -1)
                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Ve khung ROI co dinh
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    else:
        prev_box = None
        pred_history.clear()
        cv2.putText(frame, "Khong thay ban tay", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Recognition (Fixed ROI)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
