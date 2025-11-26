import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, Button, Label, filedialog, Canvas, Frame, NW
from PIL import Image, ImageTk

IMG_SIZE = 128
MODEL_PATH = "best_model.keras"
CLASS_NAMES = ["0", "1", "2", "3", "4", "5"]

model = load_model(MODEL_PATH)

def extract_roi(gray):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    pad_w, pad_h = int(bw * 0.35), int(bh * 0.35)
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(w, x + bw + pad_w), min(h, y + bh + pad_h)
    roi = gray[y1:y2, x1:x2]
    rh, rw = roi.shape
    size = max(rh, rw)
    pad_top, pad_bottom = (size - rh) // 2, size - rh - (size - rh) // 2
    pad_left, pad_right = (size - rw) // 2, size - rw - (size - rw) // 2
    return cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

def apply_clahe(gray):
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    roi = extract_roi(img)
    clahe_img = apply_clahe(roi)
    img_resized = cv2.resize(clahe_img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0
    img_norm = np.expand_dims(img_norm, axis=(0, -1))
    return roi, img_norm

def select_and_predict():
    path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return
    roi, tensor = preprocess_image(path)
    pred = model.predict(tensor, verbose=0)
    idx = np.argmax(pred)
    conf = np.max(pred) * 100
    result_label.config(text=f"Kết quả: {CLASS_NAMES[idx]}   |   Độ tin cậy: {conf:.2f}%")
    orig = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (450, 450))
    img_tk = ImageTk.PhotoImage(Image.fromarray(orig))
    canvas_img.image = img_tk
    canvas_img.create_image(0, 0, anchor=NW, image=img_tk)

root = Tk()
root.title("Hand Recognition – Neural Network")
root.configure(bg="white")
root.iconbitmap("icon.ico")
header = Frame(root, bg="white")
header.pack(pady=10)

try:
    logo_img = Image.open("logo.jpg").resize((80, 80))
    logo_photo = ImageTk.PhotoImage(logo_img)
    Label(header, image=logo_photo, bg="white").grid(row=0, column=0, rowspan=2, padx=15)
except:
    pass

Label(header, text="ĐẠI HỌC CÔNG NGHIỆP HÀ NỘI", font=("Arial", 22, "bold"), bg="white").grid(row=0, column=1, pady=3)
Label(header, text="Mạng nơ ron nhân tạo nhận diện số biểu diễn bằng ngón tay", font=("Arial", 17), bg="white").grid(row=1, column=1, pady=3)

Button(root, text="Chọn ảnh", command=select_and_predict, font=("Arial", 16)).pack(pady=15)
result_label = Label(root, text="", font=("Arial", 18, "bold"), fg="red", bg="white")
result_label.pack()
canvas_img = Canvas(root, width=450, height=450, bg="white", highlightthickness=1, highlightbackground="black")
canvas_img.pack(pady=10)

root.mainloop()
