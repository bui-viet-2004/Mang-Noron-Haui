import os
# Tắt log TensorFlow để terminal gọn hơn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense,
    Input, Dropout, BatchNormalization, LeakyReLU
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping

# EarlyStopping: tự dừng khi val_loss không giảm
early_stop = EarlyStopping(
    monitor='val_loss', patience=8,
    restore_best_weights=True
)

# ================================
# 1. TẠO BỘ SINH DỮ LIỆU ẢNH
# ================================
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,          # Chuẩn hóa về [0,1]
    rotation_range=15,          # Xoay ±15°
    width_shift_range=0.1,      # Dịch ngang 10%
    height_shift_range=0.1,     # Dịch dọc 10%
    zoom_range=0.1,             # Zoom ±10%
    brightness_range=[0.5, 1.7],# Tăng giảm độ sáng
    horizontal_flip=True,       # Lật ngang
    validation_split=0.2        # 20% dùng cho validation
)

# Bộ dữ liệu huấn luyện (ảnh grayscale)
train_data = train_gen.flow_from_directory(
    'data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'
)

# Bộ dữ liệu kiểm thử
val_data = train_gen.flow_from_directory(
    'data',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)

# ================================
# 2. KIẾN TRÚC CNN
# ================================
model = Sequential([

    # Ảnh đầu vào 128x128, 1 kênh (grayscale)
    Input(shape=(128, 128, 1)),

    # Block 1
    Conv2D(32, (3,3), padding='same'),
    LeakyReLU(0.01),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.15),

    # Block 2
    Conv2D(64, (3,3), padding='same'),
    LeakyReLU(0.01),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Block 3
    Conv2D(128, (3,3), padding='same'),
    LeakyReLU(0.01),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    # Block 4
    Conv2D(256, (3,3), padding='same'),
    LeakyReLU(0.01),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    # Block 5
    Conv2D(512, (3,3), padding='same'),
    LeakyReLU(0.01),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.35),

    # Tính trung bình theo không gian → giảm số tham số
    GlobalAveragePooling2D(),

    # FC layer
    Dense(128),
    LeakyReLU(0.01),
    Dropout(0.5),

    # Lớp đầu ra 6 lớp
    Dense(6, activation='softmax')
])

# ================================
# 3. BIÊN DỊCH MÔ HÌNH
# ================================
# AdamW giúp giảm overfitting tốt hơn Adam
model.compile(
    optimizer=AdamW(learning_rate=0.0007, weight_decay=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ================================
# 4. HUẤN LUYỆN
# ================================
history = model.fit(
    train_data,
    epochs=25,
    validation_data=val_data,
    callbacks=[early_stop]
)

# ================================
# 5. LƯU MÔ HÌNH
# ================================
model.save('cnn_model.keras')
print("Mô hình đã lưu: cnn_model.keras")


# ================================
# 6. VẼ ĐỒ THỊ TRAINING
# ================================
import matplotlib.pyplot as plt

# Lấy dữ liệu accuracy và loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Vẽ Accuracy
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Vẽ Loss
plt.subplot(1,2,2)
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
