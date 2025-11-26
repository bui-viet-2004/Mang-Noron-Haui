import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Activation,
    Dense, Input, BatchNormalization,
    Dropout, Flatten
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import regularizers

train_gen = ImageDataGenerator(
    rescale=1 / 255.,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.7, 1.3],
    shear_range=0.05,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    "data",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    "data",
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale',
    shuffle=False
)

weight_decay = 1e-4

model = Sequential([
    Input(shape=(128, 128, 1)),  # Input: (128, 128, 1)

    # Block 1
    Conv2D(32, (4, 4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),  # -> (128,128,32)
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # -> (64,64,32)
    Dropout(0.1),

    # Block 2
    Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),  # -> (64,64,64)
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # -> (32,32,64)
    Dropout(0.15),

    # Block 3
    Conv2D(128, (4, 4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),  # -> (32,32,128)
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # -> (16,16,128)
    Dropout(0.2),

    # Block 4
    Conv2D(256, (4, 4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),  # -> (16,16,256)
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # -> (8,8,256)
    Dropout(0.25),

    # Block 5
    Conv2D(512, (4, 4), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),  # -> (8,8,512)
    Activation('elu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # -> (4,4,512)
    Dropout(0.3),

    # Fully connected
    Flatten(),  # -> (4*4*512 = 8192)
    Dense(256, activation='elu'),  # -> (256)
    Dense(6, activation='softmax')  # -> (6 classes)
])


model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=5e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.35,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_data,
    epochs=40,
    validation_data=val_data,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

model.save("model_final.keras")
print("Saved: model_final.keras")