import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

DATASET_PATH = "data/leapGestRecog"

images = []
labels = []

for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        # FIXED LABELS (0-9)
        label = int(gesture.split("_")[0]) - 1

        for img_name in os.listdir(gesture_path):

            img_path = os.path.join(gesture_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))
            img = img / 255.0

            images.append(img)
            labels.append(label)

X = np.array(images)
y = np.array(labels)

# 10 classes: 0-9
y = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

os.makedirs("model", exist_ok=True)

model.save("model/hand_gesture_model.h5")

print("Model Saved Successfully!")