from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load trained model
model = load_model("model/hand_gesture_model.h5")

# Gesture Labels
gesture_labels = [
    "Palm",
    "L",
    "Fist",
    "Fist Moved",
    "Thumb",
    "Index",
    "OK",
    "Palm Moved",
    "C",
    "Down"
]

# Upload Folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file and file.filename != "":

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(image_path)

            # Read Image
            img = cv2.imread(image_path)

            # Preprocess
            img = cv2.resize(img, (64, 64))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Prediction
            pred = model.predict(img, verbose=0)

            predicted_index = np.argmax(pred)

            prediction = gesture_labels[predicted_index]
            

            with open("history.txt", "a") as f:
                 
             f.write(f"{file.filename} -> {prediction}\n")

            print("Prediction:", prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)