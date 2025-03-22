from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/marine_species.keras")

# Class names
class_names = ["Shark", "Dolphin", "Whale", "Jellyfish"]

# Image preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join("static", "uploaded.jpg")
        file.save(filepath)

        img = preprocess_image(filepath)
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template("result.html", species=predicted_class, image=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
