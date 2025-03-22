import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model
model = tf.keras.models.load_model("models/marine_species_model.h5")

# Load label names
label_dict = {0: "Fish A", 1: "Fish B", 2: "Fish C"}  # Replace with actual labels

def classify_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    
    return label_dict[class_idx], np.max(prediction) * 100
