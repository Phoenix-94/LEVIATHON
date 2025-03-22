import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("models/marine_species.keras")

# Class names (Replace with actual species names)
class_names = ["Shark", "Dolphin", "Whale", "Jellyfish"]

# Function to preprocess an image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict species
image_path = "path_to_test_image.jpg"  # Replace with actual image path
img = preprocess_image(image_path)
prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]

print(f"Predicted Species: {predicted_class}")
