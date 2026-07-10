import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("model.keras")

# Class names
classes = [
    "Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy"
]

# Check image path
if len(sys.argv) < 2:
    print("Usage: python predict.py image.jpg")
    exit()

img_path = sys.argv[1]

# Load image
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array, verbose=0)

index = np.argmax(prediction)
confidence = prediction[0][index] * 100

print("--------------------------------")
print("Prediction :", classes[index])
print("Confidence :", round(confidence,2), "%")
print("--------------------------------")