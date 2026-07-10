import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.keras")

# Class names
classes = [
    "Blight",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy"
]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press Q to Exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Copy frame for display
    display = frame.copy()

    # Resize for model
    img = cv2.resize(frame, (224, 224))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img.astype("float32") / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    confidence = prediction[0][index] * 100

    disease = classes[index]

    # Display Prediction
    cv2.putText(
        display,
        f"Disease : {disease}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display,
        f"Confidence : {confidence:.2f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Leaf Disease Detection", display)

    key = cv2.waitKey(1)

    if key == ord("q") or key == ord("Q"):
        break

cap.release()

cv2.destroyAllWindows()