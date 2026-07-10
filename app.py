from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ==========================
# Upload Folder
# ==========================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================
# Load AI Model
# ==========================

model = load_model("model.keras")


# ==========================
# Disease Classes
# ==========================

classes = [

    "Blight",

    "Common_Rust",

    "Gray_Leaf_Spot",

    "Healthy"

]


# ==========================
# Disease Information
# ==========================

disease_info = {

    "Blight":{

        "description":"Leaf Blight is caused by fungal infection.",

        "fertilizer":"Balanced NPK Fertilizer",

        "pesticide":"Mancozeb"

    },

    "Common_Rust":{

        "description":"Common Rust appears as reddish brown pustules.",

        "fertilizer":"Potassium Rich Fertilizer",

        "pesticide":"Propiconazole"

    },

    "Gray_Leaf_Spot":{

        "description":"Gray Leaf Spot is caused by Cercospora fungus.",

        "fertilizer":"Nitrogen + Zinc",

        "pesticide":"Azoxystrobin"

    },

    "Healthy":{

        "description":"Leaf is Healthy.",

        "fertilizer":"Continue Normal Fertilizer",

        "pesticide":"Not Required"

    }

}


# ==========================
# Webcam
# ==========================

camera = cv2.VideoCapture(0)
# ==========================
# Check Green Color (Leaf Detection)
# ==========================

def is_leaf(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])

    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_pixels = cv2.countNonZero(mask)

    total_pixels = frame.shape[0] * frame.shape[1]

    percentage = (green_pixels / total_pixels) * 100

    return percentage > 8
# ==========================
# Live Camera Stream
# ==========================

def generate_frames():
    

    while True:

        success, frame = camera.read()
        # Check whether a green leaf is present
        leaf_found = is_leaf(frame)

        if not success:
            break

        # Flip Image
        frame = cv2.flip(frame, 1)

        # Resize for AI Model
        img = cv2.resize(frame, (224, 224))

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype("float32") / 255.0

        # Add Batch Dimension
        img = np.expand_dims(img, axis=0)

        # Prediction
        if leaf_found:
    
            prediction = model.predict(img, verbose=0)

            index = np.argmax(prediction)

            confidence = float(prediction[0][index]) * 100

            disease = classes[index]

        else:

            disease = "No Leaf Detected"

            confidence = 0

        # Display Text
        # Text color
        if leaf_found:
            disease_color = disease_color,      # Green
            confidence_color = (0, 255, 255) # Yellow
        else:
            disease_color = (0, 0, 255)      # Red
            confidence_color = (0, 0, 255)   # Red
        cv2.putText(
            frame,
            f"Disease : {disease}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if leaf_found:
            confidence_text = f"Confidence : {confidence:.2f}%"
        else:
            confidence_text = "Confidence : --"

        cv2.putText(
            frame,
            confidence_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            confidence_color,
            2
        )
            

        # Encode Image
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        # Send Frame to Browser
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


# ==========================
# Browser Video Feed
# ==========================

@app.route("/video_feed")
def video_feed():

    return Response(

        generate_frames(),

        mimetype="multipart/x-mixed-replace; boundary=frame"

    )


# ==========================
# Live Camera Page
# ==========================

@app.route("/live")
def live():

    return render_template("live.html")
# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# Image Upload Prediction
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file selected."

    file = request.files["file"]

    if file.filename == "":
        return "No image selected."

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Load Image
    img = image.load_img(
        filepath,
        target_size=(224,224)
    )

    img = image.img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    confidence = float(prediction[0][index]) * 100

    disease = classes[index]

    info = disease_info[disease]

    return render_template(

        "result.html",

        image=file.filename,

        disease=disease,

        confidence=round(confidence,2),

        description=info["description"],

        fertilizer=info["fertilizer"],

        pesticide=info["pesticide"]

    )


# ==========================
# Run Server
# ==========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )