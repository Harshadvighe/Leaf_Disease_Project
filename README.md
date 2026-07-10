# 🌿 Leaf Disease Detection Project

An AI-powered web application and live webcam tool that detects leaf diseases (specifically for corn/maize) and provides recommendations for fertilizers and pesticides.

---

## 📸 Sample Images from Dataset

Here are a few sample images from the training dataset representing each category:

| Blight | Common Rust | Gray Leaf Spot | Healthy |
|:---:|:---:|:---:|:---:|
| <img src="dataset/Blight/Corn_Blight%20(1).jpg" width="200" /> | <img src="dataset/Common_Rust/Corn_Common_Rust%20(1).jpg" width="200" /> | <img src="dataset/Gray_Leaf_Spot/Corn_Gray_Spot%20(1).jpg" width="200" /> | <img src="dataset/Healthy/Corn_Health%20(1).jpg" width="200" /> |

*(Note: These images are dynamically loaded from the dataset directory in the repository.)*

---

## 🛠️ Technology Stack

This project is built using a modern AI and web tech stack:
- **Python 3.11**: The core programming language used for both the backend and the ML modeling.
- **TensorFlow / Keras**: Used for building, training, and running the Deep Learning model.
- **OpenCV (cv2)**: Used for real-time computer vision, webcam feed capture, and leaf detection (green pixel masking).
- **Flask**: A lightweight Python web framework used to serve the web interface and handle RESTful image upload requests.
- **NumPy**: For numerical operations and array manipulations required by OpenCV and TensorFlow.

---

## 🧠 Machine Learning Concepts

This project utilizes **Deep Learning** and **Computer Vision** to classify leaf diseases.

### 1. Transfer Learning (MobileNetV2)
Instead of training a Convolutional Neural Network (CNN) from scratch, this project uses **MobileNetV2**, a highly efficient CNN architecture pre-trained on the massive *ImageNet* dataset. 
- The base layers of MobileNetV2 are **frozen** (`base.trainable=False`), meaning we keep its pre-learned ability to detect edges, textures, and shapes.
- We replace the top layers with a `GlobalAveragePooling2D` layer and a custom `Dense` output layer with a `softmax` activation function to classify the image into our **4 specific categories**.

### 2. Image Preprocessing & Augmentation
- Images are resized to **224x224 pixels** to match MobileNetV2's expected input shape.
- Pixel values are **normalized** (rescaled by `1./255`) to fall between 0 and 1, helping the neural network converge faster during training.
- The dataset is split into **80% training** and **20% validation** using `ImageDataGenerator`.

### 3. Model Training Parameters
- **Optimizer**: `Adam` - dynamically adjusts the learning rate for efficient training.
- **Loss Function**: `categorical_crossentropy` - standard for multi-class classification problems.
- **Epochs**: Trained for `5` epochs with a batch size of `32`.

### 4. Real-time Leaf Detection (OpenCV)
Before the AI predicts a disease from the webcam feed, the system ensures a leaf is actually present in the frame:
- The webcam frame is converted from BGR to **HSV color space**.
- A mask is applied to detect **green pixels** (representing a leaf).
- If the green pixel density is greater than **8%**, the frame is passed to the AI model for prediction. Otherwise, it outputs "No Leaf Detected".

---

## 🚀 Installation

1. Make sure you have **Python 3.11** installed.
2. Clone this repository and open a terminal in the folder.
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage Guide

### 1. Run the Web Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000/`. You can upload images or use the live camera feed from the web interface.

### 2. Live Webcam Detection (Standalone)
To use your webcam for real-time disease detection in a standalone OpenCV window:
```bash
python webcam.py
```
*(Press `q` to exit the webcam stream).*

### 3. Command Line Prediction
To predict a specific image via command line:
```bash
python predict.py path/to/image.jpg
```

---

## 📂 Dataset
This project uses images from the **PlantVillage dataset**. The dataset is structured into 4 classes: `Blight`, `Common_Rust`, `Gray_Leaf_Spot`, and `Healthy`.
To train a new model, place the respective class folders inside the `dataset/` directory and run `python train.py`.
