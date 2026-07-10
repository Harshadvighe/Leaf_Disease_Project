# Leaf Disease Detection Project

An AI-powered web application and live webcam tool that detects leaf diseases (specifically for corn/maize) and provides recommendations for fertilizers and pesticides.

## Features
- **Web App**: Upload leaf images to a local Flask web server and instantly see the predicted disease, confidence level, and treatment recommendations.
- **Live Webcam**: Real-time leaf disease detection using your computer's webcam directly in the browser or via OpenCV.
- **Supported Diseases**:
  - Blight
  - Common Rust
  - Gray Leaf Spot
  - Healthy (No Disease)

## Project Structure
- `app.py`: Main Flask application that serves the web UI and handles predictions.
- `webcam.py`: Script for running live detection via webcam.
- `train.py`: Script to train the TensorFlow/Keras model.
- `predict.py`: Script to test the model on a single image via command line.
- `requirements.txt`: Python dependencies.
- `model.keras` / `model.tflite`: Pre-trained models.

## Installation

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

## Usage

### 1. Run the Web Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000/`. You can upload images or use the live camera feed from the web interface.

### 2. Live Webcam Detection
To use your webcam for real-time disease detection via OpenCV window:
```bash
python webcam.py
```
*(Press `q` to exit the webcam stream).*

### 3. Command Line Prediction
To predict a specific image via command line:
```bash
python predict.py path/to/image.jpg
```

## Dataset
This project uses images from the **PlantVillage dataset**. To train a new model, place the respective class folders inside the `dataset/` directory and run `python train.py`.
