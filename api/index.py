from flask import Blueprint, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from PIL import Image
import io
import base64
import face_recognition
import os
import requests

predict_bp = Blueprint("predict_bp", __name__)

# URL endpoint Laravel untuk mendapatkan file model dan label encoder
model_url = 'http://127.0.0.1:8000/get-model'
label_encoder_url = 'http://127.0.0.1:8000/get-label-encoder'

# Path lokal untuk menyimpan model dan label encoder
MODEL_PATH = 'face_model.h5'
LABEL_PATH = 'label_encoder.json'

# Fungsi untuk mendownload model jika belum ada atau ingin diperbarui
def download_model_files():
    try:
        response_model = requests.get(model_url)
        if response_model.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response_model.content)
        else:
            print("Error: Unable to download the model file")
            exit()

        response_label_encoder = requests.get(label_encoder_url)
        if response_label_encoder.status_code == 200:
            with open(LABEL_PATH, 'wb') as f:
                f.write(response_label_encoder.content)
        else:
            print("Error: Unable to download the label encoder file")
            exit()

        print("Model and label encoder downloaded successfully")
    except Exception as e:
        print("Error downloading model files:", e)
        exit()

# Unduh model dan label encoder saat pertama kali Flask dijalankan
download_model_files()

# Load model dan label encoder
model = load_model(MODEL_PATH)
with open(LABEL_PATH, 'r') as f:
    class_names = json.load(f)

@predict_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        img_data = data.get("image")

        if not img_data:
            return jsonify({"error": "Image data not found"}), 400

        img_bytes = base64.b64decode(img_data.split(',')[1])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        face_locations = face_recognition.face_locations(img_array)
        if len(face_locations) == 0:
            return jsonify({"error": "No faces detected"}), 400

        results = []
        for idx, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            face_img = img_array[top:bottom, left:right]
            face_img = Image.fromarray(face_img).resize((160, 160))
            face_array = image.img_to_array(face_img) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = model.predict(face_array)[0]
            max_index = np.argmax(prediction)
            confidence = float(prediction[max_index])
            label = class_names[max_index] if confidence > 0.8 else "unknown"

            results.append({
                "label": label,
                "confidence": confidence,
                "location": face_location
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
