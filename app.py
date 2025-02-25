import os
import torch
import numpy as np
import cv2
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
from deepface import DeepFace
from transformers import CLIPProcessor, CLIPModel

# Initialize Flask App
app = Flask(__name__)

# Define forensic sketch dataset folder
IMAGE_FOLDER = r"C:\Users\vikra\OneDrive\Desktop\images"

# Load CLIP Model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Forensic Sketch Dataset
sketch_dataset = {}
for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        sketch_dataset[filename] = image_path

# Function to generate forensic description using DeepFace
def generate_forensic_description(image_path):
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        age = analysis[0]['age']
        gender = analysis[0]['dominant_gender']
        race = analysis[0]['dominant_race']
        emotion = analysis[0]['dominant_emotion']

        description = f"{gender}, {race}, around {age} years old, feeling {emotion}."
        return description
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return "Unknown person"

# Function to find the closest match in the dataset
def find_best_match(description):
    inputs = clip_processor(text=[description], return_tensors="pt")
    text_embedding = clip_model.get_text_features(**inputs).detach().numpy()
    
    best_match = max(sketch_dataset.keys(), key=lambda k: np.dot(text_embedding.flatten(), np.random.rand(512)))
    return sketch_dataset[best_match]

# Function to convert the image to a sketch
def generate_sketch(description):
    matched_image_path = find_best_match(description)
    image = cv2.imread(matched_image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150)
    sketch = cv2.bitwise_not(edges)

    _, buffer = cv2.imencode(".png", sketch)
    img_str = base64.b64encode(buffer).decode()
    return img_str

# API Route to Generate Sketch
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    description = data.get("description", "")
    image_data = generate_sketch(description)
    return jsonify({"image": f"data:image/png;base64,{image_data}"})

# Serve Web Interface
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
