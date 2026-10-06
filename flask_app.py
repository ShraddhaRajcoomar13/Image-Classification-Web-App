from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import base64
from cifar10cnn import CIFAR10CNN
import config
from load_model import load_model
from image_preprocessing import preprocess_image
from predict_image import predict
from config import CLASSES
from train_model import train_model

app = Flask(__name__)
model = train_model(epochs=50, batch_size=128, lr=0.001, save_path='cifar10_cnn.pth')
model, device = load_model()
img_loaded = config.img_loaded  # Ensure img_loaded is imported from config.py
file = preprocess_image(img_loaded)  # Assuming img_loaded is defined elsewhere in your code

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html', classes=CLASSES)

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handle prediction requests"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            image_bytes = file.read()
            results = predict(image_bytes)
            
            if 'error' in results:
                return jsonify(results), 400
            
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)