import torch
import torch.nn as nn
from cifar10cnn import CIFAR10CNN
import os

# Load model
def load_model():
    """Load the trained model"""
    device = torch.device('cpu')  # Use CPU for deployment
    model = CIFAR10CNN()
    
    model_path = 'models/cifar10_cnn.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully! Accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print("Warning: Model file not found. Using untrained model.")
    
    model.eval()
    return model, device

# Initialize model globally
model, device = load_model()