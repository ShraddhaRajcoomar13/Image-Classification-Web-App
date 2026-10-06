import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from cifar10cnn import CIFAR10CNN
from train_model import train_model
from load_model import load_model
from image_preprocessing import preprocess_image
from predict_image import predict
from config import img_loaded


if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    model = train_model(epochs=50, batch_size=128, lr=0.001, save_path='models/cifar10_cnn.pth')
    print('Model saved successfully!')
    
    # Initialize model globally
    model, device = load_model()
    
    

    
    