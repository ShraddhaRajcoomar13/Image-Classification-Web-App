import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor