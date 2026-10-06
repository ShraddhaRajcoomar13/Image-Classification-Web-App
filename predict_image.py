import torch
import torch.nn as nn
from image_preprocessing import preprocess_image
from load_model import load_model


model, device = load_model()

def predict(image_bytes):
    """Make prediction on image"""
    try:
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
            # Get top 3 predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            
            results = {
                'predicted_class': CLASSES[predicted.item()],
                'confidence': float(confidence.item()),
                'top_3': [
                    {'class': CLASSES[idx], 'probability': float(prob)}
                    for idx, prob in zip(top3_idx.tolist(), top3_prob.tolist())
                ]
            }
            
            return results
    except Exception as e:
        return {'error': str(e)}
