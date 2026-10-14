import torch
from image_preprocessing import preprocess_image
from load_model import load_model
from config import CLASSES

# Load once at import time
model, device = load_model('cifar10_cnn.pth')


def predict(image_bytes):
    """Make prediction on image bytes."""
    try:
        image_tensor = preprocess_image(image_bytes).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            top3_prob, top3_idx = torch.topk(probabilities, 3)

            return {
                'predicted_class': CLASSES[predicted.item()],
                'confidence': float(confidence.item()),
                'top_3': [
                    {'class': CLASSES[idx], 'probability': float(prob)}
                    for idx, prob in zip(top3_idx.tolist(), top3_prob.tolist())
                ]
            }
    except Exception as e:
        return {'error': str(e)}