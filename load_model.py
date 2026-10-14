import torch
from cifar10cnn import CIFAR10CNN
import os

def load_model(model_path='cifar10_cnn.pth'):
    """Load the trained model from disk (non-fatal if missing)."""
    device = torch.device('cpu')
    model = CIFAR10CNN()

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Support both dict-style and raw state_dict checkpoints
            state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state)
            acc = checkpoint.get('accuracy') if isinstance(checkpoint, dict) else None
            if acc is not None:
                print(f"Model loaded successfully! Accuracy: {acc:.2f}%")
            else:
                print("Model loaded successfully (no accuracy recorded).")
        except Exception as e:
            print(f"Warning: failed to load model ({e}). Using untrained model.")
    else:
        print(f"Warning: Model file not found at '{model_path}'. Using untrained model.")

    model.eval()
    return model, device