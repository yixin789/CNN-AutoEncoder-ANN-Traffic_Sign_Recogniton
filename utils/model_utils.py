# model_utils.py
import torch
from models.traffic_sign_ann import TrafficSignCNN_AE_ANN # Import your model class

def save_model(model, path):
    """
    Saves the model's state dictionary.
    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the model.
    """
    try:
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(model_class, path, device):
    """
    Loads a model from a saved state dictionary.
    Args:
        model_class (class): The class of the model to load (e.g., TrafficSignCNN_AE_ANN).
        path (str): The file path from which to load the model.
        device (torch.device): The device to load the model onto.
    Returns:
        torch.nn.Module: The loaded model.
    """
    try:
        model = model_class(device).to(device) # Instantiate the model with the device
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval() # Set model to evaluation mode
        print(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
