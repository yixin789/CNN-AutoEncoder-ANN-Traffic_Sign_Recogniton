import torch
from torch.utils.data import DataLoader
from models.traffic_sign_ann import TrafficSignCNN_AE_ANN
from datasets.traffic_sign_dataset import TrafficSignDataset, NUM_CLASSES 
from utils.evaluation import evaluate_model, get_predictions_and_labels, calculate_topk_accuracy, calculate_f1_score, plot_confusion_matrix, plot_roc_curve
from utils.model_utils import load_model
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
import json # Used to load class names from JSON file
import matplotlib.pyplot as plt # Ensure plotting library is imported
import seaborn as sns # Ensure seaborn is imported

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for evaluation: {device}")

SAVE_MODEL_PATH = 'traffic_sign_model.pth' # Please ensure your model file is at this path
BATCH_SIZE = 32 # Use the same batch size as during training for consistency
test_json_path = 'data_stratified/test/_annotations.coco.json'
test_image_dir = 'data_stratified/test'
train_json_path_for_class_names = 'data_stratified/train/_annotations.coco.json' # Used to load class names

# --- Data Transformations (for evaluation only) ---
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load Class Names ---
class_names = []
print("Loading class names...")
try:
    with open(train_json_path_for_class_names, 'r') as f:
        coco_data = json.load(f)
        categories = coco_data['categories']
        # Sort by original ID for consistent mapping
        categories.sort(key=lambda x: x['id']) 
        
        # To get the correct class index mapping, instantiate a dummy Dataset
        # This Dataset does not need to load images, only its internal label_mapping
        dummy_dataset = TrafficSignDataset(json_path=train_json_path_for_class_names, 
                                           image_dir=test_image_dir, # Image directory is not critical here
                                           device=torch.device('cpu'), # Device is not critical here
                                           transform=None) # No transform needed
        
        temp_class_names = [None] * NUM_CLASSES
        for cat in categories:
            # Get the 0-indexed mapped label corresponding to the category ID
            mapped_idx = dummy_dataset.label_mapping.get(cat['id'])
            if mapped_idx is not None and 0 <= mapped_idx < NUM_CLASSES:
                temp_class_names[mapped_idx] = cat['name']
        
        # Fill any potential gaps or use default names
        class_names = [name if name is not None else f"Class {i}" for i, name in enumerate(temp_class_names)]
    print(f"Loaded {len(class_names)} class names.")
except Exception as e:
    print(f"Error loading class names from {train_json_path_for_class_names}: {e}")
    print("Will use generic class names. Plot labels might be incorrect.")
    class_names = [f"Class {i}" for i in range(NUM_CLASSES)]


# --- Load Test Dataset and DataLoader ---
print("Loading test dataset...")
test_dataset = TrafficSignDataset(json_path=test_json_path, image_dir=test_image_dir, device=device, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
print(f"Test dataset loaded with {len(test_dataset)} samples.")


# --- Load Model ---
print(f"Loading model from {SAVE_MODEL_PATH}...")
try:
    model = load_model(TrafficSignCNN_AE_ANN, SAVE_MODEL_PATH, device) 
    model.eval() # Set to evaluation mode
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {SAVE_MODEL_PATH}. Please ensure you have a trained model saved.")
    print("For example: Run main.py to train and save the model first.")
    exit() # Exit if model is not found
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    exit()

# --- Define Loss Function (needed for evaluation) ---
criterion = nn.CrossEntropyLoss()

# --- Perform Evaluation ---
print("\n--- Performing Final Evaluation on Test Set ---")
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# --- Get Predictions and Labels for Detailed Metrics ---
print("\nCollecting predictions and labels for detailed analysis...")
all_labels_test, all_predictions_test_logits, all_predictions_test_probabilities = \
    get_predictions_and_labels(model, test_loader, device)

# --- Calculate and Display Detailed Metrics ---
print("\n--- Detailed Metrics ---")
# Calculate Top-5 Accuracy
test_top5_accuracy = calculate_topk_accuracy(all_labels_test, all_predictions_test_logits, k=5)
print(f"Test Top-5 Accuracy: {test_top5_accuracy:.2f}%")

# Calculate F1-Score (using 'macro' average for multi-class)
test_f1_score = calculate_f1_score(all_labels_test, all_predictions_test_logits, average='macro')
print(f"Test F1-Score (Macro): {test_f1_score:.4f}")

# --- Plotting ---
print("\n--- Generating Plots ---")
# Plot Confusion Matrix
cm_predictions_test = np.argmax(all_predictions_test_logits, axis=1) # Get predicted class from logits
print("Plotting Confusion Matrix...")
plot_confusion_matrix(all_labels_test, cm_predictions_test, class_names)

# Plot ROC Curve
print("Plotting ROC Curve...")
plot_roc_curve(all_labels_test, all_predictions_test_probabilities, NUM_CLASSES, class_names)

# --- Display Example Predictions ---
print("\n--- Displaying Example Test Predictions ---")
# Ensure the model is in evaluation mode (already set)
with torch.no_grad():
    # Attempt to get a batch from the test loader
    try:
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
    except StopIteration:
        print("Test loader is empty or iterated through. Cannot display examples.")
        exit() 
    except Exception as e:
        print(f"Error loading images from test loader for display: {e}")
        exit()

    # Move images to the device
    images = images.to(device)
    
    # Get model outputs (logits)
    outputs = model(images) 
    
    # For multi-class classification, get the index of the highest probability prediction
    _, predicted_indices = torch.max(outputs, 1) 
    
    fig = plt.figure(figsize=(12, 10)) # Adjust figure size for more information
    display_count = min(8, images.shape[0]) # Display up to 8 images
    
    for i in range(display_count): 
        ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[]) # 2 rows, 4 columns
        
        # Unnormalize image for display
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean # Unnormalize
        img = np.clip(img, 0, 1) # Clip values to [0, 1] range

        ax.imshow(img)
        
        # Get names for true and predicted labels
        true_label_name = class_names[labels[i].item()] 
        predicted_label_name = class_names[predicted_indices[i].item()]
        
        # Set title color based on prediction correctness
        title_color = "green" if predicted_indices[i] == labels[i] else "red"
        ax.set_title(f"True: {true_label_name}\nPred: {predicted_label_name}", 
                     color=title_color, fontsize=8) # Reduce font size to fit
    plt.tight_layout()
    plt.show(block=True) # Use block=True to pause script until window is closed

print("\nEvaluation and plotting completed.")
