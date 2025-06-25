import torch
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # Import warnings module
from sklearn.preprocessing import label_binarize 

def evaluate_model(model, loader, criterion, device):
    """
    Evaluates the model on a given data loader for MULTI-CLASS (SINGLE-LABEL) classification.
    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The data loader for evaluation.
        criterion (torch.nn.Module): The loss function (expected CrossEntropyLoss for multi-class).
        device (torch.device): The device (CPU or GPU) to perform evaluation on.
    Returns:
        tuple: A tuple containing (average validation loss, accuracy).
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device)) # labels should be class indices (LongTensor) for CrossEntropyLoss
            val_loss += loss.item()
            
            # For multi-class classification, get the class with the highest score
            # torch.max returns (values, indices)
            _, predicted = torch.max(outputs, 1) # Get the index of the max log-probability
            
            # Compare with true labels (which are expected to be class indices)
            correct += (predicted == labels.to(device)).sum().item() 
            total += labels.size(0) # Count total samples

    avg_val_loss = val_loss / len(loader)
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return avg_val_loss, accuracy


def get_predictions_and_labels(model, loader, device):
    """
    Gets all true labels, raw prediction logits, and scores for ROC curve plotting.
    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): The data loader (e.g., test_loader).
        device (torch.device): The device (CPU or GPU).
    Returns:
        tuple: (numpy array of true labels, numpy array of raw logits, numpy array of raw logits for ROC)
    """
    model.eval()
    all_labels = []
    all_predictions_logits = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_labels.extend(labels.cpu().numpy())
            all_predictions_logits.extend(outputs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions_logits = np.array(all_predictions_logits)
    all_scores_for_roc = all_predictions_logits
    return all_labels, all_predictions_logits, all_scores_for_roc


def calculate_f1_score(all_labels, all_predictions, average='macro'):
    """
    Calculates the F1-score for MULTI-CLASS classification.
    Args:
        all_labels (np.ndarray): True labels (class indices). Shape: (num_samples,)
        all_predictions (np.ndarray): Raw model outputs (logits). Shape: (num_samples, num_classes)
        average (str): Averaging strategy for F1-score ('macro', 'micro', 'weighted', 'None').
                       'macro' is often preferred for imbalanced datasets as it calculates F1 per class
                       and then averages, giving equal weight to each class.
    Returns:
        float: F1-score.
    """
    # Convert logits to predicted class indices
    predicted_classes = np.argmax(all_predictions, axis=1)

    # Ensure all_labels is 1D for f1_score
    if all_labels.ndim > 1:
        all_labels = all_labels.squeeze()
    
    # Filter UndefinedMetricWarning for cases with no true/predicted positives
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) 
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from sklearn.exceptions import UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        
        score = f1_score(all_labels, predicted_classes, average=average, zero_division=0)
    return score


def calculate_topk_accuracy(all_labels, all_predictions, k=5):
    """
    Calculates the Top-k accuracy for MULTI-CLASS classification.
    A sample is considered correct if its true label is among the top-k predicted labels.
    
    Args:
        all_labels (np.ndarray): True labels (class indices, e.g., [34, 37, ...]).
                                 Shape: (num_samples,) or (num_samples, 1).
        all_predictions (np.ndarray): Predicted probabilities/scores (logits or softmax).
                                      Shape: (num_samples, num_classes).
        k (int): The 'k' for Top-k accuracy.
    Returns:
        float: Top-k accuracy.
    """
    num_samples = all_labels.shape[0]
    correct_predictions = 0

    # Ensure all_labels is 1D for direct indexing
    if all_labels.ndim > 1:
        all_labels = all_labels.squeeze() 

    for i in range(num_samples):
        true_label_index = all_labels[i]
        
        top_k_predicted_indices = np.argsort(all_predictions[i])[-k:][::-1]
        
        if true_label_index in top_k_predicted_indices:
            correct_predictions += 1
            
    return (correct_predictions / num_samples) * 100 if num_samples > 0 else 0.0


def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Plots the confusion matrix.
    Args:
        true_labels (np.ndarray): True class labels (indices).
        predicted_labels (np.ndarray): Predicted class labels (indices).
        class_names (list): List of class names.
    """
    cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(len(class_names)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show(block=False) # Use block=False to allow other plots to show

def plot_roc_curve(true_labels_indices, predicted_probabilities, num_classes, class_names):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for each class.
    Args:
        true_labels_indices (np.array): True labels (integer indices from 0 to num_classes-1).
                                         Shape: (num_samples,).
        predicted_probabilities (np.array): Predicted probabilities (softmax outputs) for all classes.
                                            Shape: (num_samples, num_classes).
        num_classes (int): Total number of classes.
        class_names (list): List of class names corresponding to indices.
    """
    plt.figure(figsize=(18, 14)) 

    true_labels_one_hot = label_binarize(true_labels_indices, classes=range(num_classes))
    
    warnings_encountered = False
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) 
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        from sklearn.exceptions import UndefinedMetricWarning
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        for i in range(num_classes):
            if np.sum(true_labels_one_hot[:, i]) > 0:
                fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], predicted_probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
            else:
                print(f"Skipping ROC plot for class {class_names[i]} (index {i}) due to no positive true samples in the data.)")
                warnings_encountered = True

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    if warnings_encountered:
        plt.title('ROC Curve (Some classes skipped due to no positive samples)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=True) 
