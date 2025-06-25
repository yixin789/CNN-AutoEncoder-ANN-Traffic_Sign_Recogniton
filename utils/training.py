# utils/training.py
import torch

def train_one_epoch(model, loader, optimizer, criterion, device, epoch_num=0): # Added epoch_num for conditional printing
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader): # Added batch_idx for conditional printing
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)
