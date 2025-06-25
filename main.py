import torch
from torch.utils.data import DataLoader
from models.traffic_sign_ann import TrafficSignCNN_AE_ANN
from datasets.traffic_sign_dataset import TrafficSignDataset, NUM_CLASSES 
from utils.training import train_one_epoch
from utils.evaluation import evaluate_model, get_predictions_and_labels, calculate_topk_accuracy, calculate_f1_score, plot_confusion_matrix, plot_roc_curve
from utils.model_utils import save_model, load_model
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision import transforms 

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
# Define paths to your dataset directories and annotation files
# 请确保这些路径与你的实际数据集位置相符
train_json_path = 'data_stratified/train/_annotations.coco.json'
train_image_dir = 'data_stratified/train'
valid_json_path = 'data_stratified/valid/_annotations.coco.json'
valid_image_dir = 'data_stratified/valid'
test_json_path = 'data_stratified/test/_annotations.coco.json'
test_image_dir = 'data_stratified/test'

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1 
SAVE_MODEL_PATH = 'traffic_sign_model.pth' # Path to save/load model

# Custom transform for consistent image preprocessing
# *** 重要的更改：为训练集添加数据增强 ***
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),       # 随机旋转 -10 到 10 度
    transforms.RandomHorizontalFlip(),   # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 随机颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集和测试集只进行基本的缩放和归一化，不进行随机增强
# 因为验证集和测试集应该反映真实世界的未见数据分布
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = TrafficSignDataset(json_path=train_json_path, image_dir=train_image_dir, device=device, transform=train_transform) # 使用 train_transform
valid_dataset = TrafficSignDataset(json_path=valid_json_path, image_dir=valid_image_dir, device=device, transform=eval_transform) # 使用 eval_transform
test_dataset = TrafficSignDataset(json_path=test_json_path, image_dir=test_image_dir, device=device, transform=eval_transform) # 使用 eval_transform

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# Load class names for evaluation plots
with open(train_json_path, 'r') as f:
    import json
    coco_data = json.load(f)
    categories = coco_data['categories']
    categories.sort(key=lambda x: x['id']) # Sort by original ID for consistent mapping
    
    # Re-map to 0-indexed if necessary, matching dataset's label_mapping
    dataset_categories = train_dataset.coco.loadCats(train_dataset.coco.getCatIds())
    # Sort by the mapped index to ensure class_names list matches the model's output indices
    # This requires a temporary list to hold names at their correct mapped index
    temp_class_names = [None] * NUM_CLASSES
    for cat in dataset_categories:
        mapped_idx = train_dataset.label_mapping.get(cat['id'])
        if mapped_idx is not None and mapped_idx < NUM_CLASSES:
            temp_class_names[mapped_idx] = cat['name']
    
    # Fill any potential gaps or unmapped indices with a placeholder
    class_names = [name if name is not None else f"Class {i}" for i, name in enumerate(temp_class_names)]


# Model, Optimizer, and Loss Function
model = TrafficSignCNN_AE_ANN(device=device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # 示例学习率和权重衰减
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 损失函数：针对多类别分类
criterion = nn.CrossEntropyLoss()

# Training loop
best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
    print(f"Train Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_accuracy = evaluate_model(model, valid_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Update learning rate scheduler
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
        best_val_loss = val_loss
        save_model(model, SAVE_MODEL_PATH)

print("\nTraining Finished.")

# Load the best model for final evaluation
print(f"Loading best model from {SAVE_MODEL_PATH} for final evaluation...")
model = load_model(TrafficSignCNN_AE_ANN, SAVE_MODEL_PATH, device)
model.eval() # Set to evaluation mode

# Final evaluation on test set
print("\nEvaluating on Test Set...")
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Get all predictions and labels for detailed metrics
all_labels_test, all_predictions_test_logits, all_scores_test_for_roc = get_predictions_and_labels(model, test_loader, device)

# Calculate Top-5 Accuracy
test_top5_accuracy = calculate_topk_accuracy(all_labels_test, all_predictions_test_logits, k=5)
print(f"Test Top-5 Accuracy: {test_top5_accuracy:.2f}%")

# Calculate F1-Score (using macro average for multi-class)
test_f1_score = calculate_f1_score(all_labels_test, all_predictions_test_logits, average='macro')
print(f"Test F1-Score (Macro): {test_f1_score:.4f}")

# Plot Confusion Matrix
# cm_predictions_test 需要是预测的类别索引 (0, 1, 2, ...)
cm_predictions_test = np.argmax(all_predictions_test_logits, axis=1)
plot_confusion_matrix(all_labels_test, cm_predictions_test, class_names)

# Plot ROC Curve
# all_scores_test_for_roc 已经是原始 logits，适合 roc_curve
plot_roc_curve(all_labels_test, all_scores_test_for_roc, NUM_CLASSES, class_names)

# --- 修复后的可视化预测结果 ---
print("\nDisplaying example test predictions...")
model.eval() # 确保模型处于评估模式
with torch.no_grad():
    # 尝试从测试加载器获取一个批次
    try:
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
    except StopIteration:
        print("Test loader is empty or iterated through. Cannot display examples.")
        exit() # Or handle more gracefully

    images = images.to(device)
    # labels 已经是 LongTensor，不需要 to(device) 再 to(cpu)
    
    outputs = model(images)
    
    # 对于多类别分类，获取最高概率的预测索引
    _, predicted_indices = torch.max(outputs, 1) # Get predicted class indices
    
    fig = plt.figure(figsize=(12, 10)) # 调整图表大小以适应更多信息
    # 显示图片数量，例如前8张
    display_count = min(8, images.shape[0]) 
    
    for i in range(display_count): 
        ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[]) # 2行4列
        
        # 反标准化图像以供显示
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean # 反标准化
        img = np.clip(img, 0, 1) # 裁剪值到 [0, 1] 范围

        ax.imshow(img)
        
        # 获取真实标签和预测标签的名称
        true_label_name = class_names[labels[i].item()] # .item() 转换 0-dim tensor 为标量
        predicted_label_name = class_names[predicted_indices[i].item()]
        
        # 设置标题，如果预测正确则为绿色，否则为红色
        title_color = "green" if predicted_indices[i] == labels[i] else "red"
        ax.set_title(f"True: {true_label_name}\nPred: {predicted_label_name}", 
                     color=title_color, fontsize=8) # 减小字体以适应空间
    plt.tight_layout()
    plt.show(block=True) # 使用 block=True 暂停脚本，直到窗口关闭
