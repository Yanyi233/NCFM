# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# import torchvision.transforms as transforms # 引入，如果需要在代码中添加归一化
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR # 引入学习率调度器

# --- 配置参数 ---
PT_FILE_PATH = '/home/wjh/DC/data/VOC2007/VOC2007.pt'
NUM_CLASSES = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # 微调预训练模型常用的初始学习率
WEIGHT_DECAY = 1e-5   # 使用 AdamW 时常用的权重衰减
NUM_EPOCHS = 50       # 增加训练周期
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224) # 确保与 .pt 文件中的图像尺寸匹配

print(f"Using device: {DEVICE}")

# --- 数据加载 (与你原代码类似，增加归一化检查提示) ---
print(f"Loading data from {PT_FILE_PATH}...")
try:
    data = torch.load(PT_FILE_PATH, map_location='cpu')
    class_names = data['classes']
    print(f"Dataset classes: {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"Warning: Class count mismatch!")
        # NUM_CLASSES = len(class_names)

    images_train = data['images_train']
    labels_train = data['labels_train'].float()
    images_test = data['images_test']
    labels_test = data['labels_test'].float()

    dst_train = TensorDataset(images_train, labels_train)
    dst_test = TensorDataset(images_test, labels_test)

    print(f"Training samples: {len(dst_train)}")
    print(f"Test samples: {len(dst_test)}")

    train_loader = DataLoader(dst_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# --- 模型初始化 (使用预训练 ResNet18) ---
model = models.resnet18(pretrained=True) # 使用预训练权重

# 获取 ResNet18 最后一个全连接层的输入特征数
num_ftrs = model.fc.in_features
# 替换为新的全连接层以匹配 VOC 类别数
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(DEVICE)

# --- 损失函数 ---
criterion = nn.BCEWithLogitsLoss()

# --- 优化器 (使用 AdamW) ---
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- 学习率调度器 ---
# scheduler = StepLR(optimizer, step_size=15, gamma=0.1) # 每 15 个 epoch 降低学习率
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6) # 余弦退火

# --- 训练函数 (保持不变) ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    progress_bar = tqdm(loader, desc='Training', leave=False)
    for i, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)

        # 如果需要在这里进行归一化（次优方案，但如果 TensorDataset 已固定）
        # images = normalize_transform(images) # 应用归一化

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets) # 确认 targets 是 float
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        progress_bar.set_postfix(loss=running_loss / total_samples)
    epoch_loss = running_loss / total_samples
    return epoch_loss

# --- 评估函数 (保持不变) ---
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_scores = []
    progress_bar = tqdm(loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            # 如果需要在这里进行归一化
            # images = normalize_transform(images)

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            scores = torch.sigmoid(outputs)
            preds = (scores > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_scores.append(scores.cpu())
            progress_bar.set_postfix(loss=running_loss / total_samples)
    epoch_loss = running_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()
    hamming_loss = np.mean(all_preds != all_targets)
    exact_match = np.mean(np.all(all_preds == all_targets, axis=1))
    precision_micro = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_micro = recall_score(all_targets, all_preds, average='micro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    ap_per_class = [average_precision_score(all_targets[:, i], all_scores[:, i])
                    for i in range(NUM_CLASSES) if np.sum(all_targets[:, i]) > 0] # 避免计算没有正样本的类的 AP
    mAP = np.mean(ap_per_class) if len(ap_per_class) > 0 else 0.0
    metrics = {
        'loss': epoch_loss, 'hamming_loss': hamming_loss, 'exact_match': exact_match,
        'precision_micro': precision_micro, 'precision_macro': precision_macro,
        'recall_micro': recall_micro, 'recall_macro': recall_macro,
        'f1_micro': f1_micro, 'f1_macro': f1_macro, 'mAP': mAP
    }
    return metrics

# --- 主训练循环 (加入学习率调度器步骤) ---
print("Starting training with ResNet18 (pretrained)...")
best_mAP = 0.0 # 用于早停或保存最佳模型

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    metrics = evaluate(model, test_loader, criterion, DEVICE)

    # 更新学习率调度器
    scheduler.step()

    # 打印结果
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {metrics['loss']:.4f} | mAP: {metrics['mAP']:.4f} | F1 Micro/Macro: {metrics['f1_micro']:.4f}/{metrics['f1_macro']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 可选: 早停逻辑或保存最佳模型
    if metrics['mAP'] > best_mAP:
        best_mAP = metrics['mAP']
        # torch.save(model.state_dict(), 'best_resnet18_finetuned_voc.pth')
        # print(f"Saved best model with mAP: {best_mAP:.4f}")
    # 添加早停计数器逻辑...

print("Training completed!")
# print(f"Best mAP achieved: {best_mAP:.4f}")