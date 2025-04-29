# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # 导入 TensorDataset
# import torchvision.transforms as transforms # 仍然可能需要，例如用于测试时的增强，但在此加载方式下主要用于记录
# import torchvision.datasets as datasets # 不再需要 datasets
# import xml.etree.ElementTree as ET # 不再需要解析 XML
import os
import numpy as np
from tqdm import tqdm # 用于显示进度条
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import torchvision.models as models

# --- 配置参数 ---
# DATA_ROOT = './data' # 数据根目录，现在可能不需要了，因为直接指定 .pt 文件路径
PT_FILE_PATH = '/home/wjh/DC/data/VOC2007/VOC2007.pt' # 指定你的 .pt 文件路径
NUM_CLASSES = 20      # PASCAL VOC 有 20 个类别
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15      # 根据需要调整
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224) # 图像尺寸，需要与 .pt 文件中的图像尺寸匹配

print(f"Using device: {DEVICE}")

# --- 数据加载 ---
print(f"Loading data from {PT_FILE_PATH}...")
try:
    # 加载预处理好的数据
    data = torch.load(PT_FILE_PATH, map_location='cpu') # map_location='cpu' 避免直接加载到GPU

    # 从加载的数据中提取信息
    class_names = data['classes'] # 获取类别名称列表
    print(f"Dataset classes: {class_names}")
    if len(class_names) != NUM_CLASSES:
        print(f"Warning: Number of classes in .pt file ({len(class_names)}) does not match configured NUM_CLASSES ({NUM_CLASSES})!")
        # 可以选择在这里更新 NUM_CLASSES 或抛出错误
        # NUM_CLASSES = len(class_names)

    # 训练数据
    images_train = data['images_train'] # 假设是 (N, C, H, W) 格式的 Tensor
    labels_train = data['labels_train'] # 假设是 (N, NUM_CLASSES) 格式的 Multi-hot Tensor (float)
    # 确保标签是 FloatTensor 类型，因为 BCEWithLogitsLoss 需要 float 类型的 target
    if labels_train.dtype != torch.float32:
        labels_train = labels_train.float()
    # 创建 TensorDataset
    dst_train = TensorDataset(images_train, labels_train)

    # 测试/验证数据
    images_test = data['images_test'] # 或者可能是 'images_val'，根据 .pt 文件内容调整 key
    labels_test = data['labels_test'] # 或者可能是 'labels_val'
    # 确保标签是 FloatTensor 类型
    if labels_test.dtype != torch.float32:
        labels_test = labels_test.float()
    # 创建 TensorDataset
    dst_test = TensorDataset(images_test, labels_test)

    print(f"Training samples: {len(dst_train)}")
    print(f"Test samples: {len(dst_test)}")
    print(f"Image tensor shape (first training sample): {images_train[0].shape}")
    print(f"Label tensor shape (first training sample): {labels_train[0].shape}")

    # 注意：这里的 mean 和 std 是你代码片段里定义的，但在此加载方式下【没有被显式使用】。
    # 你需要确保保存在 .pt 文件中的图像张量 'images_train', 'images_test'
    # 【已经进行了归一化处理】。如果没有，你需要在这里或训练循环前手动进行归一化。
    # mean = [0.4485, 0.4250, 0.3920] # 来自你的代码片段
    # std = [0.2692, 0.2659, 0.2788] # 来自你的代码片段

    # 创建 DataLoader
    train_loader = DataLoader(dst_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dst_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

except FileNotFoundError:
    print(f"Error: Data file not found {PT_FILE_PATH}")
    exit()
except KeyError as e:
    print(f"Error: Missing key in .pt file: {e}. Please check if the file contains 'classes', 'images_train', 'labels_train', 'images_test', 'labels_test'")
    exit()
except Exception as e:
    print(f"Unknown error occurred while loading data: {e}")
    exit()


# --- 简单的 3 层 CNN 模型 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        # 输入通道为 3 (RGB)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 224 -> 112 (假设输入是224x224)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 112 -> 56

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 56 -> 28

        # 使用自适应平均池化层，可以自动处理不同大小的特征图，输出固定大小 (1x1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层输入特征数等于最后一个卷积层的输出通道数 (128)
        self.fc1 = nn.Linear(128, 512)

        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # 使用 Dropout 防止过拟合
        self.fc2 = nn.Linear(512, num_classes)
        # 注意：多标签分类这里【不需要】加 Sigmoid 激活函数，
        # 因为 BCEWithLogitsLoss 会自动处理 Sigmoid 计算

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.adaptive_pool(x)      # 应用自适应池化
        x = torch.flatten(x, 1)        # 展平成 (batch_size, num_features)

        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # 输出原始的 logits
        return x

# --- 初始化模型、损失函数、优化器 ---
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
print(model)
model = model.to(DEVICE)

input()
# 损失函数：BCEWithLogitsLoss，适用于多标签分类问题
# 它结合了 Sigmoid 层和二元交叉熵损失，数值上更稳定
criterion = nn.BCEWithLogitsLoss()

# 优化器：Adam
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 训练函数 ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    total_samples = 0

    # 使用 tqdm 显示进度条
    progress_bar = tqdm(loader, desc='Training', leave=False)
    for i, (images, targets) in enumerate(progress_bar):
        # 将数据移动到指定设备 (GPU 或 CPU)
        images = images.to(device)
        targets = targets.to(device) # 标签也需要移动

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        # 确保 targets 是 Float 类型，BCEWithLogitsLoss 需要
        loss = criterion(outputs, targets.float()) # 再次确认 targets 是 float

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累加损失
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        # 更新进度条显示信息
        progress_bar.set_postfix(loss=running_loss / total_samples)

    epoch_loss = running_loss / total_samples
    return epoch_loss

# --- 评估函数 ---
def evaluate(model, loader, criterion, device):
    model.eval() # 设置模型为评估模式
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_scores = []  # 存储原始预测分数（用于计算mAP）

    # 使用 tqdm 显示进度条
    progress_bar = tqdm(loader, desc='Evaluating', leave=False)
    with torch.no_grad(): # 评估时不需要计算梯度
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device) # 标签也需要移动

            outputs = model(images)
            loss = criterion(outputs, targets.float()) # 确保 targets 是 float

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # 获取预测结果
            # 1. 应用 Sigmoid 将 logits 转换为 0-1 之间的概率
            scores = torch.sigmoid(outputs)
            # 2. 设置阈值 (例如 0.5) 将概率转换为二元预测 (0 或 1)
            preds = (scores > 0.5).float()
            
            all_preds.append(preds.cpu())      # 将预测结果移回 CPU 保存
            all_targets.append(targets.cpu())  # 将真实标签移回 CPU 保存
            all_scores.append(scores.cpu())    # 保存原始预测分数

            # 更新进度条显示信息
            progress_bar.set_postfix(loss=running_loss / total_samples)

    epoch_loss = running_loss / total_samples

    # 将所有批次的预测和标签连接起来
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()

    # 计算评估指标
    # Hamming Loss = 每个样本中标签预测错误的比例 的 平均值
    hamming_loss = np.mean(all_preds != all_targets)

    # Exact Match Ratio (子集准确率) = 完全预测正确的样本所占的比例
    exact_match = np.mean(np.all(all_preds == all_targets, axis=1))

    # 计算更多指标
    precision_micro = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    
    recall_micro = recall_score(all_targets, all_preds, average='micro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # 计算每个类别的AP，然后取平均得到mAP
    ap_per_class = [average_precision_score(all_targets[:, i], all_scores[:, i]) 
                    for i in range(NUM_CLASSES)]
    mAP = np.mean(ap_per_class)

    metrics = {
        'loss': epoch_loss,
        'hamming_loss': hamming_loss,
        'exact_match': exact_match,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'mAP': mAP
    }
    
    return metrics


# --- 主训练循环 ---
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    # 训练一个 epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

    # 在测试集上评估
    metrics = evaluate(model, test_loader, criterion, DEVICE)

    # 打印当前 epoch 的结果
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {metrics['loss']:.4f} | "
          f"Hamming Loss: {metrics['hamming_loss']:.4f} | "
          f"Exact Match: {metrics['exact_match']:.4f} | "
          f"Precision (micro/macro): {metrics['precision_micro']:.4f}/{metrics['precision_macro']:.4f} | "
          f"Recall (micro/macro): {metrics['recall_micro']:.4f}/{metrics['recall_macro']:.4f} | "
          f"F1 (micro/macro): {metrics['f1_micro']:.4f}/{metrics['f1_macro']:.4f} | "
          f"mAP: {metrics['mAP']:.4f}")

print("Training completed!")

# --- 可选: 保存模型 ---
# torch.save(model.state_dict(), 'voc_simple_cnn_model_from_pt.pth')
# print("Model saved to voc_simple_cnn_model_from_pt.pth")