import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.voc2007_dataset import VOC2007_PT_Dataset
import numpy as np
from tqdm import tqdm

def check_data(data_path, name=""):
    print(f"\n{'='*20} 检查{name}数据 {'='*20}")
    print(f"Loading data from {data_path}...")
    try:
        data = torch.load(data_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # 检查数据格式
    print("\n=== 检查数据格式 ===")
    
    if isinstance(data, list):
        print("数据格式: 列表")
        print(f"列表长度: {len(data)}")
        if len(data) > 0:
            print(f"第一个元素类型: {type(data[0])}")
            if len(data) == 2 and isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor):
                images = data[0]
                labels = data[1]
            elif isinstance(data[0], tuple):
                images = torch.stack([x[0] for x in data])
                labels = torch.stack([x[1] for x in data])
            else:
                images = torch.stack(data)
                labels = None
    elif isinstance(data, tuple):
        print("数据格式: 元组")
        images, labels = data
    elif isinstance(data, dict):
        print("数据格式: 字典")
        print(f"Keys in data dict: {list(data.keys())}")
        if 'images_train' in data:
            # 原始数据格式
            print("\n--- 训练数据 ---")
            print(f"Training images shape: {data['images_train'].shape}")
            print(f"Training labels shape: {data['labels_train'].shape}")
            print(f"Test images shape: {data['images_test'].shape}")
            print(f"Test labels shape: {data['labels_test'].shape}")
            images = data['images_train']
            labels = data['labels_train']
        else:
            # 其他字典格式
            images = data['images']
            labels = data['labels']
    else:
        print(f"未知的数据格式: {type(data)}")
        return

    print(f"\nImages shape: {images.shape}")
    if labels is not None:
        print(f"Labels shape: {labels.shape}")

    if 'classes' in data if isinstance(data, dict) else False:
        print(f"Classes: {data['classes']}")
        print(f"Number of classes: {len(data['classes'])}")
        class_names = data['classes']
    else:
        if labels is not None:
            if len(labels.shape) == 1:
                num_classes = labels.max().item() + 1
            else:
                num_classes = labels.shape[1]
            class_names = [f"Class_{i}" for i in range(num_classes)]
        else:
            class_names = []

    # 检查图像数据的基本统计信息
    print("\n=== 图像数据统计 ===")
    print(f"Image data type: {images.dtype}")
    print(f"Image value range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Image mean: {images.mean():.2f}")
    print(f"Image std: {images.std():.2f}")

    # 检查标签的统计信息（如果有标签）
    if labels is not None:
        print("\n=== 标签统计 ===")
        print(f"Label data type: {labels.dtype}")
        
        if len(labels.shape) == 1:
            # 如果标签是一维的，计算每个类别的样本数
            unique_labels, counts = torch.unique(labels, return_counts=True)
            print("\nSamples per class:")
            for label, count in zip(unique_labels, counts):
                class_idx = int(label.item())
                print(f"{class_names[class_idx]}: {count}")
            print(f"\nNumber of classes found in labels: {len(unique_labels)}")
        else:
            # 多标签情况
            positive_samples = labels.sum(dim=0)
            print("\nPositive samples per class:")
            for cls_name, count in zip(class_names, positive_samples):
                print(f"{cls_name}: {count}")

            print(f"\nAverage labels per sample: {labels.sum(dim=1).mean():.2f}")
            print(f"Max labels per sample: {labels.sum(dim=1).max():.0f}")
            print(f"Min labels per sample: {labels.sum(dim=1).min():.0f}")

            # 检查是否有空标签的样本
            empty_samples = (labels.sum(dim=1) == 0).sum()
            print(f"\nSamples with no labels: {empty_samples}")

    # 如果是原始数据，测试数据集加载器
    if isinstance(data, dict) and 'images_train' in data:
        print("\n=== 测试数据集加载 ===")
        try:
            train_dataset = VOC2007_PT_Dataset(data_path, train=True)
            print(f"Successfully created dataset with {len(train_dataset)} samples")
            
            # 测试获取一个样本
            image, label = train_dataset[0]
            print(f"Sample image shape: {image.shape}")
            print(f"Sample label shape: {label.shape}")
            
            # 测试数据加载器
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            print(f"\nCreated DataLoader with {len(train_loader)} batches")
            
            # 获取一个批次并测试
            batch_images, batch_labels = next(iter(train_loader))
            print(f"Batch images shape: {batch_images.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
            
            # 测试损失函数
            print("\n=== 测试损失函数 ===")
            criterion = nn.BCEWithLogitsLoss()
            dummy_outputs = torch.randn_like(batch_labels)
            loss = criterion(dummy_outputs, batch_labels)
            print(f"Test loss value: {loss.item():.4f}")
            
        except Exception as e:
            print(f"Error during dataset testing: {str(e)}")

if __name__ == "__main__":
    # 检查原始数据
    # ORIGINAL_DATA_PATH = '/home/wjh/DC/data/VOC2007/VOC2007.pt'
    # check_data(ORIGINAL_DATA_PATH, "原始")
    
    # 检查蒸馏数据
    DISTILLED_DATA_PATH = '/home/wjh/NCFM/results/condense/condense/voc2007/ipc10/adamw_lr_img_0.0010_numr_reqs4096_factor2_20250429-0141/converted_data/converted_20000.pt'
    check_data(DISTILLED_DATA_PATH, "蒸馏") 