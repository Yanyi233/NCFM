import torch

def verify_converted_data(data_path):
    print(f"Loading converted data from {data_path}...")
    data = torch.load(data_path, map_location='cpu')
    
    if isinstance(data, list) and len(data) == 2:
        images, multi_labels = data
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {multi_labels.shape}")
        print(f"Labels dtype: {multi_labels.dtype}")
        
        # 检查标签格式
        print("\n=== 标签统计 ===")
        print(f"Number of samples: {len(images)}")
        print(f"Number of classes: {multi_labels.shape[1]}")
        
        # 检查每个样本的标签
        print("\n=== 样本标签检查 ===")
        for i in range(min(5, len(images))):
            label = multi_labels[i]
            print(f"\nSample {i}:")
            print(f"Label shape: {label.shape}")
            print(f"Label values: {label}")
            print(f"One-hot position: {label.argmax().item()}")
            print(f"Number of positive labels: {label.sum().item()}")
            
            # 验证是否只有一个正标签
            assert label.sum().item() == 1.0, f"Sample {i} has multiple positive labels"
            
        # 检查类别分布
        print("\n=== 类别分布 ===")
        class_counts = multi_labels.sum(dim=0)
        for class_idx, count in enumerate(class_counts):
            print(f"Class {class_idx}: {count.item()} samples")
            
        # 验证所有类别都有相同数量的样本
        unique_counts = torch.unique(class_counts)
        assert len(unique_counts) == 1, "Classes have different numbers of samples"
        print(f"\nAll classes have {unique_counts[0].item()} samples each")
        
    else:
        print("Error: Unexpected data format")

if __name__ == "__main__":
    data_path = "test/data_20000_multi.pt"
    verify_converted_data(data_path) 