import torch
import os

def convert_single_to_multi_label(single_labels, num_classes):
    """
    将单标签转换为多标签格式，但保持每个样本只有一个正标签
    Args:
        single_labels: 形状为[N]的整数张量，每个元素是类别索引
        num_classes: 类别总数
    Returns:
        形状为[N]的单标签张量
    """
    return single_labels  # 保持原始的单标签格式

def process_distilled_data(input_path, output_path, num_classes=20):
    """
    处理蒸馏数据，保持单标签格式但确保数据类型正确
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        num_classes: 类别总数
    """
    print(f"Loading distilled data from {input_path}...")
    data = torch.load(input_path, map_location='cpu')
    
    if isinstance(data, list) and len(data) == 2:
        images, single_labels = data
        print(f"Original images shape: {images.shape}")
        print(f"Original labels shape: {single_labels.shape}")
        print(f"Labels dtype: {single_labels.dtype}")
        
        # 确保标签是整数类型
        if single_labels.dtype != torch.int64:
            single_labels = single_labels.long()
        
        # 保存转换后的数据
        converted_data = [images, single_labels]
        torch.save(converted_data, output_path)
        print(f"Converted data saved to {output_path}")
        
        # 验证转换
        loaded_data = torch.load(output_path, map_location='cpu')
        loaded_images, loaded_labels = loaded_data
        print(f"Loaded images shape: {loaded_images.shape}")
        print(f"Loaded labels shape: {loaded_labels.shape}")
        print(f"Loaded labels dtype: {loaded_labels.dtype}")
        
        # 检查转换是否正确
        for i in range(min(5, len(single_labels))):
            original_label = single_labels[i].item()
            converted_label = loaded_labels[i].item()
            print(f"\nSample {i}:")
            print(f"Original label: {original_label}")
            print(f"Converted label: {converted_label}")
            assert converted_label == original_label, "Conversion error!"
            
        # 检查类别分布
        print("\n=== 类别分布 ===")
        unique_labels, counts = torch.unique(single_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Class {label.item()}: {count.item()} samples")
            
        # 验证所有类别都有相同数量的样本
        unique_counts = torch.unique(counts)
        assert len(unique_counts) == 1, "Classes have different numbers of samples"
        print(f"\nAll classes have {unique_counts[0].item()} samples each")
        
    else:
        print("Error: Unexpected data format")

if __name__ == "__main__":
    input_path = "results/condense/condense/voc2007/ipc10/adamw_lr_img_0.0010_numr_reqs4096_factor2_20250421-0852/distilled_data/data_20000.pt"
    output_path = "test/data_20000_single.pt"  # 修改输出文件名以反映单标签格式
    
    process_distilled_data(input_path, output_path) 