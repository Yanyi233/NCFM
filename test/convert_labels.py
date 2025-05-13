import torch
import os

def convert_single_to_multi_label(single_labels, num_classes):
    """
    将单标签转换为多标签格式的multi-hot编码
    Args:
        single_labels: 形状为[N]的整数张量，每个元素是类别索引
        num_classes: 类别总数
    Returns:
        形状为[N, num_classes]的multi-hot张量，每行只有一个1
    """
    batch_size = single_labels.size(0)
    multi_labels = torch.zeros(batch_size, num_classes, dtype=torch.float)
    for i in range(batch_size):
        multi_labels[i, single_labels[i]] = 1.0
    return multi_labels

def process_distilled_data(input_path, output_path, num_classes=20):
    """
    处理蒸馏数据，将单标签转换为多标签的multi-hot形式
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
        
        # 将单标签转换为multi-hot形式
        multi_labels = convert_single_to_multi_label(single_labels, num_classes)
        print(f"Converted multi-hot labels shape: {multi_labels.shape}")
        
        # 保存转换后的数据
        converted_data = [images, multi_labels]
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
            random_idx = torch.randint(0, len(single_labels), (1,)).item()
            original_label = single_labels[random_idx].item()
            converted_label = loaded_labels[random_idx].argmax().item()  # 找到multi-hot编码中的1所在位置
            print(f"\nSample {i}:")
            print(f"Original label (单标签): {original_label}")
            print(f"Converted label (multi-hot): {loaded_labels[random_idx]}")
            print(f"Argmax of converted label: {converted_label}")
            assert converted_label == original_label, "Conversion error!"
            
        # 检查类别分布
        print("\n=== 类别分布 ===")
        multi_hot_sum = multi_labels.sum(dim=0)  # 每个类别的样本数
        for class_idx in range(num_classes):
            count = multi_hot_sum[class_idx].item()
            if count > 0:  # 只显示有样本的类别
                print(f"Class {class_idx}: {count} samples")
            
        # 验证所有类别都有相同数量的样本
        unique_counts = torch.unique(multi_hot_sum[multi_hot_sum > 0])
        assert len(unique_counts) == 1, "Classes have different numbers of samples"
        print(f"\nAll classes have {unique_counts[0].item()} samples each")
        
    else:
        print("Error: Unexpected data format")

if __name__ == "__main__":
    base_dir = "/home/wjh/NCFM/results/condense/condense/voc2007/ipc10/adamw_lr_img_0.0010_numr_reqs4096_factor2_20250509-0620"
    input_dir = os.path.join(base_dir, "distilled_data")
    
    # 创建输出目录（与distilled_data同级）
    output_dir = os.path.join(base_dir, "converted_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取distilled_data文件夹中所有pt文件
    pt_files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
    
    print(f"找到 {len(pt_files)} 个PT文件需要处理")
    
    # 处理每个文件
    for pt_file in pt_files:
        input_path = os.path.join(input_dir, pt_file)
        output_path = os.path.join(output_dir, pt_file.replace('data_', 'converted_'))
        
        print(f"\n处理文件: {pt_file}")
        process_distilled_data(input_path, output_path) 