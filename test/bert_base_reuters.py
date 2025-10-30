# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# --- 配置参数 ---
# 数据集文件路径 (请根据您的实际路径修改)
DATA_FILES = {
    'train': 'dataset/reuters/training_data.csv',
    'test': 'dataset/reuters/test_data.csv',
    'validation': 'dataset/reuters/val_data.csv'
}
MODEL_PATH = "models/model/bert-base-uncased" # 或者您本地的 "models/model/bert-base-uncased"
# NUM_CLASSES 将在数据加载后根据标签列确定 (应为90)
BATCH_SIZE = 32  # 减小批次大小，提高训练稳定性
LEARNING_RATE = 3e-5  # 降低学习率，BERT微调的标准学习率
NUM_EPOCHS = 30      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512    # BERT最大序列长度
TEXT_COLUMN = 'sentence' # 更新：用户指定文本列名
THRESHOLD = 0.3  # 降低预测阈值，提高召回率
WARMUP_STEPS = 100  # 学习率预热步数

print(f"使用设备: {DEVICE}")

# --- 数据加载与预处理 ---
print("加载和预处理数据...")
try:
    # 1. 加载数据集
    datasets = load_dataset("csv", data_files=DATA_FILES)

    # 2. 确定文本列和标签列
    all_column_names = datasets['train'].column_names
    if TEXT_COLUMN not in all_column_names:
        raise ValueError(f"文本列 '{TEXT_COLUMN}' 未在数据集中找到。可用列: {all_column_names}")

    # 标签在'labels'列中，是multi-hot向量
    label_column = 'labels'
    if label_column not in all_column_names:
        raise ValueError(f"标签列 '{label_column}' 未在数据集中找到。可用列: {all_column_names}")
    
    # 获取标签向量的长度作为类别数
    NUM_CLASSES = len(eval(datasets['train'][0][label_column]))  # 假设labels是字符串形式的列表

    print(f"文本列: '{TEXT_COLUMN}'")
    print(f"识别到 {NUM_CLASSES} 个类别。")

    # 3. 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 4. 定义预处理函数
    def preprocess_function(examples):
        # 编码文本
        tokenized_inputs = tokenizer(
            examples[TEXT_COLUMN],
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH,
        )
        
        # 处理标签 - 将字符串形式的列表转换为浮点数数组
        labels_batch = []
        for label_str in examples[label_column]:
            # 将字符串形式的列表转换为实际的列表
            if isinstance(label_str, str):
                label_list = eval(label_str)
            else:
                label_list = label_str  # 如果已经是列表，直接使用
            labels_batch.append(label_list)
        
        # 确保标签是浮点类型
        tokenized_inputs["labels"] = torch.tensor(labels_batch, dtype=torch.float32)
        return tokenized_inputs

    # 5. 应用预处理
    print("对数据集进行分词...")
    # 移除原始的文本列和标签列，保留预处理后的数据
    tokenized_datasets = datasets.map(
        preprocess_function, 
        batched=True, 
        remove_columns=[TEXT_COLUMN, label_column]
    )
    
    # 设置数据集的格式 - 告诉 datasets 库我们希望返回 PyTorch 张量
    # 注意：这不会自动将所有内容转换为张量，只是标记数据集的输出格式
    tokenized_datasets.set_format("torch")
    
    # 6. 创建 DataLoader
    def collate_fn(batch):
        # 初始化返回的批次字典
        batch_dict = {}
        
        # 处理所有非标签键 (input_ids, attention_mask 等)
        if "input_ids" in batch[0]:
            batch_dict["input_ids"] = torch.stack([example["input_ids"] for example in batch])
        if "attention_mask" in batch[0]:
            batch_dict["attention_mask"] = torch.stack([example["attention_mask"] for example in batch])
        if "token_type_ids" in batch[0]:
            batch_dict["token_type_ids"] = torch.stack([example["token_type_ids"] for example in batch])
            
        # 特殊处理标签 - 确保它们是张量
        if "labels" in batch[0]:
            # 如果 labels 已经是张量，直接堆叠
            if isinstance(batch[0]["labels"], torch.Tensor):
                batch_dict["labels"] = torch.stack([example["labels"] for example in batch])
            else:
                # 如果 labels 是列表，先转换为张量
                batch_dict["labels"] = torch.tensor([example["labels"] for example in batch], dtype=torch.float32)
                
        return batch_dict

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_datasets['validation'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(tokenized_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"训练样本数: {len(tokenized_datasets['train'])}")
    print(f"验证样本数: {len(tokenized_datasets['validation'])}")
    print(f"测试样本数: {len(tokenized_datasets['test'])}")
    
    # 打印一个批次的数据形状以供检查
    try:
        sample_batch = next(iter(train_loader))
        print(f"样本批次 input_ids 形状: {sample_batch['input_ids'].shape}")
        if 'attention_mask' in sample_batch:
            print(f"样本批次 attention_mask 形状: {sample_batch['attention_mask'].shape}")
        if 'token_type_ids' in sample_batch:
            print(f"样本批次 token_type_ids 形状: {sample_batch['token_type_ids'].shape}")
        print(f"样本批次 labels 形状: {sample_batch['labels'].shape}")
        
        # 检查标签分布
        labels_sample = sample_batch['labels']
        print(f"标签值范围: [{labels_sample.min():.3f}, {labels_sample.max():.3f}]")
        print(f"正标签比例: {labels_sample.mean():.3f}")
        print(f"每个样本的平均标签数: {labels_sample.sum(dim=1).mean():.3f}")
        
    except Exception as e:
        print(f"检查 DataLoader 样本批次时出错: {e}")
        import traceback
        traceback.print_exc()
        print("这可能表明预处理函数或 collate_fn 存在问题。")


except FileNotFoundError:
    print(f"错误: 找不到一个或多个数据文件。请检查 DATA_FILES 路径: {DATA_FILES}")
    exit()
except ValueError as e:
    print(f"数据加载/预处理过程中的值错误: {e}")
    import traceback
    traceback.print_exc()
    exit()
except Exception as e:
    print(f"加载数据时发生未知错误: {e}")
    import traceback
    traceback.print_exc()
    exit()


# --- BERT 模型 ---
print(f"加载 BERT 模型: {MODEL_PATH} 用于 {NUM_CLASSES} 个类别...")
try:
    bert_model_config = AutoConfig.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_CLASSES,
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        config=bert_model_config,
    )
    model = model.to(DEVICE)

except Exception as e:
    print(f"加载 BERT 模型时出错: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 损失函数、优化器 ---
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 添加学习率调度器
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

# --- 训练函数 ---
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc='训练中', leave=False)
    for batch in progress_bar:  # 去掉 batch_idx 参数，避免解包问题
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)
        
        model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if 'token_type_ids' in batch: # BERT uses token_type_ids
            model_inputs['token_type_ids'] = batch['token_type_ids'].to(device)

        optimizer.zero_grad()
        
        outputs = model(**model_inputs)
        logits = outputs.logits
        loss = criterion(logits, targets)

        loss.backward()
        
        # 添加梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # 更新学习率

        running_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)
        progress_bar.set_postfix(loss=running_loss / total_samples)

    epoch_loss = running_loss / total_samples
    return epoch_loss

# --- 评估函数 ---
def evaluate(model, loader, criterion, device, num_classes_eval): # Pass num_classes for mAP
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_scores = []

    progress_bar = tqdm(loader, desc='评估中', leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if 'token_type_ids' in batch:
                model_inputs['token_type_ids'] = batch['token_type_ids'].to(device)
            
            outputs = model(**model_inputs)
            logits = outputs.logits
            loss = criterion(logits, targets)

            running_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            scores = torch.sigmoid(logits)
            preds = (scores > THRESHOLD).float()
            
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_scores.append(scores.cpu())

            progress_bar.set_postfix(loss=running_loss / total_samples)

    epoch_loss = running_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()

    if all_targets.shape[1] == 0 or all_preds.shape[1] == 0:
        print("警告: 在目标或预测中未找到用于指标计算的标签列。")
        return {'loss': epoch_loss, 'mAP': 0.0, 'f1_micro': 0.0, 'f1_macro': 0.0}


    hamming_loss = np.mean(all_preds != all_targets) if all_targets.size > 0 else 0.0
    exact_match = np.mean(np.all(all_preds == all_targets, axis=1)) if all_targets.size > 0 else 0.0
    
    precision_micro = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_micro = recall_score(all_targets, all_preds, average='micro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_targets, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    ap_per_class = []
    # Ensure num_classes_eval is used here, which should be NUM_CLASSES
    for i in range(num_classes_eval): 
        # Check if all_targets has this class and if there are any positive examples for it
        if i < all_targets.shape[1] and np.sum(all_targets[:, i]) > 0:
            try:
                ap = average_precision_score(all_targets[:, i], all_scores[:, i])
                ap_per_class.append(ap)
            except ValueError: # Handles cases where a class might only have one value (e.g. all zeros)
                ap_per_class.append(0.0) # Or np.nan, then filter nans before mean
        # else:
            # If a class has no positive examples in the gold set, its AP is often undefined or 0.
            # ap_per_class.append(0.0) # Or skip, or use np.nan

    mAP = np.mean([ap for ap in ap_per_class if not np.isnan(ap)]) if ap_per_class else 0.0

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
print("开始训练...")
best_f1_sum = 0.0
best_metrics = None

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE)
    
    print(f"\n在验证集上评估第 {epoch+1} 轮...")
    # Pass NUM_CLASSES to evaluate for mAP calculation consistency
    metrics = evaluate(model, val_loader, criterion, DEVICE, NUM_CLASSES) 

    print(f"轮次 [{epoch+1}/{NUM_EPOCHS}] | "
          f"训练损失: {train_loss:.4f} | "
          f"验证损失: {metrics.get('loss', float('nan')):.4f} | "
          f"Hamming: {metrics.get('hamming_loss', float('nan')):.4f} | "
          f"精确匹配: {metrics.get('exact_match', float('nan')):.4f} | "
          f"mAP: {metrics.get('mAP', float('nan')):.4f}")
    print(f"    验证 P(micro/macro): {metrics.get('precision_micro', float('nan')):.4f}/{metrics.get('precision_macro', float('nan')):.4f}")
    print(f"    验证 R(micro/macro): {metrics.get('recall_micro', float('nan')):.4f}/{metrics.get('recall_macro', float('nan')):.4f}")
    print(f"    验证 F1(micro/macro): {metrics.get('f1_micro', float('nan')):.4f}/{metrics.get('f1_macro', float('nan')):.4f}")
    
    # 每3个epoch在测试集上评估一次
    # if (epoch + 1) % 3 == 0:
    print(f"\n在测试集上评估第 {epoch+1} 轮...")
    test_metrics = evaluate(model, test_loader, criterion, DEVICE, NUM_CLASSES)
    print(f"测试结果: | "
            f"损失: {test_metrics.get('loss', float('nan')):.4f} | "
            f"Hamming: {test_metrics.get('hamming_loss', float('nan')):.4f} | "
            f"精确匹配: {test_metrics.get('exact_match', float('nan')):.4f} | "
            f"mAP: {test_metrics.get('mAP', float('nan')):.4f}")
    print(f"    测试 P(micro/macro): {test_metrics.get('precision_micro', float('nan')):.4f}/{test_metrics.get('precision_macro', float('nan')):.4f}")
    print(f"    测试 R(micro/macro): {test_metrics.get('recall_micro', float('nan')):.4f}/{test_metrics.get('recall_macro', float('nan')):.4f}")
    print(f"    测试 F1(micro/macro): {test_metrics.get('f1_micro', float('nan')):.4f}/{test_metrics.get('f1_macro', float('nan')):.4f}")
    
    # 计算micro-f1和macro-f1的和，并记录最佳结果
    current_f1_sum = test_metrics.get('f1_micro', 0.0) + test_metrics.get('f1_macro', 0.0)
    if current_f1_sum > best_f1_sum:
        best_f1_sum = current_f1_sum
        best_metrics = test_metrics.copy()

print("\n训练完成!")

# 打印最佳评估结果
if best_metrics:
    print("\n最佳测试结果:")
    print(f"损失: {best_metrics.get('loss', float('nan')):.4f} | "
          f"Hamming: {best_metrics.get('hamming_loss', float('nan')):.4f} | "
          f"精确匹配: {best_metrics.get('exact_match', float('nan')):.4f} | "
          f"mAP: {best_metrics.get('mAP', float('nan')):.4f}")
    print(f"P(micro/macro): {best_metrics.get('precision_micro', float('nan')):.4f}/{best_metrics.get('precision_macro', float('nan')):.4f}")
    print(f"R(micro/macro): {best_metrics.get('recall_micro', float('nan')):.4f}/{best_metrics.get('recall_macro', float('nan')):.4f}")
    print(f"F1(micro/macro): {best_metrics.get('f1_micro', float('nan')):.4f}/{best_metrics.get('f1_macro', float('nan')):.4f}")
    print(f"F1总和(micro+macro): {best_metrics.get('f1_micro', 0.0) + best_metrics.get('f1_macro', 0.0):.4f}")

# --- 可选: 保存模型 ---
# output_dir = './bert_reuters_model_multi_label'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)
# print(f"模型和分词器已保存到 {output_dir}")