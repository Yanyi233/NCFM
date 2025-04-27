import re
from statistics import mean

# 读取日志文件
with open('pretrained_models/ipc10_0422/voc2007/print.log', 'r') as f:
    log_content = f.read()

# 匹配每个预训练模型的最后一个epoch（Epoch 59）的Val指标行
pattern = r'<Pretraining\s+(\d+)-th model>...\[Epoch 59\].*Val P: ([\d\.]+) R: ([\d\.]+) F1: ([\d\.]+) Hamming: ([\d\.]+) mAP: ([\d\.]+)'
matches = re.findall(pattern, log_content)

# 准备数据结构来存储结果
results = []
for match in matches:
    model_idx, precision, recall, f1, hamming, map_value = match
    results.append({
        'Model': int(model_idx),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1': float(f1),
        'Hamming': float(hamming),
        'mAP': float(map_value)
    })

# 计算平均值
precisions = [result['Precision'] for result in results]
recalls = [result['Recall'] for result in results]
f1s = [result['F1'] for result in results]
hammings = [result['Hamming'] for result in results]
maps = [result['mAP'] for result in results]

# 打印Markdown表格格式的平均值
print("## 预训练模型最后一个Epoch的Val指标平均值\n")
print("| Precision | Recall | F1 | Hamming | mAP |")
print("|-----------|--------|----|---------|----|")
print(f"| {mean(precisions):.4f} | {mean(recalls):.4f} | {mean(f1s):.4f} | {mean(hammings):.4f} | {mean(maps):.4f} |")

# 打印每个模型的结果
print("\n## 每个预训练模型的最后一个Epoch的Val指标\n")
print("| 模型 | Precision | Recall | F1 | Hamming | mAP |")
print("|------|-----------|--------|----|---------|----|")
for result in results:
    print(f"| {result['Model']} | {result['Precision']:.4f} | {result['Recall']:.4f} | {result['F1']:.4f} | {result['Hamming']:.4f} | {result['mAP']:.4f} |") 