# Condense（数据蒸馏）模块实现说明

## 概述

`condense` 模块实现了基于 NCFM（Neural Characteristic Function Matching）方法的数据蒸馏（Data Condensation）功能，主要用于文本分类任务（如 Reuters 数据集）。该模块通过优化合成数据的 embedding，使其在特征空间中匹配真实数据的特征分布。

## 目录结构

```
condense/
├── condense_script_text.py    # 文本数据蒸馏主脚本
├── condense_script.py          # 图像数据蒸馏主脚本（参考）
└── README.md                   # 本文档
```

## 核心依赖模块

- `condenser/Condenser_text.py` - 数据蒸馏器核心类
- `condenser/compute_loss_text.py` - 损失计算函数
- `NCFM/NCFM_text.py` - NCFM 损失函数实现
- `utils/utils_text.py` - 文本处理工具函数
- `utils/init_script.py` - 分布式训练初始化
- `utils/ddp.py` - 分布式训练辅助函数

## 启动命令

在 `condense` 目录下执行：

```bash
torchrun --nproc_per_node=6 --nnodes=1 condense_script_text.py \
  --gpu="0,2,3,4,5,6" \
  --config_path=../config/ipc10/reuters.yaml
```

### 参数说明

- `--nproc_per_node=6`: 每个节点的进程数（GPU数量）
- `--nnodes=1`: 节点数量
- `--gpu="0,2,3,4,5,6"`: 使用的GPU编号（逗号分隔）
- `--config_path`: YAML 配置文件路径

### 其他可选参数

- `--ipc`: 每个类别的样本数（默认从配置文件读取）
- `--init`: 初始化类型（`random`, `noise`, `mix`, `load`），默认 `mix`
- `--load_path`: 加载已有合成数据的路径（用于 `init=load`）
- `--sampling_net`: 启用采样网络（默认 False）
- `--debug`: 启用调试模式（更频繁的日志输出）
- `--tf32`: 启用 TensorFloat-32（默认 True）

## 配置文件结构

配置文件位于 `config/ipc*/reuters.yaml`，主要包含以下部分：

### 1. 数据集配置
```yaml
dataset:
  dataset: reuters
  nclass: 90
  data_dir: '../dataset/reuters'
  load_memory: True
  batch_real: 256
  is_multilabel: True
```

### 2. 网络配置
```yaml
network:
  model_path: '../models/model/bert-base-uncased'
  net_type: BERT
  norm_type: layernorm
  depth: 12
  layer_index: [3, 6, 9, 12]
  max_length: 512
```

### 3. 优化配置
```yaml
optimization:
  optimizer: sgd              # 或 adam
  lr_img: 0.01               # 合成数据学习率（SGD）
  mom_img: 0.5               # 动量（SGD）
  lr_scale_adam: 0.1         # Adam学习率缩放因子
  lr_sampling_net: 1e-3      # 采样网络学习率
```

### 4. 蒸馏配置
```yaml
condense:
  ipc: 10                    # 每个类别的样本数
  num_premodel: 10           # 使用的预训练模型数量
  niter: 400                 # 蒸馏迭代次数
  iter_calib: 1              # 校准迭代次数
  calib_weight: 1            # 校准损失权重
  sampling_net: False        # 是否使用采样网络
  num_freqs: 4096            # NCFM 频率数量
  dis_metrics: "NCFM"        # 蒸馏度量（NCFM 或 MMD）
  factor: 2                  # NCFM 因子
  alpha_for_loss: 0.5        # NCFM 损失 alpha（幅度权重）
  beta_for_loss: 0.5         # NCFM 损失 beta（相位权重）
  decode_type: 'single'      # 解码类型
  teacher_model_epoch: 20    # 使用的教师模型 epoch
```

## 代码实现流程

### 1. 初始化阶段 (`condense_script_text.py`)

```python
# 1. 解析命令行参数和配置文件
args = args_processor.add_args_from_yaml(args)

# 2. 初始化分布式训练环境
init_script(args)

# 3. 分配类别到各个进程（类别级并行）
args.class_list = distribute_class(args.nclass, args.debug)

# 4. 初始化数据加载器
loader_real, _ = get_loader(args)

# 5. 初始化 Condenser
condenser = Condenser(args, nclass_list=args.class_list, device='cuda')
```

### 2. Condenser 类结构 (`condenser/Condenser_text.py`)

#### 初始化数据
- **合成 embedding**: `(nclass * ipc, max_length, 768)` - BERT embedding 维度
- **attention_mask**: `(nclass * ipc, max_length)` - 注意力掩码
- **标签**: `(nclass * ipc, total_nclass)` - 多标签分类标签（one-hot）

#### 初始化方法
- `random`: 从真实数据中随机采样初始化
- `noise`: 随机噪声初始化（默认）
- `mix`: 混合初始化
- `load`: 从文件加载已有合成数据

### 3. 蒸馏循环 (`condenser.condense()`)

每次迭代包含以下步骤：

1. **更新特征提取器**: 加载不同 epoch 的教师模型
   ```python
   model_init, model_final, model_interval = update_feature_extractor(
       args, model_init, model_final, model_interval, a=0, b=1
   )
   ```

2. **计算匹配损失** (`compute_match_loss`):
   - 对每个类别，从真实数据和合成数据中采样批次
   - 使用教师模型提取特征（`model_interval`）
   - 计算 NCFM 损失（特征匹配）

3. **计算校准损失** (`compute_calib_loss`, 可选):
   - 使用最终模型（`model_final`）进行分类损失
   - 帮助合成数据保持分类能力

4. **优化合成数据**:
   - 反向传播更新合成 embedding
   - 可选：同时优化采样网络

5. **保存和日志**:
   - 定期保存合成数据到 `distilled_data/`
   - 记录损失曲线到 `images/`

### 4. NCFM 损失函数 (`NCFM/NCFM_text.py`)

#### 单层匹配 (`match_loss`)
```python
# 提取特征
feat_real = model.get_feature_single(batch_real)
feat_syn = model.get_feature_single(embedding=batch_syn)

# 归一化
feat_real = F.normalize(feat_real, dim=1)
feat_syn = F.normalize(feat_syn, dim=1)

# NCFM 损失
loss = 300 * args.cf_loss_func(feat_real, feat_syn, t, args)
```

#### 多层匹配 (`mutil_layer_match_loss`)
- 在多个 Transformer 层提取特征（`layer_index`）
- 对每层计算 NCFM 损失并求和平均

#### CF 损失 (`CFLossFunc`)
- **幅度差异**: `amp_diff = t_target_norm - t_x_norm`
- **相位差异**: 基于实部和虚部的余弦相似度
- **总损失**: `sqrt(alpha * loss_amp + beta * loss_pha)`

### 5. 分布式训练

- **类别级并行**: 每个进程处理不同的类别子集
- **数据同步**: 使用 `dist.barrier()` 同步各进程
- **梯度聚合**: 使用 `sync_distributed_metric()` 聚合指标
- **数据保存**: 仅在 rank 0 保存合成数据

## 输出结果

训练完成后，结果保存在 `results/condense/condense/{dataset}/ipc{ipc}/...` 目录下：

```
results/condense/condense/reuters/ipc10/.../
├── distilled_data/        # 合成数据（embedding + labels）
│   ├── syn_data_0.pth     # 第 0 次迭代的合成数据
│   ├── syn_data_200.pth   # 第 200 次迭代的合成数据
│   └── ...
├── images/                # 损失曲线图
│   └── loss_curve.png
├── print.log              # 训练日志
└── args.log               # 配置文件副本
```

## 关键设计要点

### 1. 文本数据表示
- 直接优化 BERT embedding（`768` 维），而非原始 token IDs
- 使用 `attention_mask` 处理变长序列
- 支持多标签分类（`is_multilabel=True`）

### 2. 特征提取策略
- **单层**: 仅使用 `pooler_output`（适用于浅层模型）
- **多层**: 使用多个 Transformer 层的 `[CLS]` token（`layer_index`）

### 3. 损失函数组合
- **匹配损失**: 特征空间对齐（NCFM）
- **校准损失**: 分类能力保持（CrossEntropy）

### 4. 优化器选择
- **SGD**: 使用 `lr_img` 和 `mom_img`
- **Adam/AdamW**: 使用 `lr_img * lr_scale_adam`

## 常见问题

### Q: 如何调整合成数据数量？
A: 修改配置文件中的 `condense.ipc` 参数或使用 `--ipc` 命令行参数。

### Q: 如何提高蒸馏质量？
A: 
- 增加 `niter`（迭代次数）
- 增加 `num_freqs`（NCFM 频率数）
- 使用多层特征匹配（`layer_index`）
- 调整 `alpha_for_loss` 和 `beta_for_loss`

### Q: 如何处理内存不足？
A: 
- 减少 `batch_real`
- 减少 `num_premodel`
- 减少 `num_freqs`

### Q: 如何使用已保存的合成数据继续训练？
A: 使用 `--init load --load_path <path>` 参数。

## 参考

- 项目根目录: `/home/wjh/NCFM`
- 配置文件: `config/ipc*/reuters.yaml`
- 预训练模型: `pretrained_models/reuters/ipc10/bert_20251030`
- 结果保存: `results/condense/condense/reuters/ipc10/...`

