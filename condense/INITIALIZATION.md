# 合成数据集初始化详解

本文档详细说明合成数据集的初始化过程，包括标签和样本内容的初始化方式。

## 概述

合成数据集的初始化发生在 `Condenser` 类的两个阶段：
1. **`__init__` 方法**: 创建数据结构并设置标签
2. **`load_condensed_data` 方法**: 根据初始化类型设置样本内容

## 数据结构

合成数据集包含三个核心组件：

```python
# 1. 合成 embedding (可优化参数)
self.data: (nclass * ipc, max_length, 768)
  - 数据类型: torch.float32
  - requires_grad: True (用于反向传播优化)
  - 设备: cuda

# 2. 注意力掩码 (固定)
self.attention_mask: (nclass * ipc, max_length)
  - 数据类型: torch.long
  - requires_grad: False
  - 设备: cuda
  - 默认值: 全 1（表示所有位置都有效）

# 3. 多标签向量 (固定)
self.targets: (nclass * ipc, total_nclass)
  - 数据类型: torch.float
  - requires_grad: False
  - 设备: cuda
```

## 标签初始化

### 初始化位置
标签在 `Condenser.__init__` 方法中初始化（第 49-72 行）：

```python
# 1. 创建全零标签矩阵
self.targets = torch.zeros(
    (self.nclass * self.ipc, self.total_nclass),
    dtype=torch.float,
    requires_grad=False,
    device=self.device
)

# 2. 为每个类别设置标签（one-hot）
for i, c in enumerate(self.nclass_list):
    start_idx = i * self.ipc
    end_idx = (i + 1) * self.ipc
    self.targets[start_idx:end_idx, c] = 1.0
```

### 标签格式说明

- **多标签分类**: 虽然代码支持多标签（`is_multilabel=True`），但在初始化时，每个合成样本只被分配一个主类别标签
- **标签分配规则**: 
  - 每个进程处理 `nclass` 个类别（例如：6个进程，每个处理15个类别，共90个类别）
  - 对于类别 `c`（在 `nclass_list` 中的第 `i` 个位置），其对应的 `ipc` 个样本的标签向量中，第 `c` 位被设置为 `1.0`，其余为 `0.0`
  - 例如：类别 5（第3个位置）的10个样本，标签为 `[0, 0, 0, 0, 0, 1, 0, ..., 0]`

### 类别索引映射

```python
# 记录每个类别的样本索引
self.cls_idx = [[] for _ in range(self.nclass)]
for i in range(self.data.shape[0]):
    # 对于多标签情况，找到第一个为1的标签作为主类别
    main_class = torch.where(self.targets[i] == 1)[0][0].item()
    # 将主类别映射到本地类别索引
    local_class_idx = self.nclass_list.index(main_class)
    self.cls_idx[local_class_idx].append(i)
```

## 样本内容初始化

样本内容的初始化通过 `load_condensed_data` 方法完成，支持四种初始化类型：

### 1. `noise` 初始化（默认）

**代码位置**: `Condenser.__init__` 第 34-39 行

```python
# 使用正态分布初始化 embedding
self.data = torch.randn(
    (self.nclass * self.ipc, args.max_length, 768),
    dtype=torch.float32,
    requires_grad=True,
    device=self.device
)
```

**特点**:
- 从标准正态分布 `N(0, 1)` 随机采样
- 不依赖真实数据
- 从随机噪声开始优化

**适用场景**: 
- 探索合成数据的可学习性
- 避免真实数据的偏差

### 2. `random` 初始化

**代码位置**: `Condenser.load_condensed_data` 第 84-97 行

```python
if init_type == "random":
    for c in self.nclass_list:
        # 从真实数据中采样
        data, target = loader.class_sample(c, self.ipc)
        
        # 确定位置
        start_idx = self.ipc * self.nclass_list.index(c)
        end_idx = self.ipc * (self.nclass_list.index(c) + 1)
        
        # 更新数据
        self.data[start_idx:end_idx] = data['embeddings']  # 注意：这里期望 embeddings
        self.attention_mask[start_idx:end_idx] = data['attention_mask']
        self.targets[start_idx:end_idx] = target
```

**特点**:
- 从真实数据集中随机采样每个类别的 `ipc` 个样本
- 使用真实样本的 embedding（而非 token IDs）

**注意事项**:
- ⚠️ **潜在问题**: `MultiLabelClassMemDataLoader.class_sample()` 返回的是 `{'input_ids': ..., 'attention_mask': ...}`（见 `data/dataloader.py:599-613`），但代码期望 `data['embeddings']`（见 `Condenser_text.py:95`）
- **需要修复**: 如果使用 `random` 初始化，需要在 `load_condensed_data` 中添加转换逻辑：
  ```python
  # 修复建议
  if init_type == "random":
      # 获取BERT模型（需要从args或全局获取）
      model = get_feature_extractor(args)[0]  # 获取模型
      with torch.no_grad():
          # 将 input_ids 转换为 embeddings
          if 'input_ids' in data:
              embeddings = model.bert.embeddings(
                  input_ids=data['input_ids'],
                  token_type_ids=data.get('token_type_ids', None)
              )
              data['embeddings'] = embeddings
      # 然后使用 data['embeddings']
      self.data[start_idx:end_idx] = data['embeddings']
  ```
- **当前状态**: 如果直接使用会报 `KeyError: 'embeddings'` 错误，需要先修复代码

**适用场景**:
- 希望从真实数据开始优化（更好的起点）
- 需要更快的收敛速度

### 3. `mix` 初始化

**代码位置**: `Condenser.load_condensed_data` 第 99-102 行

**实际行为**:
- 当前代码中，`mix` 初始化实际上会执行 `noise` 分支（因为 `mix` 不在条件判断中）
- 查看代码逻辑，`mix` 会fallthrough到 `noise`，即等同于 `noise` 初始化
- 因此 `mix` 和 `noise` 在功能上是相同的

```python
elif init_type == "noise":  # mix 也会进入这里（因为没有 mix 的 elif）
    if dist.get_rank() == 0:
        self.logger("===================Noise initialize condensed dataset===================")
    pass  # 保持 __init__ 中的随机噪声
```

**注意**: 如果需要实现真正的混合初始化，需要添加：
```python
elif init_type == "mix":
    # 混合噪声和随机采样
    for c in self.nclass_list:
        random_count = self.ipc // 2
        noise_count = self.ipc - random_count
        
        # 部分从真实数据采样
        random_data, random_target = loader.class_sample(c, random_count)
        
        # 部分保持噪声
        start_idx = self.ipc * self.nclass_list.index(c)
        noise_data = self.data[start_idx:start_idx + noise_count]
        
        # 合并并更新
        # ...
```

### 4. `load` 初始化

**代码位置**: `Condenser.load_condensed_data` 第 104-111 行

```python
elif init_type == "load":
    if load_path is None:
        raise ValueError("Please provide the path of the initialization data")
    
    data_dict = torch.load(load_path)
    self.data = data_dict[0].to(self.device)
    self.targets = data_dict[1].to(self.device)
```

**特点**:
- 从保存的 `.pth` 文件加载已训练的合成数据
- 文件格式: `(embeddings, targets)` 的元组或列表
- 用于继续训练或微调已有合成数据

**文件格式**:
```python
# 保存格式
torch.save(
    (self.data.cpu(), self.targets.cpu()),
    save_path
)

# 加载格式
data_dict = torch.load(load_path)
# data_dict[0]: embeddings (nclass * ipc, max_length, 768)
# data_dict[1]: targets (nclass * ipc, total_nclass)
```

**适用场景**:
- 从之前的训练checkpoint继续
- 微调合成数据集

## 初始化流程

完整初始化流程如下：

```python
# 步骤 1: 创建 Condenser 实例
condenser = Condenser(args, nclass_list=args.class_list, device='cuda')
  ├─ 初始化 self.data (随机噪声)
  ├─ 初始化 self.attention_mask (全 1)
  └─ 初始化 self.targets (one-hot 标签)

# 步骤 2: 根据初始化类型加载数据
condenser.load_condensed_data(
    loader_real,
    init_type=args.init,  # 'noise', 'random', 'mix', 'load'
    load_path=args.load_path
)
  ├─ 'noise': 保持随机噪声（默认）
  ├─ 'random': 从真实数据采样并转换为 embeddings
  ├─ 'mix': 混合初始化（需确认实现）
  └─ 'load': 从文件加载
```

## 关键代码引用

### 标签初始化
```32:72:condenser/Condenser_text.py
# 初始化合成数据
# 使用正态分布初始化embedding，范围与BERT的embedding层初始化范围一致
self.data = torch.randn(
    (self.nclass * self.ipc, args.max_length, 768),  # 768是BERT的隐藏层维度
    dtype=torch.float32,
    requires_grad=True,
    device=self.device
)

# attention_mask作为类属性单独存储
self.attention_mask = torch.ones(
    (self.nclass * self.ipc, args.max_length),
    dtype=torch.long,
    requires_grad=False,
    device=self.device
)

# 修改标签维度为总类别数（90）
self.targets = torch.zeros(
    (self.nclass * self.ipc, self.total_nclass),
    dtype=torch.float,
    requires_grad=False,
    device=self.device
)

# ... 为每个类别设置标签
for i, c in enumerate(self.nclass_list):
    start_idx = i * self.ipc
    end_idx = (i + 1) * self.ipc
    self.targets[start_idx:end_idx, c] = 1.0
```

### 数据内容初始化
```83:111:condenser/Condenser_text.py
def load_condensed_data(self, loader, init_type="noise", load_path=None):
    if init_type == "random":
        # ... 从真实数据采样
    elif init_type == "noise":
        # ... 保持随机噪声
    elif init_type == "load":
        # ... 从文件加载
```

## 常见问题

### Q1: 为什么使用 embedding 而不是 token IDs？
**A**: 
- Token IDs 是离散的，无法直接梯度优化
- Embedding 是连续的，可以直接优化
- 直接在 embedding 空间优化可以绕过 tokenization 的限制

### Q2: attention_mask 为什么全设为 1？
**A**: 
- 简化初始化，所有位置都视为有效
- 在实际训练中，可以根据需要动态调整
- 对于固定长度序列，全 1 是合理的

### Q3: random 初始化时如何处理 input_ids 到 embeddings 的转换？
**A**: 
- 如果数据加载器返回 `input_ids`，需要通过 BERT 的 embedding 层转换
- 转换代码示例：
  ```python
  with torch.no_grad():
      model = BERTWithFeatures(...)
      embeddings = model.bert.embeddings(
          input_ids=data['input_ids'],
          token_type_ids=data.get('token_type_ids', None)
      )
  ```
- 注意：当前代码中 `data['embeddings']` 可能需要修复为处理 `input_ids` 的情况

### Q4: 如何验证初始化是否正确？
**A**: 
```python
# 检查数据形状
assert self.data.shape == (nclass * ipc, max_length, 768)
assert self.attention_mask.shape == (nclass * ipc, max_length)
assert self.targets.shape == (nclass * ipc, total_nclass)

# 检查标签
for i, c in enumerate(self.nclass_list):
    start = i * self.ipc
    end = (i + 1) * self.ipc
    assert self.targets[start:end, c].sum() == self.ipc  # 每个类别的标签应该全为1
```

## 总结

1. **标签初始化**: 在 `__init__` 中创建 one-hot 标签，每个样本对应一个主类别
2. **样本初始化**: 通过 `load_condensed_data` 根据 `init_type` 选择：
   - `noise`: 随机噪声（默认）
   - `random`: 从真实数据采样
   - `mix`: 混合方式（需确认）
   - `load`: 从文件加载
3. **分布式**: 每个进程处理不同的类别子集，初始化是并行进行的

