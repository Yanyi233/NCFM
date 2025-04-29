import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18WithFeatures(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True): # 默认类别数可以设为 ImageNet 的
        super().__init__()
        # 加载 ResNet18 模型
        self.resnet = models.resnet18(pretrained=pretrained)

        # --- 保存对 ResNet 内部关键层/块的引用 ---
        # 这使得我们可以在 get_feature_mutil 中按顺序调用它们
        self.layer0_conv1 = self.resnet.conv1
        self.layer0_bn1 = self.resnet.bn1
        self.layer0_relu = self.resnet.relu
        self.layer0_maxpool = self.resnet.maxpool

        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        self.avgpool = self.resnet.avgpool
        # 获取原始 fc 层的输入特征数，以便正确替换
        num_ftrs = self.resnet.fc.in_features
        # 替换最后一层以匹配你的任务类别数（如果需要的话）
        # 注意：这里假设最终目的是用于你的 VOC 任务，所以替换掉
        # 如果只是用作特征提取器而不修改最后一层，则可以注释掉下一行
        self.fc = nn.Linear(num_ftrs, num_classes)
        # 将修改后的 fc 层也赋给内部的 resnet 实例（可选，但保持一致性）
        self.resnet.fc = self.fc

    def forward(self, x, return_features=False):
        """
        可以根据 return_features 标志选择性地返回 avgpool 后的特征。
        """
        # Layer 0
        x = self.layer0_conv1(x)
        x = self.layer0_bn1(x)
        x = self.layer0_relu(x)
        x = self.layer0_maxpool(x)

        # Layer 1-4
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global AvgPool and Flatten
        features_before_fc = self.avgpool(x)
        features_before_fc = torch.flatten(features_before_fc, 1) # (batch_size, num_features)

        # Final FC layer (Logits)
        out = self.fc(features_before_fc)

        if return_features:
            # 返回 logits 和 avgpool 后展平的特征
            return out, features_before_fc
        else:
            # 只返回 logits
            return out
        
    def get_feature_mutil(self, x):
        """
        执行前向传播并收集指定中间层的特征。
        返回一个包含 7 个特征张量的列表 (索引 0-6):
        - Index 0: Output after initial MaxPool
        - Index 1: Output after Layer 1
        - Index 2: Output after Layer 2
        - Index 3: Output after Layer 3
        - Index 4: Output after Layer 4
        - Index 5: Output after Global AvgPool (before FC)
        - Index 6: Output after FC (Logits)
        所有空间特征图都被展平为 (batch_size, num_features)。
        """
        features = []

        # Layer 0 features
        x = self.layer0_conv1(x)
        x = self.layer0_bn1(x)
        x = self.layer0_relu(x)
        x = self.layer0_maxpool(x)
        features.append(x.view(x.size(0), -1)) # Flatten and add (Index 0)

        # Layer 1 features
        x = self.layer1(x)
        features.append(x.view(x.size(0), -1)) # Flatten and add (Index 1)

        # Layer 2 features
        x = self.layer2(x)
        features.append(x.view(x.size(0), -1)) # Flatten and add (Index 2)

        # Layer 3 features
        x = self.layer3(x)
        features.append(x.view(x.size(0), -1)) # Flatten and add (Index 3)

        # Layer 4 features
        x = self.layer4(x)
        features.append(x.view(x.size(0), -1)) # Flatten and add (Index 4)

        # After AvgPool
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten
        features.append(x) # Add (Index 5)

        # After FC (Logits)
        x = self.fc(x)
        features.append(x) # Add (Index 6)

        # 确保返回了 7 个特征
        assert len(features) == 7
        return features

# # --- 如何使用 ---
# nclass = NUM_CLASSES # 你的 VOC 类别数
# # 实例化包装后的模型
# model_wrapped = ResNet18WithFeatures(num_classes=nclass, pretrained=True).to(DEVICE)

# 在你的训练或评估代码中，使用 model_wrapped 替代原始的 model
# 例如，在计算 match loss 时:
# feat_tg_list = model_wrapped.get_feature_mutil(img_real)
# feat_list = model_wrapped.get_feature_mutil(img_syn)
# ...然后使用 mutil_layer_match_loss 函数...

# 如果你需要原始的 ResNet18 对象（例如，用于加载/保存 state_dict），
# 可以访问 model_wrapped.resnet
# 注意：保存和加载 state_dict 时要小心，因为包装类的 state_dict 包含了 resnet 的参数
# 可能需要这样加载：
# state_dict = torch.load('your_checkpoint.pth')
# model_wrapped.load_state_dict(state_dict)
# 或者只加载 resnet 部分：
# model_wrapped.resnet.load_state_dict(torch.load('resnet_only_checkpoint.pth'))