import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERTWithFeatures(nn.Module):
    def __init__(self, num_classes=2, pretrained_model_name='/home/wjh/NCFM/models/model/bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True, local_files_only=True)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_features=False):
        # BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        pooled_output = outputs.pooler_output  # [batch, hidden]
        logits = self.classifier(pooled_output)
        if return_features:
            return logits, pooled_output
        else:
            return logits

    def get_feature_mutil(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        返回一个包含12层transformer encoder输出的列表（每层shape: [batch, seq_len, hidden]），
        以及pooler_output和logits。
        返回：
        features: [hidden_state_1, ..., hidden_state_12, pooler_output, logits]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        features = []
        for i in range(len(outputs.hidden_states)):
            # 使用[CLS]token的特征
            feature = outputs.hidden_states[i][:, 0, :]
            features.append(feature)
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)  # [batch, num_classes]
        features = features + [pooler_output, logits]
        assert len(features) == 15  # embeddings+12层+pooler+logits
        return features

# 用法示例：
# model = BERTWithFeatures(num_classes=NUM_CLASSES)
# features = model.get_feature_mutil(input_ids, attention_mask, token_type_ids)

if __name__ == "__main__":
    model = BERTWithFeatures(num_classes=90)
    input_ids = torch.randint(0, 100, (1, 128))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    features = model.get_feature_mutil(input_ids, attention_mask, token_type_ids)
    for i, feature in enumerate(features):
        print(f"layer {i+1} shape: {feature.shape}")