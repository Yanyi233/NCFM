import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERTWithFeatures(nn.Module):
    def __init__(self, num_classes=2, pretrained_model_name='../models/model/bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True, local_files_only=True)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

        # 在初始化时直接禁用 BERT 内部 embedding 层的梯度
        if hasattr(self.bert, 'embeddings'):
            for param in self.bert.embeddings.parameters():
                param.requires_grad_(False)
            print("Disabled gradients for BERT embeddings layer.") # 可选的日志输出

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, embedding=None, return_features=False):
        if embedding is not None:
            # 处理合成样本 (embedding 形状: [batch_size, seq_len, hidden_size])
            if input_ids is not None:
                raise ValueError("Cannot provide both 'input_ids' and 'embedding'.")
            
            # 自动为合成样本生成 attention_mask (如果未提供)
            if attention_mask is None:
                batch_size, seq_len, _ = embedding.shape
                attention_mask = torch.ones(batch_size, seq_len, device=embedding.device, dtype=torch.long)
            
            # 注意：这里不再需要动态修改 embedding 层的 requires_grad
            outputs = self.bert(inputs_embeds=embedding, 
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids,
                                output_hidden_states=True)
        elif input_ids is not None:
            # 处理真实样本
            # 注意：这里也不再需要动态修改 embedding 层的 requires_grad
            # 因为在 __init__ 中已经将其设置为 False
            # 如果这里传入 input_ids，BERT 内部的 embedding 层仍然会被调用，
            # 但由于其参数的 requires_grad 为 False，所以不会计算和累积梯度。
            outputs = self.bert(input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids, 
                                output_hidden_states=True)
        else:
            raise ValueError("Either 'input_ids' or 'embedding' must be provided.")
        
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        if return_features:
            return logits, pooled_output
        else:
            return logits
        
    def get_feature(self, input_ids=None, attention_mask=None, token_type_ids=None, embedding=None):
        # 在这个实现中，get_feature 不需要再单独处理梯度，
        # 因为bert.embeddings的梯度已经在__init__中被禁用了。
        if embedding is not None:
            batch_size, seq_len, _ = embedding.shape
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_len, device=embedding.device, dtype=torch.long)
            
            outputs = self.bert(inputs_embeds=embedding, 
                                attention_mask=attention_mask, 
                                token_type_ids=token_type_ids,
                                output_hidden_states=True)
        elif input_ids is not None:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                output_hidden_states=True)
        else:
            raise ValueError("Either 'input_ids' or 'embedding' must be provided to get_feature.")
        
        return outputs.hidden_states, outputs.pooler_output

    def get_feature_last_layer(self, input_ids=None, attention_mask=None, token_type_ids=None, embedding=None):
        hidden_states, pooler_output = self.get_feature(input_ids, attention_mask, token_type_ids, embedding)
        
        # 返回最后一层Transformer Encoder的[CLS] token的特征
        return hidden_states[-1][:, 0, :]

    def get_feature_mutil(self, input_ids=None, attention_mask=None, token_type_ids=None, embedding=None):
        """
        返回一个包含13层hidden_states（包括初始embedding和12层transformer encoder输出的[CLS]特征），
        以及pooler_output和logits。
        返回：
        features: [initial_embedding_cls, hidden_state_1_cls, ..., hidden_state_12_cls, pooler_output, logits]
        """
        hidden_states, pooler_output = self.get_feature(input_ids, attention_mask, token_type_ids, embedding)

        features = []
        # hidden_states 包含:
        # 0: initial embeddings
        # 1-12: output of each of the 12 BERT layers
        # Total 13 hidden states
        for hidden_state in hidden_states:
            # 使用[CLS]token的特征
            feature = hidden_state[:, 0, :]
            features.append(feature)
        
        logits = self.classifier(pooler_output)
        
        features.append(pooler_output)
        features.append(logits)
        
        # 初始embedding + 12层Transformer输出 + pooler_output + logits = 13 + 1 + 1 = 15
        assert len(features) == (self.bert.config.num_hidden_layers + 1 + 2), \
            f"Expected {self.bert.config.num_hidden_layers + 1 + 2} features, but got {len(features)}"
        return features

# 用法示例：
# model = BERTWithFeatures(num_classes=NUM_CLASSES)
# # 对于真实样本
# features_real = model.get_feature_mutil(input_ids, attention_mask, token_type_ids)
# last_layer_feature_real = model.get_feature_last_layer(input_ids, attention_mask, token_type_ids)
# # 对于合成样本 (假设 syn_embedding 是 [batch, seq_len, hidden_size])
# # syn_attention_mask = torch.ones(syn_embedding.shape[0], syn_embedding.shape[1], device=syn_embedding.device) # 可选
# # features_syn = model.get_feature_mutil(embedding=syn_embedding, attention_mask=syn_attention_mask)
# # last_layer_feature_syn = model.get_feature_last_layer(embedding=syn_embedding, attention_mask=syn_attention_mask)

if __name__ == "__main__":
    model = BERTWithFeatures(num_classes=90, pretrained_model_name='models/model/bert-base-uncased')
    input_ids = torch.randint(0, 100, (1, 128))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    features = model.get_feature_mutil(input_ids, attention_mask, token_type_ids)
    for i, feature in enumerate(features):
        print(f"layer {i+1} shape: {feature.shape}")