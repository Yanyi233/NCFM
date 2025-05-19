import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_norm(x_r, x_i):
    # 根据实部和虚部，计算幅度
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    # 计算虚部
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    # 计算实部
    return torch.mean(torch.cos(x), dim=1)


class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference.
    Args:
        alpha_for_loss: the weight for amplitude in CF loss, from 0-1
        beta_for_loss: the weight for phase in CF loss, from 0-1
    """

    def __init__(self, alpha_for_loss=0.5, beta_for_loss=0.5):
        super(CFLossFunc, self).__init__()
        self.alpha = alpha_for_loss
        self.beta = beta_for_loss

    def forward(self, feat_tg, feat, t=None, args=None):
        """
        Calculate CF loss between target and synthetic features.
        Args:
            feat_tg: target features from real data [B1 x D]
            feat: synthetic features [B2 x D]
            args: additional arguments containing num_freqs
        """
        # Generate random frequencies
        if t is None:
            t = torch.randn((args.num_freqs, feat.size(1)), device=feat.device)
        t_x_real = calculate_real(torch.matmul(t, feat.t()))
        t_x_imag = calculate_imag(torch.matmul(t, feat.t()))
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target_real = calculate_real(torch.matmul(t, feat_tg.t()))
        t_target_imag = calculate_imag(torch.matmul(t, feat_tg.t()))
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        # Calculate amplitude difference and phase difference
        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (
            torch.mul(t_target_norm, t_x_norm)
            - torch.mul(t_x_real, t_target_real)
            - torch.mul(t_x_imag, t_target_imag)
        )

        loss_pha = loss_pha.clamp(min=1e-12)  # Ensure numerical stability

        # Combine losses
        loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss


def match_loss(batch_real, batch_syn, model, args=None):
    """Matching losses (feature or gradient)"""
    with torch.no_grad():
        # Extract pooled_output for real batch
        # model.get_feature_mutil returns: [emb_cls, h1_cls, ..., h12_cls, pooler_output, logits]
        # pooled_output is at index -2
        feat_real = model.get_feature_single(
            input_ids=batch_real['input_ids'],
            attention_mask=batch_real.get('attention_mask'),
            token_type_ids=batch_real.get('token_type_ids')
        )

    # Extract pooled_output for synthetic batch
    feat_syn = model.get_feature_single(
        embedding=batch_syn
    )

    feat_real = F.normalize(feat_real, dim=1)
    feat_syn = F.normalize(feat_syn, dim=1)
    t = None # t is handled by CFLossFunc if it's None and args.num_freqs is present
    loss = 300 * args.cf_loss_func(feat_real, feat_syn, t, args)
    return loss


def mutil_layer_match_loss(batch_real, batch_syn, model, args=None):

    # Ensure layer_index is a list
    assert isinstance(
        args.layer_index, list
    ), "args.layer_index must be a list of layer indices"
    
    # Initialize loss as a tensor on the correct device
    # Assuming batch_real['input_ids'] is always present and a tensor
    loss = torch.tensor(0.0).to(batch_real['input_ids'].device)

    # Extract features for both real and synthetic images/batches
    with torch.no_grad():
        feat_tg_list = model.get_feature_mutil(
            input_ids=batch_real['input_ids'],
            attention_mask=batch_real.get('attention_mask'),
            token_type_ids=batch_real.get('token_type_ids')
        )
    feat_list = model.get_feature_mutil(
        embedding=batch_syn
    )

    for layer_index in args.layer_index:
        # The assertion for layer_index (0-6) is kept as per original code.
        # BERT.py's get_feature_mutil returns 15 features for bert-base:
        # 0: initial_embedding_cls
        # 1-12: hidden_state_1_cls to hidden_state_12_cls
        # 13: pooler_output
        # 14: logits
        # If you intend to use up to the 6th transformer layer's [CLS] token,
        # layer_index 6 would correspond to feat_list[6] (output of 6th encoder layer).
        assert (
             0 <= layer_index < len(feat_list) - 2 # Ensure index is valid and not pooler or logits for layer-wise comparison
        ), f"layer_index {layer_index} must be a valid index for hidden states. Max layer index: {len(feat_list) - 3}"


        if args.dis_metrics == "MMD":
            # If the metric is MMD, calculate the MMD loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            loss += torch.sum((feat.mean(0) - feat_tg.mean(0)) ** 2)
        else:
            # Otherwise, calculate the feature matching loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            feat = F.normalize(feat, dim=1)  # Normalize the feature
            feat_tg = F.normalize(feat_tg, dim=1)  # Normalize the target feature
            t = None  # Adjust this based on your CFLossFunc usage
            loss += 300 * args.cf_loss_func(feat_tg, feat, t, args) / len(args.layer_index)

    return loss


def cailb_loss(img_syn, label_syn, trained_model):
    logits = trained_model(img_syn, return_features=False)
    loss = F.cross_entropy(logits, label_syn)
    return loss