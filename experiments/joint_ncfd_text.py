import argparse
import os
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from argsprocessor.args import ArgsProcessor
from utils.utils_text import load_reuters_data, define_language_model


ALPHA = 10.0


def get_joint_vector(embeds: torch.Tensor, labels: torch.Tensor, alpha: float = ALPHA) -> torch.Tensor:
    weighted_labels = labels * alpha
    return torch.cat([embeds, weighted_labels], dim=1)


def kmeans_initialize(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    num_syn: int,
    num_iters: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = embeds.device
    num_samples = embeds.size(0)
    if num_syn > num_samples:
        raise ValueError("num_syn cannot exceed available samples for initialization")

    perm = torch.randperm(num_samples, device=device)
    centers = labels[perm[:num_syn]].clone()
    center_embeds = embeds[perm[:num_syn]].clone()

    for _ in range(num_iters):
        dists = torch.cdist(labels, centers, p=2)
        assignments = torch.argmin(dists, dim=1)
        for c in range(num_syn):
            mask = assignments == c
            if mask.sum() == 0:
                continue
            centers[c] = labels[mask].mean(dim=0)
            center_embeds[c] = embeds[mask].mean(dim=0)

    return center_embeds, centers


class BalancedClassSampler(Sampler[int]):
    """Class-balanced sampler for multi-label data."""

    def __init__(self, labels: torch.Tensor, samples_per_epoch: int):
        self.labels = labels
        self.samples_per_epoch = samples_per_epoch
        class_freq = labels.sum(dim=0) + 1e-6
        inv_freq = 1.0 / class_freq
        self.sample_weights = (inv_freq.unsqueeze(0) * labels).sum(dim=1)

    def __iter__(self):
        indices = torch.multinomial(self.sample_weights, self.samples_per_epoch, replacement=True)
        return iter(indices.tolist())

    def __len__(self):
        return self.samples_per_epoch


class FrequencySampler(nn.Module):
    def __init__(self, joint_dim: int, hidden_dim: int = 512, num_freqs: int = 64):
        super().__init__()
        self.num_freqs = num_freqs
        self.joint_dim = joint_dim
        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_dim * num_freqs),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        freqs = self.net(noise)
        freqs = freqs.view(noise.size(0), self.num_freqs, self.joint_dim)
        return freqs


def calculate_ncfd(real_joint: torch.Tensor, syn_joint: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    real_proj = real_joint @ freqs.t()
    syn_proj = syn_joint @ freqs.t()
    real_cf = torch.stack([torch.cos(real_proj), torch.sin(real_proj)], dim=-1).mean(dim=0)
    syn_cf = torch.stack([torch.cos(syn_proj), torch.sin(syn_proj)], dim=-1).mean(dim=0)
    diff = real_cf - syn_cf
    return diff.pow(2).sum()


class SimpleClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointNCFMSynthesizer(nn.Module):
    def __init__(
        self,
        init_embeds: torch.Tensor,
        init_labels: torch.Tensor,
        pretrained_classifier: nn.Module,
        num_freqs: int = 64,
    ):
        super().__init__()
        self.syn_embeds = nn.Parameter(init_embeds.clone())
        self.syn_labels = nn.Parameter(init_labels.clone())
        joint_dim = init_embeds.size(1) + init_labels.size(1)
        self.freq_sampler = FrequencySampler(joint_dim, num_freqs=num_freqs)
        self.pretrained_classifier = pretrained_classifier.eval()
        for p in self.pretrained_classifier.parameters():
            p.requires_grad = False

    def forward_step(
        self,
        real_embeds: torch.Tensor,
        real_labels: torch.Tensor,
        noise: torch.Tensor,
        alpha: float = ALPHA,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real_joint = get_joint_vector(real_embeds, real_labels, alpha=alpha)
        syn_joint = get_joint_vector(self.syn_embeds, self.syn_labels, alpha=alpha)
        freqs = self.freq_sampler(noise).mean(dim=0)
        ncfm_loss = calculate_ncfd(real_joint, syn_joint, freqs)
        preds = torch.sigmoid(self.pretrained_classifier(self.syn_embeds))
        consistency = F.mse_loss(preds, self.syn_labels)
        total_loss = ncfm_loss + 0.1 * consistency
        return total_loss, ncfm_loss


def collate_hf_batch(batch):
    batch_dict = {}
    keys = batch[0].keys()
    for key in keys:
        elems = [example[key] for example in batch]
        if isinstance(elems[0], torch.Tensor):
            batch_dict[key] = torch.stack(elems)
        else:
            batch_dict[key] = torch.tensor(elems)
    return batch_dict


def collect_all_embeddings(
    dataset: Dataset,
    feature_extractor: nn.Module,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_hf_batch)
    embeds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            features = feature_extractor.get_feature_last_layer(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            embeds.append(features.cpu())
            labels.append(batch["labels"].cpu())
    return torch.cat(embeds, dim=0), torch.cat(labels, dim=0)


def build_train_loader(
    dataset: Dataset,
    batch_size: int,
    sampler: Sampler,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_hf_batch)


def train_joint_ncfd(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    pretrained_classifier: nn.Module,
    init_embeds: torch.Tensor,
    init_labels: torch.Tensor,
    num_iters: int,
    lr: float,
    device: torch.device,
):
    feature_extractor = feature_extractor.to(device).eval()
    pretrained_classifier = pretrained_classifier.to(device).eval()
    synthesizer = JointNCFMSynthesizer(init_embeds.to(device), init_labels.to(device), pretrained_classifier).to(device)

    synth_opt = torch.optim.Adam([synthesizer.syn_embeds, synthesizer.syn_labels], lr=lr)
    freq_opt = torch.optim.Adam(synthesizer.freq_sampler.parameters(), lr=lr)

    joint_dim = synthesizer.syn_embeds.size(1) + synthesizer.syn_labels.size(1)
    iterator = iter(dataloader)
    for it in range(num_iters):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            real_embeds = feature_extractor.get_feature_last_layer(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
        real_labels = batch["labels"]

        noise = torch.randn(real_embeds.size(0), joint_dim, device=device)
        synth_opt.zero_grad()
        freq_opt.zero_grad()
        total_loss, ncfm_loss = synthesizer.forward_step(real_embeds, real_labels, noise)
        total_loss.backward()
        synth_opt.step()
        freq_opt.step()

        if (it + 1) % 50 == 0:
            print(f"[Iter {it+1}] total_loss: {total_loss.item():.4f}, ncfm: {ncfm_loss.item():.4f}")

    return synthesizer.syn_embeds.detach().cpu(), synthesizer.syn_labels.detach().cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="Joint NCFM condensation for Reuters text")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML configuration")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--num_syn", type=int, default=200, help="Number of synthetic samples")
    parser.add_argument("--num_iters", type=int, default=500, help="Condensation iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for real data")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for synthesizer")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save synthetic tensors")
    parser.add_argument("--classifier_ckpt", type=str, default=None, help="Optional classifier checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    cli_args = vars(args).copy()
    args_processor = ArgsProcessor(args.config_path)
    cfg = args_processor.add_args_from_yaml(argparse.Namespace())
    for k, v in cli_args.items():
        setattr(cfg, k, v)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _, _ = load_reuters_data(
        cfg.data_dir,
        cfg.model_path,
        max_length=getattr(cfg, "max_length", 512),
        text_column=getattr(cfg, "text_column", "sentence"),
        label_column=getattr(cfg, "label_column", "labels"),
    )
    num_classes = train_dataset.nclass
    labels_tensor = torch.stack([train_dataset[i]["labels"] for i in range(len(train_dataset))])
    sampler = BalancedClassSampler(labels_tensor, samples_per_epoch=len(train_dataset))
    train_loader = build_train_loader(train_dataset, cfg.batch_size, sampler)

    feature_extractor = define_language_model(cfg.model_path, cfg.net_type, num_classes)
    embed_dim = feature_extractor.bert.config.hidden_size
    classifier = SimpleClassifier(embed_dim, num_classes)
    if cfg.classifier_ckpt and os.path.isfile(cfg.classifier_ckpt):
        classifier.load_state_dict(torch.load(cfg.classifier_ckpt, map_location="cpu"))

    all_embeds, all_labels = collect_all_embeddings(
        train_dataset,
        feature_extractor,
        device,
        batch_size=cfg.batch_size,
    )
    init_embeds, init_labels = kmeans_initialize(all_embeds, all_labels, num_syn=cfg.num_syn)

    syn_embeds, syn_labels = train_joint_ncfd(
        feature_extractor=feature_extractor,
        dataloader=train_loader,
        pretrained_classifier=classifier,
        init_embeds=init_embeds,
        init_labels=init_labels,
        num_iters=cfg.num_iters,
        lr=cfg.lr,
        device=device,
    )

    print(f"Synthetic embeddings: {syn_embeds.shape}, Synthetic labels: {syn_labels.shape}")
    if cfg.save_path:
        torch.save((syn_embeds, syn_labels), cfg.save_path)
        print(f"Saved synthetic tensors to {cfg.save_path}")


if __name__ == "__main__":
    main()
