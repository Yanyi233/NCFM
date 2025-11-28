import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Tuple, Callable, Optional


ALPHA = 10.0


def get_joint_vector(embeds: torch.Tensor, labels: torch.Tensor, alpha: float = ALPHA) -> torch.Tensor:
    """Concatenate embeddings and scaled labels to form joint vectors."""
    if embeds.shape[0] != labels.shape[0]:
        raise ValueError("embeds and labels must share the batch dimension")
    weighted_labels = labels * alpha
    return torch.cat([embeds, weighted_labels], dim=1)


def kmeans_initialize(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    num_syn: int,
    num_iters: int = 25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple k-means on label space to obtain initial synthetic samples."""
    device = labels.device
    num_samples = labels.shape[0]
    if num_syn > num_samples:
        raise ValueError("num_syn cannot exceed available samples")

    # randomly pick seeds
    perm = torch.randperm(num_samples, device=device)
    centers = labels[perm[:num_syn]].clone()
    center_embeds = embeds[perm[:num_syn]].clone()

    for _ in range(num_iters):
        # compute L2 distance in label space
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
        self.sample_weights = (1.0 / class_freq).matmul(labels.T)

    def __iter__(self):
        idx = torch.multinomial(self.sample_weights, self.samples_per_epoch, replacement=True)
        return iter(idx.tolist())

    def __len__(self):
        return self.samples_per_epoch


class FrequencySampler(nn.Module):
    """Learns frequency vectors for the joint space."""

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
        freqs = freqs.view(noise.shape[0], self.num_freqs, self.joint_dim)
        return freqs


def calculate_ncfd(real_joint: torch.Tensor, syn_joint: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Approximate characteristic function distance."""
    # freqs: [num_freqs, joint_dim]
    real_proj = real_joint @ freqs.t()
    syn_proj = syn_joint @ freqs.t()
    real_cf = torch.stack([torch.cos(real_proj), torch.sin(real_proj)], dim=-1).mean(dim=0)
    syn_cf = torch.stack([torch.cos(syn_proj), torch.sin(syn_proj)], dim=-1).mean(dim=0)
    diff = real_cf - syn_cf
    return diff.pow(2).sum()


class JointNCFMSynthesizer(nn.Module):
    """Joint feature-label NCFM synthesizer."""

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
        joint_dim = init_embeds.shape[1] + init_labels.shape[1]
        self.freq_sampler = FrequencySampler(joint_dim=joint_dim, num_freqs=num_freqs)
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
        freqs = self.freq_sampler(noise).mean(dim=0)  # [num_freqs, joint_dim]
        ncfm_loss = calculate_ncfd(real_joint, syn_joint, freqs)
        preds = torch.sigmoid(self.pretrained_classifier(self.syn_embeds))
        consistency_loss = F.mse_loss(preds, self.syn_labels)
        total_loss = ncfm_loss + 0.1 * consistency_loss
        return total_loss, ncfm_loss


class SyntheticDemoDataset(Dataset):
    """Random multi-label dataset to exercise the pipeline."""

    def __init__(self, num_samples: int, input_dim: int, num_classes: int, tail_pow: float = 2.0):
        self.inputs = torch.randn(num_samples, input_dim)
        # create class probabilities with long-tail distribution
        class_probs = torch.arange(1, num_classes + 1, dtype=torch.float)
        class_probs = (1.0 / class_probs.pow(tail_pow))
        class_probs = class_probs / class_probs.sum()
        labels = torch.bernoulli(class_probs.unsqueeze(0).repeat(num_samples, 1))
        # ensure at least one class per sample
        empty_rows = labels.sum(dim=1) == 0
        while empty_rows.any():
            labels[empty_rows] = torch.bernoulli(class_probs.unsqueeze(0).repeat(empty_rows.sum(), 1))
            empty_rows = labels.sum(dim=1) == 0
        self.labels = labels

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


class SimpleFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def train_joint_ncfd(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    pretrained_classifier: nn.Module,
    num_syn: int = 500,
    num_iters: int = 2000,
    lr: float = 1e-2,
    device: Optional[torch.device] = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device).eval()
    pretrained_classifier = pretrained_classifier.to(device).eval()

    # gather features for initialization
    all_embeds, all_labels = [], []
    with torch.no_grad():
        for real_x, real_y in dataloader:
            real_x = real_x.to(device)
            real_y = real_y.to(device)
            embeds = feature_extractor(real_x)
            all_embeds.append(embeds)
            all_labels.append(real_y)
    all_embeds = torch.cat(all_embeds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    init_embeds, init_labels = kmeans_initialize(all_embeds, all_labels, num_syn=num_syn)

    synthesizer = JointNCFMSynthesizer(init_embeds, init_labels, pretrained_classifier).to(device)
    synth_optimizer = torch.optim.Adam([synthesizer.syn_embeds, synthesizer.syn_labels], lr=lr)
    freq_optimizer = torch.optim.Adam(synthesizer.freq_sampler.parameters(), lr=lr)

    dataloader_iter = iter(dataloader)
    joint_dim = synthesizer.syn_embeds.size(1) + synthesizer.syn_labels.size(1)
    for it in range(num_iters):
        try:
            real_x, real_y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            real_x, real_y = next(dataloader_iter)

        real_x = real_x.to(device)
        real_y = real_y.to(device)
        real_embeds = feature_extractor(real_x)

        noise = torch.randn(real_embeds.size(0), joint_dim, device=device)

        synth_optimizer.zero_grad()
        freq_optimizer.zero_grad()
        total_loss, ncfm_loss = synthesizer.forward_step(real_embeds, real_y, noise)
        total_loss.backward()
        synth_optimizer.step()
        freq_optimizer.step()

        if (it + 1) % 50 == 0:
            print(f"[Iter {it+1}] total loss: {total_loss.item():.4f}, ncfm: {ncfm_loss.item():.4f}")

    return synthesizer.syn_embeds.detach(), synthesizer.syn_labels.detach()


def main():
    parser = argparse.ArgumentParser(description="Joint NCFM demo")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of real samples to simulate")
    parser.add_argument("--num_classes", type=int, default=90, help="Number of classes for multi-label data")
    parser.add_argument("--input_dim", type=int, default=1024, help="Input dimension before feature extractor")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--num_syn", type=int, default=200, help="Synthetic samples")
    parser.add_argument("--num_iters", type=int, default=500, help="Training iterations")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    args = parser.parse_args()

    dataset = SyntheticDemoDataset(args.num_samples, args.input_dim, args.num_classes)
    sampler = BalancedClassSampler(dataset.labels, samples_per_epoch=len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    feature_extractor = SimpleFeatureExtractor(args.input_dim, args.embed_dim)
    classifier = SimpleClassifier(args.embed_dim, args.num_classes)

    syn_embeds, syn_labels = train_joint_ncfd(
        feature_extractor=feature_extractor,
        dataloader=dataloader,
        pretrained_classifier=classifier,
        num_syn=args.num_syn,
        num_iters=args.num_iters,
        lr=args.lr,
    )

    print(f"Synthetic embeddings shape: {syn_embeds.shape}")
    print(f"Synthetic labels shape: {syn_labels.shape}")


if __name__ == "__main__":
    main()
