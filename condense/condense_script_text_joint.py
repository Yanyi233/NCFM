import argparse
import math
import numpy as np
import os
import sys
import socket
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, hamming_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from argsprocessor.args import ArgsProcessor
from utils.init_script import init_script
from utils.utils_text import load_reuters_data, define_language_model
from utils.ddp import load_state_dict
import torch.distributed as dist
import time


ALPHA = 10.0


def check_args(args):
    if not hasattr(args, "dataset"):
        args.dataset = "reuters"
    if not hasattr(args, "norm_type"):
        args.norm_type = "layernorm"
    if not hasattr(args, "net_type"):
        args.net_type = "BERT"
    if not hasattr(args, "depth"):
        args.depth = 12
    if not hasattr(args, "nclass"):
        args.nclass = 90
    if not hasattr(args, "max_length"):
        args.max_length = 512
    if not hasattr(args, "is_multilabel"):
        args.is_multilabel = True
    if not hasattr(args, "pretrain_dir"):
        args.pretrain_dir = "../pretrained_models"
    if not hasattr(args, "save_dir"):
        args.save_dir = "../results/condense"
    if not hasattr(args, "backend"):
        args.backend = "gloo"
    if not hasattr(args, "init_method"):
        args.init_method = "env://"
    if not hasattr(args, "lr_img"):
        args.lr_img = 0.01
    if not hasattr(args, "lr_scale_adam"):
        args.lr_scale_adam = 0.1
    if not hasattr(args, "optimizer"):
        args.optimizer = "adamw"
    if not hasattr(args, "factor"):
        args.factor = 2
    if not hasattr(args, "num_freqs"):
        args.num_freqs = 4096
    if not hasattr(args, "lr"):
        args.lr = 1e-3
    if not hasattr(args, "joint_lr"):
        args.joint_lr = 1e-2
    if not hasattr(args, "synth_lr") or args.synth_lr is None:
        args.synth_lr = args.joint_lr * 0.25
    if not hasattr(args, "freq_lr") or args.freq_lr is None:
        args.freq_lr = max(args.joint_lr * 0.1, 5e-5)
    if not hasattr(args, "kmeans_iters"):
        args.kmeans_iters = 25
    if not hasattr(args, "joint_init_path"):
        args.joint_init_path = None
    if not hasattr(args, "save_joint_init"):
        args.save_joint_init = None
    if not hasattr(args, "workers"):
        args.workers = 8
    if not hasattr(args, "text_column"):
        args.text_column = "sentence"
    if not hasattr(args, "label_column"):
        args.label_column = "labels"
    if not hasattr(args, "freq_noise_samples"):
        args.freq_noise_samples = 256
    if not hasattr(args, "grad_clip"):
        args.grad_clip = 1.0
    if not hasattr(args, "it_log"):
        args.it_log = 20
    if not hasattr(args, "eval_after_train"):
        args.eval_after_train = False
    if not hasattr(args, "eval_before_train"):
        args.eval_before_train = False
    return args


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        elems = [example[key] for example in batch]
        if isinstance(elems[0], torch.Tensor):
            out[key] = torch.stack(elems)
        else:
            out[key] = torch.tensor(elems)
    return out


class BalancedClassSampler(Sampler[int]):
    def __init__(self, labels: torch.Tensor, samples_per_epoch: int):
        self.labels = labels
        self.samples_per_epoch = samples_per_epoch
        class_freq = labels.sum(dim=0) + 1e-6
        inv_freq = 1.0 / class_freq
        self.sample_weights = (inv_freq.unsqueeze(0) * labels).sum(dim=1)

    def __iter__(self):
        idx = torch.multinomial(self.sample_weights, self.samples_per_epoch, replacement=True)
        return iter(idx.tolist())

    def __len__(self):
        return self.samples_per_epoch


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total += param_norm * param_norm
    return math.sqrt(total)


def get_joint_vector(embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    weighted_labels = labels * ALPHA
    return torch.cat([embeds, weighted_labels], dim=1)


def kmeans_initialize(
    embeds: torch.Tensor,
    labels: torch.Tensor,
    num_syn: int,
    num_iters: int = 25,
    logger=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = embeds.device
    num_samples = embeds.size(0)
    if num_syn > num_samples:
        raise ValueError("num_syn cannot exceed available samples")

    perm = torch.randperm(num_samples, device=device)
    centers = labels[perm[:num_syn]].clone()
    center_embeds = embeds[perm[:num_syn]].clone()

    for it in range(num_iters):
        dists = torch.cdist(labels, centers, p=2)
        assignments = torch.argmin(dists, dim=1)
        for c in range(num_syn):
            mask = assignments == c
            if mask.sum() == 0:
                continue
            centers[c] = labels[mask].mean(dim=0)
            center_embeds[c] = embeds[mask].mean(dim=0)
        # msg = f"[KMeans] iteration {it+1}/{num_iters} finished"
        # if logger:
        #     logger(msg)
        # else:
        #     print(msg)

    return center_embeds, centers


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


class JointNCFMSynthesizer(nn.Module):
    def __init__(
        self,
        init_embeds: torch.Tensor,
        init_labels: torch.Tensor,
        classifier_head: nn.Module,
        num_freqs: int = 64,
    ):
        super().__init__()
        self.syn_embeds = nn.Parameter(init_embeds.clone())
        clamped_labels = init_labels.clamp(1e-4, 1 - 1e-4)
        self.syn_label_logits = nn.Parameter(torch.logit(clamped_labels))
        joint_dim = init_embeds.size(1) + init_labels.size(1)
        self.freq_sampler = FrequencySampler(joint_dim, num_freqs=num_freqs)
        self.classifier_head = classifier_head.eval()
        for p in self.classifier_head.parameters():
            p.requires_grad = False

    def forward_step(self, real_embeds, real_labels, noise):
        real_joint = get_joint_vector(real_embeds, real_labels)
        syn_labels = torch.sigmoid(self.syn_label_logits)
        syn_joint = get_joint_vector(self.syn_embeds, syn_labels)
        freqs = self.freq_sampler(noise).mean(dim=0)
        ncfm_loss = calculate_ncfd(real_joint, syn_joint, freqs)
        preds = torch.sigmoid(self.classifier_head(self.syn_embeds))
        consistency = F.mse_loss(preds, syn_labels)
        total = ncfm_loss + 0.1 * consistency
        return total, ncfm_loss, consistency

    @property
    def syn_labels(self):
        return torch.sigmoid(self.syn_label_logits)


def collect_embeddings(dataset, model, device, batch_size, num_workers=0):
    wrapper = HFDatasetWrapper(dataset)
    loader = DataLoader(
        wrapper,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    embeds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            features = model.get_feature_last_layer(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            embeds.append(features.cpu())
            labels.append(batch["labels"].cpu())
    return torch.cat(embeds, dim=0), torch.cat(labels, dim=0)


def load_teacher_components(args):
    feature_model = define_language_model(args.model_path, args.net_type, args.nclass).to(args.device)
    ckpt_path = getattr(args, "teacher_ckpt", None)
    num_premodel = getattr(args, "num_premodel", 1)
    if ckpt_path is None:
        for idx in range(num_premodel):
            candidate = os.path.join(args.pretrain_dir, f"premodel{idx}_trained.pth.tar")
            if os.path.isfile(candidate):
                ckpt_path = candidate
                break
    if ckpt_path is None:
        raise FileNotFoundError("No pretrained classifier checkpoint found in pretrain_dir.")
    load_state_dict(ckpt_path, feature_model)
    feature_model.eval()
    for p in feature_model.parameters():
        p.requires_grad = False

    classifier_head = nn.Linear(
        feature_model.classifier.in_features,
        feature_model.classifier.out_features,
    ).to(args.device)
    classifier_head.load_state_dict(feature_model.classifier.state_dict())
    classifier_head.eval()
    for p in classifier_head.parameters():
        p.requires_grad = False
    return feature_model, classifier_head


def train_joint_pipeline(args):
    train_dataset, val_dataset, _ = load_reuters_data(
        args.data_dir,
        args.model_path,
        max_length=args.max_length,
        text_column=getattr(args, "text_column", "sentence"),
        label_column=getattr(args, "label_column", "labels"),
    )

    # load经过pre-train的model和分类头
    feature_model, classifier_head = load_teacher_components(args)

    args.logger(f"Caching {len(train_dataset)} Reuters embeddings for efficient training")
    start_time = time.time()
    all_embeds, all_labels = collect_embeddings(
        train_dataset,
        feature_model,
        args.device,
        args.batch_real,
        num_workers=args.workers,
    )
    elapsed = time.time() - start_time
    args.logger(f"Caching done (elapsed {elapsed:.2f}s)")

    def log_sample_labels(tag, label_tensor):
        if args.rank != 0 or not getattr(args, "debug", False):
            return
        preview = label_tensor[:3].detach().cpu().tolist()
        args.logger(f"[{tag}] sample labels (first 3): {preview}")

    use_cached_init = getattr(args, "joint_init_path", None)
    if use_cached_init and os.path.exists(use_cached_init):
        args.logger(f"Loading joint initialization from {use_cached_init}")
        cache = torch.load(use_cached_init, map_location="cpu")
        init_embeds, init_labels = cache["embeds"], cache["labels"]
    else:
        args.logger("Running K-Means initialization")
        start_time = time.time()
        init_embeds, init_labels = kmeans_initialize(
            all_embeds,
            all_labels,
            num_syn=args.num_syn,
            num_iters=args.kmeans_iters,
        )
        elapsed = time.time() - start_time
        args.logger(f"K-Means initialization done (elapsed {elapsed:.2f}s)")
        if args.rank == 0 and getattr(args, "save_joint_init", None):
            torch.save({"embeds": init_embeds, "labels": init_labels}, args.save_joint_init)
            args.logger(f"Saved joint initialization to {args.save_joint_init}")

    log_sample_labels("pre-train", init_labels)

    if val_dataset is not None:
        val_embeds, val_labels = collect_embeddings(
            val_dataset,
            feature_model,
            args.device,
            args.batch_real,
            num_workers=args.workers,
        )
        val_loader = DataLoader(TensorDataset(val_embeds, val_labels), batch_size=args.batch_real, shuffle=False)
    else:
        val_loader = None

    cached_dataset = TensorDataset(all_embeds, all_labels)
    sampler = BalancedClassSampler(all_labels, samples_per_epoch=len(cached_dataset))
    loader = DataLoader(
        cached_dataset,
        batch_size=args.batch_real,
        sampler=sampler,
        drop_last=False,
    )

    synthesizer = JointNCFMSynthesizer(
        init_embeds.to(args.device),
        init_labels.to(args.device),
        classifier_head,
    ).to(args.device)
    synth_opt = torch.optim.Adam([synthesizer.syn_embeds, synthesizer.syn_label_logits], lr=args.synth_lr)
    freq_opt = torch.optim.Adam(synthesizer.freq_sampler.parameters(), lr=args.freq_lr)
    joint_dim = synthesizer.syn_embeds.size(1) + synthesizer.syn_labels.size(1)

    evaluate_synthetic_embeddings(
        args,
        init_embeds,
        init_labels,
        val_loader,
        run_eval=args.eval_before_train,
        stage="pre-train",
    )

    iterator = iter(loader)
    progress = tqdm(range(args.num_iters), disable=args.rank != 0, dynamic_ncols=True)
    for it in progress:
        try:
            batch_embeds, batch_labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_embeds, batch_labels = next(iterator)
        real_embeds = batch_embeds.to(args.device)
        real_labels = batch_labels.to(args.device)
        noise = torch.randn(args.freq_noise_samples, joint_dim, device=args.device)

        synth_opt.zero_grad()
        freq_opt.zero_grad()
        total_loss, ncfm_loss, consistency = synthesizer.forward_step(real_embeds, real_labels, noise)
        total_loss.backward()

        synth_params = [synthesizer.syn_embeds, synthesizer.syn_label_logits]
        freq_params = list(synthesizer.freq_sampler.parameters())
        if args.grad_clip and args.grad_clip > 0:
            synth_grad_norm = torch.nn.utils.clip_grad_norm_(synth_params, args.grad_clip).item()
            freq_grad_norm = torch.nn.utils.clip_grad_norm_(freq_params, args.grad_clip).item()
        else:
            synth_grad_norm = grad_norm(synth_params)
            freq_grad_norm = grad_norm(freq_params)

        synth_opt.step()
        freq_opt.step()

        if args.rank == 0:
            progress.set_description(
                f"Loss {total_loss.item():.4f} | NCFM {ncfm_loss.item():.4f} | Cons {consistency.item():.4f}"
            )
        if (it + 1) % args.it_log == 0 and args.rank == 0:
            args.logger(
                f"[Iter {it+1}/{args.num_iters}] total: {total_loss.item():.4f} ncfm: {ncfm_loss.item():.4f} "
                f"cons: {consistency.item():.4f} grad_syn: {synth_grad_norm:.3f} grad_freq: {freq_grad_norm:.3f}"
            )

    syn_embeds = synthesizer.syn_embeds.detach().cpu()
    syn_labels = synthesizer.syn_labels.detach().cpu()
    save_path = os.path.join(args.save_dir, "joint_syn_dataset.pt")
    if args.rank == 0:
        torch.save((syn_embeds, syn_labels), save_path)
        args.logger(f"Saved synthetic tensors to {save_path}")
        log_sample_labels("post-train", syn_labels)

    return syn_embeds, syn_labels, val_loader


def evaluate_classifier(classifier, loader, criterion, args, threshold=0.5):
    classifier.eval()
    total_loss = 0.0
    total_samples = 0
    prob_chunks, target_chunks = [], []
    with torch.no_grad():
        for embeds, labels in loader:
            embeds = embeds.to(args.device)
            labels = labels.to(args.device)
            loss_target = labels.float() if args.is_multilabel else torch.argmax(labels, dim=1)
            logits = classifier(embeds)
            loss = criterion(logits, loss_target)
            total_loss += loss.item() * embeds.size(0)
            total_samples += embeds.size(0)

            if args.is_multilabel:
                prob_chunks.append(torch.sigmoid(logits).cpu().numpy())
                target_chunks.append(labels.cpu().numpy())
            else:
                prob_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
                target_chunks.append(torch.argmax(labels, dim=1).cpu().numpy())

    metrics = {"loss": total_loss / max(total_samples, 1)}
    if not prob_chunks:
        return metrics

    probs = np.concatenate(prob_chunks, axis=0)
    targets = np.concatenate(target_chunks, axis=0)
    if args.is_multilabel:
        preds = (probs > threshold).astype(int)
        metrics.update(
            {
                "prec_micro": precision_score(targets, preds, average="micro", zero_division=0),
                "rec_micro": recall_score(targets, preds, average="micro", zero_division=0),
                "f1_micro": f1_score(targets, preds, average="micro", zero_division=0),
                "hamming": hamming_loss(targets, preds),
            }
        )
        ap_scores = []
        for cls_idx in range(targets.shape[1]):
            if targets[:, cls_idx].sum() > 0:
                ap_scores.append(average_precision_score(targets[:, cls_idx], probs[:, cls_idx]))
        metrics["mAP"] = float(np.mean(ap_scores)) if ap_scores else 0.0
    else:
        preds = np.argmax(probs, axis=1)
        metrics["top1"] = float((preds == targets).mean())
    return metrics


def evaluate_synthetic_embeddings(args, syn_embeds, syn_labels, val_loader, run_eval=False, stage="post-train"):
    if not run_eval:
        return
    if val_loader is None:
        args.logger(f"Skipping {stage} evaluation because validation data is unavailable")
        return

    args.logger(f"Running {stage} evaluation on cached embeddings")
    syn_dataset = TensorDataset(syn_embeds, syn_labels)
    syn_loader = DataLoader(syn_dataset, batch_size=args.batch_real, shuffle=True)

    embed_dim = syn_embeds.size(1)
    hidden_dim = max(embed_dim // 2, 256)
    classifier = nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, syn_labels.size(1)),
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=getattr(args, "adamw_lr", args.synth_lr),
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    criterion = nn.BCEWithLogitsLoss() if args.is_multilabel else nn.CrossEntropyLoss()
    eval_epochs = getattr(args, "evaluation_epochs", 50)
    eval_interval = max(1, getattr(args, "epoch_eval_interval", 5))
    pred_threshold = getattr(args, "pred_threshold", 0.5)

    best_score = -float("inf")
    best_snapshot = None
    progress = tqdm(range(1, eval_epochs + 1), disable=args.rank != 0, dynamic_ncols=True)
    for epoch in progress:
        classifier.train()
        total_loss = 0.0
        total_samples = 0
        for embeds, labels in syn_loader:
            embeds = embeds.to(args.device)
            labels = labels.to(args.device)
            loss_target = labels.float() if args.is_multilabel else torch.argmax(labels, dim=1)
            logits = classifier(embeds)
            loss = criterion(logits, loss_target)
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item() * embeds.size(0)
            total_samples += embeds.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        if args.rank == 0:
            progress.set_description(f"EvalEp {epoch}/{eval_epochs} TrainLoss {avg_loss:.4f}")

        if epoch % eval_interval == 0 or epoch == eval_epochs:
            metrics = evaluate_classifier(classifier, val_loader, criterion, args, threshold=pred_threshold)
            score = metrics.get("mAP", metrics.get("top1", 0.0))
            if score > best_score:
                best_score = score
                best_snapshot = {"epoch": epoch, **metrics}
            if args.rank == 0:
                metric_log = " ".join([f"{k}:{v:.4f}" for k, v in metrics.items() if isinstance(v, float)])
                args.logger(
                    f"[Eval Epoch {epoch}/{eval_epochs}] train_loss: {avg_loss:.4f} val_loss: {metrics['loss']:.4f} {metric_log}"
                )

    if best_snapshot and args.rank == 0:
        summary = " ".join([f"{k}:{v:.4f}" for k, v in best_snapshot.items() if k != "epoch" and isinstance(v, float)])
        args.logger(f"Best {stage} evaluation epoch {best_snapshot['epoch']}: {summary}")


def build_parser():
    parser = argparse.ArgumentParser(description="Joint feature-label condensation")
    parser.add_argument('--debug', dest='debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--run_mode', type=str, choices=['Condense', 'Evaluation', 'Pretrain'], default='Condense', help='Run mode')
    parser.add_argument('-a', '--aug_type', type=str, default='color_crop_cutout', help='augmentation strategy')
    parser.add_argument('--init', type=str, default='mix', choices=['random', 'noise', 'mix', 'load'], help='condensed data initialization type')
    parser.add_argument('--load_path', type=str, default=None, help="Path to load the synset")
    parser.add_argument('--gpu', type=str, default="0", required=True, help='GPUs to use')
    parser.add_argument('-i', '--ipc', type=int, default=10, help='number of condensed data per class')
    parser.add_argument('--tf32', action='store_true', default=True, help='Enable TF32')
    parser.add_argument('--sampling_net', action='store_true', default=False, help='Enable sampling net')
    parser.add_argument('--num_syn', type=int, default=200, help='Number of synthetic samples')
    parser.add_argument('--num_iters', type=int, default=500, help='Training iterations')
    parser.add_argument('--joint_lr', type=float, default=1e-2, help='Learning rate for joint condensation')
    parser.add_argument('--synth_lr', type=float, default=None, help='Learning rate for synthetic tensors (defaults to joint_lr)')
    parser.add_argument('--freq_lr', type=float, default=None, help='Learning rate for frequency sampler (defaults to 0.25 * joint_lr)')
    parser.add_argument('--kmeans_iters', type=int, default=25, help='Iterations for k-means initialization')
    parser.add_argument('--joint_init_path', type=str, default=None, help='Path to cached joint initialization')
    parser.add_argument('--save_joint_init', type=str, default=None, help='Path to save joint initialization cache')
    parser.add_argument('--batch_real', type=int, default=64, help='Real batch size')
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='Optional teacher checkpoint override')
    parser.add_argument('--freq_noise_samples', type=int, default=64, help='Number of noise draws for frequency sampler')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value (<=0 disables)')
    parser.add_argument('--eval_after_train', action='store_true', default=True, help='Run evaluation after condensation')
    parser.add_argument('--eval_before_train', action='store_true', default=True, help='Evaluate initialization before optimization')
    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    cli_overrides = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if dest and dest not in ("help",):
            val = getattr(cli_args, dest, None)
            default = parser.get_default(dest)
            if val != default:
                cli_overrides[dest] = val

    args_processor = ArgsProcessor(cli_args.config_path)
    args = args_processor.add_args_from_yaml(argparse.Namespace())
    for key, value in cli_overrides.items():
        setattr(args, key, value)

    # Ensure parser defaults are still applied when the config file omits them.
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if dest and dest not in ("help",) and not hasattr(args, dest):
            setattr(args, dest, getattr(cli_args, dest))

    args = check_args(args)
    if not hasattr(args, "tf32"):
        args.tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    if not hasattr(args, "gpu"):
        args.gpu = "0"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            free_port = s.getsockname()[1]
        os.environ["MASTER_PORT"] = str(free_port)
    if not hasattr(args, "run_mode"):
        args.run_mode = "Condense"
    if not hasattr(args, "num_iters"):
        args.num_iters = getattr(args, "niter", 400)
    args.niter = args.num_iters
    if not hasattr(args, "num_premodel"):
        args.num_premodel = 1
    if not hasattr(args, "batch_real"):
        args.batch_real = 64
    if not hasattr(args, "debug"):
        args.debug = False
    if not hasattr(args, "load_path"):
        args.load_path = None
    if not hasattr(args, "backend"):
        args.backend = "gloo"
    if not hasattr(args, "init_method"):
        args.init_method = "env://"
    init_script(args)
    if args.world_size != 1:
        raise RuntimeError("Joint pipeline currently supports single GPU execution for now.")

    # Print all arguments for debugging
    if args.rank == 0:
        args.logger("=" * 80)
        args.logger("Configuration Parameters:")
        args.logger("=" * 80)
        sorted_args = sorted([(k, v) for k, v in vars(args).items() if not k.startswith('_')])
        for key, value in sorted_args:
            # Truncate very long values for readability
            if isinstance(value, str) and len(value) > 100:
                display_value = value[:100] + "..."
            elif isinstance(value, (list, tuple)) and len(value) > 10:
                display_value = str(value[:10]) + f"... (length={len(value)})"
            else:
                display_value = value
            args.logger(f"  {key:30s} = {display_value}")
        args.logger("=" * 80)

    args.logger("Starting joint condensation pipeline")
    syn_embeds, syn_labels, val_loader = train_joint_pipeline(args)
    evaluate_synthetic_embeddings(
        args,
        syn_embeds,
        syn_labels,
        val_loader,
        run_eval=args.eval_after_train,
        stage="post-train",
    )
    dist.destroy_process_group()
