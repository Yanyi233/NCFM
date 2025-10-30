import time
import torch
import numpy as np
from utils.experiment_tracker import AverageMeter, accuracy
from utils.mix_cut_up import random_indices, rand_bbox
from utils.ddp import sync_distributed_metric
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, average_precision_score


def sync_distributed_metric_dict(metrics_dict, device):
    """ Helper to sync a dictionary of scalar metrics across DDP processes. """
    if not dist.is_available() or not dist.is_initialized():
        return metrics_dict

    world_size = dist.get_world_size()
    if world_size == 1:
        return metrics_dict

    # 确保所有进程都有相同的键顺序
    metric_keys = sorted(metrics_dict.keys())
    metric_values = [metrics_dict[k] for k in metric_keys]

    # Convert dict values to tensor for reduction
    tensor = torch.tensor(metric_values, dtype=torch.float64, device=device)

    # Average across processes
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

    # Convert back to dict
    synced_metrics = {k: tensor[i].item() for i, k in enumerate(metric_keys)}
    return synced_metrics


def train_epoch(
    args,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    aug=None,
    mixup="cut",
    scheduler=None,
    grad_clip_value=None,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if args.is_multilabel:
        prec_micro = AverageMeter()
        rec_micro = AverageMeter()
        f1_micro = AverageMeter()
        hamming = AverageMeter()
        mAP = AverageMeter()
        all_targets_list = []
        all_outputs_list = []
    else:
        top1 = AverageMeter()
        top5 = AverageMeter()

    model.train()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_ids_arg = None
        attention_mask_arg = None
        token_type_ids_arg = None
        embeddings_arg = None
        target_labels = None
        current_batch_size = 0

        if isinstance(batch, dict):
            # Real data (dictionary format)
            input_ids_arg = batch['input_ids'].cuda(non_blocking=True)
            attention_mask_arg = batch['attention_mask'].cuda(non_blocking=True)
            token_type_ids_arg = batch.get('token_type_ids', None)
            if token_type_ids_arg is not None:
                token_type_ids_arg = token_type_ids_arg.cuda(non_blocking=True)
            target_labels = batch['labels'].cuda(non_blocking=True)
            current_batch_size = input_ids_arg.size(0)
        else:
            # Synthetic data (tuple: embeddings, targets)
            data_input, target_labels = batch
            embeddings_arg = data_input.cuda(non_blocking=True)
            target_labels = target_labels.cuda(non_blocking=True)
            current_batch_size = embeddings_arg.size(0)
            # For embeddings, attention_mask and token_type_ids are often managed by the model if None
            if embeddings_arg.ndim != 3:
                raise ValueError(f"Expected 3D embeddings for synthetic data, got shape {embeddings_arg.shape}")

        if args.is_multilabel:
            target_float = target_labels.float()
        else:
            target_float = target_labels

        # Augmentation (if applicable and input_ids_arg is present)
        if aug is not None and input_ids_arg is not None:
            with torch.no_grad():
                input_ids_arg = aug(input_ids_arg)
        
        # Model forward pass
        # BERT.py's forward method handles `embedding` or `input_ids`
        output = model(
            input_ids=input_ids_arg,
            attention_mask=attention_mask_arg,
            token_type_ids=token_type_ids_arg,
            embedding=embeddings_arg
        )
        loss = criterion(output, target_float)

        if args.is_multilabel:
            with torch.no_grad():
                scores = torch.sigmoid(output.detach())
                # Ensure prediction threshold is appropriate, e.g. 0.5 for general cases
                # Using args.pred_threshold if available, otherwise a common default like 0.5
                pred_threshold = getattr(args, 'pred_threshold', 0.5)
                preds = (scores > pred_threshold).cpu().numpy().astype(int)
                targets_np = target_labels.cpu().numpy().astype(int)

                batch_prec_micro = precision_score(targets_np, preds, average='micro', zero_division=0)
                batch_rec_micro = recall_score(targets_np, preds, average='micro', zero_division=0)
                batch_f1_micro = f1_score(targets_np, preds, average='micro', zero_division=0)
                batch_hamming = hamming_loss(targets_np, preds)

                prec_micro.update(batch_prec_micro, current_batch_size)
                rec_micro.update(batch_rec_micro, current_batch_size)
                f1_micro.update(batch_f1_micro, current_batch_size)
                hamming.update(batch_hamming, current_batch_size)

                all_outputs_list.append(output.cpu())
                all_targets_list.append(target_labels.cpu())
        else:
            acc1, acc5 = accuracy(output.data, target_labels, topk=(1, 5))
            top1.update(acc1.item(), current_batch_size)
            top5.update(acc5.item(), current_batch_size)

        losses.update(loss.item(), current_batch_size)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

    metrics = {'loss': losses.avg}
    if args.is_multilabel:
        metrics['prec_micro'] = prec_micro.avg
        metrics['rec_micro'] = rec_micro.avg
        metrics['f1_micro'] = f1_micro.avg
        metrics['hamming'] = hamming.avg

        if all_targets_list and all_outputs_list: # Ensure lists are not empty
            all_outputs = torch.cat([out.detach().cpu() for out in all_outputs_list], dim=0)
            all_targets = torch.cat([t.detach().cpu() for t in all_targets_list], dim=0)
            scores = torch.sigmoid(all_outputs).numpy()
            targets_np = all_targets.numpy().astype(int)
            
            try:
                # Filter out classes with no true samples for AP calculation
                ap_per_class = []
                for i in range(targets_np.shape[1]):
                    if np.sum(targets_np[:, i]) > 0:
                        ap_per_class.append(average_precision_score(targets_np[:, i], scores[:, i]))
                
                mAP_value = np.mean([ap for ap in ap_per_class if not np.isnan(ap)]) if ap_per_class else 0.0
            except ValueError as e:
                print(f"Warning: Could not compute mAP during training. Error: {e}")
                mAP_value = 0.0
                
            metrics['mAP'] = mAP_value
        else:
            metrics['mAP'] = 0.0 # Default if no data to compute
    else:
        metrics['top1'] = top1.avg
        metrics['top5'] = top5.avg

    synced_metrics = sync_distributed_metric_dict(metrics, args.device)
    return synced_metrics


def get_softlabel(img, teacher_model, target=None):
    # Get the soft labels
    softlabel = teacher_model(img).detach()  # [n, class]

    # If target is None, directly return the soft labels
    if target is None:
        return softlabel

    # Get the predicted class for each sample in the soft labels
    predicted = torch.argmax(softlabel, dim=1)  # [n]

    # Find the indices of misclassified samples
    incorrect_indices = predicted != target  # [n], True indicates misclassified samples

    # Replace the misclassified parts with the correct labels
    # Initialize the soft labels to all zeros
    corrected_softlabel = softlabel.clone()
    corrected_softlabel[incorrect_indices] = (
        0  # Set all class probabilities to 0 for misclassified samples
    )
    corrected_softlabel[incorrect_indices, target[incorrect_indices]] = (
        1  # Set the correct class probability to 1
    )

    return corrected_softlabel


def train_epoch_softlabel(
    args,
    train_loader,
    model,
    teacher_model,
    criterion,
    optimizer,
    epoch,
    aug=None,
    mixup="cut",
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if args.is_multilabel:
        prec_micro = AverageMeter()
        rec_micro = AverageMeter()
        f1_micro = AverageMeter()
        hamming = AverageMeter()
    else:
        top1 = AverageMeter()
        top5 = AverageMeter()

    model.train()
    teacher_model.eval()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if args.is_multilabel:
            target_float = target.float()
        else:
            target_float = target

        data_time.update(time.time() - end)

        with torch.no_grad():
            soft_label_logits = teacher_model(input).detach()
            if isinstance(criterion, torch.nn.KLDivLoss):
                soft_label_prob = F.softmax(soft_label_logits / args.temperature, dim=1)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                soft_label = soft_label_logits
            else:
                soft_label_prob = torch.sigmoid(soft_label_logits) if args.is_multilabel else F.softmax(soft_label_logits / args.temperature, dim=1)
                soft_label = soft_label_prob

        if aug is not None:
            with torch.no_grad():
                input = aug(input)
        r = np.random.rand(1)
        if r < args.mix_p and mixup == "cut":
            if args.is_multilabel:
                # print("Warning: Mixup/Cutmix for multi-label soft labels is not fully implemented/verified. Using standard loss.")
                output = model(input)
                if isinstance(criterion, torch.nn.KLDivLoss):
                    loss = criterion(F.log_softmax(output / args.temperature, dim=1), soft_label_prob)
                elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                    loss = criterion(output, soft_label)
                else:
                    loss = criterion(output, soft_label)
            else:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = random_indices(target, nclass=args.nclass)
                soft_label_prob_b = soft_label_prob[rand_index, :]

                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[
                    rand_index, :, bbx1:bbx2, bby1:bby2
                ]
                ratio = 1 - (
                    (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
                )
                output = model(input)
                if isinstance(criterion, torch.nn.KLDivLoss):
                    log_probs = F.log_softmax(output / args.temperature, dim=1)
                    loss = criterion(log_probs, soft_label_prob) * ratio + criterion(log_probs, soft_label_prob_b) * (1.0 - ratio)
                else:
                    loss = criterion(output, soft_label)
        else:
            output = model(input)
            if isinstance(criterion, torch.nn.KLDivLoss):
                loss = criterion(F.log_softmax(output / args.temperature, dim=1), soft_label_prob)
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                loss = criterion(output, soft_label)
            else:
                loss = criterion(output, soft_label)

        if args.is_multilabel:
            with torch.no_grad():
                scores = torch.sigmoid(output.data)
                preds = (scores > 0.5).cpu().numpy().astype(int)
                targets_np = target.cpu().numpy().astype(int)

                batch_prec_micro = precision_score(targets_np, preds, average='micro', zero_division=0)
                batch_rec_micro = recall_score(targets_np, preds, average='micro', zero_division=0)
                batch_f1_micro = f1_score(targets_np, preds, average='micro', zero_division=0)
                batch_hamming = hamming_loss(targets_np, preds)

                prec_micro.update(batch_prec_micro, input.size(0))
                rec_micro.update(batch_rec_micro, input.size(0))
                f1_micro.update(batch_f1_micro, input.size(0))
                hamming.update(batch_hamming, input.size(0))
        else:
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    metrics = {'loss': losses.avg}
    if args.is_multilabel:
        metrics['prec_micro'] = prec_micro.avg
        metrics['rec_micro'] = rec_micro.avg
        metrics['f1_micro'] = f1_micro.avg
        metrics['hamming'] = hamming.avg
    else:
        metrics['top1'] = top1.avg
        metrics['top5'] = top5.avg

    synced_metrics = sync_distributed_metric_dict(metrics, args.device)
    return synced_metrics


def validate(args, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # Metrics initialization (moved from inside if/else for clarity)
    all_targets_list = []
    all_outputs_list = []
    
    # For single-label, top1/top5 will be populated if not args.is_multilabel
    top1 = AverageMeter() 
    top5 = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids_arg = None
            attention_mask_arg = None
            token_type_ids_arg = None
            embeddings_arg = None
            target_labels = None
            current_batch_size = 0

            if isinstance(batch, dict): # Real data (dictionary format)
                input_ids_arg = batch['input_ids'].to(args.device, non_blocking=True)
                attention_mask_arg = batch['attention_mask'].to(args.device, non_blocking=True)
                token_type_ids_arg = batch.get('token_type_ids', None)
                if token_type_ids_arg is not None:
                    token_type_ids_arg = token_type_ids_arg.to(args.device, non_blocking=True)
                target_labels = batch['labels'].to(args.device, non_blocking=True)
                current_batch_size = input_ids_arg.size(0)
            else: # Synthetic data or other tuple format (data_input, target_labels)
                data_input, target_labels = batch
                data_input_cuda = data_input.to(args.device, non_blocking=True)
                target_labels = target_labels.to(args.device, non_blocking=True)
                current_batch_size = data_input_cuda.size(0)

                if data_input_cuda.ndim == 3: # Synthetic Embeddings
                    embeddings_arg = data_input_cuda
                elif data_input_cuda.ndim == 2: # Fallback for tokenized data in tuple
                    input_ids_arg = data_input_cuda
                else:
                    raise ValueError(f"Unexpected data_input shape in validate: {data_input_cuda.shape}")
                # attention_mask and token_type_ids are assumed to be handled by model for embeddings

            target_float = target_labels.float() if args.is_multilabel else target_labels

            output = model(
                input_ids=input_ids_arg,
                attention_mask=attention_mask_arg,
                token_type_ids=token_type_ids_arg,
                embedding=embeddings_arg
            )
            loss = criterion(output, target_float)

            losses.update(loss.item(), current_batch_size)

            if args.is_multilabel:
                all_outputs_list.append(output.cpu())
                all_targets_list.append(target_labels.cpu())
            else:
                acc1, acc5 = accuracy(output.data, target_labels, topk=(1, 5))
                top1.update(acc1.item(), current_batch_size)
                top5.update(acc5.item(), current_batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

    metrics = {'loss': losses.avg}

    if args.is_multilabel:
        if not all_targets_list or not all_outputs_list: # Check if lists are empty
            print("Warning: Validation set yielded no data for metric calculation.")
            metrics.update({
                'prec_micro': 0.0, 'rec_micro': 0.0, 'f1_micro': 0.0,
                'prec_macro': 0.0, 'rec_macro': 0.0, 'f1_macro': 0.0,
                'hamming': 1.0, 'mAP': 0.0 # Default to worst/neutral values
            })
        else:
            all_outputs_tensor = torch.cat(all_outputs_list, dim=0)
            all_targets_tensor = torch.cat(all_targets_list, dim=0)

            scores = torch.sigmoid(all_outputs_tensor).numpy()
            # Ensure prediction threshold is appropriate, e.g. 0.5 for general cases
            pred_threshold = getattr(args, 'pred_threshold', 0.5)
            preds = (scores > pred_threshold).astype(int)
            targets_np = all_targets_tensor.numpy().astype(int)

            prec_micro = precision_score(targets_np, preds, average='micro', zero_division=0)
            rec_micro = recall_score(targets_np, preds, average='micro', zero_division=0)
            f1_micro = f1_score(targets_np, preds, average='micro', zero_division=0)
            prec_macro = precision_score(targets_np, preds, average='macro', zero_division=0)
            rec_macro = recall_score(targets_np, preds, average='macro', zero_division=0)
            f1_macro = f1_score(targets_np, preds, average='macro', zero_division=0)
            h_loss = hamming_loss(targets_np, preds)

            try:
                # Filter out classes with no true samples for AP calculation
                ap_per_class = []
                for i in range(targets_np.shape[1]):
                    if np.sum(targets_np[:, i]) > 0: # Only calculate AP for classes with positive labels
                         ap_per_class.append(average_precision_score(targets_np[:, i], scores[:, i]))
                
                mAP = np.mean([ap for ap in ap_per_class if not np.isnan(ap)]) if ap_per_class else 0.0
            except ValueError as e:
                print(f"Warning: Could not compute mAP during validation. Error: {e}")
                mAP = 0.0

            metrics.update({
                'prec_micro': prec_micro, 'rec_micro': rec_micro, 'f1_micro': f1_micro,
                'prec_macro': prec_macro, 'rec_macro': rec_macro, 'f1_macro': f1_macro,
                'hamming': h_loss, 'mAP': mAP
            })
    else:
        metrics['top1'] = top1.avg
        metrics['top5'] = top5.avg

    synced_metrics = sync_distributed_metric_dict(metrics, args.device)

    return synced_metrics
