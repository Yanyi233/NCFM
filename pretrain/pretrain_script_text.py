import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch.optim as optim
from utils.utils import define_model
from utils.utils import get_loader
from utils.train_val import train_epoch, validate
from utils.diffaug import diffaug

def check_args(args):
    # 检查args中是否有define_model使用到的参数，如果没有则设置缺省值
    if not hasattr(args, 'dataset'):
        args.dataset = 'reuters'  # 默认数据集
        if args.rank == 0:
            print(f"Warning: 'dataset' not found in args, using default: {args.dataset}")
    
    if not hasattr(args, 'norm_type'):
        args.norm_type = 'layernorm'  # 默认归一化类型
        if args.rank == 0:
            print(f"Warning: 'norm_type' not found in args, using default: {args.norm_type}")
    
    if not hasattr(args, 'net_type'):
        args.net_type = 'BERT'  # 默认网络类型
        if args.rank == 0:
            print(f"Warning: 'net_type' not found in args, using default: {args.net_type}")
    
    if not hasattr(args, 'depth'):
        args.depth = 12  # 默认网络深度
        if args.rank == 0:
            print(f"Warning: 'depth' not found in args, using default: {args.depth}")
    
    if not hasattr(args, 'width'):
        args.width = 1.0  # 默认网络宽度
        if args.rank == 0:
            print(f"Warning: 'width' not found in args, using default: {args.width}")
    
    if not hasattr(args, 'nclass'):
        args.nclass = 10  # 默认类别数
        if args.rank == 0:
            print(f"Warning: 'nclass' not found in args, using default: {args.nclass}")
    
    if not hasattr(args, 'size'):
        args.size = 32  # 默认图像大小
        if args.rank == 0:
            print(f"Warning: 'size' not found in args, using default: {args.size}")
    
    if not hasattr(args, 'is_multilabel'):
        args.is_multilabel = False  # 默认为单标签分类
        if args.rank == 0:
            print(f"Warning: 'is_multilabel' not found in args, using default: {args.is_multilabel}")
    return args

def get_available_model_id(pretrain_dir, model_id):
    while True:
        init_path = os.path.join(pretrain_dir, f"premodel{model_id}_init.pth.tar")
        trained_path = os.path.join(pretrain_dir, f"premodel{model_id}_trained.pth.tar")
        # Check if both files do not exist, if both are missing, return the current model_id
        if not os.path.exists(init_path) and not os.path.exists(trained_path):
            return model_id  # Return the first available model_id
        model_id += 1  # If files exist, try the next model_id


def count_existing_models(pretrain_dir):
    """
    Count the number of initial model files (premodel{model_id}_init.pth.tar)
    that exist in pretrain_dir.
    """
    model_count = 0
    for filename in os.listdir(pretrain_dir):
        if filename.startswith("premodel") and filename.endswith("_init.pth.tar"):
            model_count += 1  # Increment count if the file matches the criteria

    return model_count  # Return the count of matching files


def main_worker(args):
    train_loader, val_loader, train_sampler = get_loader(args)
    for model_id in range(args.model_num):
        if count_existing_models(args.pretrain_dir) >= args.model_num:
            break
        model_id = get_available_model_id(args.pretrain_dir, model_id)
        if args.rank == 0:
            print(f"Training model {model_id + 1}/{args.model_num}")
        args = check_args(args) ## 检查args中是否有define_model使用到的参数，如果没有则设置缺省值
        model = define_model(
            args.dataset,
            args.norm_type,
            args.net_type,
            args.nch,
            args.depth,
            args.width,
            args.nclass,
            args.logger,
            args.size,
        ).to(args.device)
        model = model.to(args.device)
        model = DDP(model, device_ids=[args.rank])

        # Save initial model state
        init_path = os.path.join(args.pretrain_dir, f"premodel{model_id}_init.pth.tar")
        if args.rank == 0 and not os.path.exists(init_path):
            torch.save(model.state_dict(), init_path)
            print(f"Model {model_id} initial state saved at {init_path}")

        # Define loss function, optimizer, and scheduler
        if args.is_multilabel:
            criterion = torch.nn.BCEWithLogitsLoss().to(args.device)
            # 定义用于追踪最佳模型的指标 (例如 mAP 或 F1-micro)
            best_metric_key = 'mAP' # 或者 'f1_micro'
            best_metric = 0.0       # 初始化为 0 (越高越好)
            print(f"Using multi-label criterion (BCEWithLogitsLoss) and tracking best '{best_metric_key}'.")
        else:
            criterion = torch.nn.CrossEntropyLoss().to(args.device)
            # 定义用于追踪最佳模型的指标 (例如 Top-1 Accuracy)
            best_metric_key = 'top1'
            best_metric = 0.0       # 初始化为 0 (越高越好)
            print(f"Using single-label criterion (CrossEntropyLoss) and tracking best '{best_metric_key}'.")

        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        ) ## TODO:若改成BERT，可考虑修改成Adam等
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[2 * args.pertrain_epochs // 3, 5 * args.pertrain_epochs // 6], ## 在epoch 2/3和5/6处进行学习率衰减
            gamma=0.2,
        )
        _, aug_rand = diffaug(args)
        for epoch in range(0, args.pertrain_epochs):
            start_time = time.time()
            train_sampler.set_epoch(epoch)
            train_metrics = train_epoch(
                args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                aug_rand,
                mixup=args.mixup,
            )
            train_loss = train_metrics['loss']

            # 打印训练指标
            if args.rank == 0:
                print(f"Epoch [{epoch+1}/{args.pertrain_epochs}] Training Metrics:")
                log_str = f"  Loss: {train_loss:.4f}"
                if args.is_multilabel:
                    log_str += f" | Prec (micro): {train_metrics.get('prec_micro', 0.0):.4f}"
                    log_str += f" | Recall (micro): {train_metrics.get('rec_micro', 0.0):.4f}"
                    log_str += f" | F1 (micro): {train_metrics.get('f1_micro', 0.0):.4f}"
                    log_str += f" | Hamming: {train_metrics.get('hamming', 0.0):.4f}"
                else:
                    log_str += f" | Top-1 Acc: {train_metrics.get('top1', 0.0):.2f}%"
                    log_str += f" | Top-5 Acc: {train_metrics.get('top5', 0.0):.2f}%"
                print(log_str)

            val_metrics = validate(args, val_loader, model, criterion)
            val_loss = val_metrics['loss']

            # 打印验证指标
            if args.rank == 0:
                print(f"Epoch [{epoch+1}/{args.pertrain_epochs}] Validation Metrics:")
                log_str = f"  Loss: {val_loss:.4f}"
                current_metric = 0.0 # 初始化当前轮次的最佳指标值
                if args.is_multilabel:
                    log_str += f" | mAP: {val_metrics.get('mAP', 0.0):.4f}"
                    log_str += f" | F1 (micro): {val_metrics.get('f1_micro', 0.0):.4f}"
                    log_str += f" | F1 (macro): {val_metrics.get('f1_macro', 0.0):.4f}"
                    log_str += f" | Hamming: {val_metrics.get('hamming_loss', 0.0):.4f}"
                    current_metric = val_metrics.get(best_metric_key, 0.0) # 获取用于比较的指标
                else:
                    log_str += f" | Top-1 Acc: {val_metrics.get('top1', 0.0):.2f}%"
                    log_str += f" | Top-5 Acc: {val_metrics.get('top5', 0.0):.2f}%"
                    current_metric = val_metrics.get(best_metric_key, 0.0) # 获取用于比较的指标
                print(log_str)

            epoch_time = time.time() - start_time
            if args.rank == 0:
                if args.is_multilabel:
                    args.logger(
                        "<Pretraining {:2d}-th model>...[Epoch {:2d}] Train P: {:.4f} R: {:.4f} F1: {:.4f} Hamming: {:.4f} mAP: {:.4f}, Val P: {:.4f} R: {:.4f} F1: {:.4f} Hamming: {:.4f} mAP: {:.4f}, Time: {:.2f} seconds".format(
                            model_id, epoch, 
                            train_metrics.get('prec_micro', 0.0), train_metrics.get('rec_micro', 0.0), 
                            train_metrics.get('f1_micro', 0.0), train_metrics.get('hamming', 0.0), train_metrics.get('mAP', 0.0),
                            val_metrics.get('prec_micro', 0.0), val_metrics.get('rec_micro', 0.0), 
                            val_metrics.get('f1_micro', 0.0), val_metrics.get('hamming', 0.0), val_metrics.get('mAP', 0.0),
                            epoch_time
                        )
                    )
                else:
                    args.logger(
                        "<Pretraining {:2d}-th model>...[Epoch {:2d}] Train acc: {:.1f} (loss: {:.3f}), Val acc: {:.1f}, Time: {:.2f} seconds".format(
                            model_id, epoch, train_metrics.get('top1', 0.0), train_loss, val_metrics.get('top1', 0.0), epoch_time
                        )
                    )
            scheduler.step()

        # Save trained model state
        trained_path = os.path.join(
            args.pretrain_dir, f"premodel{model_id}_trained.pth.tar"
        )
        if args.rank == 0:
            torch.save(model.state_dict(), trained_path)
            print(f"Model {model_id} trained state saved at {trained_path}")

    dist.destroy_process_group()


def main():
    import os
    from utils.init_script import init_script
    import argparse
    from argsprocessor.args import ArgsProcessor

    parser = argparse.ArgumentParser(description="Configuration parser")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="When dataset is very large , you should get it",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["Condense", "Evaluation", "Pretrain"],
        default="Pretrain",
        help="Condense or Evaluation",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        required=True,
        help='GPUs to use, e.g., "0,1,2,3"',
    )
    parser.add_argument(
        "-i", "--ipc", type=int, default=1, help="number of condensed data per class"
    )
    parser.add_argument("--load_path", type=str, help="Path to load the synset")
    parser.add_argument("--tf32", action="store_true", default=True, help="Enable TF32")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    init_script(args)

    main_worker(args)


if __name__ == "__main__":
    main()
