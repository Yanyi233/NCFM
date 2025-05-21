import os
import json
import numpy as np
import torch
import datetime
from .experiment_tracker import Logger
from .diffaug import remove_aug
import torch.distributed as dist
import datetime
from datetime import timedelta
from torch.backends import cudnn
from .ddp import initialize_distribution_training


def init_script(args):
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    rank, world_size, local_rank, local_world_size, device = (
        initialize_distribution_training(args.backend, args.init_method)
    )
    args.rank, args.world_size, args.local_rank, args.local_world_size, args.device = (
        rank,
        world_size,
        local_rank,
        local_world_size,
        device,
    )

    # 由 Rank 0 生成时间戳
    if args.rank == 0:
        timestamp_obj = torch.tensor(list(map(ord, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))), dtype=torch.int8, device=args.device)
    else:
        # 其他 rank 创建一个同样大小的 placeholder tensor
        timestamp_obj = torch.empty(15, dtype=torch.int8, device=args.device) # "%Y%m%d-%H%M%S" 是 15 个字符

    # 广播时间戳
    dist.broadcast(timestamp_obj, src=0)
    
    # 将 tensor 转回字符串
    timestamp_str = "".join([chr(x) for x in timestamp_obj.cpu().tolist()])
    args.timestamp_str = timestamp_str # 将时间戳字符串存储到 args 中，供后续函数使用

    args.it_save, args.it_log = set_iteration_parameters(args.niter, args.debug)

    # args.pretrain_dir = set_Pretrain_Directory(
    #     args.pretrain_dir, args.dataset, args.depth, args.ipc, args.net_type, args.timestamp_str
    # )

    args.exp_name, args.save_dir, args.lr_img = set_experiment_name_and_save_Dir(
        args.run_mode,
        args.dataset,
        args.pretrain_dir, # 如果 set_Pretrain_Directory 被调用，它会使用 args.timestamp_str
        args.save_dir,
        args.lr_img,
        args.lr_scale_adam,
        args.ipc,
        args.optimizer,
        args.load_path,
        args.factor,
        args.lr,
        args.num_freqs,
        args.timestamp_str # 传递时间戳字符串
    )

    set_random_seeds(args.seed)

    args.mixup, args.dsa_strategy, args.dsa, args.augment = (
        adjust_augmentation_strategy(args.mixup, args.dsa_strategy, args.dsa)
    )

    args.logger = setup_logging_and_directories(args, args.run_mode, args.save_dir)
    
    if args.rank == 0:
        args.logger("TF32 is enabled") if args.tf32 else args.logger("TF32 is disabled") # logger 在 rank 0 才打印
        args.logger(
            f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}"
        )


def set_iteration_parameters(niter, debug):

    it_save = np.arange(0, niter + 1, 200).tolist()
    it_log = 1 if debug else 20
    return it_save, it_log


def set_Pretrain_Directory(pretrain_dir, dataset, depth, ipc, net_type, timestamp_str): # 接收 timestamp_str
    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M") # 不再在这里生成
    if dataset.lower() == "imagenet":
        pretrain_dir = f"./{pretrain_dir}/{dataset}/ipc{ipc}/ResNet-{depth}_{timestamp_str}"
    else:
        pretrain_dir = f"./{pretrain_dir}/{dataset}/ipc{ipc}/{net_type}_{timestamp_str}"
    return pretrain_dir


def set_experiment_name_and_save_Dir(
    run_mode,
    dataset,
    pretrain_dir,
    save_dir,
    lr_img,
    lr_scale_adam,
    ipc,
    optimizer,
    load_path,
    factor,
    lr,
    num_freqs,
    timestamp_str # 接收 timestamp_str
):
    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M") # 不再在这里生成
    # Set the base save directory path according to the run_mode
    if run_mode == "Condense":
        assert ipc > 0, "IPC must be greater than 0"
        if optimizer.lower() == "sgd":
            lr_img = lr_img
        else:
            lr_img = lr_img * lr_scale_adam

        # Generate experiment name
        exp_name = f"./condense/{dataset}/ipc{ipc}/{optimizer}_lr_img_{lr_img:.4f}_numr_reqs{num_freqs}_factor{factor}_{timestamp_str}"
        if load_path:
            exp_name += f"Reload_SynData_Path_{load_path}"
        save_dir = os.path.join(save_dir, exp_name)

    elif run_mode == "Evaluation":
        assert ipc > 0, "IPC must be greater than 0"
        exp_name = (
            f"./evaluate/{dataset}/ipc{ipc}/_lr{lr:.4f}__factor{factor}_{timestamp_str}"
        )
        save_dir = os.path.join(save_dir, exp_name)
    elif run_mode == "Pretrain":
        save_dir = pretrain_dir # pretrain_dir 应该也已经包含了 timestamp_str
        exp_name = pretrain_dir
    else:
        raise ValueError(
            "Invalid run_mode. Choose 'Condense', 'Evaluation' or 'Pretrain'."
        )

    # Create save directory if the rank is 0
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier() # 确保 rank 0 创建完目录后其他进程再继续，避免日志等写入问题

    return exp_name, save_dir, lr_img


def set_random_seeds(seed):

    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        if dist.get_rank() == 0:
            print(f"Set Random Seed as {seed}")


def setup_logging_and_directories(args, run_mode, save_dir):
    # 确保所有进程都等待目录创建完成 (set_experiment_name_and_save_Dir 中已有 barrier)
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True) # 主保存目录在这里创建
        if run_mode == "Condense":
            subdirs = ["images", "distilled_data"] # "distilled_data" 在这里
            for subdir in subdirs:
                os.makedirs(os.path.join(save_dir, subdir), exist_ok=True) # 子目录创建
    
    # 使用修改后的参数序列化
    if dist.get_rank() == 0:
        args_log_path = os.path.join(save_dir, "args.log")
        serializable_args = {
            k: str(v) if isinstance(v, (Logger, torch.device)) else v 
            for k, v in vars(args).items()
        }
        with open(args_log_path, "w") as f:
            json.dump(serializable_args, f, indent=3)

    dist.barrier() 
    
    logger = Logger(save_dir) # 每个进程都创建自己的 Logger 实例，但都写入同一个目录
    
    dist.barrier()
    
    if dist.get_rank() == 0:
        logger(f"Save dir: {save_dir}")
    
    return logger


def adjust_augmentation_strategy(mixup, dsa_strategy, dsa):

    if mixup == "cut":
        dsa_strategy = remove_aug(dsa_strategy, "cutout")

    if dsa:
        augment = False
        if dist.get_rank() == 0:
            print(
                "DSA strategy: ",
                dsa_strategy,
            )
    else:
        augment = True
    return mixup, dsa_strategy, dsa, augment
