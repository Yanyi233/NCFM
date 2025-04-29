import os
import torch
import random
import torch.distributed as dist
import torch.functional as F
import numpy as np
import contextlib
from utils.ddp import load_state_dict
from utils.experiment_tracker import LossPlotter
from data.dataloader import (
    ClassDataLoader,
    ClassMemDataLoader,
    MultiLabelClassDataLoader,
    MultiLabelClassMemDataLoader,
    AsyncLoader,
    ImageNetMemoryDataLoader,
)
from torch.utils.data import DataLoader, DistributedSampler
import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN
import models.resnet18_pretrain as RN18_PRETRAIN
# from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from data.transform import transform_imagenet
from data.dataset import ImageFolder, ImageFolder_mtt
from data.dataset_statistics import MEANS, STDS
from data.voc2007_dataset import VOC2007_PT_Dataset

import torchvision.models


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (
            np.max(child.stride) > 1 and child.in_channels >= 16
        ):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def define_model(dataset, norm_type, net_type, nch, depth, width, nclass, logger, size):
    ## TODO:看是否需要添加新的模型类型，如BERT等
    if net_type == 'resnet18':
        model = RN18_PRETRAIN.ResNet18WithFeatures(num_classes=nclass, pretrained=True)
    elif net_type == "resnet":
        model = RN.ResNet(
            dataset, depth, nclass, norm_type=norm_type, size=size, nch=nch
        )
    elif net_type == "resnet_ap":
        model = RNAP.ResNetAP(
            dataset, depth, nclass, width=width, norm_type=norm_type, size=size, nch=nch
        )
        apply_blurpool(model)
    # elif net_type == "efficient":
    #     model = EfficientNet.from_name("efficientnet-b0", num_classes=nclass)
    elif net_type == "densenet":
        model = DN.densenet_cifar(nclass)
    elif net_type == "convnet":
        width = int(128 * width)
        model = CN.ConvNet(
            nclass,
            net_norm=norm_type,
            net_depth=depth,
            net_width=width,
            channel=nch,
            im_size=(size, size),
        )
    else:
        raise Exception("unknown network architecture: {}".format(net_type))

    # if logger is not None:
    #     if dist.get_rank() == 0:
    #         logger(f"=> creating model {net_type}-{depth}, norm: {norm_type}")
    #         logger('# model parameters: {:.1f}M'.format(sum([p.data.nelement() for p in model.parameters()]) / 10**6))
    return model


def load_resized_data(
    dataset, data_dir, size=None, nclass=None, load_memory=False, seed=0
):

    normalize = transforms.Normalize(mean=MEANS[dataset], std=STDS[dataset])
    # Initialize datasets to None
    train_dataset, val_dataset = None, None
    is_multilabel = False # 初始化多标签标志

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        if dataset == "cifar10":
            train_dataset = datasets.CIFAR10(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.CIFAR10(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "cifar100":
            train_dataset = datasets.CIFAR100(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.CIFAR100(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 100

        elif dataset == "svhn":
            train_dataset = datasets.SVHN(
                os.path.join(data_dir, "SVHN"),
                download=True,
                split="train",
                transform=transforms.ToTensor(),
            )
            train_dataset.targets = train_dataset.labels
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.SVHN(
                os.path.join(data_dir, "SVHN"), split="test", transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "mnist":
            train_dataset = datasets.MNIST(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.MNIST(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "fashion":
            train_dataset = datasets.FashionMNIST(
                data_dir, download=True, train=True, transform=transforms.ToTensor()
            )
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            val_dataset = datasets.FashionMNIST(
                data_dir, train=False, transform=transform_test
            )
            train_dataset.nclass = 10

        elif dataset == "tinyimagenet":
            data_path = os.path.join(data_dir, "tinyimagenet")
            transform_test = (
                transforms.Compose([transforms.ToTensor(), normalize])
                if normalize
                else transforms.ToTensor()
            )
            train_dataset = datasets.ImageFolder(
                os.path.join(data_path, "train"), transform=transforms.ToTensor()
            )
            val_dataset = datasets.ImageFolder(
                os.path.join(data_path, "val"), transform=transform_test
            )
            train_dataset.nclass = 200

        elif dataset in [
            "imagenette",
            "imagewoof",
            "imagemeow",
            "imagesquawk",
            "imagefruit",
            "imageyellow",
        ]:
            traindir = os.path.join(data_dir, "train")
            valdir = os.path.join(data_dir, "val")
            resize = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.PILToTensor(),
                ]
            )
            if load_memory:
                transform = None
                load_transform = resize
            else:
                transform = transforms.Compose(
                    [resize, transforms.ConvertImageDtype(torch.float)]
                )
                load_transform = None

            _, test_transform = transform_imagenet(size=size)
            train_dataset = ImageFolder_mtt(
                traindir,
                transform=transform,
                type=dataset,
                load_memory=load_memory,
                load_transform=load_transform,
            )
            val_dataset = ImageFolder_mtt(
                valdir, test_transform, type=dataset, load_memory=False
            )

        elif dataset == "imagenet":
            traindir = os.path.join(data_dir, "train")
            valdir = os.path.join(data_dir, "val")
            resize = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.PILToTensor(),
                ]
            )
            if load_memory:
                transform = None
                load_transform = resize
            else:
                transform = transforms.Compose(
                    [resize, transforms.ConvertImageDtype(torch.float)]
                )
                load_transform = None

            _, test_transform = transform_imagenet(size=size)
            train_dataset = ImageFolder(
                traindir,
                transform=transform,
                nclass=nclass,
                seed=seed,
                load_memory=load_memory,
                load_transform=load_transform,
            )
            val_dataset = ImageFolder(
                valdir, test_transform, nclass=nclass, seed=seed, load_memory=False
            )

        elif dataset == "voc2007":
            # 仅加载数据集对象，不创建 DataLoader
            # 确保 data_dir 指向 VOC2007.pt 文件或包含该文件的目录
            pt_file_path = data_dir
            if not os.path.isfile(pt_file_path):
                 default_pt_name = "VOC2007/VOC2007.pt"
                 pt_file_path = os.path.join(data_dir, default_pt_name)
                 if not os.path.isfile(pt_file_path):
                     raise FileNotFoundError(f"VOC2007 .pt file not found at {data_dir} or {pt_file_path}")

            # 注意: .pt 文件通常已包含 Tensor，transform 可能不需要或需要特殊处理
            # 假设 .pt 文件已处理好，transform 设为 None
            transform_train = None
            transform_test = None

            train_dataset = VOC2007_PT_Dataset(
                pt_file_path=pt_file_path,
                train=True,
                transform=transform_train
            )
            val_dataset = VOC2007_PT_Dataset(
                pt_file_path=pt_file_path,
                train=False,
                transform=transform_test
            )
            # 设置多标签标志
            is_multilabel = True
            # 动态获取类别数和图像大小 (将在 get_loader 中赋给 args)
            # train_dataset.nclass = train_dataset.get_num_classes()
            # train_dataset.img_size = train_dataset.images.shape[-1] # 假设 H=W

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # 移动断言到函数末尾，确保 train_dataset 和 val_dataset 已被赋值
        if train_dataset is not None and val_dataset is not None:
             # 对于 VOC2007，我们从同一个文件加载，尺寸必然匹配
             if dataset != "voc2007":
                 # 原始检查，可能需要调整以适应不同数据集的返回结构
                 # 例如，检查 train_dataset.data[0].shape 或 train_dataset[0][0].shape
                 # 这里假设它们都有一个可索引的结构返回图像张量
                 try:
                     train_img_shape = train_dataset[0][0].shape
                     val_img_shape = val_dataset[0][0].shape
                     assert train_img_shape[-1] == val_img_shape[-1], \
                         f"Train ({train_img_shape}) and Val ({val_img_shape}) dataset sizes do not match"
                 except Exception as e:
                     print(f"Warning: Could not verify train/val image size consistency for {dataset}. Error: {e}")
        else:
             print(f"Warning: train_dataset or val_dataset is None for dataset {dataset}. Skipping size check.")


    # 返回数据集对象和多标签标志
    return train_dataset, val_dataset, is_multilabel


def get_plotter(args):
    base_filename = f"{args.dataset}_ipc{args.ipc}_factor{args.factor}_{args.optimizer}_alpha{args.alpha_for_loss}_beta{args.beta_for_loss}_dis{args.dis_metrics}_freqs{args.num_freqs}_calib{args.iter_calib}"
    optimizer_info = {
        "type": args.optimizer,
        "lr": (
            args.lr_img * args.lr_scale_adam
            if args.optimizer.lower() in ["adam", "adamw"]
            else args.lr_img
        ),
        "weight_decay": args.weight_decay if args.optimizer.lower() == "adamw" else 0.0,
    }

    plotter = LossPlotter(
        save_path=args.save_dir,
        filename_pattern=base_filename,
        dataset=args.dataset,
        ipc=args.ipc,
        dis_metrics=args.dis_metrics,
        optimizer_info=optimizer_info,
    )
    return plotter


def get_optimizer(optimizer: str= "sgd", parameters=None,lr=0.01, mom_img=0.5,weight_decay=5e-4,logger=None):
    if optimizer.lower() == "sgd":
        optim_img = torch.optim.SGD(parameters, lr=lr, momentum=mom_img)
        if logger and dist.get_rank() == 0:
            logger(f"Using SGD optimizer with learning rate: {lr}")
    elif optimizer.lower() == "adam":
        optim_img = torch.optim.Adam(parameters, lr=lr)
        if logger and dist.get_rank() == 0:
            logger(f"Using Adam optimizer with learning rate: {lr}")
    elif optimizer.lower() == "adamw":
        optim_img = torch.optim.AdamW(
            parameters, lr=lr, weight_decay=weight_decay
        )
        if logger and dist.get_rank() == 0:
            logger(f"Using AdamW optimizer with learning rate: {lr}")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer.lower()}")
    return optim_img


def get_loader(args):
    # 首先加载数据集对象
    train_dataset, val_dataset, is_multilabel = load_resized_data(
        args.dataset,
        args.data_dir,
        size=args.size,
        nclass=args.nclass,
        load_memory=args.load_memory,
        seed=getattr(args, 'seed', 0) # 确保 seed 存在
    )

    # 将多标签标志添加到 args
    args.is_multilabel = is_multilabel

    # 如果是 VOC2007，动态更新 args 中的 nclass 和 img_size
    if args.dataset == "voc2007" and train_dataset is not None:
        args.nclass = train_dataset.get_num_classes()
        # 假设图像是方形的，取最后一个维度作为大小
        args.size = train_dataset.images.shape[-1]
        args.img_size = (args.size, args.size) # 更新 img_size 元组
        if args.rank == 0: # 只在主进程打印
             print(f"Dynamically set nclass={args.nclass}, img_size={args.img_size} from VOC2007 .pt file")


    # 根据 run_mode 创建 DataLoader
    if args.run_mode == "Condense":
        if args.dataset == "imagenet":
            # ... (ImageNet 特殊处理逻辑) ...
            loader_real = ImageNetMemoryDataLoader(
                args.imagenet_prepath, class_list=args.class_list
            )
            dist.barrier()
            _ = None # val_loader 在 Condense 模式下通常不需要
        elif args.dataset == "voc2007":
            if args.load_memory:
                loader_real = MultiLabelClassMemDataLoader(train_dataset, batch_size=args.batch_real)
            else:
                # 使用 ClassDataLoader 进行类别采样
                loader_real = MultiLabelClassDataLoader(
                    train_dataset,
                    batch_size=args.batch_real,
                    num_workers=args.workers,
                    shuffle=True, # ClassDataLoader 内部处理 shuffle
                    pin_memory=True,
                    drop_last=True,
                )
            _ = None # val_loader 在 Condense 模式下通常不需要
        else:
            # 处理其他单标签数据集 (CIFAR, etc.)
            if args.load_memory:
                loader_real = ClassMemDataLoader(train_dataset, batch_size=args.batch_real)
            else:
                # 使用 ClassDataLoader 进行类别采样
                loader_real = ClassDataLoader(
                    train_dataset,
                    batch_size=args.batch_real,
                    num_workers=args.workers,
                    shuffle=True, # ClassDataLoader 内部处理 shuffle
                    pin_memory=True,
                    drop_last=True,
                )
            _ = None # val_loader 在 Condense 模式下通常不需要

        return loader_real, _

    elif args.run_mode == "Evaluation":
        # 评估模式总是需要验证集加载器
        if val_dataset is None:
             raise ValueError("Validation dataset could not be loaded for Evaluation mode.")

        val_sampler = DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False # 验证时不 shuffle
        )
        val_loader = DataLoader(
            val_dataset,
            # batch_size=int(args.batch_size / args.world_size), # 使用 args.batch_size
            batch_size=int(args.batch_size / args.world_size), # 使用测试批次大小
            sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False # 验证时通常不 drop_last
        )
        # 评估模式通常不需要训练加载器，返回 None
        return None, val_loader

    elif args.run_mode == "Pretrain":
        # 预训练模式需要训练集和验证集加载器
        if train_dataset is None or val_dataset is None:
             raise ValueError("Train or Validation dataset could not be loaded for Pretrain mode.")

        # 验证集加载器
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False
        )
        val_loader = DataLoader(
            val_dataset,
            # batch_size=int(args.batch_size / args.world_size), # 使用 args.batch_size
            batch_size=int(args.batch_size / args.world_size), # 使用测试批次大小
            sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False
        )

        # 训练集加载器
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size / args.world_size), # 使用训练批次大小
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True # 预训练时通常 drop_last
        )
        return train_loader, val_loader, train_sampler

    else:
        # 如果有其他 run_mode 或者逻辑错误
        raise ValueError(f"Unknown run_mode: {args.run_mode}")


def get_feature_extractor(args):
    model_init = define_model(
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
    model_final = define_model(
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
    model_interval = define_model(
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
    return model_init, model_interval, model_final


def update_feature_extractor(args, model_init, model_final, model_interval, a=0, b=1):
    if args.num_premodel > 0:
        # Select pre-trained model ID
        slkt_model_id = random.randint(0, args.num_premodel - 1)

        # Construct the paths
        init_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_init.pth.tar"
        )
        final_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
        )
        # Load the pre-trained models
        load_state_dict(init_path, model_init)
        load_state_dict(final_path, model_final)
        l = (b - a) * torch.rand(1).to(args.device) + a
        # Interpolate to initialize `model_interval`
        for model_interval_param, model_init_param, model_final_param in zip(
            model_interval.parameters(),
            model_init.parameters(),
            model_final.parameters(),
        ):
            model_interval_param.data.copy_(
                l * model_init_param.data + (1 - l) * model_final_param.data
            )

    else:
        if args.iter_calib > 0:
            slkt_model_id = random.randint(0, 9)
            final_path = os.path.join(
                args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
            )
            load_state_dict(final_path, model_final)
        # model_interval = define_model(args.dataset, args.norm_type, args.net_type, args.nch, args.depth, args.width, args.nclass, args.logger, args.size).to(args.device)
        slkt_model_id = random.randint(0, 9)
        interval_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
        )
        load_state_dict(interval_path, model_interval)

    return model_init, model_final, model_interval
