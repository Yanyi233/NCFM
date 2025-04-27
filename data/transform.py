import torchvision.transforms as transforms
from data.dataset_statistics import MEANS, STDS
from data.augment import ColorJitter, Lighting


def transform_cifar(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        print("Dataset with basic Cifar augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["cifar"], std=STDS["cifar"])]
    else:
        normal_fn = []
    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_svhn(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(32, padding=4)]
        print("Dataset with basic SVHN augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["svhn"], std=STDS["svhn"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_mnist(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(28, padding=4)]
        print("Dataset with basic MNIST augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["mnist"], std=STDS["mnist"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_fashion(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(28, padding=4)]
        print("Dataset with basic FashionMNIST augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["fashion"], std=STDS["fashion"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_tiny(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip()]
        print("Dataset with basic Cifar augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [
            transforms.Normalize(mean=MEANS["tinyimagenet"], std=STDS["tinyimagenet"])
        ]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_imagenet(
    size=-1,
    augment=False,
    from_tensor=False,
    normalize=True,
    rrc=True,
    rrc_size=-1,
    device="cpu",
):
    if size > 0:
        resize_train = [transforms.Resize(size), transforms.CenterCrop(size)]
        resize_test = [transforms.Resize(size), transforms.CenterCrop(size)]
        # print(f"Resize and crop training images to {size}")
    elif size == 0:
        resize_train = []
        resize_test = []
        assert rrc_size > 0, "Set RRC size!"
    else:
        resize_train = [transforms.RandomResizedCrop(224)]
        resize_test = [transforms.Resize(256), transforms.CenterCrop(224)]

    if not augment:
        aug = []
        # print("Loader with DSA augmentation")
    else:
        jittering = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = Lighting(
            alphastd=0.1,
            eigval=[0.2175, 0.0188, 0.0045],
            eigvec=[
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ],
            device=device,
        )
        aug = [transforms.RandomHorizontalFlip(), jittering, lighting]

        if rrc and size >= 0:
            if rrc_size == -1:
                rrc_size = size
            rrc_fn = transforms.RandomResizedCrop(rrc_size, scale=(0.5, 1.0))
            aug = [rrc_fn] + aug
            print("Dataset with basic imagenet augmentation and RRC")
        else:
            print("Dataset with basic imagenet augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS["imagenet"], std=STDS["imagenet"])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(resize_train + cast + aug + normal_fn)
    test_transform = transforms.Compose(resize_test + cast + normal_fn)

    return train_transform, test_transform


def transform_voc2007(augment=False, from_tensor=False, normalize=True, size=224):
    """
    Defines transforms for VOC2007 dataset.
    Assumes MEANS["voc2007"] and STDS["voc2007"] are defined in dataset_statistics.
    """
    # --- 训练集转换 ---
    train_aug = []
    if augment:
        # 为 VOC 添加常见的增强：随机调整大小裁剪和水平翻转
        train_aug = [
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)), # 类似 ImageNet 的 RRC
        ]
        print("Dataset with basic VOC2007 augmentation (RRC, Flip)")

    train_cast = [] if from_tensor else [transforms.ToTensor()]

    train_normal_fn = []
    if normalize:
        # 假设 voc2007 的均值和标准差已在 dataset_statistics 中定义
        try:
            mean = MEANS["voc2007"]
            std = STDS["voc2007"]
            train_normal_fn = [transforms.Normalize(mean=mean, std=std)]
        except KeyError:
            print("Warning: MEANS['voc2007'] or STDS['voc2007'] not found. Skipping normalization.")
            train_normal_fn = []


    # --- 测试集/验证集转换 ---
    # 通常测试集只做 Resize 和 CenterCrop
    test_resize = [
        transforms.Resize(int(size * 256 / 224)), # 保持 ImageNet 比例
        transforms.CenterCrop(size),
    ]
    test_cast = [] if from_tensor else [transforms.ToTensor()]
    test_normal_fn = train_normal_fn # 使用与训练集相同的归一化

    # --- 组合转换 ---
    train_transform = transforms.Compose(train_cast + train_aug + train_normal_fn)
    test_transform = transforms.Compose(test_resize + test_cast + test_normal_fn)

    return train_transform, test_transform
