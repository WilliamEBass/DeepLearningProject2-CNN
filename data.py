"""cifar-10 loaders. same seeded split for both models so the comparison is fair.

train 45k / val 5k (carved out of the official 50k train set) / test 10k (official).
test only gets touched by the --final runs. all tuning happens on val.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# cifar train-set stats, for the scratch model
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
# pretrained weights expect imagenet normalization, don't mix these up
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(augment, pretrained):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained else (CIFAR_MEAN, CIFAR_STD)
    tf = []
    if augment:
        tf += [transforms.RandomCrop(32, padding=4),
               transforms.RandomHorizontalFlip()]
    tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(tf)


def get_split_indices(val_size=5000, subset_frac=1.0, seed=42):
    """seeded perm -> val/train idx. subset_frac only shrinks train, val stays 5k."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(50000, generator=g).tolist()
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if subset_frac < 1.0:
        train_idx = train_idx[:int(len(train_idx) * subset_frac)]
    return train_idx, val_idx


def get_loaders(data_dir="./data", batch_size=128, augment=True, pretrained=False,
                subset_frac=1.0, seed=42, num_workers=4):
    train_tf = get_transforms(augment, pretrained)
    eval_tf = get_transforms(False, pretrained)

    # two dataset objects over the same files bc Subset can't swap transforms,
    # and val must not get augmented
    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = get_split_indices(subset_frac=subset_frac, seed=seed)
    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    # persistent_workers errors out with 0 workers
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **common)
    return train_loader, val_loader, test_loader
