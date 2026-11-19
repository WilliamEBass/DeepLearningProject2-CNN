"""part 1: CustomCNN, scratch vgg-ish net. part 2: build_pretrained, torchvision model w/ new 10-class head."""
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CustomCNN(nn.Module):
    """blocks of [conv3x3 -> bn? -> relu] x convs_per_block, then 2x2 maxpool.

    knobs exist so run_experiments can sweep them without touching this file.
    """

    def __init__(self, channels=(32, 64, 128), convs_per_block=2, batch_norm=True,
                 dropout=0.5, fc_units=256, num_classes=10):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in channels:
            for _ in range(convs_per_block):
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                in_ch = out_ch
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)

        # one pool per block, so 3 blocks -> 4x4. if you change the pooling this breaks
        spatial = 32 // (2 ** len(channels))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * spatial * spatial, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResizeWrapper(nn.Module):
    """upsample 32 -> 224 on the gpu. resizing in the dataloader was the bottleneck.
    also these nets downsample 32x, so a raw 32px input ends up 1x1 halfway through."""

    def __init__(self, net, img_size=224):
        super().__init__()
        self.net = net
        self.img_size = img_size

    def forward(self, x):
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return self.net(x)


def build_pretrained(arch="resnet18", num_classes=10, freeze_backbone=False, img_size=224):
    # every torchvision family names its head differently, hence the branches
    if arch == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head = net.fc
    elif arch == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head = net.fc
    elif arch == "mobilenet_v3_small":
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        head = net.classifier[3]
    elif arch == "efficientnet_b0":
        # classifier[0] is the dropout, leave it alone
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
        head = net.classifier[1]
    elif arch == "densenet121":
        # never actually ran this one, kept it in case
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        head = net.classifier
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if freeze_backbone:
        # linear probe: freeze everything, then re-enable just the new head
        for p in net.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True

    return ResizeWrapper(net, img_size)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
