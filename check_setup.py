"""Quick setup check: is PyTorch using the GPU, and how fast does training run?
Not part of the submission - delete before zipping.

Usage:  python check_setup.py
"""
import platform
import time

import torch
import torch.nn as nn

from data import get_loaders
from models import CustomCNN


def time_one_pass(num_workers, device):
    train_loader, _, _ = get_loaders("data", batch_size=128, augment=True,
                                     subset_frac=0.2, num_workers=num_workers)  # 9,000 images
    model = CustomCNN(channels=(32, 64, 128)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    t_start = time.time()
    first_batch = None
    n = 0
    for x, y in train_loader:
        if first_batch is None:
            first_batch = time.time() - t_start
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
        n += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total = time.time() - t_start
    print(f"  num_workers={num_workers}: {n} images in {total:.1f}s "
          f"({n / total:.0f} img/s, first batch after {first_batch:.1f}s)")


if __name__ == "__main__":
    print(f"Python      {platform.python_version()} on {platform.node()}")
    print(f"PyTorch     {torch.__version__}  (built for CUDA: {torch.version.cuda})")
    print(f"CUDA ready  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU         {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("\n*** PyTorch can't see a GPU, so everything runs on the CPU. ***")
        if torch.version.cuda is None:
            print("    You have the CPU-only PyTorch build. Reinstall with:")
            print("    pip uninstall -y torch torchvision")
            print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
        else:
            print("    PyTorch has CUDA support but found no GPU. Check `nvidia-smi` / the NVIDIA driver.")
        device = torch.device("cpu")
    print(f"CPU threads {torch.get_num_threads()}\n")

    print("Timing one pass over 9,000 training images:")
    time_one_pass(0, device)
    time_one_pass(4, device)
