"""train one model on cifar-10. everything lands in results/<name>/:

  config.json   - settings used
  history.json  - per-epoch loss/acc + epoch times
  best.pt       - weights from the best-val epoch
  test.json     - --final only: test acc, per-class acc, confusion, params, time

test set is only touched with --final. pick stuff on val, not test.

  python train.py --name my_cnn --model custom --epochs 30
  python train.py --name resnet --model pretrained --arch resnet18 --epochs 8 --lr 0.01 --final
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import CLASSES, get_loaders
from models import CustomCNN, build_pretrained, count_params

DEFAULTS = dict(
    name="run",
    model="custom",            # custom | pretrained
    # custom cnn
    channels=[32, 64, 128],
    convs_per_block=2,
    batch_norm=True,
    dropout=0.5,
    fc_units=256,
    # pretrained
    arch="resnet18",
    freeze_backbone=False,
    img_size=224,
    # training
    epochs=15,
    batch_size=128,
    optimizer="sgd",           # sgd | adam (actually adamw)
    lr=0.05,
    weight_decay=5e-4,
    scheduler="cosine",        # cosine | none
    augment=True,
    subset_frac=1.0,
    seed=42,
    num_workers=0,             # windows worker spawn is slow and data's already in ram. 0 was ~4x faster
    final=False,               # eval on test at the end
    results_dir="results",
    data_dir="data",
)


def build_model(cfg):
    if cfg["model"] == "custom":
        return CustomCNN(channels=tuple(cfg["channels"]), convs_per_block=cfg["convs_per_block"],
                         batch_norm=cfg["batch_norm"], dropout=cfg["dropout"],
                         fc_units=cfg["fc_units"])
    return build_pretrained(cfg["arch"], freeze_backbone=cfg["freeze_backbone"],
                            img_size=cfg["img_size"])


def build_optimizer(cfg, model):
    # only trainable params, so the frozen-backbone run doesn't carry 4M dead params in the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"], nesterov=True)
    if cfg["optimizer"] == "adam":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError(cfg["optimizer"])


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    """trains if you pass an optimizer, else evals. eval also returns preds/labels for the confusion matrix."""
    training = optimizer is not None
    total_loss, correct, count = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            preds = out.argmax(1)
            total_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            count += y.size(0)
            if not training:
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
    result = {"loss": total_loss / count, "acc": correct / count}
    if not training:
        result["preds"] = torch.cat(all_preds).numpy()
        result["labels"] = torch.cat(all_labels).numpy()
    return result


def run(cfg_overrides):
    cfg = {**DEFAULTS, **cfg_overrides}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # cpu autocast is really a bf16 thing, keep fp16 amp gpu-only
    torch.backends.cudnn.benchmark = True

    out_dir = os.path.join(cfg["results_dir"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    # dump config up front so a crashed run still says what it was
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train_loader, val_loader, test_loader = get_loaders(
        cfg["data_dir"], cfg["batch_size"], augment=cfg["augment"],
        pretrained=cfg["model"] == "pretrained", subset_frac=cfg["subset_frac"],
        seed=cfg["seed"], num_workers=cfg["num_workers"])

    model = build_model(cfg).to(device)
    total_params, trainable_params = count_params(model)
    print(f"\n=== {cfg['name']} | {total_params:,} params ({trainable_params:,} trainable) "
          f"| {len(train_loader.dataset)} train imgs | device={device} ===")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
                 if cfg["scheduler"] == "cosine" else None)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"]}
    best_val, best_epoch = -1.0, -1
    best_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        if cfg["freeze_backbone"]:
            # frozen backbone still updates bn running stats in train mode, so force eval.
            # grads still flow to the head
            model.eval()
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        model.eval()
        va = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        if scheduler:
            scheduler.step()
        dt = time.time() - t0  # includes the val pass

        for k, v in [("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                     ("val_loss", va["loss"]), ("val_acc", va["acc"]), ("epoch_time", dt)]:
            history[k].append(v)
        if va["acc"] > best_val:
            best_val, best_epoch = va["acc"], epoch
            torch.save(model.state_dict(), best_path)
        print(f"epoch {epoch:3d}/{cfg['epochs']} | train loss {tr['loss']:.3f} acc {tr['acc']:.4f} "
              f"| val loss {va['loss']:.3f} acc {va['acc']:.4f} | {dt:.1f}s")

    history.update(best_val_acc=best_val, best_epoch=best_epoch,
                   total_train_time=sum(history["epoch_time"]),
                   total_params=total_params, trainable_params=trainable_params,
                   num_train_images=len(train_loader.dataset))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if cfg["final"]:
        # score the best-val checkpoint, not whatever the last epoch happened to be
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()
        te = run_epoch(model, test_loader, criterion, device, use_amp=use_amp)
        conf = np.zeros((10, 10), dtype=int)
        for t, p in zip(te["labels"], te["preds"]):
            conf[t, p] += 1
        class_acc = {c: float(conf[i, i] / conf[i].sum()) for i, c in enumerate(CLASSES)}
        test_res = dict(test_acc=te["acc"], test_loss=te["loss"], class_acc=class_acc,
                        confusion=conf.tolist(), best_epoch=best_epoch,
                        total_params=total_params, trainable_params=trainable_params,
                        total_train_time=history["total_train_time"])
        with open(os.path.join(out_dir, "test.json"), "w") as f:
            json.dump(test_res, f, indent=2)
        print(f"TEST accuracy: {te['acc']:.4f} (best val epoch {best_epoch})")

    return history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=DEFAULTS["name"])
    p.add_argument("--model", choices=["custom", "pretrained"], default=DEFAULTS["model"])
    p.add_argument("--channels", default="32,64,128", help="comma separated, e.g. 64,128,256")
    p.add_argument("--convs-per-block", type=int, default=DEFAULTS["convs_per_block"])
    p.add_argument("--no-bn", action="store_true")
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--fc-units", type=int, default=DEFAULTS["fc_units"])
    p.add_argument("--arch", default=DEFAULTS["arch"])
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--img-size", type=int, default=DEFAULTS["img_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--optimizer", choices=["sgd", "adam"], default=DEFAULTS["optimizer"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--scheduler", choices=["cosine", "none"], default=DEFAULTS["scheduler"])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--subset-frac", type=float, default=DEFAULTS["subset_frac"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--final", action="store_true", help="evaluate best checkpoint on the test set")
    a = p.parse_args()
    return dict(name=a.name, model=a.model, channels=[int(c) for c in a.channels.split(",")],
                convs_per_block=a.convs_per_block, batch_norm=not a.no_bn, dropout=a.dropout,
                fc_units=a.fc_units, arch=a.arch, freeze_backbone=a.freeze_backbone,
                img_size=a.img_size, epochs=a.epochs, batch_size=a.batch_size,
                optimizer=a.optimizer, lr=a.lr, weight_decay=a.weight_decay,
                scheduler=a.scheduler, augment=not a.no_augment, subset_frac=a.subset_frac,
                seed=a.seed, num_workers=a.num_workers, final=a.final)


if __name__ == "__main__":  # windows spawns workers by re-importing this file, so the guard is mandatory
    run(parse_args())
