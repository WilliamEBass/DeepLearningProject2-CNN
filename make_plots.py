"""builds every figure + table for the report from results/. no training here.

  figures/curves_final_custom.png       train/val loss + acc
  figures/curves_final_pretrained.png
  figures/arch_val_curves.png           stage 1 archs
  figures/experiments_val_acc.png       best val acc, every run
  figures/class_accuracy.png            per-class test acc, both models
  figures/confusion_custom.png          row-normalized confusion matrices
  figures/confusion_pretrained.png
  results/summary.md                    every number the report uses

  python make_plots.py            (or --results results_quick after a --quick run)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # no display needed, just write pngs
import matplotlib.pyplot as plt
import numpy as np

from data import CLASSES

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MODEL_COLOR = {"final_custom": PALETTE[0], "final_pretrained": PALETTE[1]}
MODEL_LABEL = {"final_custom": "Custom CNN", "final_pretrained": "Pretrained CNN"}

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
})


def load(results, name, fname):
    # None = run hasn't happened yet, callers just skip it
    path = os.path.join(results, name, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def describe(cfg):
    if cfg["model"] == "custom":
        return (f"custom ch={cfg['channels']} x{cfg['convs_per_block']} "
                f"bn={cfg['batch_norm']} drop={cfg['dropout']} aug={cfg['augment']} | "
                f"{cfg['optimizer']} lr={cfg['lr']} bs={cfg['batch_size']} wd={cfg['weight_decay']}")
    return (f"{cfg['arch']} frozen={cfg['freeze_backbone']} | {cfg['optimizer']} "
            f"lr={cfg['lr']} bs={cfg['batch_size']}")


def plot_curves(results, name, out):
    h = load(results, name, "history.json")
    if h is None:
        return
    color = MODEL_COLOR[name]
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    # same color per model, solid = train, dashed = val
    for ax, key, title in [(axes[0], "loss", "Loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(ep, h[f"train_{key}"], color=color, lw=2, label="train")
        ax.plot(ep, h[f"val_{key}"], color=color, lw=2, ls="--", label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(f"{MODEL_LABEL[name]}: {title}")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"curves_{name}.png"))
    plt.close(fig)


def plot_arch_curves(results, out):
    names = sorted(n for n in os.listdir(results) if n.startswith("arch"))
    if not names:
        return
    # shared y so train vs val gaps are comparable at a glance
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for i, n in enumerate(names):
        h = load(results, n, "history.json")
        if h is None:
            continue
        ep = np.arange(1, len(h["val_acc"]) + 1)
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(ep, h["train_acc"], color=c, lw=2, label=n)
        axes[1].plot(ep, h["val_acc"], color=c, lw=2, label=n)
    axes[0].set_title("Stage 1 architectures: train accuracy")
    axes[1].set_title("Stage 1 architectures: validation accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "arch_val_curves.png"))
    plt.close(fig)


def plot_experiments(rows, out):
    # finals are 40/8 ep runs, not comparable to the 15/3 ep screening runs
    rows = [r for r in rows if not r["name"].startswith("final")]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["val"])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1))
    colors = [PALETTE[0] if r["model"] == "custom" else PALETTE[1] for r in rows]
    ax.barh([r["name"] for r in rows], [r["val"] * 100 for r in rows], color=colors, height=0.6)
    for i, r in enumerate(rows):
        ax.text(r["val"] * 100 + 0.3, i, f"{r['val'] * 100:.1f}", va="center", fontsize=8)
    # zoom in, otherwise every bar looks the same length
    lo = min(r["val"] for r in rows) * 100
    ax.set_xlim(max(0, lo - 5), 100)
    ax.set_xlabel("best validation accuracy (%)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE[0], label="custom CNN"),
                       Patch(color=PALETTE[1], label="pretrained CNN")],
              frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "experiments_val_acc.png"))
    plt.close(fig)


def plot_class_acc(tests, out):
    if len(tests) == 0:
        return
    x = np.arange(len(CLASSES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.4))
    lowest = 100
    for i, (name, t) in enumerate(tests.items()):
        vals = [t["class_acc"][c] * 100 for c in CLASSES]
        lowest = min(lowest, min(vals))
        ax.bar(x + (i - 0.5) * w, vals, width=w - 0.02, color=MODEL_COLOR[name],
               label=MODEL_LABEL[name])
    ax.set_xticks(x, CLASSES, rotation=30)
    ax.set_ylabel("test accuracy (%)")
    # floor the axis a bit below the worst class. a fixed 50 cut off low bars on the quick run
    ax.set_ylim(max(0, 10 * int(lowest // 10) - 10), 100)
    ax.set_title("Class-wise test accuracy", pad=22)
    ax.grid(axis="x", visible=False)
    # legend above the plot so it doesn't sit on top of bars
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "class_accuracy.png"))
    plt.close(fig)


def plot_confusion(name, t, out):
    conf = np.array(t["confusion"], dtype=float)
    # row-normalize -> % of each true class
    norm = conf / conf.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    for i in range(10):
        for j in range(10):
            # white text on the dark cells or it's unreadable
            ax.text(j, i, f"{norm[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 60 else "#333333")
    ax.set_title(f"{MODEL_LABEL[name]}: confusion matrix (% of true class)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"confusion_{name.replace('final_', '')}.png"))
    plt.close(fig)


def top_confusions(t, k=5):
    # biggest off-diagonal cells = the mistakes worth talking about
    conf = np.array(t["confusion"])
    pairs = [(conf[i, j], CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if i != j]
    return sorted(pairs, reverse=True)[:k]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for d in sorted(glob.glob(os.path.join(args.results, "*", "history.json"))):
        name = os.path.basename(os.path.dirname(d))
        h = load(args.results, name, "history.json")
        cfg = load(args.results, name, "config.json")
        rows.append(dict(name=name, model=cfg["model"], desc=describe(cfg), val=h["best_val_acc"],
                         best_epoch=h["best_epoch"], epochs=len(h["val_acc"]),
                         final_train_acc=h["train_acc"][-1], final_val_acc=h["val_acc"][-1],
                         time=h["total_train_time"], params=h["total_params"],
                         trainable=h["trainable_params"], n_train=h["num_train_images"]))

    tests = {n: load(args.results, n, "test.json") for n in ["final_custom", "final_pretrained"]}
    tests = {n: t for n, t in tests.items() if t is not None}

    for n in ["final_custom", "final_pretrained"]:
        plot_curves(args.results, n, args.out)
    plot_arch_curves(args.results, args.out)
    plot_experiments(rows, args.out)
    plot_class_acc(tests, args.out)
    for n, t in tests.items():
        plot_confusion(n, t, args.out)

    # ---------- summary.md: every number the report needs, so nothing gets copied by hand ----------
    L = ["# Results summary", ""]
    if tests:
        L += ["## Final models (test set)", "",
              "| model | config | test acc | params | trainable | epochs | train time | time/epoch |",
              "|---|---|---|---|---|---|---|---|"]
        for n, t in tests.items():
            r = next(r for r in rows if r["name"] == n)
            L.append(f"| {MODEL_LABEL[n]} | {r['desc']} | {t['test_acc'] * 100:.2f}% | "
                     f"{t['total_params']:,} | {t['trainable_params']:,} | {r['epochs']} "
                     f"(best {t['best_epoch']}) | {t['total_train_time'] / 60:.1f} min | "
                     f"{t['total_train_time'] / r['epochs']:.1f} s |")
        L += ["", "## Class-wise test accuracy", "",
              "| class | " + " | ".join(MODEL_LABEL[n] for n in tests) + " |",
              "|---|" + "---|" * len(tests)]
        for c in CLASSES:
            L.append(f"| {c} | " + " | ".join(f"{t['class_acc'][c] * 100:.1f}%"
                                              for t in tests.values()) + " |")
        L += ["", "## Most common misclassifications (true -> predicted, count out of 1000)", ""]
        for n, t in tests.items():
            L.append(f"**{MODEL_LABEL[n]}:** " + "; ".join(f"{a} -> {b} ({cnt})"
                                                          for cnt, a, b in top_confusions(t)))
            L.append("")
    L += ["## All experiments (validation set)", "",
          "| run | settings | best val acc | best epoch | final train acc | final val acc "
          "| train imgs | params | time |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['desc']} | {r['val'] * 100:.2f}% | {r['best_epoch']}/"
                 f"{r['epochs']} | {r['final_train_acc'] * 100:.2f}% | {r['final_val_acc'] * 100:.2f}%"
                 f" | {r['n_train']:,} | {r['params']:,} | {r['time'] / 60:.1f} min |")
    summary_path = os.path.join(args.results, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {summary_path} and figures to {args.out}/")


if __name__ == "__main__":
    main()
