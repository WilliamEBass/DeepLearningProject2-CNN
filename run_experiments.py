"""runs the whole project in order. winners are picked on VAL acc, test only gets used by the two final runs.

  stage 1  custom arch search             6 runs x 15 ep
  stage 2  custom hparams                 lr/optimizer, then bs/wd
  stage 3  final custom                   best config, 40 ep, test eval
  stage 4  pretrained arch comparison     4 models x 3 ep, full fine-tune
  stage 5  frozen vs full fine-tune       on the best pretrained model
  stage 6  final pretrained               8 ep, test eval

finished runs get skipped, so if it dies just rerun it.

  python run_experiments.py           # full run (~2 h on an RTX 3070 Laptop GPU)
  python run_experiments.py --quick   # 1 epoch on 5% of data, just to test the pipeline
"""
import argparse
import json
import os

import train

QUICK = False
RESULTS = "results"


def result_path(name, fname):
    return os.path.join(RESULTS, name, fname)


def do_run(cfg):
    cfg = dict(cfg, results_dir=RESULTS)
    if QUICK:
        cfg.update(epochs=1, subset_frac=0.05)
    # final runs aren't done until test.json exists, everything else just needs history
    needed = "test.json" if cfg.get("final") else "history.json"
    if os.path.exists(result_path(cfg["name"], needed)):
        print(f"[skip] {cfg['name']} already done")
        return
    train.run(cfg)


def load_cfg(name):
    with open(result_path(name, "config.json")) as f:
        return json.load(f)


def best_of(names):
    """highest best_val_acc wins. ties go to whichever comes first in names."""
    scores = {}
    for n in names:
        with open(result_path(n, "history.json")) as f:
            scores[n] = json.load(f)["best_val_acc"]
    best = max(scores, key=scores.get)
    print("\nValidation accuracy:")
    for n, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {n:28s} {s:.4f}{'   <-- selected' if n == best else ''}")
    return best


def strip(cfg):
    """drop bookkeeping keys so a saved config can seed the next stage."""
    return {k: v for k, v in cfg.items() if k not in ("name", "final", "results_dir")}


def main():
    # ---------------- stage 1: custom arch ----------------
    # one change per step so each one's effect is isolated.
    # lr 0.01 on purpose: the no-bn nets were likely to blow up at 0.05
    base = dict(model="custom", epochs=15, optimizer="sgd", lr=0.01, batch_size=128,
                weight_decay=5e-4)
    arch_runs = [
        dict(name="arch1_small", channels=[32, 64], convs_per_block=1, batch_norm=False,
             dropout=0.0, augment=False),
        dict(name="arch2_deeper", channels=[32, 64, 128], convs_per_block=2, batch_norm=False,
             dropout=0.0, augment=False),
        dict(name="arch3_deeper_bn", channels=[32, 64, 128], convs_per_block=2, batch_norm=True,
             dropout=0.0, augment=False),
        dict(name="arch4_bn_dropout", channels=[32, 64, 128], convs_per_block=2, batch_norm=True,
             dropout=0.5, augment=False),
        dict(name="arch5_bn_dropout_aug", channels=[32, 64, 128], convs_per_block=2,
             batch_norm=True, dropout=0.5, augment=True),
        dict(name="arch6_wide", channels=[64, 128, 256], convs_per_block=2, batch_norm=True,
             dropout=0.5, augment=True),
    ]
    for r in arch_runs:
        do_run({**base, **r})
    best_arch = best_of([r["name"] for r in arch_runs])
    arch_cfg = strip(load_cfg(best_arch))

    # ---------------- stage 2a: optimizer + lr ----------------
    # adamw wants a much bigger wd than sgd since it's decoupled
    lr_runs = [
        dict(name="hp_sgd_lr0.05", optimizer="sgd", lr=0.05),
        dict(name="hp_sgd_lr0.1", optimizer="sgd", lr=0.1),
        dict(name="hp_adamw_lr1e-3", optimizer="adam", lr=1e-3, weight_decay=5e-2),
        dict(name="hp_adamw_lr3e-3", optimizer="adam", lr=3e-3, weight_decay=5e-2),
    ]
    for r in lr_runs:
        do_run({**arch_cfg, **r})
    # stage 1 winner already covers sgd @ 0.01, no point rerunning it
    best_lr = best_of([best_arch] + [r["name"] for r in lr_runs])
    lr_cfg = strip(load_cfg(best_lr))

    # ---------------- stage 2b: batch size + wd ----------------
    bs_runs = [
        dict(name="hp_bs64", batch_size=64),
        dict(name="hp_bs256", batch_size=256),
        dict(name="hp_wd0", weight_decay=0.0),
    ]
    for r in bs_runs:
        do_run({**lr_cfg, **r})
    best_hp = best_of([best_lr] + [r["name"] for r in bs_runs])
    final_custom_cfg = strip(load_cfg(best_hp))

    # ---------------- stage 3: final custom ----------------
    # 40 ep bc every 15 ep run was still improving at the end
    do_run({**final_custom_cfg, "name": "final_custom", "epochs": 40, "final": True})

    # ---------------- stage 4: pretrained comparison ----------------
    # bs 64 so resnet50 @ 224 fits in 8gb
    pt_base = dict(model="pretrained", epochs=3, optimizer="sgd", lr=0.01, batch_size=64,
                   weight_decay=5e-4, augment=True, freeze_backbone=False)
    pt_runs = [dict(name=f"pt_{a}", arch=a) for a in
               ["resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0"]]
    for r in pt_runs:
        do_run({**pt_base, **r})
    best_pt = best_of([r["name"] for r in pt_runs])
    pt_cfg = strip(load_cfg(best_pt))

    # ---------------- stage 5: frozen vs full fine-tune ----------------
    # higher lr for the frozen run since only the head is learning
    frozen_name = f"pt_{pt_cfg['arch']}_frozen"
    do_run({**pt_cfg, "name": frozen_name, "freeze_backbone": True, "lr": 0.05})
    best_strategy = best_of([best_pt, frozen_name])
    final_pt_cfg = strip(load_cfg(best_strategy))

    # ---------------- stage 6: final pretrained ----------------
    do_run({**final_pt_cfg, "name": "final_pretrained", "epochs": 8, "final": True})

    print("\nAll experiments done. Now run:  python make_plots.py"
          + (" --results results_quick" if QUICK else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="1 epoch on 5%% of the data (pipeline test)")
    args = p.parse_args()
    if args.quick:
        # separate dir so quick results never get mixed into the real ones
        QUICK = True
        RESULTS = "results_quick"
    main()
