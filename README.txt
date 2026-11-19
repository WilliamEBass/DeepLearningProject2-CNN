CNN Image Classification on CIFAR-10 (CSCI 490-DPL Project 2)
==============================================================

William Bass

Part 1 trains a custom VGG-style CNN from scratch. Part 2 fine-tunes an
ImageNet-pretrained CNN from torchvision. Both use the same fixed data split.


FILES
-----

  data.py              CIFAR-10 loading, preprocessing/augmentation, fixed train/val/test split
  models.py            CustomCNN (Part 1) and build_pretrained (Part 2)
  train.py             Trains one model; saves history, best checkpoint, and test metrics
  run_experiments.py   Runs every experiment in order and picks the best settings on validation accuracy
  make_plots.py        Builds all figures and results/summary.md from the saved results
  requirements.txt     Python dependencies


DATA SPLIT
----------

  Train:       45,000 images, randomly chosen (seed 42) from the official 50,000 training images
  Validation:  the other 5,000 training images. All model selection and hyperparameter
               choices use this set.
  Test:        the official 10,000 test images. Only the two final runs evaluate on it.

The full dataset is used (no subset). --subset-frac exists in train.py if a
smaller run is ever needed.

Preprocessing: per-channel normalization (CIFAR-10 statistics for the custom
CNN, ImageNet statistics for the pretrained CNN). When augmentation is on, it
adds a random 32x32 crop with 4 px padding and a random horizontal flip. The
pretrained models upsample the 32x32 images to 224x224 on the GPU, inside the
model.


SETUP
-----

Tested with Python 3.10.11 and PyTorch 2.5.1 (CUDA 12.1). To use the GPU,
install the CUDA build of PyTorch. Get the exact command for your system from
https://pytorch.org/get-started/locally/

    python -m venv venv
    venv\Scripts\activate          (Windows)
    source venv/bin/activate       (Linux/macOS)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install numpy matplotlib

CIFAR-10 (~170 MB) downloads automatically into ./data on the first run. The
pretrained weights download automatically into the PyTorch cache.


REPRODUCING THE EXPERIMENTS
---------------------------

  1) Optional: quick check of the whole pipeline (1 epoch, 5% of the data, ~5 min)

       python run_experiments.py --quick

  2) All experiments (roughly 2 hours on an RTX 3070 Laptop GPU)

       python run_experiments.py

  3) Figures + summary tables

       python make_plots.py

run_experiments.py skips runs that have already finished. If it gets
interrupted, run it again and it continues where it stopped.

What it runs:

  1. Custom CNN architecture search (15 epochs each, SGD lr 0.01). Each run adds
     one change: small 2-block net -> deeper 3-block/6-conv net -> + BatchNorm
     -> + dropout -> + data augmentation -> wider channels.
  2. Hyperparameter tuning on the best architecture: SGD lr 0.05/0.1 vs AdamW
     lr 1e-3/3e-3, then batch size 64/256 and weight decay 0.
  3. Final custom CNN: the best configuration trained for 40 epochs, then
     evaluated on the test set.
  4. Pretrained model comparison (3 epochs each, full fine-tuning): ResNet-18,
     ResNet-50, MobileNetV3-Small, EfficientNet-B0.
  5. Transfer strategy: frozen backbone (only the new head is trained) vs full
     fine-tuning, on the best pretrained model.
  6. Final pretrained CNN: the best model and strategy trained for 8 epochs,
     then evaluated on the test set.

Running a single model by hand:

    python train.py --name my_cnn --model custom --channels 64,128,256 --epochs 40 --lr 0.05 --final
    python train.py --name my_resnet --model pretrained --arch resnet18 --epochs 8 --lr 0.01 --batch-size 64 --final
    python train.py --help        (all options)


OUTPUTS
-------

  results/<run>/history.json   per-epoch train/val loss and accuracy, plus timing and parameter counts
  results/<run>/test.json      test accuracy, class-wise accuracy, and confusion matrix (final runs only)
  results/<run>/best.pt        weights from the epoch with the best validation accuracy
  results/summary.md           every number used in the report
  figures/*.png                loss/accuracy curves, architecture comparison, class-wise accuracy,
                               confusion matrices


NOTES
-----

  - num_workers defaults to 0. CIFAR-10 fits in memory, and on Windows worker
    processes took ~12 s per loader to start and made training slower
    (705 vs 3,046 img/s measured on an RTX 3070 Laptop GPU).
  - If the GPU runs out of memory on the pretrained models, lower batch_size in
    pt_base in run_experiments.py.
  - Mixed precision (fp16 autocast) is used automatically on CUDA.


REFERENCES
----------

  CIFAR-10:                      https://www.cs.toronto.edu/~kriz/cifar.html
  PyTorch CIFAR-10 tutorial:     https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
  Torchvision pretrained models: https://docs.pytorch.org/vision/stable/models.html
