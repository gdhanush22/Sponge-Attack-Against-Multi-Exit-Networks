# Sponge Attack Defense with Multi-Exit Networks

A lightweight pipeline to detect and defend against high-frequency (“sponge”) data-poisoning in image classifiers. We train a Multi-Exit CNN on CIFAR-10, poison the data with rich high-frequency noise, observe its impact, then detect & repair poisoned samples via 2D-FFT analysis and lightweight augmentation—finally retraining to recover accuracy and speed.

## Files

- **train_clean.py**  
  Defines and trains the `MultiExitNet` on clean CIFAR-10. Uses four early-exit branches, averaged cross-entropy loss, Adam optimizer, and LR scheduling. Saves the best clean model as `clean_model.pth`.

- **poison_data.py**  
  Wraps the CIFAR-10 dataset in `PoisonedCIFAR10`, randomly selects a fraction of images, injects multiple high-frequency noise patterns (checkerboard, sinusoids, edge masks, Gaussian noise), optionally flips labels, and saves metadata to `poison_info.pth`.

- **train_poisoned.py**  
  Same training script as `train_clean.py`, but uses the poisoned dataset from `poison_data.py`. Demonstrates degraded early-exit rates, higher inference latency, and reduced accuracy.

- **repair_dataset.py**  
  1. **Detects** suspicious samples by computing per-image high-frequency energy via 2D-FFT.  
  2. **Repairs** them by replacing each flagged image with a randomly selected clean example plus simple augmentations (flip, rotation, color jitter).  
  3. **Retrains** a fresh `MultiExitNet` on the repaired data, checkpoints the best model as `repaired_model.pth`, and logs detection metrics in `repair_info.pth`.

## Quick Start

1. Install dependencies:
   ```bash
   pip install torch torchvision numpy scipy pillow tqdm
After Installing the dependencies run the files in the below order:
1)train_clean.py  which trains a clean model and saves it.
2)poison_data.py  which poisoned the cifar 10 dataset.
3)train_poisoned.py which trains a new model on the poisoned dataset.
4)repair_dataset.py which repairs the poisoned dataset and trains a new model on it.

There you go you have all the three models now clean, poisoned and repaired!!
