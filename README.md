# -Sponge-Attack-Against-Multi-Exit-Networks

A lightweight pipeline to detect and defend against high-frequency (“sponge”) data-poisoning in image classifiers. We train a Multi-Exit CNN on CIFAR-10, poison the data with rich high-frequency noise, observe its impact, then detect & repair poisoned samples via 2D-FFT analysis and lightweight augmentation—finally retraining to recover accuracy and speed.

Files
train_clean.py
Defines and trains the MultiExitNet on clean CIFAR-10. Uses four early-exit branches, averaged cross-entropy loss, Adam optimizer, and LR scheduling. Saves the best clean model as clean_model.pth.

poison_data.py
Wraps the CIFAR-10 dataset in PoisonedCIFAR10, randomly selects a fraction of images, injects multiple high-frequency noise patterns (checkerboard, sinusoids, edge masks, Gaussian noise), optionally flips labels, and saves metadata to poison_info.pth.

train_poisoned.py
Same training script as train_clean.py, but uses the poisoned dataset from poison_data.py. Demonstrates degraded early-exit rates, higher inference latency, and reduced accuracy.

repair_dataset.py

Detects suspicious samples by computing per-image high-frequency energy via 2D-FFT.
Repairs them by replacing each flagged image with a randomly selected clean example plus simple augmentations (flip, rotation, color jitter).
Retrains a fresh MultiExitNet on the repaired data, checkpoints the best model as repaired_model.pth, and logs detection metrics in repair_info.pth.
