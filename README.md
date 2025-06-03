# LUD-VAE Image Denoising Training Framework

This repository implements a conditional latent-variable denoising variational autoencoder (LUD-VAE) for image denoising, built on PyTorch. It provides a flexible training pipeline, configurable via a JSONC file, with utilities for data handling, logging, and model management.

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Training](#running-training)
- [Logging and Outputs](#logging-and-outputs)
- [Code Overview](#code-overview)
  - [train.py](#trainpy)
  - [utils](#utils)
  - [models](#models)
- [Customizing](#customizing)
- [License](#license)

## Features
- Patch-based training with synthetic noise injection
- Flexible JSONC-based configuration with comment support
- Weighted sampler for class balancing
- Checkpointing, periodic testing, and logging
- Modular design (data, model, training wrapper)

## Repository Structure
```
.
├── models/
│   ├── network.py      # LUDVAE architecture (encoder, decoder, Gaussian blocks)
│   └── ludvae.py       # Training wrapper (optimizer, scheduler, I/O)
├── utils/
│   ├── dataset.py      # UPMDataset for patch sampling and noise injection
│   ├── image.py        # Image I/O, conversions, augmentation
│   ├── logger.py       # Logging setup for file and console
│   └── options.py      # JSONC parsing and option management
├── train.py            # Main training script
├── train_sidd.jsonc    # Example configuration (SIDD Small dataset)
└── requirements.txt    # Python dependencies
```

## Requirements
- Python 3.7 or higher
- PyTorch 1.x (with CUDA support for GPU training)
- torchvision
- numpy
- opencv-python
- matplotlib

Install via pip:
```bash
pip install -r requirements.txt
```

## Configuration
Training and testing behavior is controlled by a JSONC file (e.g., `train_sidd.jsonc`). JSONC allows inline comments using `//`.

Key sections:
- **task**: Unique name for this experiment (used to build output directories).
- **gpu_ids**: List of GPU indices (e.g., `[0]`).
- **n_channels**: Number of image channels (1 or 3).
- **path**:
  - **root**: Root folder under which `task` directory is created.
  - **pretrained_net**: Optional path to a pretrained `.pth` model.
- **datasets**:
  - **train** and **test** phases, each with:
    - **dataroot**: List of paths (clean, noisy).
    - **n_max**: Max number of images (slice list).
    - **H_size**: Patch size for H (high/noise) images.
    - **dataloader_***: Batch size, shuffle, num_workers.
    - **H_noise_level**, **L_noise_level**: Noise standard deviations.
    - **normalize** (optional): Mean/std maps for clean/noisy normalization.
- **train**:
  - **optimizer_type**, **optimizer_lr**
  - **KL_anneal**, **KL_anneal_maxiter**, **KL_weight**
  - **scheduler_type**, **scheduler_milestones**, **scheduler_gamma**
  - **checkpoint_test**, **checkpoint_save**, **checkpoint_print**

## Running Training
```bash
python train.py -opt train_sidd.jsonc
```

This performs:
1. Parse and broadcast options, set `CUDA_VISIBLE_DEVICES`.
2. Create directories:
   - `translate/<task>/models` (checkpoints)
   - `translate/<task>/images` (test outputs)
   - `translate/<task>/options` (copies of config)
   - `translate/<task>/log` (training logs)
3. Initialize logger (`train.log`).
4. Prepare `UPMDataset` and `DataLoader` for train/test.
5. Instantiate `LUDVAE` model, load pretrained if available.
6. Train loop: update LR, feed data, optimize, log losses, save checkpoints, and run periodic tests.

## Logging and Outputs
- **Logs**: `translate/<task>/train.log` (console + file).
- **Checkpoints**: `translate/<task>/models/{iteration}.pth`.
- **Test Images**: `translate/<task>/images/<img_name>/{img_name}_{label}_to_{1-label}_{iter}.png`.
- **Config Copies**: `translate/<task>/options/{config_name}_<timestamp>.jsonc`.

## Code Overview

### train.py
- Parses command-line args and JSONC config (`utils/options.py`).
- Sets random seeds, logger.
- Builds data loaders (`utils/dataset.py`).
- Wraps model (`models/ludvae.py`) and runs training/testing loops.

### utils/
- **options.py**: Reads JSONC, handles comments, broadcasts options, sets up paths and GPU env.
- **logger.py**: Initializes Python `logging` to file and stdout.
- **image.py**: CV2-based I/O, tensor ⇄ numpy conversions, patch extraction, augmentation.
- **dataset.py**: `UPMDataset` provides paired clean/noisy patches, applies noise and normalization.

### models/
- **network.py**: Defines `LUDVAE` PyTorch `nn.Module`, with encoder/decoder stacks and Gaussian latent blocks.
- **ludvae.py**: `LUDVI` class (not to be confused with `LUDVAE`): wraps training utilities including optimizer, scheduler, save/load, training and testing routines.

## Customizing
- To train on your own data, prepare two folders of clean and noisy images and update `dataroot` in a new JSONC config.
- Adjust hyperparameters (learning rate, batch size, noise levels, scheduler milestones) in the `train` section.
- You can resume training by setting `pretrained_net` to a saved `.pth` file.

## License
This project does not include a license. Please contact the author or add a suitable open-source license.