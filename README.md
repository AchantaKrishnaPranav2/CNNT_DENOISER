# CNNT Denoiser (Denoising-only)

This repository contains a compact, human-written implementation of a CNNT-style image denoiser.

## Files

- `model.py` — Model definitions: InstanceNormalization, local Conv-QK attention, CNNT cells, and `build_cnnt_denoiser`.
- `utils.py` — Minimal dataset utilities (`create_denoise_datasets`). No previews or prints.
- `train.py` — Minimal training script. No visualizations or dataset outputs. Saves best model to disk.

## Quick start

1. Prepare two folders with the *same number of* images and matching filenames/order:
   - `noisy_dir/` — noisy inputs
   - `clean_dir/` — clean ground-truth images

2. Install dependencies:
```bash
pip install tensorflow opencv-python numpy
