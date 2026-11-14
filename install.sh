#!/usr/bin/env bash
# Run in a shell where Conda is initialized.

conda create -n DeepfakeBench-MM python=3.10.19 -y
conda activate DeepfakeBench-MM

# Install the matching PyTorch packages, then the remaining project dependencies.
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
