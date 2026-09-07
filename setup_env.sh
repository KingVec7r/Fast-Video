#!/bin/bash
# ============================================================
# One-command environment setup for fast_onevision
# Usage: bash setup_env.sh
# ============================================================
set -e

echo "=== Installing fast_onevision with all dependencies ==="
cd "$(dirname "$0")"
pip install -e ".[train,build]"

echo ""
echo "=== Fixing torchaudio CUDA compatibility with NVIDIA PyTorch ==="
# NVIDIA PyTorch (2.7.0a0, CUDA 12.9) is ABI-compatible with
# torchaudio 2.7.1 (CUDA 12.6), but pip installs the latest (2.11.0)
# which is incompatible. Downgrade without dep-check.
TORCHAUDIO_VER=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null || echo "none")
if [[ "$TORCHAUDIO_VER" != "2.7.1"* ]]; then
    pip uninstall torchaudio -y 2>/dev/null
    pip install torchaudio==2.7.1 --no-deps
    echo "torchaudio fixed: 2.7.1 installed"
else
    echo "torchaudio already correct: $TORCHAUDIO_VER"
fi

echo ""
echo "=== Verifying installation ==="
python3 -c "
import torch;           print(f'  torch:       {torch.__version__} | CUDA: {torch.cuda.is_available()}')
import torchvision;     print(f'  torchvision: {torchvision.__version__}')
import torchaudio;      print(f'  torchaudio:  {torchaudio.__version__}')
import triton;          print(f'  triton:      {triton.__version__}')
import transformers;    print(f'  transformers:{transformers.__version__}')
import accelerate;      print(f'  accelerate:  {accelerate.__version__}')
import datasets;        print(f'  datasets:    {datasets.__version__}')
import timm;            print(f'  timm:        {timm.__version__}')
import cv2;             print(f'  cv2:         {cv2.__version__}')
import deepspeed;       print(f'  deepspeed:   {deepspeed.__version__}')
import fast_onevision;  print(f'  fast_onevision: OK')
print('All packages ready!')
"

echo ""
echo "=== Environment setup complete! ==="
