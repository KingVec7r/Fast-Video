# Monkey-patch to fix torchaudio CUDA version mismatch with NVIDIA PyTorch
import torch as _torch
_ORIG_CUDA = _torch.version.cuda
_torch.version.cuda = "12.6"  # torchaudio 2.7.1 was compiled with CUDA 12.6
import torchaudio as _torchaudio  # noqa: E402
_torch.version.cuda = _ORIG_CUDA
del _torch, _ORIG_CUDA, _torchaudio

try:
    from .model import LlavaLlamaForCausalLM
except:
    pass
