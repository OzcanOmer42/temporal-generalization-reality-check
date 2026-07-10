"""Utility helpers: seeding, device selection, and flat parameter-vector ops.

The core trick used throughout this repo is to represent every model checkpoint
as a single flat 1-D tensor of its *learnable* parameters, plus a copy of its
buffers (e.g. BatchNorm running statistics). This makes the parameter
interpolation / extrapolation math (averaging, downscaling, Taylor steps)
trivial and unambiguous, while keeping buffers out of the arithmetic (scaling a
BatchNorm running-variance toward zero would be meaningless).
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Pick the best available device. On an Apple-Silicon Mac this is 'mps'."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_param_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all learnable parameters into one 1-D CPU tensor (detached)."""
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def layer_slices(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """Map each parameter name -> (start, end) index range in the flat vector.

    Used by the per-layer downscaling extension so we can shrink each layer by
    its own factor.
    """
    slices: Dict[str, Tuple[int, int]] = {}
    idx = 0
    for name, p in model.named_parameters():
        n = p.numel()
        slices[name] = (idx, idx + n)
        idx += n
    return slices


def set_param_vector(model: nn.Module, vec: torch.Tensor) -> None:
    """Load a flat parameter vector back into the model in-place."""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[idx: idx + n].reshape(p.shape).to(p.device, p.dtype))
            idx += n


def clone_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: buf.detach().clone().cpu() for name, buf in model.named_buffers()}


def load_buffers(model: nn.Module, buffers: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name in buffers:
                buf.copy_(buffers[name].to(buf.device, buf.dtype))


class Checkpoint:
    """A frozen snapshot of a model at one timestep."""

    def __init__(self, model: nn.Module):
        self.vec: torch.Tensor = get_param_vector(model)
        self.buffers: Dict[str, torch.Tensor] = clone_buffers(model)

    def apply_to(self, model: nn.Module, vec: torch.Tensor | None = None) -> None:
        """Load a (possibly modified) parameter vector + this checkpoint's buffers."""
        set_param_vector(model, self.vec if vec is None else vec)
        load_buffers(model, self.buffers)
