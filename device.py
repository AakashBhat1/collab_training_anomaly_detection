"""Device detection with TPU/GPU/CPU fallback for Colab compatibility."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available device: TPU > CUDA > CPU."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

        return xm.xla_device()
    except (ImportError, RuntimeError):
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_xla_device(device: torch.device) -> bool:
    """Check whether *device* is a torch_xla TPU device."""
    return "xla" in str(device)


def optimizer_step(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Perform an optimizer step, using xla barrier when on TPU."""
    if is_xla_device(device):
        import torch_xla.core.xla_model as xm  # type: ignore[import-untyped]

        xm.optimizer_step(optimizer)
    else:
        optimizer.step()
