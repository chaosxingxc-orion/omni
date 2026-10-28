"""Device resolution for the retrieval encoders.

The study hard-codes ``cuda`` in two places (``core/speech_embedding.py`` line 94
``model.to("cuda")`` and line 113 ``device: str = "cuda"``). We run on Ascend
910B, so the device is the first thing the adapter has to absorb. It is kept in
its own module because nothing else about the encoder depends on it, and because
``import torch_npu`` has to happen before ``torch.npu`` exists at all -- that
import is a side effect, not a value, and burying it inside the encoder module
makes the ordering easy to break by accident.

Resolution order, most explicit first:
  1. ``SAEA_ENCODER_DEVICE`` -- an exact device string, honoured as given.
  2. Ascend, if ``torch_npu`` imports and reports a visible device.
  3. CUDA, if available (so the same code still runs on the author's stack).
  4. CPU, which is correct but slow; the caller is told, never silently demoted.
"""

from __future__ import annotations

import os


class DeviceError(RuntimeError):
    """Refuse to guess when an explicit device request cannot be honoured."""


def resolve(requested: str | None = None) -> str:
    """Return a torch device string, importing the Ascend extension if needed."""

    import torch

    requested = requested or os.environ.get("SAEA_ENCODER_DEVICE") or ""
    if requested:
        # An explicit request is a statement of intent: if the backend is not
        # there, that is an error, not an invitation to fall back to CPU and
        # produce the right numbers 40x slower than the operator expected.
        if requested.startswith("npu"):
            _import_torch_npu(hard=True)
            if not torch.npu.is_available():  # type: ignore[attr-defined]
                raise DeviceError(f"{requested} requested but torch.npu reports no device")
        elif requested.startswith("cuda") and not torch.cuda.is_available():
            raise DeviceError(f"{requested} requested but torch.cuda reports no device")
        return requested

    if _import_torch_npu(hard=False) and torch.npu.is_available():  # type: ignore[attr-defined]
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _import_torch_npu(*, hard: bool) -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        if hard:
            raise DeviceError("an npu device was requested but torch_npu is not installed")
        return False
    return True


def describe(device: str) -> dict[str, object]:
    """Facts worth landing in a receipt beside every vector file."""

    import torch

    out: dict[str, object] = {"device": device, "torch": torch.__version__}
    if device.startswith("npu"):
        import torch_npu

        out["torch_npu"] = torch_npu.__version__
        out["device_count"] = int(torch.npu.device_count())  # type: ignore[attr-defined]
    elif device.startswith("cuda"):
        out["device_count"] = int(torch.cuda.device_count())
        out["device_name"] = torch.cuda.get_device_name(0)
    return out
