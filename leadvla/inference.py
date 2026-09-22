"""Backend-neutral inference adapter; no checkpoint or backbone is included."""

from typing import Protocol
import torch

from .inputs import PolicyInput
from .path_composer import ArcLengthPathComposer


class PolicyBackend(Protocol):
    def predict(self, images: tuple, prompt: str) -> dict[str, torch.Tensor]:
        """One multimodal pass: return pace [1] and residual [1,24,3].

        Backend must implement 24 shared action queries. Pace uses their original
        pooled representation; Residual alone receives soft lateral+formation
        modulation. The six auxiliary tasks remain training supervision. This
        interface cannot verify backend internals or checkpoint provenance.
        """
        ...


@torch.inference_mode()
def predict_waypoints(backend: PolicyBackend, sample: PolicyInput):
    prediction = backend.predict(sample.images, sample.prompt)
    pace, residual = prediction["pace"], prediction["residual"]
    if pace.shape != (1,) or residual.shape != (1, 24, 3):
        raise ValueError("Backend must return pace [1] and residual [1,24,3]")
    if not pace.is_floating_point() or not residual.is_floating_point():
        raise ValueError("Predictions must be floating point")
    if not torch.isfinite(pace).all() or not torch.isfinite(residual).all() or not ((pace >= 0) & (pace <= 1)).all():
        raise ValueError("Predictions must be finite; Pace must be in [0,1]")
    if pace.device != residual.device:
        raise ValueError("Predictions must share a device")
    def tensor(array):
        return torch.as_tensor(array, device=residual.device, dtype=residual.dtype).unsqueeze(0)
    return ArcLengthPathComposer()(
        tensor(sample.path_reference), tensor(sample.path_progress), pace,
        residual.new_tensor([sample.executed_pace]), residual,
        path_origin=tensor(sample.path_origin),
    )
