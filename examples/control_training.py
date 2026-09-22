"""Synthetic tensor exercise of control heads and losses, not a trained policy."""

import torch
from torch import nn

from leadvla.model import ControlHeads, SEMANTIC_CLASSES
from leadvla.losses import joint_loss
from leadvla.path_composer import ArcLengthPathComposer


def run():
    torch.manual_seed(7)
    # This small decoder is for a tensor smoke test, not the paper's trained model.
    heads = ControlHeads(32, nn.Linear(32, 3), auxiliary_size=16)
    queries = torch.randn(2, 24, 32)
    predictions = heads(queries)
    progress = torch.arange(1, 25).float().repeat(2, 1) * 0.15
    route = torch.stack((progress, torch.zeros_like(progress), torch.zeros_like(progress)), dim=-1)
    predictions.update(ArcLengthPathComposer()(route, progress, predictions["pace"], torch.ones(2), predictions["residual"]))
    targets = {
        "waypoint": route, "pace": torch.ones(2), "residual": torch.zeros_like(route),
        "trajectory_valid": torch.ones(2, 24), "reaction_valid": torch.ones(2),
        "bbox": torch.zeros(2, 4), "bbox_valid": torch.ones(2),
    }
    for name in SEMANTIC_CLASSES:
        targets[name] = torch.zeros(2, dtype=torch.long)
        targets[f"{name}_valid"] = torch.ones(2)
    losses = joint_loss(predictions, targets)
    losses["total"].backward()
    return {name: float(value.detach()) for name, value in losses.items()}


if __name__ == "__main__":
    import json
    print(json.dumps({"kind": "synthetic_gradient_check", "losses": run()}, indent=2))
