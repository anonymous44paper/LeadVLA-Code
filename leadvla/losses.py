"""Masked trajectory and follower-semantic objectives (Appendix A.4)."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import SEMANTIC_CLASSES
from .path_composer import ArcLengthPathComposer, wrap_angle


@dataclass(frozen=True)
class LossWeights:
    waypoint: float = 1.0
    pace: float = 1.0
    residual: float = 1.0
    residual_norm: float = 0.001
    residual_smooth: float = 0.02
    bbox: float = 1.0
    classification: float = 0.5
    semantic: float = 0.25


def expanded_mask(mask, value):
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Validity masks must be binary")
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return torch.broadcast_to(mask, value.shape)


def masked_mean(value, mask):
    mask = expanded_mask(mask, value)
    selected = torch.where(mask.bool(), value, torch.zeros_like(value))
    if not torch.isfinite(selected).all():
        raise ValueError("Valid supervised elements must be finite")
    return selected.sum() / mask.sum().clamp_min(1.0)


def regression(prediction, target, mask):
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    valid = expanded_mask(mask, prediction).bool()
    # Undefined labels may contain NaNs; remove them before evaluating Smooth L1.
    safe_pred = torch.where(valid, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    return masked_mean(F.smooth_l1_loss(safe_pred, safe_target, reduction="none"), mask)


def classification(logits, labels, valid):
    if logits.ndim != 2 or labels.shape != logits.shape[:1] or valid.shape != labels.shape:
        raise ValueError("Expected logits [B,C], labels [B], validity [B]")
    mask = expanded_mask(valid, labels).bool()
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels)).long()
    if torch.any((safe_labels < 0) | (safe_labels >= logits.shape[-1])):
        raise ValueError("Valid class index is out of range")
    safe_logits = torch.where(mask[:, None], logits, torch.zeros_like(logits))
    return masked_mean(F.cross_entropy(safe_logits, safe_labels, reduction="none"), valid)


def joint_loss(predictions, targets, weights=None):
    """Return differentiable total and each constituent loss.

    Targets: waypoint/residual [B,24,3], pace [B], trajectory_valid [B,24],
    reaction_valid [B], bbox [B,4], bbox_valid [B], and a class index [B]
    with its own <name>_valid mask for each of the five semantic objectives.
    """
    w = weights or LossWeights()
    valid = targets["trajectory_valid"]
    if valid.shape != predictions["residual"].shape[:2]:
        raise ValueError("trajectory_valid must be [B,24]")
    residual = predictions["residual"]
    safe = torch.where(expanded_mask(valid, residual).bool(), residual, torch.zeros_like(residual))
    losses = {
        "waypoint": regression(predictions["waypoint"], targets["waypoint"], valid),
        "pace": regression(predictions["pace"], targets["pace"], targets["reaction_valid"]),
        "residual": regression(residual, targets["residual"], valid),
        "residual_norm": masked_mean(safe.abs(), valid),
        "residual_smooth": masked_mean((safe[:, 1:] - safe[:, :-1]).abs(), valid[:, 1:] * valid[:, :-1]),
        "bbox": regression(predictions["bbox"], targets["bbox"], targets["bbox_valid"]),
    }
    for name in SEMANTIC_CLASSES:
        losses[name] = classification(predictions[name], targets[name], targets[f"{name}_valid"])
    losses["action"] = sum(getattr(w, name) * losses[name] for name in ("waypoint", "pace", "residual", "residual_norm", "residual_smooth"))
    losses["semantic"] = w.bbox * losses["bbox"] + w.classification * sum(losses[name] for name in SEMANTIC_CLASSES)
    losses["total"] = losses["action"] + w.semantic * losses["semantic"]
    return losses


@torch.no_grad()
def residual_targets(route, progress, origin, executed_pace, expert_pace, expert_waypoints, reaction_valid):
    """Construct deterministic residual targets relative to expert-paced Path."""
    if reaction_valid.shape != expert_pace.shape:
        raise ValueError("reaction_valid must match expert_pace [B]")
    pace = torch.where(reaction_valid.bool(), expert_pace, torch.ones_like(expert_pace))
    composed = ArcLengthPathComposer()(route, progress, pace, executed_pace, torch.zeros_like(route), path_origin=origin)
    residual = expert_waypoints - composed["scaled_reference"]
    return torch.cat((residual[..., :2], wrap_angle(residual[..., 2:3])), dim=-1)
