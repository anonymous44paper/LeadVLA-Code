"""Optimizer groups and stage-transition utilities (Appendix A.4)."""

import math

import torch


def freeze_vision_tower(vision_tower):
    for parameter in vision_tower.parameters():
        parameter.requires_grad_(False)


def optimizer_for(model):
    """Frozen vision parameters are excluded; each trainable parameter occurs once."""
    groups = {"backbone": [], "residual": [], "control": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "backbone" if name.startswith("backbone.") else "residual" if name.startswith("heads.residual_decoder.") else "control"
        groups[group].append(parameter)
    rates = {"backbone": 2e-5, "residual": 1e-5, "control": 1e-4}
    parameters = [{"params": values, "lr": rates[name], "name": name} for name, values in groups.items() if values]
    return torch.optim.AdamW(parameters, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-8)


def constant_with_warmup(optimizer, warmup_steps):
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps) if warmup_steps else 1.0)


def start_stage(model, *, warmup_steps, initial_weights=None):
    """Load model weights only; always create fresh optimizer and scheduler state."""
    if initial_weights is not None:
        model.load_state_dict(initial_weights, strict=True)
    optimizer = optimizer_for(model)
    return optimizer, constant_with_warmup(optimizer, warmup_steps)


def train_step(model, loss, optimizer, scheduler):
    """One optimizer step; distributed/AMP orchestration belongs to the caller."""
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Expected a finite scalar loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return float(norm)


class ValidationSelector:
    """Select the checkpoint with minimum finite validation total loss."""

    def __init__(self):
        self.best = math.inf

    def improves(self, total_loss):
        if not math.isfinite(total_loss):
            raise ValueError("Validation total loss must be finite")
        if total_loss < self.best:
            self.best = float(total_loss)
            return True
        return False
