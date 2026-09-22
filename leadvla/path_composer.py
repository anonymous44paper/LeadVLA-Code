from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi)."""
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def unwrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Unwrap a batched angle sequence along its last dimension."""
    if angle.shape[-1] < 2:
        return angle
    delta = wrap_angle(angle[..., 1:] - angle[..., :-1])
    return torch.cat((angle[..., :1], angle[..., :1] + torch.cumsum(delta, dim=-1)), dim=-1)


@dataclass(frozen=True)
class ComposerConfig:
    horizon: int = 24
    nominal_speed_mps: float = 0.75
    trajectory_dt: float = 0.2
    acceleration_mps2: float = 0.5
    deceleration_mps2: float = 0.6
    prepend_robot_origin: bool = True
    eps: float = 1e-6

    def __post_init__(self):
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        for value in (self.nominal_speed_mps, self.trajectory_dt, self.acceleration_mps2, self.deceleration_mps2, self.eps):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Composer rates, interval and epsilon must be finite and positive")


class ArcLengthPathComposer(nn.Module):
    """Compose waypoints by progressing along route arc length.

    The model predicts one target scale. A deterministic rate limiter expands it
    into a horizon profile. The profile scales arc-length increments, never the
    robot-local XY coordinates.
    """

    def __init__(self, config: ComposerConfig | None = None, **kwargs) -> None:
        super().__init__()
        self.config = config or ComposerConfig(**kwargs)

    def rate_limited_profile(
        self,
        scale_target: torch.Tensor,
        scale_executed: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        if scale_target.numel() != scale_executed.numel():
            raise ValueError("Target and executed Pace must have the same batch size")
        if not torch.isfinite(scale_target).all() or not torch.isfinite(scale_executed).all():
            raise ValueError("Pace must be finite")
        target = scale_target.reshape(-1).clamp(0.0, 1.0)
        current = scale_executed.reshape(-1).clamp(0.0, 1.0)
        target_speed = target * cfg.nominal_speed_mps
        speed = current * cfg.nominal_speed_mps
        up_step = cfg.acceleration_mps2 * cfg.trajectory_dt
        down_step = cfg.deceleration_mps2 * cfg.trajectory_dt
        profile = []
        for _ in range(cfg.horizon):
            delta = target_speed - speed
            delta = torch.clamp(delta, min=-down_step, max=up_step)
            speed = speed + delta
            profile.append((speed / cfg.nominal_speed_mps).clamp(0.0, 1.0))
        return torch.stack(profile, dim=1)

    def integrated_progress(
        self,
        path_progress: torch.Tensor,
        scale_profile: torch.Tensor,
    ) -> torch.Tensor:
        if path_progress.shape != scale_profile.shape:
            raise ValueError(
                f"path_progress and scale_profile must match, got "
                f"{tuple(path_progress.shape)} and {tuple(scale_profile.shape)}"
            )
        zeros = torch.zeros_like(path_progress[:, :1])
        nominal_delta = torch.diff(torch.cat((zeros, path_progress), dim=1), dim=1)
        if torch.any(nominal_delta < -self.config.eps):
            raise ValueError("path_progress must be non-decreasing")
        return torch.cumsum(nominal_delta.clamp_min(0.0) * scale_profile, dim=1)

    def interpolate(
        self,
        path_reference: torch.Tensor,
        path_progress: torch.Tensor,
        query_progress: torch.Tensor,
        path_origin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if path_reference.ndim != 3 or path_reference.shape[-1] != 3:
            raise ValueError(f"path_reference must be [B,T,3], got {tuple(path_reference.shape)}")
        if path_progress.shape != path_reference.shape[:2]:
            raise ValueError("path_progress must match path_reference [B,T]")
        if query_progress.shape != path_progress.shape:
            raise ValueError("query_progress must match path_progress [B,T]")

        if self.config.prepend_robot_origin:
            if path_origin is None:
                origin_pose = torch.zeros_like(path_reference[:, :1])
            else:
                if path_origin.shape != (path_reference.shape[0], 3):
                    raise ValueError("path_origin must be [B,3]")
                origin_pose = path_origin.unsqueeze(1)
            origin_progress = torch.zeros_like(path_progress[:, :1])
            knots = torch.cat((origin_pose, path_reference), dim=1)
            progress = torch.cat((origin_progress, path_progress), dim=1)
        else:
            knots = path_reference
            progress = path_progress

        if torch.any(torch.diff(progress, dim=1) < -self.config.eps):
            raise ValueError("path_progress knots must be non-decreasing")

        query = torch.minimum(query_progress.clamp_min(0.0), progress[:, -1:])
        right = torch.searchsorted(progress.contiguous(), query.contiguous(), right=False)
        right = right.clamp(min=1, max=progress.shape[1] - 1)
        left = right - 1

        left_s = torch.gather(progress, 1, left)
        right_s = torch.gather(progress, 1, right)
        denom = (right_s - left_s).clamp_min(self.config.eps)
        ratio = ((query - left_s) / denom).clamp(0.0, 1.0)

        xy = knots[..., :2]
        left_xy = torch.gather(xy, 1, left.unsqueeze(-1).expand(-1, -1, 2))
        right_xy = torch.gather(xy, 1, right.unsqueeze(-1).expand(-1, -1, 2))
        interp_xy = left_xy + ratio.unsqueeze(-1) * (right_xy - left_xy)

        yaw = unwrap_angle(knots[..., 2])
        left_yaw = torch.gather(yaw, 1, left)
        right_yaw = torch.gather(yaw, 1, right)
        interp_yaw = wrap_angle(left_yaw + ratio * (right_yaw - left_yaw))
        return torch.cat((interp_xy, interp_yaw.unsqueeze(-1)), dim=-1)

    def forward(
        self,
        path_reference: torch.Tensor,
        path_progress: torch.Tensor,
        scale_target: torch.Tensor,
        scale_executed: torch.Tensor,
        residual: torch.Tensor,
        *,
        scale_profile: torch.Tensor | None = None,
        path_origin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if path_reference.ndim != 3 or path_reference.shape[-1] != 3 or path_reference.shape[0] < 1:
            raise ValueError("path_reference must be a nonempty [B,T,3] tensor")
        if path_progress.shape != path_reference.shape[:2]:
            raise ValueError("path_progress must match [B,T]")
        for value in (path_reference, path_progress, residual, scale_target, scale_executed):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Composer inputs must be finite floating-point tensors")
            if value.device != path_reference.device:
                raise ValueError("Composer inputs must share a device")
        if scale_target.numel() != path_reference.shape[0] or scale_executed.numel() != path_reference.shape[0]:
            raise ValueError("Pace must have one value per batch element")
        if path_origin is not None and (not torch.isfinite(path_origin).all() or path_origin.device != path_reference.device):
            raise ValueError("path_origin must be finite and on the route device")
        if path_reference.shape[1] != self.config.horizon:
            raise ValueError(
                f"composer horizon={self.config.horizon}, path has {path_reference.shape[1]} steps"
            )
        if residual.shape != path_reference.shape:
            raise ValueError("residual must match path_reference [B,T,3]")
        profile = scale_profile
        if profile is None:
            profile = self.rate_limited_profile(scale_target, scale_executed)
        elif not torch.isfinite(profile).all() or torch.any((profile < 0) | (profile > 1)):
            raise ValueError("Explicit Pace profile must be finite and in [0,1]")
        query_progress = self.integrated_progress(path_progress, profile)
        scaled_reference = self.interpolate(
            path_reference, path_progress, query_progress, path_origin=path_origin
        )
        waypoint = scaled_reference + residual
        waypoint = torch.cat((waypoint[..., :2], wrap_angle(waypoint[..., 2:3])), dim=-1)
        return {
            "scale_profile": profile,
            "query_progress": query_progress,
            "scaled_reference": scaled_reference,
            "waypoint": waypoint,
        }
