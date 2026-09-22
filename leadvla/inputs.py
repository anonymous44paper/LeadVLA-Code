"""Causal RGB preprocessing and policy-visible input boundary (Appendix A.1)."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PolicyInput:
    images: tuple
    prompt: str
    path_reference: np.ndarray
    path_progress: np.ndarray
    path_origin: np.ndarray
    executed_pace: float


class InputBuilder:
    """Feed synchronized FRONT/REAR RGB frames at 10 Hz, once per observation.

    Five past offsets are evenly spaced over the preceding 31 frames. Missing
    early history repeats the first observation; the sixth image is current.
    Reset on every episode. No target IDs or future schedules enter this API.
    """

    def __init__(self):
        self._frames = deque(maxlen=32)
        self._index = 0

    def reset(self):
        self._frames.clear()
        self._index = 0

    def observe(self, front_rgb, rear_rgb):
        pair = []
        for frame in (front_rgb, rear_rgb):
            array = np.asarray(frame)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
                raise ValueError("Images must be nonempty uint8 HxWx3 RGB arrays")
            pair.append(array.copy())
        self._frames.append((self._index, *pair))
        self._index += 1

    def history_indices(self):
        if not self._frames:
            raise ValueError("Observe a synchronized pair before building input")
        current = self._index - 1
        past = np.rint(np.linspace(current - 31, current - 1, 5)).astype(int)
        return [max(0, int(i)) for i in past] + [current]

    def build(self, *, instruction, path_reference, path_progress, executed_pace, path_origin=None):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A natural-language instruction is required")
        route = np.asarray(path_reference, dtype=np.float32)
        progress = np.asarray(path_progress, dtype=np.float32)
        origin = np.zeros(3, dtype=np.float32) if path_origin is None else np.asarray(path_origin, dtype=np.float32)
        if route.shape != (24, 3) or progress.shape != (24,) or origin.shape != (3,):
            raise ValueError("Expected route [24,3], progress [24], origin [3]")
        if not all(np.isfinite(a).all() for a in (route, progress, origin)):
            raise ValueError("Route geometry must be finite")
        if progress[0] < 0 or (np.diff(progress) < 0).any():
            raise ValueError("Relative route progress must be nonnegative and nondecreasing")
        if not np.isfinite(executed_pace) or not 0 <= executed_pace <= 1:
            raise ValueError("Executed Pace must be in [0,1]")
        indices = self.history_indices()
        frames = {i: (front, rear) for i, front, rear in self._frames}
        images = tuple(Image.fromarray(frames[i][view]).resize((384, 384), Image.Resampling.BILINEAR) for view in (0, 1) for i in indices)
        hint = ", ".join(f"({x:.2f}, {y:.2f})" for x, y, _ in route[:10])
        prompt = (
            "You are a guiding robot. The first 6 images are the FRONT history and current view; "
            "the next 6 images are the REAR history and current view. "
            f"{instruction} Route waypoints are in robot-local coordinates, in meters: "
            f"x is forward and y is left. {hint}. Predict the next 24 robot waypoints."
        )
        return PolicyInput(images, prompt, route.copy(), progress.copy(), origin.copy(), float(executed_pace))
