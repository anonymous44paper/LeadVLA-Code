"""Fixed interaction contract from Appendix B.3, Table 13."""

from dataclasses import dataclass
import math

DISTANCES = {"near": (1.10, 1.60), "far": (2.60, 4.00)}
BEARINGS = {"left": (15.0, 25.0), "center": (-5.0, 5.0), "right": (-25.0, -15.0)}


@dataclass(frozen=True)
class Contract:
    distance: str = "far"
    bearing: str = "center"

    def __post_init__(self):
        if self.distance not in DISTANCES or self.bearing not in BEARINGS:
            raise ValueError("Unknown distance or bearing relation")

    def distance_error(self, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Target distance must be finite and nonnegative")
        low, high = DISTANCES[self.distance]
        return max(low - value, 0.0, value - high)

    def contains(self, distance: float, bearing_deg: float) -> bool:
        if not math.isfinite(bearing_deg):
            raise ValueError("Target bearing must be finite")
        low, high = BEARINGS[self.bearing]
        return self.distance_error(distance) == 0 and low <= bearing_deg <= high
