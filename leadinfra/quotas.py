"""Exact integer allocation for balanced composition plans."""

import math


def largest_remainder(total, weights):
    """Allocate total items proportionally; break ties by semantic key.

    This utility does not imply exhaustive coverage of the 8 x 5 x 6 interaction
    combinations, which still require physical and semantic compatibility checks.
    """
    if type(total) is not int or total < 0 or not weights:
        raise ValueError("Expected nonnegative integer total and nonempty weights")
    if any(not isinstance(key, str) for key in weights):
        raise ValueError("Weight names must be strings")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    maximum = max(weights.values())
    if maximum <= 0:
        raise ValueError("At least one weight must be positive")
    normalized = {key: value / maximum for key, value in weights.items()}
    norm = sum(normalized.values())
    exact = {key: total * value / norm for key, value in normalized.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    ranked = sorted(exact, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in ranked[:total - sum(allocated.values())]:
        allocated[key] += 1
    return allocated
