"""Post-smoothing route checks on the navigability grid."""

import numpy as np


def validate_route(route, grid, *, min_clearance=0.0, max_curvature=None):
    spacing = min(grid.resolution / 4, 0.1)
    progress = np.linspace(0, route.length, max(2, int(np.ceil(route.length / spacing)) + 1))
    samples = route.sample(progress)
    if not np.all(grid.contains(samples[:, :2])):
        raise ValueError("Route leaves navigable space")
    cells = grid.world_to_cell(samples[:, :2])
    clearances = grid.clearance[cells[:, 1], cells[:, 0]]
    if np.any(clearances < min_clearance):
        raise ValueError("Route violates requested clearance")
    curvature = np.abs(np.gradient(np.unwrap(samples[:, 2]), progress))
    if max_curvature is not None and np.any(curvature > max_curvature):
        raise ValueError("Route exceeds curvature bound")
    return {"length_m": route.length, "minimum_clearance_m": float(clearances.min()), "maximum_curvature_rad_per_m": float(curvature.max()), "samples": len(samples)}


def route_distance(first, second, spacing=0.25):
    """Symmetric sampled Chamfer distance for geometric redundancy checks."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive")
    def points(route):
        return route.sample(np.linspace(0, route.length, max(2, int(np.ceil(route.length / spacing)) + 1)))[:, :2]
    a, b = points(first), points(second)
    def nearest(source, target):
        # Chunk source points to bound memory on long routes.
        result = []
        for offset in range(0, len(source), 256):
            distance = np.linalg.norm(source[offset:offset + 256, None] - target[None], axis=-1)
            result.extend(distance.min(axis=1))
        return np.mean(result)
    return float((nearest(a, b) + nearest(b, a)) / 2)
