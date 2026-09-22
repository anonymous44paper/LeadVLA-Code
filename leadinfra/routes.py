from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class GridMap:
    planning_grid: np.ndarray
    world_min: np.ndarray
    resolution: float

    def __post_init__(self):
        if self.planning_grid.ndim != 2 or self.planning_grid.dtype != bool or not self.planning_grid.size:
            raise ValueError("planning_grid must be a nonempty boolean [rows,columns] array")
        if np.shape(self.world_min) != (2,) or not np.isfinite(self.world_min).all():
            raise ValueError("world_min must be finite XY")
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("resolution must be positive and finite")

    @classmethod
    def load(cls, path: Path) -> "GridMap":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                planning_grid=data["planning_grid"].astype(bool),
                world_min=np.asarray(data["world_min"], dtype=np.float64),
                resolution=float(data["resolution"]),
            )

    @property
    def clearance(self) -> np.ndarray:
        padded = np.pad(self.planning_grid, 1, constant_values=False)
        return ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * self.resolution

    def cell_to_world(self, cells: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells, dtype=np.float64)
        x = self.world_min[0] + (cells[:, 0] + 0.5) * self.resolution
        y = self.world_min[1] + (cells[:, 1] + 0.5) * self.resolution
        return np.stack([x, y], axis=1)

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        if not np.isfinite(xy).all():
            raise ValueError("Path points must be finite")
        col = np.floor((xy[..., 0] - self.world_min[0]) / self.resolution)
        row = np.floor((xy[..., 1] - self.world_min[1]) / self.resolution)
        return np.stack([col, row], axis=-1).astype(np.int64)

    def contains(self, xy: np.ndarray) -> np.ndarray:
        cells = self.world_to_cell(xy)
        col, row = cells[..., 0], cells[..., 1]
        inside = (
            (row >= 0)
            & (row < self.planning_grid.shape[0])
            & (col >= 0)
            & (col < self.planning_grid.shape[1])
        )
        result = np.zeros(inside.shape, dtype=bool)
        result[inside] = self.planning_grid[row[inside], col[inside]]
        return result


class ArcLengthPath:
    def __init__(self, xy: np.ndarray, closed: bool = False):
        xy = np.asarray(xy, dtype=np.float64)
        if not np.isfinite(xy).all():
            raise ValueError("Path points must be finite")
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
            raise ValueError("path must have shape [N,2] with N >= 2")
        self.closed = bool(closed)
        if self.closed and np.linalg.norm(xy[0] - xy[-1]) > 1e-8:
            xy = np.vstack((xy, xy[0]))
        segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        keep = np.concatenate([[True], segment > 1e-8])
        self.xy = xy[keep]
        segment = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(segment)])
        if self.s[-1] <= 0:
            raise ValueError("path length must be positive")
        if self.closed:
            core = self.xy[:-1]
            delta = np.roll(core, -1, axis=0) - np.roll(core, 1, axis=0)
            dx = delta[:, 0]
            dy = delta[:, 1]
        else:
            dx = np.gradient(self.xy[:, 0], self.s)
            dy = np.gradient(self.xy[:, 1], self.s)
        norm = np.maximum(np.hypot(dx, dy), 1e-9)
        self.tx = dx / norm
        self.ty = dy / norm
        self.yaw_unwrapped = np.unwrap(np.arctan2(self.ty, self.tx))
        if self.closed:
            closure_yaw = self.yaw_unwrapped[0] + 2.0 * np.pi * np.round(
                (self.yaw_unwrapped[-1] - self.yaw_unwrapped[0]) / (2.0 * np.pi)
            )
            self.tx = np.append(self.tx, self.tx[0])
            self.ty = np.append(self.ty, self.ty[0])
            self.yaw_unwrapped = np.append(self.yaw_unwrapped, closure_yaw)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def sample(self, progress, lateral=0.0) -> np.ndarray:
        progress = np.asarray(progress, dtype=np.float64)
        if self.closed:
            progress = np.mod(progress, self.length)
        else:
            progress = np.clip(progress, 0.0, self.length)
        lateral = np.broadcast_to(np.asarray(lateral, dtype=np.float64), progress.shape)
        x = np.interp(progress, self.s, self.xy[:, 0])
        y = np.interp(progress, self.s, self.xy[:, 1])
        yaw = np.interp(progress, self.s, self.yaw_unwrapped)
        x = x - np.sin(yaw) * lateral
        y = y + np.cos(yaw) * lateral
        return np.stack([x, y, _wrap_angle(yaw)], axis=-1)

    def project(self, xy: np.ndarray) -> tuple[float, float]:
        """Nearest-segment projection as (arc progress, signed cross-track error).

        Equidistant overlapping segments select the first segment. Applications
        with ambiguous loops should constrain the active route section upstream.
        """
        point = np.asarray(xy, dtype=np.float64).reshape(2)
        if not np.isfinite(point).all():
            raise ValueError("Projection point must be finite")
        vectors = np.diff(self.xy, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        fractions = np.clip(np.sum((point - self.xy[:-1]) * vectors, axis=1) / lengths ** 2, 0, 1)
        projected = self.xy[:-1] + fractions[:, None] * vectors
        index = int(np.argmin(np.sum((projected - point) ** 2, axis=1)))
        tangent = vectors[index] / lengths[index]
        error = point - projected[index]
        lateral = tangent[0] * error[1] - tangent[1] * error[0]
        return float(self.s[index] + fractions[index] * lengths[index]), float(lateral)

    def unwrap_progress(self, projected_progress: float, previous_progress: float) -> float:
        """Lift closed-path progress onto the lap nearest the previous unwrapped value."""
        projected = float(projected_progress)
        if not self.closed:
            return projected
        lap = round((float(previous_progress) - projected) / self.length)
        return projected + lap * self.length


def _wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _snap_anchor(anchor, traversable: np.ndarray, clearance: np.ndarray, radius: int = 20):
    col0, row0 = (int(anchor[0]), int(anchor[1]))
    candidates = []
    for row in range(max(0, row0 - radius), min(traversable.shape[0], row0 + radius + 1)):
        for col in range(max(0, col0 - radius), min(traversable.shape[1], col0 + radius + 1)):
            if traversable[row, col]:
                distance = math.hypot(col - col0, row - row0)
                candidates.append((distance, -float(clearance[row, col]), col, row))
    if not candidates:
        raise ValueError(f"no traversable cell near anchor {(col0, row0)}")
    _, _, col, row = min(candidates)
    return col, row


def _astar(traversable: np.ndarray, clearance: np.ndarray, start, goal):
    width = traversable.shape[1]
    start_id = start[1] * width + start[0]
    goal_id = goal[1] * width + goal[0]
    queue = [(0.0, start_id)]
    came_from = {start_id: -1}
    cost_so_far = {start_id: 0.0}
    neighbors = (
        (-1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, math.sqrt(2.0)),
    )
    while queue:
        _, current_id = heapq.heappop(queue)
        if current_id == goal_id:
            break
        row, col = divmod(current_id, width)
        current_cost = cost_so_far[current_id]
        for dc, dr, step in neighbors:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < traversable.shape[0] and 0 <= nc < width and traversable[nr, nc]):
                continue
            if dc and dr and not (traversable[row, nc] and traversable[nr, col]):
                continue
            next_id = nr * width + nc
            clearance_cost = 0.15 / max(float(clearance[nr, nc]), 0.1)
            new_cost = current_cost + step * (1.0 + clearance_cost)
            if new_cost >= cost_so_far.get(next_id, float("inf")):
                continue
            cost_so_far[next_id] = new_cost
            came_from[next_id] = current_id
            heuristic = math.hypot(goal[0] - nc, goal[1] - nr)
            heapq.heappush(queue, (new_cost + heuristic, next_id))
    if goal_id not in came_from:
        raise RuntimeError(f"no route between {start} and {goal}")
    ids = []
    current_id = goal_id
    while current_id != -1:
        row, col = divmod(current_id, width)
        ids.append((col, row))
        current_id = came_from[current_id]
    return np.asarray(ids[::-1], dtype=np.int64)


def _resample_xy(xy: np.ndarray, spacing: float, closed: bool = False) -> np.ndarray:
    if closed and np.linalg.norm(xy[0] - xy[-1]) > 1e-8:
        xy = np.vstack((xy, xy[0]))
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segment)])
    segment_count = max(1, int(np.ceil(s[-1] / float(spacing))))
    samples = np.linspace(0.0, s[-1], segment_count + 1)
    result = np.stack([np.interp(samples, s, xy[:, 0]), np.interp(samples, s, xy[:, 1])], axis=1)
    if closed:
        result[-1] = result[0]
    return result


def plan_route(
    grid_map: GridMap,
    anchors_cells,
    min_clearance: float = 1.0,
    spacing: float = 0.1,
    smooth_sigma_cells: float = 2.0,
    closed: bool = False,
) -> ArcLengthPath:
    if not math.isfinite(spacing) or spacing <= 0 or min_clearance < 0 or not math.isfinite(min_clearance):
        raise ValueError("Spacing must be positive; clearance finite and nonnegative")
    if not math.isfinite(smooth_sigma_cells) or smooth_sigma_cells < 0:
        raise ValueError("Smoothing scale must be finite and nonnegative")
    anchors_cells = list(anchors_cells)
    if len(anchors_cells) < 2:
        raise ValueError("At least two anchors are required")
    if closed and tuple(anchors_cells[0]) != tuple(anchors_cells[-1]):
        anchors_cells.append(anchors_cells[0])
    clearance = grid_map.clearance
    traversable = grid_map.planning_grid & (clearance >= min_clearance)
    anchors = [_snap_anchor(a, traversable, clearance) for a in anchors_cells]
    pieces = []
    for start, goal in zip(anchors[:-1], anchors[1:]):
        piece = _astar(traversable, clearance, start, goal)
        pieces.append(piece if not pieces else piece[1:])
    cells = np.concatenate(pieces, axis=0)
    xy = grid_map.cell_to_world(cells)
    if smooth_sigma_cells > 0:
        if closed:
            core = xy[:-1] if np.linalg.norm(xy[0] - xy[-1]) <= 1e-8 else xy
            smooth = np.empty_like(core)
            smooth[:, 0] = ndimage.gaussian_filter1d(core[:, 0], smooth_sigma_cells, mode="wrap")
            smooth[:, 1] = ndimage.gaussian_filter1d(core[:, 1], smooth_sigma_cells, mode="wrap")
            xy = np.vstack((smooth, smooth[0]))
        else:
            smooth = np.empty_like(xy)
            smooth[:, 0] = ndimage.gaussian_filter1d(xy[:, 0], smooth_sigma_cells, mode="nearest")
            smooth[:, 1] = ndimage.gaussian_filter1d(xy[:, 1], smooth_sigma_cells, mode="nearest")
            smooth[0], smooth[-1] = xy[0], xy[-1]
            xy = smooth
    result = ArcLengthPath(_resample_xy(xy, spacing, closed=closed), closed=closed)
    from .validation import validate_route
    validate_route(result, grid_map, min_clearance=min_clearance)
    return result
