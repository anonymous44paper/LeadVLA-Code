"""Follower-label encoding and validity masks for recorded observations."""

import math
import numpy as np

from leadbench.contract import DISTANCES

CLASSES = {
    "lateral": ("left", "center", "right"),
    "distance": ("too_close", "in_range", "too_far"),
    "observation": ("tracked", "missing", "lost", "reacquiring"),
    "motion": ("moving", "slowing", "stopped"),
    "formation": ("acquiring", "maintaining", "correcting"),
}


def normalized_box(xyxy, width, height):
    box = np.asarray(xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or width <= 0 or height <= 0:
        raise ValueError("Expected a finite XYXY box and positive image dimensions")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Box must use valid half-open image coordinates")
    return np.asarray(((x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height), dtype=np.float32)


def spatial_labels(distance_m, bearing_deg, requested_distance):
    if not math.isfinite(distance_m) or distance_m < 0 or not math.isfinite(bearing_deg):
        raise ValueError("Geometry must be finite and distance nonnegative")
    low, high = DISTANCES[requested_distance]
    distance = "too_close" if distance_m < low else "too_far" if distance_m > high else "in_range"
    lateral = "left" if bearing_deg > 5 else "right" if bearing_deg < -5 else "center"
    return {"distance": distance, "lateral": lateral}


def encode_semantics(labels, *, bbox_xyxy=None, image_size=(384, 384)):
    """Encode realized semantic labels; None means undefined, not class zero.

    Observation/motion/formation labels must be provided from the causal data
    recorder. Hidden motion-program names are not accepted as semantic labels.
    """
    if set(labels) != set(CLASSES):
        raise ValueError("Provide all five semantic labels, using None when undefined")
    output = {}
    for name, classes in CLASSES.items():
        value = labels[name]
        if value is not None and value not in classes:
            raise ValueError(f"Unknown {name} label")
        output[name] = np.int64(-1 if value is None else classes.index(value))
        output[f"{name}_valid"] = np.float32(value is not None)
    output["bbox"] = np.zeros(4, np.float32) if bbox_xyxy is None else normalized_box(bbox_xyxy, *image_size)
    output["bbox_valid"] = np.float32(bbox_xyxy is not None)
    return output
