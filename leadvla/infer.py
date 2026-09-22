"""Single-query inference using a user-supplied model backend and checkpoint."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from .inputs import InputBuilder
from .inference import predict_waypoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, help="Trusted Python module:factory; factory(checkpoint=Path) returns a PolicyBackend")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--observations", required=True, help="NPZ: front_rgb/rear_rgb [N,H,W,3], path_reference [24,3], path_progress [24], path_origin [3]")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--executed-pace", required=True, type=float)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint not found. Trained weights are not included in this snapshot.")
    module, separator, factory_name = args.backend.partition(":")
    if not separator or not module or not factory_name:
        parser.error("Expected --backend module:factory")
    builder = InputBuilder()
    with np.load(args.observations, allow_pickle=False) as observations:
        front, rear = observations["front_rgb"], observations["rear_rgb"]
        if front.ndim != 4 or rear.ndim != 4 or len(front) != len(rear) or not len(front):
            parser.error("Expected equally long, nonempty synchronized RGB histories")
        for f, r in zip(front, rear):
            builder.observe(f, r)
        sample = builder.build(
            instruction=args.instruction,
            path_reference=observations["path_reference"],
            path_progress=observations["path_progress"],
            path_origin=observations["path_origin"],
            executed_pace=args.executed_pace,
        )
    factory = getattr(importlib.import_module(module), factory_name)
    backend = factory(checkpoint=args.checkpoint)
    prediction = predict_waypoints(backend, sample)
    print(json.dumps({key: value.cpu().tolist() for key, value in prediction.items()}, allow_nan=False))


if __name__ == "__main__":
    main()
