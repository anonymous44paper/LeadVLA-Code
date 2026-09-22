"""Evaluate a portable JSON trace without CARLA, checkpoints, or network access."""

import argparse
import json
from .metrics import evaluate_episode
from .report import aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="JSON object with schema_version=1 and episodes")
    args = parser.parse_args()
    with open(args.trace, encoding="utf-8") as handle:
        data = json.load(handle)
    if set(data) != {"schema_version", "episodes"} or data["schema_version"] != 1:
        raise ValueError("Unsupported trace schema")
    results = [evaluate_episode(episode) for episode in data["episodes"]]
    print(json.dumps({"schema_version": 1, "summary": aggregate(results), "episodes": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
