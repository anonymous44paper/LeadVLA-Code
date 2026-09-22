"""Deterministic, asset-independent composition of episode specifications.

This does not execute behavior programs or claim geometric compatibility.
The caller supplies compatibility-validated combinations and semantic asset IDs.
"""

import argparse
import hashlib
import json
import random
from itertools import product

from leadbench.contract import Contract

MOTIONS = (
    "steady_tracking", "mild_lag", "persistent_lag", "short_pause",
    "gradual_slowdown", "multi_hesitate", "stop_recover", "start_failure",
)
PRESENCES = (
    "normal_visible", "stationary_fall_behind", "temporary_occlusion",
    "target_absent_distractor_visible", "empty_rear_reacquisition",
)
FORMATIONS = tuple(f"{d}_{b}" for d, b in product(("near", "far"), ("left", "center", "right")))


def semantic_id(value):
    """Reject paths/URLs; asset resolution belongs to a separate private registry."""
    if not isinstance(value, str) or not value or not all(c.isascii() and (c.isalnum() or c in "_-") for c in value):
        raise ValueError("Asset identifiers must contain only ASCII letters, digits, '_' or '-'")
    return value


def compose(config):
    required = {"seed", "episodes", "environments", "routes", "robots", "actors", "scenes", "sensors", "compatible_interactions"}
    if set(config) != required:
        raise ValueError(f"Expected configuration fields: {sorted(required)}")
    if type(config["seed"]) is not int or type(config["episodes"]) is not int or config["episodes"] <= 0:
        raise ValueError("seed and positive episodes must be integers")
    for key in ("environments", "routes", "robots", "actors", "scenes", "sensors"):
        if not isinstance(config[key], list) or not config[key]:
            raise ValueError(f"{key} must be a nonempty list")
        for value in config[key]:
            semantic_id(value)
    if not config["compatible_interactions"]:
        raise ValueError("Provide explicitly compatible interactions; a Cartesian product is not a validity check")
    for item in config["compatible_interactions"]:
        if set(item) != {"motion", "presence", "formation", "multi_person"}:
            raise ValueError("Invalid interaction fields")
        if item["motion"] not in MOTIONS or item["presence"] not in PRESENCES or item["formation"] not in FORMATIONS:
            raise ValueError("Unknown interaction setting")
        if type(item["multi_person"]) is not bool:
            raise ValueError("multi_person must be boolean")
        if item["multi_person"] and len(set(config["actors"])) < 2:
            raise ValueError("Multi-person episodes require two distinct appearance IDs")
        if item["presence"] == "target_absent_distractor_visible" and not item["multi_person"]:
            raise ValueError("This presence condition requires a distractor")
        Contract(*item["formation"].split("_"))
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    rng = random.Random(config["seed"])
    episodes = []
    for index in range(config["episodes"]):
        interaction = rng.choice(config["compatible_interactions"])
        people = rng.sample(sorted(set(config["actors"])), 2 if interaction["multi_person"] else 1)
        episodes.append({
            "episode_id": f"example_{index:04d}",
            "environment": rng.choice(config["environments"]),
            "nominal_route": rng.choice(config["routes"]),
            "robot": rng.choice(config["robots"]),
            "scene": rng.choice(config["scenes"]),
            "sensors": list(config["sensors"]),
            "target_appearance": people[0],
            "distractor_appearances": people[1:],
            "interaction": dict(interaction),
        })
    return {"schema_version": 1, "kind": "composition_example", "config_sha256": hashlib.sha256(canonical.encode()).hexdigest(), "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Composition JSON; paths are not stored in the manifest")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        result = compose(json.load(handle))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
