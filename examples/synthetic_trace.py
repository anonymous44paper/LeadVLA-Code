"""Print a synthetic unit-test trace, NOT a policy rollout or paper result."""

from dataclasses import asdict
import json

from leadbench.metrics import Frame


def example():
    frames = [asdict(Frame(
        time=i / 10, x=i / 20, y=0.0, route_progress=i / 20,
        route_deviation=0.0, target_distance=3.0, target_bearing_deg=0.0,
        target_visible=True, formation_valid=True, interference=False,
        robot_goal_distance=5.0, target_goal_distance=5.0,
    )) for i in range(41)]
    return {"schema_version": 1, "episodes": [{
        "episode_id": "synthetic_001", "case_id": "synthetic_case", "cell": "Core-S", "variant": "S",
        "contract": {"distance": "far", "bearing": "center"}, "termination": "route_timeout",
        "frames": frames, "events": [{"source": "target", "response": "lead", "onset": 1.0, "grace": 0.0}],
    }]}


if __name__ == "__main__":
    print(json.dumps(example(), indent=2, allow_nan=False))
