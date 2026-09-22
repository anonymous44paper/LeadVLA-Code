"""Replay exactly the physical rollouts named by a frozen offline manifest."""

import argparse
import json
from pathlib import Path

from .manifest import validate_manifest, validate_episode_identity
from .metrics import evaluate_episode
from .report import aggregate


def replay(manifest, trace_root, *, expected_digest=None, require_full_suite=False):
    metadata = validate_manifest(manifest, require_full_suite=require_full_suite)
    if expected_digest is not None and expected_digest != metadata["sha256"]:
        raise ValueError("Frozen manifest digest does not match")
    root = Path(trace_root).resolve()
    results = []
    for specification in manifest["episodes"]:
        path = (root / specification["trace_file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Trace resolves outside the supplied root")
        with path.open(encoding="utf-8") as handle:
            episode = json.load(handle)
        validate_episode_identity(episode, specification)
        results.append(evaluate_episode(episode))
    return {"manifest": metadata, "summary": aggregate(results), "episodes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("trace_root")
    parser.add_argument("--expected-digest")
    parser.add_argument("--require-full-suite", action="store_true")
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    print(json.dumps(replay(manifest, args.trace_root, expected_digest=args.expected_digest, require_full_suite=args.require_full_suite), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
