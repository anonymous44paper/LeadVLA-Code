"""Frozen offline-evaluation manifests, identities, and completeness checks."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import PurePosixPath

from .contract import Contract
from .metrics import CELLS, Event


def digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_manifest(manifest, *, require_full_suite=False):
    if set(manifest) != {"schema_version", "episodes"} or manifest["schema_version"] != 1:
        raise ValueError("Manifest must contain schema_version=1 and episodes")
    if not isinstance(manifest["episodes"], list) or not manifest["episodes"]:
        raise ValueError("Manifest requires a nonempty episode list")
    ids, files, cases, counts = set(), set(), defaultdict(list), Counter()
    required = {"episode_id", "case_id", "cell", "variant", "contract", "events", "trace_file"}
    for episode in manifest["episodes"]:
        if set(episode) != required:
            raise ValueError("Invalid manifest episode fields")
        for field in ("episode_id", "case_id"):
            if not isinstance(episode[field], str) or not episode[field] or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in episode[field]):
                raise ValueError("Episode and case IDs must be portable identifiers")
        if episode["episode_id"] in ids:
            raise ValueError("Duplicate episode ID")
        ids.add(episode["episode_id"])
        if episode["cell"] not in CELLS:
            raise ValueError("Unknown cell")
        file = episode["trace_file"]
        if not isinstance(file, str) or not file:
            raise ValueError("trace_file must be a relative JSON path")
        path = PurePosixPath(file)
        if path.is_absolute() or ".." in path.parts or ":" in file or "\\" in file or path.suffix != ".json":
            raise ValueError("trace_file must be a relative JSON path")
        if str(path) in files:
            raise ValueError("Each physical rollout must use a distinct trace file")
        files.add(str(path))
        Contract(**episode["contract"])
        if not episode["events"]:
            raise ValueError("Every episode needs a frozen event schedule")
        for event in episode["events"]:
            Event(**event)
        cases[(episode["cell"], episode["case_id"])].append(episode["variant"])
        counts[episode["cell"]] += 1
    for (cell, _), variants in cases.items():
        if sorted(variants) != (["S"] if cell.endswith("-S") else ["A", "B"]):
            raise ValueError("Cases must contain S or a complete A/B pair")
    if require_full_suite:
        expected = {cell: 12 if cell.endswith("-S") else 24 for cell in CELLS}
        if dict(counts) != expected:
            raise ValueError("Full suite requires 12 S and 24 M physical episodes per difficulty")
    return {"sha256": digest(manifest), "physical_episodes": len(ids), "logical_cases": len(cases), "cell_counts": dict(counts)}


def validate_episode_identity(episode, specification):
    for field in ("episode_id", "case_id", "cell", "variant", "contract", "events"):
        if episode[field] != specification[field]:
            raise ValueError(f"Trace differs from frozen manifest: {field}")
