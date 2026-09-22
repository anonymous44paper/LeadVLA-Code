"""Append-only episode recorder; refuses to overwrite existing outputs."""

from dataclasses import asdict
import json
from pathlib import Path

from leadbench.metrics import Frame
from .compose import semantic_id


class EpisodeRecorder:
    """Write evaluator geometry as JSONL; images/behavior programs are separate."""

    def __init__(self, output_root, episode_id):
        semantic_id(episode_id)
        self.directory = Path(output_root) / episode_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self._handle = (self.directory / "frames.jsonl").open("x", encoding="utf-8")
        self._last_time = None
        self.frames = 0

    def append(self, frame):
        if not isinstance(frame, Frame):
            raise TypeError("Expected a validated physical Frame")
        if self._last_time is not None and abs(frame.time - self._last_time - 0.1) > 1e-6:
            raise ValueError("Recorder requires contiguous 10 Hz frames")
        self._handle.write(json.dumps(asdict(frame), allow_nan=False) + "\n")
        self._handle.flush()
        self._last_time = frame.time
        self.frames += 1

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
