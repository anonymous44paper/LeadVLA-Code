"""Physical closed-loop metrics; see docs/metrics.md for sampling conventions.

Inputs are evaluator-only physical traces, never model-specific action outputs.
"""

from dataclasses import dataclass
import math
from statistics import mean

from .contract import Contract

RESPONSES = ("lead", "slow", "wait", "resume")
TERMINATIONS = ("success", "sustained_collision", "terminal_sync_timeout", "route_stall", "route_timeout")
CELLS = tuple(f"{difficulty}-{people}" for difficulty in ("Core", "Easy", "Constrained") for people in ("S", "M"))


@dataclass(frozen=True)
class Frame:
    time: float
    x: float
    y: float
    route_progress: float
    route_deviation: float
    target_distance: float
    target_bearing_deg: float
    target_visible: bool
    formation_valid: bool
    interference: bool
    robot_goal_distance: float
    target_goal_distance: float

    def __post_init__(self):
        for name in ("time", "x", "y", "route_progress", "route_deviation", "target_distance", "target_bearing_deg", "robot_goal_distance", "target_goal_distance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if min(self.time, self.route_deviation, self.target_distance, self.robot_goal_distance, self.target_goal_distance) < 0:
            raise ValueError("Time and distance fields cannot be negative")
        for name in ("target_visible", "formation_valid", "interference"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class Event:
    source: str
    response: str
    onset: float
    grace: float

    def __post_init__(self):
        if self.source not in ("target", "distractor") or self.response not in RESPONSES:
            raise ValueError("Invalid event source or response")
        if self.source == "distractor" and self.response != "lead":
            raise ValueError("Distractor control events must expect lead")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in (self.onset, self.grace)):
            raise ValueError("Event onset and grace must be finite and nonnegative")


def validate_trace(frames):
    if not frames:
        raise ValueError("A physical trace must contain frames")
    for previous, current in zip(frames, frames[1:]):
        if not math.isclose(current.time - previous.time, 0.1, abs_tol=1e-6):
            raise ValueError("Reference evaluator requires contiguous 10 Hz samples")


def route_fidelity(frames):
    retained = []
    for frame in frames:
        if not retained or math.hypot(frame.x - retained[-1].x, frame.y - retained[-1].y) >= 0.05:
            retained.append(frame)
    if len(retained) < 2:
        return 0.0
    quality = mean(math.exp(-max(0.0, f.route_deviation - 1.0) / 0.5) for f in retained)
    forward = mean(b.route_progress - a.route_progress >= -0.05 - 1e-9 for a, b in zip(retained, retained[1:]))
    return quality * forward


def social_contract_score(frames, contract):
    valid = [f for f in frames if f.formation_valid]
    if not valid:
        raise ValueError("No frozen formation-valid frames; SCS is undefined")
    return mean(f.target_visible and contract.contains(f.target_distance, f.target_bearing_deg) for f in valid)


def lead_success(frames, termination):
    if termination not in TERMINATIONS:
        raise ValueError("Infrastructure failures must be retried, not scored")
    if termination != "success":
        return 0.0
    start = None
    for f in frames:
        joint = f.robot_goal_distance <= 1.0 and f.target_goal_distance <= 1.0 and 0.8 <= f.target_distance <= 4.0
        if not joint:
            start = None
        elif start is None:
            start = f.time
    # A success record must end with an uninterrupted terminal hold.
    return float(start is not None and frames[-1].time - start >= 0.5 - 1e-9)


def response_score(frames, event, contract):
    start, end = event.onset + event.grace, event.onset + event.grace + 2.0
    if frames[-1].time < end - 1e-9:
        return 0.0  # Preserve scheduled events after early termination.
    pre = [f for f in frames if event.onset - 0.5 - 1e-9 <= f.time < event.onset - 1e-9 and not f.interference]
    post = [f for f in frames if end - 0.5 + 1e-9 < f.time <= end + 1e-9 and not f.interference]
    window = [f for f in frames if start - 1e-9 <= f.time <= end + 1e-9]
    intervals = [(a, b) for a, b in zip(window, window[1:]) if not a.interference and not b.interference]
    if not pre or not post or not intervals:
        raise ValueError("Frozen event has no usable pre/post/response support")
    before = mean(contract.distance_error(f.target_distance) for f in pre)
    after = mean(contract.distance_error(f.target_distance) for f in post)
    progress = max(0.0, sum(b.route_progress - a.route_progress for a, b in intervals))
    duration = sum(b.time - a.time for a, b in intervals)
    qd = math.exp(-after / 0.75)
    qc = math.exp(-max(0.0, after - before) / 0.75)
    qp = min(1.0, max(0.0, progress / duration / 0.5))
    qs = math.exp(-progress / 0.5)
    if event.response in ("lead", "resume"):
        return math.sqrt(qp * qd)
    if event.response == "slow":
        return math.sqrt(qd * qc)
    return (qd * qc * qs) ** (1.0 / 3.0)


def macro_response(events):
    if not events:
        raise ValueError("TRS requires frozen response events")
    groups = {}
    for event in events:
        response, score = event["response"], event["score"]
        if response not in RESPONSES or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid scored event")
        groups.setdefault(response, []).append(score)
    return mean(mean(scores) for scores in groups.values())


def evaluate_episode(data):
    required = {"episode_id", "case_id", "cell", "variant", "contract", "termination", "frames", "events"}
    if set(data) != required:
        raise ValueError(f"Episode fields must be {sorted(required)}")
    if data["cell"] not in CELLS:
        raise ValueError("Unknown evaluation cell")
    for key in ("episode_id", "case_id"):
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if data["variant"] not in (("S",) if data["cell"].endswith("-S") else ("A", "B")):
        raise ValueError("Use variant S for single-person and A/B for counterfactual pairs")
    contract = Contract(**data["contract"])
    frames = [Frame(**item) for item in data["frames"]]
    events = [Event(**item) for item in data["events"]]
    validate_trace(frames)
    scored = [{"response": e.response, "score": response_score(frames, e, contract)} for e in events]
    result = {key: data[key] for key in ("episode_id", "case_id", "cell", "variant")}
    result.update(LSR=lead_success(frames, data["termination"]), RF=route_fidelity(frames), SCS=social_contract_score(frames, contract), TRS=macro_response(scored), events=scored)
    return result
