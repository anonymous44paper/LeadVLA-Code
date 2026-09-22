"""Stateful 10 Hz termination monitors for Appendix C.5, Table 16."""

from collections import deque
import math

from .metrics import Frame


class TerminationMonitor:
    def __init__(self, timeout_seconds):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Episode timeout must be finite and positive")
        self.timeout = timeout_seconds
        self.previous_time = None
        self.collision_frames = 0
        self.arrival_time = None
        self.joint_since = None
        self.active_time = 0.0
        self.progress_history = deque()
        self.reason = None

    def update(self, frame, *, penetration_m, expected_wait):
        """expected_wait comes from the frozen schedule, never a policy decision.

        An interval ending at a waiting-marked sample does not advance the stall
        clock. The episode clock starts at time zero. Return None while active.
        """
        if self.reason is not None:
            raise RuntimeError("A terminated monitor cannot accept more frames")
        if not isinstance(frame, Frame) or type(expected_wait) is not bool:
            raise ValueError("Expected Frame and boolean frozen waiting state")
        if not math.isfinite(penetration_m) or penetration_m < 0:
            raise ValueError("Penetration must be finite and nonnegative")
        if self.previous_time is None and abs(frame.time) > 1e-6:
            raise ValueError("Start the monitor at episode time zero")
        dt = 0.0 if self.previous_time is None else frame.time - self.previous_time
        if self.previous_time is not None and abs(dt - 0.1) > 1e-6:
            raise ValueError("Termination monitor requires contiguous 10 Hz frames")
        self.previous_time = frame.time
        self.collision_frames = self.collision_frames + 1 if penetration_m >= 0.10 else 0
        if frame.robot_goal_distance <= 1 and self.arrival_time is None:
            self.arrival_time = frame.time
        joint = frame.robot_goal_distance <= 1 and frame.target_goal_distance <= 1 and 0.8 <= frame.target_distance <= 4.0
        if not joint:
            self.joint_since = None
        elif self.joint_since is None:
            self.joint_since = frame.time
        if not expected_wait:
            self.active_time += dt
        # Waiting need not retain one record per simulation frame indefinitely.
        if self.progress_history and self.active_time == self.progress_history[-1][0]:
            pass
        else:
            self.progress_history.append((self.active_time, frame.route_progress))
        while len(self.progress_history) > 1 and self.progress_history[1][0] <= self.active_time - 20 + 1e-9:
            self.progress_history.popleft()
        elapsed = self.active_time - self.progress_history[0][0]
        progress = frame.route_progress - self.progress_history[0][1]
        if self.collision_frames > 20:
            self.reason = "sustained_collision"
        elif self.joint_since is not None and frame.time - self.joint_since >= 0.5 - 1e-9:
            self.reason = "success"
        elif self.arrival_time is not None and frame.time - self.arrival_time >= 3 - 1e-9:
            self.reason = "terminal_sync_timeout"
        elif not expected_wait and elapsed >= 20 - 1e-9 and progress < 0.25:
            self.reason = "route_stall"
        elif frame.time >= self.timeout:
            self.reason = "route_timeout"
        return self.reason
