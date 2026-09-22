import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from examples.synthetic_trace import example
from leadbench.export import render
from leadbench.manifest import digest, validate_manifest, validate_episode_identity
from leadbench.replay import replay
from leadbench.termination import TerminationMonitor
from tests.test_metrics import frames


def manifest_for(episode):
    fields = ("episode_id", "case_id", "cell", "variant", "contract", "events")
    return {"schema_version": 1, "episodes": [dict({key: episode[key] for key in fields}, trace_file="trace.json")]}


class BenchToolTests(unittest.TestCase):
    def test_manifest_stability_and_completeness(self):
        manifest = manifest_for(example()["episodes"][0])
        self.assertEqual(validate_manifest(manifest)["logical_cases"], 1)
        self.assertEqual(digest(manifest), digest(dict(reversed(list(manifest.items())))))
        with self.assertRaises(ValueError):
            validate_manifest(manifest, require_full_suite=True)
        manifest["episodes"].append(copy.deepcopy(manifest["episodes"][0]))
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_manifest_no_traversal_or_unpaired_rollout(self):
        manifest = manifest_for(example()["episodes"][0])
        manifest["episodes"][0]["trace_file"] = "../trace.json"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)
        manifest["episodes"][0].update(trace_file="trace.json", cell="Core-M", variant="A")
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_batch_replay_export_and_manifest_mutation(self):
        episode = example()["episodes"][0]
        manifest = manifest_for(episode)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "trace.json").write_text(json.dumps(episode))
            result = replay(manifest, directory, expected_digest=digest(manifest))
            self.assertIn("| Core-S | 0.000 | 1.000 | 1.000 | 1.000 |", render(result["summary"]))
            self.assertIn("Overall,—,—,—,—", render(result["summary"], "csv"))
            with self.assertRaises(ValueError):
                replay(manifest, directory, expected_digest="incorrect")
            manifest["episodes"][0]["events"][0]["grace"] = 0.5
            with self.assertRaises(ValueError):
                replay(manifest, directory)

    def test_joint_success_hold(self):
        monitor = TerminationMonitor(60)
        for index in range(6):
            frame = replace(frames()[index], robot_goal_distance=0.5, target_goal_distance=0.5)
            reason = monitor.update(frame, penetration_m=0, expected_wait=False)
            self.assertEqual(reason, "success" if index == 5 else None)

    def test_collision_needs_more_than_20_frames(self):
        monitor = TerminationMonitor(60)
        for index in range(21):
            reason = monitor.update(frames()[index], penetration_m=0.10, expected_wait=False)
            self.assertEqual(reason, "sustained_collision" if index == 20 else None)

    def test_collision_counter_resets(self):
        monitor = TerminationMonitor(60)
        for index in range(40):
            reason = monitor.update(frames()[index], penetration_m=0 if index == 19 else 0.10, expected_wait=False)
            self.assertIsNone(reason)

    def test_terminal_sync_timeout(self):
        monitor = TerminationMonitor(60)
        for index in range(31):
            frame = replace(frames()[index], robot_goal_distance=0.5)
            reason = monitor.update(frame, penetration_m=0, expected_wait=False)
        self.assertEqual(reason, "terminal_sync_timeout")

    def test_stall_timer_pauses_only_for_frozen_waits(self):
        monitor = TerminationMonitor(100)
        base = frames(speed=0)[0]
        for index in range(201):
            reason = monitor.update(replace(base, time=index / 10), penetration_m=0, expected_wait=False)
        self.assertEqual(reason, "route_stall")
        monitor = TerminationMonitor(100)
        for index in range(301):
            reason = monitor.update(replace(base, time=index / 10), penetration_m=0, expected_wait=True)
            self.assertIsNone(reason)

    def test_route_timeout_and_no_visibility_termination(self):
        monitor = TerminationMonitor(2)
        for index in range(21):
            frame = replace(frames()[index], target_visible=False, target_distance=100)
            reason = monitor.update(frame, penetration_m=0, expected_wait=False)
        self.assertEqual(reason, "route_timeout")
        with self.assertRaises(RuntimeError):
            monitor.update(frames()[21], penetration_m=0, expected_wait=False)
