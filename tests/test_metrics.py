from dataclasses import replace
import math
import unittest

from leadbench.contract import Contract
from leadbench.metrics import Frame, Event, evaluate_episode, lead_success, route_fidelity, social_contract_score, response_score, macro_response, validate_trace
from leadbench.report import aggregate
from examples.synthetic_trace import example


def frames(speed=0.5, distance=3.0):
    return [Frame(i / 10, i / 10 * speed, 0, i / 10 * speed, 0, distance, 0, True, True, False, 10, 10) for i in range(41)]


class MetricTests(unittest.TestCase):
    def test_exact_contract(self):
        self.assertTrue(Contract("near", "left").contains(1.1, 15))
        self.assertTrue(Contract("near", "left").contains(1.6, 25))
        self.assertFalse(Contract("near", "center").contains(1.7, 0))
        self.assertFalse(Contract("far", "center").contains(3, 6))
        self.assertTrue(Contract("far", "right").contains(4, -25))

    def test_rf_forward_and_deviation(self):
        self.assertAlmostEqual(route_fidelity(frames()), 1)
        self.assertAlmostEqual(route_fidelity([replace(f, route_deviation=1.5) for f in frames()]), math.exp(-1))
        self.assertEqual(route_fidelity(frames(speed=-1)), 0)

    def test_rf_waiting_does_not_accumulate_penalty(self):
        trace = frames()
        self.assertEqual(route_fidelity(trace), route_fidelity([trace[0]] * 100 + trace))
        self.assertEqual(route_fidelity(frames(speed=0)), 0)

    def test_scs_visibility_remains_in_denominator(self):
        trace = [replace(f, target_visible=i % 2 == 0) for i, f in enumerate(frames()[:40])]
        self.assertEqual(social_contract_score(trace, Contract()), 0.5)
        self.assertEqual(social_contract_score([replace(f, formation_valid=i < 2) for i, f in enumerate(trace)], Contract()), 0.5)

    def test_scs_empty_mask_is_not_success(self):
        with self.assertRaises(ValueError):
            social_contract_score([replace(f, formation_valid=False) for f in frames()], Contract())

    def test_lsr_joint_hold_and_hard_failure(self):
        trace = [replace(f, robot_goal_distance=0.5, target_goal_distance=0.5) for f in frames()[:6]]
        self.assertEqual(lead_success(trace, "success"), 1)
        self.assertEqual(lead_success(trace[:5], "success"), 0)
        self.assertEqual(lead_success(trace, "sustained_collision"), 0)
        self.assertEqual(lead_success([replace(f, target_goal_distance=1.1) for f in trace], "success"), 0)
        self.assertEqual(lead_success([replace(f, target_distance=4.1) for f in trace], "success"), 0)

    def test_response_formulas(self):
        for response in ("lead", "resume", "slow"):
            self.assertAlmostEqual(response_score(frames(), Event("target", response, 1, 0), Contract()), 1)
        self.assertAlmostEqual(response_score(frames(speed=0), Event("target", "wait", 1, 0), Contract()), 1)
        self.assertAlmostEqual(response_score(frames(speed=0.25), Event("target", "lead", 1, 0), Contract()), math.sqrt(0.5))
        self.assertAlmostEqual(response_score(frames(), Event("target", "wait", 1, 0), Contract()), math.exp(-2 / 3))
        self.assertAlmostEqual(response_score(frames(distance=4.75), Event("target", "slow", 1, 0), Contract()), math.exp(-0.5))

    def test_error_worsening_penalizes_slow(self):
        trace = [replace(f, target_distance=4.75 if f.time >= 1 else 3) for f in frames()]
        self.assertAlmostEqual(response_score(trace, Event("target", "slow", 1, 0), Contract()), math.exp(-1))

    def test_early_termination_events_receive_zero(self):
        self.assertEqual(response_score(frames()[:15], Event("target", "resume", 2, 0), Contract()), 0)
        self.assertEqual(response_score(frames()[:30], Event("target", "lead", 1, 0), Contract()), 0)

    def test_distractor_control(self):
        with self.assertRaises(ValueError):
            Event("distractor", "wait", 1, 0)
        self.assertEqual(response_score(frames(speed=0), Event("distractor", "lead", 1, 0), Contract()), 0)

    def test_grace_is_not_part_of_response_progress(self):
        trace = [replace(f, route_progress=min(f.time, 2.0), x=min(f.time, 2.0)) for f in frames()]
        self.assertEqual(response_score(trace, Event("target", "lead", 1, 1), Contract()), 0)

    def test_scheduled_early_event_not_dropped_from_macro(self):
        data = example()["episodes"][0]
        data["events"].append({"source": "target", "response": "wait", "onset": 10, "grace": 0})
        result = evaluate_episode(data)
        self.assertEqual(result["TRS"], 0.5)
        self.assertEqual(len(result["events"]), 2)

    def test_interference_excluded_without_bridging(self):
        trace = [replace(f, interference=1.5 <= f.time <= 2.0, target_distance=100 if 1.5 <= f.time <= 2 else 3) for f in frames()]
        self.assertAlmostEqual(response_score(trace, Event("target", "lead", 1, 0), Contract()), 1)

    def test_macro_does_not_overweight_frequent_lead(self):
        events = [{"response": "lead", "score": 1}] * 10 + [{"response": "wait", "score": 0}]
        self.assertEqual(macro_response(events), 0.5)

    def test_log_validation(self):
        with self.assertRaises(ValueError):
            validate_trace(frames()[::2])
        with self.assertRaises(ValueError):
            replace(frames()[0], target_distance=float("nan"))
        with self.assertRaises(ValueError):
            replace(frames()[0], target_visible="false")
        with self.assertRaises(ValueError):
            lead_success(frames(), "infrastructure_error")

    def test_evaluate_portable_example(self):
        result = evaluate_episode(example()["episodes"][0])
        self.assertEqual([result[k] for k in ("LSR", "RF", "TRS", "SCS")], [0, 1, 1, 1])
        self.assertIsNone(aggregate([result])["overall"])

    def test_pair_first_six_cells_and_incomplete_pair(self):
        base = evaluate_episode(example()["episodes"][0])
        rows = []
        for difficulty in ("Core", "Easy", "Constrained"):
            for people in ("S", "M"):
                for variant in (("S",) if people == "S" else ("A", "B")):
                    cell = f"{difficulty}-{people}"
                    score = 1.0 if variant in ("S", "A") else 0.0
                    rows.append(dict(base, episode_id=f"{cell}-{variant}", case_id=cell, cell=cell, variant=variant, LSR=score))
        summary = aggregate(rows)
        self.assertEqual(summary["physical_episodes"], 9)
        self.assertEqual(summary["logical_cases"], 6)
        self.assertEqual(summary["overall"]["LSR"], 0.75)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])
        with self.assertRaises(ValueError):
            aggregate(rows + rows[:1])

    def test_pair_trs_classes_are_balanced(self):
        base = evaluate_episode(example()["episodes"][0])
        a = dict(base, cell="Core-M", variant="A", events=[{"response": "lead", "score": 1}] * 10)
        b = dict(a, episode_id="second", variant="B", events=[{"response": "wait", "score": 0}])
        self.assertEqual(aggregate([a, b])["cells"]["Core-M"]["TRS"], 0.5)
