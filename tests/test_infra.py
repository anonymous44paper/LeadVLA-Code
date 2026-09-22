import copy
import json
from pathlib import Path
import unittest

from leadinfra.compose import compose, semantic_id, MOTIONS, PRESENCES, FORMATIONS
from leadinfra.quotas import largest_remainder


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1] / "examples" / "composition.json").read_text())

    def test_deterministic_and_portable(self):
        first = compose(self.config)
        self.assertEqual(first, compose(copy.deepcopy(self.config)))
        self.assertEqual(len(first["episodes"]), 4)
        self.assertEqual((len(MOTIONS), len(PRESENCES), len(FORMATIONS)), (8, 5, 6))
        reordered = dict(reversed(list(self.config.items())))
        self.assertEqual(first, compose(reordered))

    def test_ids_are_not_paths(self):
        for value in ("../private", "drive:asset", "https://example.invalid/asset", "nested/asset", ""):
            with self.assertRaises(ValueError):
                semantic_id(value)

    def test_distractor_presence_requires_multiple_people(self):
        self.config["compatible_interactions"][0]["presence"] = "target_absent_distractor_visible"
        with self.assertRaises(ValueError):
            compose(self.config)

    def test_quotas(self):
        self.assertEqual(largest_remainder(5, {"b": 1, "a": 1}), {"a": 3, "b": 2})
        self.assertEqual(sum(largest_remainder(108, {"s": 1, "m": 2}).values()), 108)
        for weights in ({"a": 0}, {"a": -1}, {"a": float("inf")}):
            with self.assertRaises(ValueError):
                largest_remainder(3, weights)
