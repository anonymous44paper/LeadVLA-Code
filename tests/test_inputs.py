import unittest

import numpy as np
import torch

from leadvla.inputs import InputBuilder
from leadvla.inference import predict_waypoints


class InputTests(unittest.TestCase):
    def build(self, builder):
        progress = np.arange(1, 25, dtype=np.float32) * 0.15
        route = np.column_stack((progress, np.zeros(24), np.zeros(24)))
        return builder.build(instruction="Lead the person in a teal top at far center.", path_reference=route, path_progress=progress, executed_pace=1.0)

    def test_history_and_view_order(self):
        builder = InputBuilder()
        for i in range(40):
            builder.observe(np.full((2, 2, 3), i, np.uint8), np.full((2, 2, 3), 100 + i, np.uint8))
        sample = self.build(builder)
        self.assertEqual(builder.history_indices(), [8, 16, 23, 30, 38, 39])
        self.assertEqual(len(sample.images), 12)
        self.assertEqual(sample.images[0].size, (384, 384))
        self.assertEqual(sample.images[0].getpixel((0, 0)), (8, 8, 8))
        self.assertEqual(sample.images[6].getpixel((0, 0)), (108, 108, 108))
        self.assertIn("(1.50, 0.00)", sample.prompt)
        self.assertNotIn("(3.60, 0.00)", sample.prompt)

    def test_missing_history_repeats_earliest(self):
        builder = InputBuilder()
        frame = np.zeros((2, 2, 3), np.uint8)
        builder.observe(frame, frame)
        self.assertEqual(builder.history_indices(), [0] * 6)
        builder.observe(frame, frame)
        self.assertEqual(builder.history_indices(), [0, 0, 0, 0, 0, 1])
        builder.reset()
        with self.assertRaises(ValueError):
            self.build(builder)

    def test_backend_called_once_and_composed(self):
        class TestBackend:
            calls = 0
            def predict(self, images, prompt):
                self.calls += 1
                return {"pace": torch.ones(1), "residual": torch.zeros(1, 24, 3)}
        builder = InputBuilder()
        frame = np.zeros((2, 2, 3), np.uint8)
        builder.observe(frame, frame)
        sample = self.build(builder)
        backend = TestBackend()
        result = predict_waypoints(backend, sample)
        self.assertEqual(backend.calls, 1)
        torch.testing.assert_close(result["waypoint"][0], torch.from_numpy(sample.path_reference))
