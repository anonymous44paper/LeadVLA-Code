import math
import unittest

import torch

from leadvla.path_composer import ArcLengthPathComposer, ComposerConfig


def _straight_path(batch: int = 1, horizon: int = 24):
    progress = torch.arange(1, horizon + 1, dtype=torch.float32).unsqueeze(0).repeat(batch, 1) * 0.15
    path = torch.zeros(batch, horizon, 3)
    path[..., 0] = progress
    return path, progress


class PathComposerTest(unittest.TestCase):
    def test_table_seven_rate_limits(self):
        composer = ArcLengthPathComposer()
        slowing = composer.rate_limited_profile(torch.zeros(1), torch.ones(1))
        starting = composer.rate_limited_profile(torch.ones(1), torch.zeros(1))
        self.assertEqual(slowing.shape, (1, 24))
        self.assertAlmostEqual(float(slowing[0, 0]), 0.84, places=6)
        self.assertAlmostEqual(float(starting[0, 0]), 0.1 / 0.75, places=6)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            ComposerConfig(nominal_speed_mps=0)
        path, progress = _straight_path()
        with self.assertRaises(ValueError):
            ArcLengthPathComposer()(path, progress, torch.ones(2), torch.ones(1), torch.zeros_like(path))
        with self.assertRaises(ValueError):
            ArcLengthPathComposer()(path, progress, torch.ones(1), torch.ones(1), torch.full_like(path, float("nan")))
        with self.assertRaises(ValueError):
            ArcLengthPathComposer()(path, progress, torch.ones(1), torch.ones(1), torch.zeros_like(path), scale_profile=torch.ones_like(progress) * 2)

    def test_scale_one_copies_reference(self):
        path, progress = _straight_path(batch=2)
        composer = ArcLengthPathComposer()
        result = composer(path, progress, torch.ones(2), torch.ones(2), torch.zeros_like(path))
        torch.testing.assert_close(result["scaled_reference"], path, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result["waypoint"], path, atol=1e-5, rtol=1e-5)

    def test_stopped_scale_stays_at_origin(self):
        path, progress = _straight_path()
        composer = ArcLengthPathComposer()
        result = composer(path, progress, torch.zeros(1), torch.zeros(1), torch.zeros_like(path))
        torch.testing.assert_close(result["waypoint"], torch.zeros_like(path))

    def test_rate_limiter_decelerates_monotonically(self):
        composer = ArcLengthPathComposer()
        profile = composer.rate_limited_profile(torch.zeros(1), torch.ones(1))[0]
        self.assertTrue(bool(torch.all(profile[1:] <= profile[:-1])))
        self.assertTrue(0.0 <= float(profile[-1]) < float(profile[0]) < 1.0)

    def test_curve_is_interpolated_on_path_not_xy_scaled(self):
        cfg = ComposerConfig(horizon=4, nominal_speed_mps=1.0, trajectory_dt=1.0)
        composer = ArcLengthPathComposer(cfg)
        progress = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        path = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, math.pi / 2],
                              [0.0, 1.0, math.pi], [-1.0, 1.0, math.pi]]])
        query = torch.tensor([[0.5, 1.5, 2.5, 3.5]])
        scaled = composer.interpolate(path, progress, query)[0]
        expected_xy = torch.tensor([[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [-0.5, 1.0]])
        torch.testing.assert_close(scaled[:, :2], expected_xy)

    def test_scale_gradient_reaches_target(self):
        path, progress = _straight_path()
        composer = ArcLengthPathComposer()
        scale = torch.tensor([0.45], requires_grad=True)
        result = composer(path, progress, scale, torch.tensor([0.45]), torch.zeros_like(path))
        result["waypoint"][..., 0].sum().backward()
        self.assertIsNotNone(scale.grad)
        self.assertTrue(bool(torch.isfinite(scale.grad).all()))
        self.assertGreater(float(scale.grad.abs()), 0.0)

    def test_yaw_wrap_interpolates_across_pi_boundary(self):
        cfg = ComposerConfig(horizon=2)
        composer = ArcLengthPathComposer(cfg)
        progress = torch.tensor([[1.0, 2.0]])
        path = torch.tensor([[[1.0, 0.0, math.radians(179.0)],
                              [2.0, 0.0, math.radians(-179.0)]]])
        query = torch.tensor([[1.0, 1.5]])
        yaw = composer.interpolate(path, progress, query)[0, :, 2]
        self.assertLess(abs(abs(float(yaw[1])) - math.pi), math.radians(2.0))

    def test_explicit_path_origin_preserves_lateral_route_offset(self):
        cfg = ComposerConfig(horizon=2, nominal_speed_mps=1.0, trajectory_dt=1.0)
        composer = ArcLengthPathComposer(cfg)
        progress = torch.tensor([[1.0, 2.0]])
        path = torch.tensor([[[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]])
        origin = torch.tensor([[0.0, 1.0, 0.0]])
        profile = torch.zeros(1, 2)
        result = composer(
            path,
            progress,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros_like(path),
            scale_profile=profile,
            path_origin=origin,
        )
        expected = origin.unsqueeze(1).expand_as(path)
        torch.testing.assert_close(result["scaled_reference"], expected)


if __name__ == "__main__":
    unittest.main()
