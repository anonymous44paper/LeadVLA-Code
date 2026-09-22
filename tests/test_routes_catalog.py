from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from leadinfra.routes import GridMap, ArcLengthPath, plan_route, _astar
from leadinfra.validation import validate_route, route_distance
from leadinfra.catalog import Asset, Catalog, appearance_combinations
from leadinfra.recording import EpisodeRecorder
from tests.test_metrics import frames


class RouteCatalogTests(unittest.TestCase):
    def grid(self):
        return GridMap(np.ones((30, 30), dtype=bool), np.array([0., 0.]), 0.5)

    def test_grid_round_trip_and_bounds(self):
        grid = self.grid()
        cells = np.array([[2, 4], [8, 9]])
        np.testing.assert_array_equal(grid.world_to_cell(grid.cell_to_world(cells)), cells)
        self.assertFalse(grid.contains(np.array([-1.0, -1.0])))
        self.assertTrue(grid.contains(np.array([1.0, 1.0])))
        self.assertEqual(grid.clearance[0, 0], 0.5)

    def test_arc_length_sample_project(self):
        route = ArcLengthPath(np.array([[1., 1.], [5., 1.], [5., 5.]]))
        self.assertEqual(route.length, 8)
        np.testing.assert_allclose(route.sample([2])[:, :2], [[3, 1]])
        self.assertEqual(route.project(np.array([3., 2.])), (2., 1.))

    def test_duplicates_and_invalid_path(self):
        route = ArcLengthPath(np.array([[0., 0.], [0., 0.], [1., 0.]]))
        self.assertEqual(route.length, 1)
        with self.assertRaises(ValueError):
            ArcLengthPath(np.array([[0., 0.], [float("nan"), 0.]]))
        with self.assertRaises(ValueError):
            ArcLengthPath(np.zeros((3, 2)))

    def test_closed_path_and_unwrapped_progress(self):
        route = ArcLengthPath(np.array([[1., 1.], [4., 1.], [4., 4.], [1., 4.]]), closed=True)
        self.assertEqual(route.length, 12)
        self.assertEqual(route.unwrap_progress(0.2, 11.9), 12.2)
        np.testing.assert_allclose(route.sample([0]), route.sample([12]))

    def test_astar_does_not_cut_blocked_corners(self):
        traversable = np.array([[True, False], [False, True]])
        with self.assertRaises(RuntimeError):
            _astar(traversable, np.ones((2, 2)), (0, 0), (1, 1))

    def test_planning_and_post_validation(self):
        grid = self.grid()
        route = plan_route(grid, [(4, 4), (20, 20)], min_clearance=1, smooth_sigma_cells=1)
        self.assertGreater(route.length, 5)
        self.assertGreaterEqual(validate_route(route, grid, min_clearance=1)["minimum_clearance_m"], 1)
        grid.planning_grid[10:15, 10:15] = False
        with self.assertRaises(ValueError):
            validate_route(route, grid)

    def test_route_redundancy(self):
        first = ArcLengthPath(np.array([[0., 0.], [2., 0.]]))
        same = ArcLengthPath(np.array([[0., 0.], [1., 0.], [2., 0.]]))
        other = ArcLengthPath(np.array([[0., 1.], [2., 1.]]))
        self.assertAlmostEqual(route_distance(first, same), 0)
        self.assertAlmostEqual(route_distance(first, other), 1)

    def test_catalog_does_not_escape_asset_root(self):
        with self.assertRaises(ValueError):
            Asset("route_a", "route", "../private.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "route.json").write_text("{}")
            asset = Asset("route_a", "route", "route.json")
            catalog = Catalog([asset])
            self.assertEqual(catalog.resolve("route_a", root), root / "route.json")
            self.assertEqual(catalog.inventory()["route"], 1)
            with self.assertRaises(ValueError):
                Catalog([asset, asset])
            (root / "escape").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                Catalog([Asset("bad", "route", "escape/file.json")]).resolve("bad", root)

    def test_appearance_filter(self):
        parts = {"body": ["body_a"], "clothing": ["top_a", "top_b"], "accessory": ["none"], "scale": ["medium"]}
        result = list(appearance_combinations(parts, lambda item: item["clothing"] == "top_b"))
        self.assertEqual(len(result), 1)

    def test_recorder_monotonic_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            with EpisodeRecorder(directory, "episode_01") as recorder:
                recorder.append(frames()[0])
                recorder.append(frames()[1])
                with self.assertRaises(ValueError):
                    recorder.append(frames()[3])
            stored = (Path(directory) / "episode_01" / "frames.jsonl").read_text().splitlines()
            self.assertEqual(len(stored), 2)
            self.assertEqual(json.loads(stored[1])["time"], 0.1)
            with self.assertRaises(FileExistsError):
                EpisodeRecorder(directory, "episode_01")
