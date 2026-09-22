import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from leadvla.model import FollowerHeads, ControlHeads, LeadVLA, SEMANTIC_CLASSES, gather_action_queries
from leadvla.losses import LossWeights, classification, joint_loss, regression, residual_targets
from leadvla.targets import encode_semantics, normalized_box, spatial_labels
from leadvla.training import optimizer_for, constant_with_warmup, freeze_vision_tower, start_stage, train_step, ValidationSelector
from examples.control_training import run


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.query = torch.randn(2, 24, 16, requires_grad=True)

    def test_six_heads_zero_initial_modulation(self):
        model = FollowerHeads(16, 8)
        output = model(self.query)
        for name, labels in SEMANTIC_CLASSES.items():
            self.assertEqual(output[name].shape, (2, len(labels)))
        self.assertEqual(output["bbox"].shape, (2, 4))
        torch.testing.assert_close(output["modulation"], torch.zeros(2, 16))
        torch.testing.assert_close(output["pace_queries"], self.query)
        torch.testing.assert_close(output["residual_queries"], self.query)

    def test_only_lateral_formation_directly_modulate(self):
        model = FollowerHeads(16, 8)
        with torch.no_grad():
            model.residual_fusion[-1].weight.normal_()
        prediction = model(self.query)
        modified = dict(prediction)
        for name in ("bbox", "distance", "observation", "motion"):
            modified[name] = torch.randn_like(modified[name]) * 100
        torch.testing.assert_close(model.modulation(prediction), model.modulation(modified))
        modified["lateral"] = torch.randn_like(modified["lateral"]) * 100
        self.assertFalse(torch.allclose(model.modulation(prediction), model.modulation(modified)))

    def test_semantics_do_not_change_pace(self):
        model = ControlHeads(16, nn.Linear(16, 3), auxiliary_size=8)
        with torch.no_grad():
            model.pace_head[-1].weight.normal_()
        first = model(self.query)
        with torch.no_grad():
            model.semantics.residual_fusion[-1].weight.normal_()
        second = model(self.query)
        torch.testing.assert_close(first["pace"], second["pace"])
        self.assertFalse(torch.allclose(first["residual"], second["residual"]))

    def test_soft_coupling_is_differentiable(self):
        heads = ControlHeads(16, nn.Linear(16, 3), auxiliary_size=8)
        with torch.no_grad():
            heads.semantics.residual_fusion[-1].weight.normal_()
        heads(self.query)["residual"].sum().backward()
        self.assertGreater(float(heads.semantics.lateral_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(heads.semantics.formation_head.weight.grad.abs().sum()), 0)
        self.assertIsNone(heads.semantics.motion_head.weight.grad)

    def test_exactly_24_action_queries(self):
        hidden = torch.randn(2, 30, 16)
        ids = torch.zeros(2, 30, dtype=torch.long)
        ids[:, 6:] = 5
        torch.testing.assert_close(gather_action_queries(hidden, ids, 5), hidden[:, 6:])
        ids[0, 0] = 5
        with self.assertRaises(ValueError):
            gather_action_queries(hidden, ids, 5)

    def test_single_backbone_pass_and_composition(self):
        class Backbone(nn.Module):
            calls = 0
            def forward(self, input_ids, **kwargs):
                self.calls += 1
                return SimpleNamespace(hidden_states=(torch.ones(*input_ids.shape, 16),))
        backbone = Backbone()
        model = LeadVLA(backbone, nn.Linear(16, 3), 16, 7, auxiliary_size=8)
        progress = torch.arange(1, 25).float().repeat(2, 1) * 0.15
        route = torch.stack((progress, torch.zeros_like(progress), torch.zeros_like(progress)), -1)
        result = model({"input_ids": torch.full((2, 24), 7)}, route, progress, torch.zeros(2, 3), torch.ones(2))
        self.assertEqual(backbone.calls, 1)
        self.assertEqual(result["waypoint"].shape, (2, 24, 3))

    def test_joint_backward_example(self):
        losses = run()
        expected = losses["waypoint"] + losses["pace"] + losses["residual"] + 0.001 * losses["residual_norm"] + 0.02 * losses["residual_smooth"]
        self.assertAlmostEqual(losses["action"], expected, places=5)
        expected_semantic = losses["bbox"] + 0.5 * sum(losses[k] for k in SEMANTIC_CLASSES)
        self.assertAlmostEqual(losses["semantic"], expected_semantic, places=5)
        self.assertAlmostEqual(losses["total"], expected + 0.25 * expected_semantic, places=5)

    def test_masked_undefined_targets_have_zero_gradient(self):
        pred = torch.tensor([1.0, 1.0], requires_grad=True)
        loss = regression(pred, torch.tensor([0.0, float("nan")]), torch.tensor([1.0, 0.0]))
        loss.backward()
        self.assertEqual(float(pred.grad[1]), 0)
        self.assertAlmostEqual(float(loss.detach()), 0.5)
        logits = torch.randn(2, 3, requires_grad=True)
        classification(logits, torch.tensor([-1, -1]), torch.zeros(2)).backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        with self.assertRaises(ValueError):
            classification(logits, torch.tensor([0, 1]), torch.tensor([0.5, 1.0]))

    def test_residual_target_unit_pace_and_wrapped_heading(self):
        progress = torch.arange(1, 25).float().unsqueeze(0) * 0.15
        route = torch.stack((progress, torch.zeros_like(progress), torch.zeros_like(progress)), -1)
        expert = route.clone()
        expert[..., 2] = 2 * torch.pi + 0.1
        result = residual_targets(route, progress, torch.zeros(1, 3), torch.ones(1), torch.zeros(1), expert, torch.zeros(1))
        torch.testing.assert_close(result[..., :2], torch.zeros_like(result[..., :2]))
        torch.testing.assert_close(result[..., 2], torch.full((1, 24), 0.1), atol=1e-6, rtol=1e-5)

    def test_optimizer_groups_stage_reset_and_freeze(self):
        backbone = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        model = LeadVLA(backbone, nn.Linear(16, 3), 16, 5, auxiliary_size=8)
        freeze_vision_tower(backbone[0])
        optimizer = optimizer_for(model)
        self.assertEqual({g["name"]: g["lr"] for g in optimizer.param_groups}, {"backbone": 2e-5, "residual": 1e-5, "control": 1e-4})
        params = [p for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(params), len(set(map(id, params))))
        self.assertFalse(any(id(backbone[0].weight) == id(p) for p in params))
        scheduler = constant_with_warmup(optimizer, 0)
        loss = sum(p.square().sum() for p in params)
        train_step(model, loss, optimizer, scheduler)
        self.assertTrue(optimizer.state)
        stage_two, _ = start_stage(model, warmup_steps=10, initial_weights=model.state_dict())
        self.assertFalse(stage_two.state)

    def test_validation_selection(self):
        selector = ValidationSelector()
        self.assertTrue(selector.improves(2))
        self.assertFalse(selector.improves(3))
        self.assertTrue(selector.improves(1))

    def test_configuration_constants(self):
        config = json.loads((Path(__file__).parents[1] / "configs" / "training.json").read_text())
        self.assertEqual(config["loss"], LossWeights().__dict__)
        self.assertEqual(config["explicit_semantic_fusion"], ["lateral", "formation"])

    def test_label_encoding_and_contract(self):
        labels = {key: None for key in SEMANTIC_CLASSES}
        labels.update(spatial_labels(1.65, 20, "near"))
        result = encode_semantics(labels)
        self.assertEqual(result["distance"], 2)
        self.assertEqual(result["observation_valid"], 0)
        np.testing.assert_allclose(normalized_box([0, 0, 100, 50], 100, 50), [0.5, 0.5, 1, 1])
