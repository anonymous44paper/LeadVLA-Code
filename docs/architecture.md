# Architecture

LeadVLA separates nominal route geometry from follower-aware execution. A single
multimodal pass produces 24 shared action queries. Their pooled representation
predicts Pace and six follower-related objectives. Soft lateral and formation
distributions modulate only the Residual branch. The deterministic Path Composer
combines rate-limited Pace, nominal route geometry and local Residual corrections.

| Module | Responsibility |
| --- | --- |
| `leadvla.inputs` | Causal front/rear RGB history and language route hint |
| `leadvla.model.FollowerHeads` | Six semantic heads and zero-initialized selective fusion |
| `leadvla.model.ControlHeads` | Unfused Pace branch and supplied Residual decoder |
| `leadvla.model.LeadVLA` | One backbone forward pass, action-token gathering and trajectory composition |
| `leadvla.path_composer` | Rate limits, arc-length interpolation and wrapped heading addition |
| `leadvla.targets` | Normalized boxes, semantic classes and undefined-label masks |
| `leadvla.losses` | Trajectory/Pace/Residual losses, regularization and semantic objectives |
| `leadvla.training` | Optimizer groups, warmup, stage reset and validation selection |

The application supplies a compatible multimodal backbone, tokenizer/processor,
Residual decoder and checkpoint. The package does not bundle weights or the
complete backbone-specific training/deployment integration. The synthetic control
example uses a small decoder for tensor/gradient checks, not a trained policy.

## Training objectives

`L_action = L_waypoint + L_pace + L_residual + 0.001 L_norm + 0.02 L_smooth`

`L_semantic = L_bbox + 0.5 (L_lateral + L_distance + L_observation + L_motion + L_formation)`

`L_total = L_action + 0.25 L_semantic`

Regression uses Smooth L1; classification uses cross entropy. Every objective
has an explicit validity mask. Smoothness requires both adjacent waypoints to be
valid. Undefined Pace supervision is masked, with unit Pace used in composition.

Stage 1 uses approximately 200K samples; Stage 2 uses approximately 4M. Stage 2
loads selected Stage 1 model weights but starts fresh optimizer/scheduler state.
The vision tower remains frozen. `configs/training.json` records the settings.
Call `freeze_vision_tower()` on the application's actual vision module before
optimizer construction. Distributed training and data orchestration are external.

## Validation

Tests cover formulas, masks, gradients, selective fusion, single-pass query
extraction, route geometry, termination and aggregation. Full training and replay
of the original 108 experimental traces are not included in this validation.
