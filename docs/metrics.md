# Offline metrics and trace schema

The evaluator implements the formulas in Appendix C.4 and Tables 13/15 using
recorded physical geometry, frozen event schedules and formation-valid masks.
See [architecture.md](architecture.md#validation) for validation scope.

## Metric definitions

- **LSR**: joint robot/target terminal geometry, separation and ≥0.5 s hold;
  all policy-failure termination labels score zero. It does not use RF/TRS/SCS
  as gates. The final continuous terminal interval must satisfy the contract.
- **RF**: after stationary compression, average
  `exp(-max(0, deviation-1.0)/0.5)` and multiply by the fraction of successive
  projected progress increments ≥−0.05 m. Waiting does not repeatedly contribute
  identical positions. The paper's formula is evaluated over the route trace;
  this implementation does not add an interference exclusion to RF.
- **TRS**: compute realized target distance error and robot route progress;
  never inspect a predicted Pace, action token or predicted waypoint as a proxy
  for physical response. Exclude frozen scene-interference frames. For a response
  window starting after the event's grace period and lasting 2.0 s, let `pre` and
  `post` be mean interval-distance error in the final 0.5 s before onset and
  before the response-window end. Define
  `Qd=exp(-post/0.75)`, `Qc=exp(-max(0,post-pre)/0.75)`,
  `Qp=clip((progress/duration)/0.5,0,1)`, `Qs=exp(-progress/0.5)`.
  Lead/resume score `sqrt(Qp*Qd)`, slow scores `sqrt(Qd*Qc)`, and wait scores
  `(Qd*Qc*Qs)^(1/3)`. Distractor-control events expect lead but still use the
  designated target's distance. Scheduled events lacking their full window due
  to early termination receive zero rather than being dropped.
- **SCS**: fraction of frozen formation-valid frames satisfying visibility,
  requested distance and requested bearing together. Losing visibility does
  **not** remove a frame from the denominator.

Distance intervals: near `[1.10,1.60]` m, far `[2.60,4.00]` m. Bearing intervals:
left `[15,25]`, center `[-5,5]`, right `[-25,-15]` degrees. Bounds are inclusive.
Distance error is `max(lower-distance,0,distance-upper)`.

## Portable JSON interface

The top-level object has exactly `schema_version: 1` and an `episodes` list.
Each episode has:

| Field | Definition |
| --- | --- |
| `episode_id` | Unique physical-rollout identifier |
| `case_id` | Shared identifier of the logical A/B pair, or single-person case |
| `cell` | Core-S, Core-M, Easy-S, Easy-M, Constrained-S or Constrained-M |
| `variant` | S for single-person; A or B for a counterfactual pair |
| `contract` | `distance`: near/far; `bearing`: left/center/right |
| `termination` | success, sustained_collision, terminal_sync_timeout, route_stall or route_timeout |
| `frames` | Contiguous chronological 10 Hz physical trace |
| `events` | Complete frozen schedule, including events after an early termination |

Each frame contains these required fields:

| Field | Units / semantics |
| --- | --- |
| `time` | Seconds on the episode clock |
| `x`, `y` | Actual robot position in a fixed metric frame |
| `route_progress` | Actual robot position projected onto nominal-route arc length, meters |
| `route_deviation` | Nonnegative lateral distance to nominal route, meters |
| `target_distance` | Actual robot–designated-target distance, meters |
| `target_bearing_deg` | Contract-relative follower bearing in degrees: positive left, zero center, negative right |
| `target_visible` | Evaluator visibility boolean; false is a scored failure for SCS |
| `formation_valid` | Frozen membership in formation-valid intervals, not a policy-derived mask |
| `interference` | Frozen scene-interference interval membership |
| `robot_goal_distance`, `target_goal_distance` | Distance to the corresponding frozen terminal region, meters |

The external trace exporter must consistently convert simulator coordinates into
the contract-relative bearing convention; do not pass an unconverted world yaw
or front-camera azimuth. Projection disambiguation at loops/intersections also
belongs to the exporter. Fields must represent realized geometry, not predictions.
Invisible target geometry may be available to the evaluator; this does not make
it a policy-visible observation.

An event has exactly `source` (target/distractor), `response` (lead/slow/wait/resume),
`onset` and `grace` in seconds. Masks and schedules must originate from the same
frozen manifest for all policies. The schema does not itself prove their origin.

## Sampling and edge cases

The evaluator uses the following sampling and edge-case conventions:

- Require complete 10 Hz traces; reject missing samples instead of interpolating.
- Compress a position if it is within 0.05 m of the last retained position.
  Fewer than two retained positions yields RF=0 because forward consistency is
  undefined. This threshold is an implementation convention.
- TRS pre-statistics use `[onset−0.5,onset)`; post-statistics use
  `(window_end−0.5,window_end]`. Sample means approximate temporal means at 10 Hz.
  Event boundaries should align to the frozen sampling grid.
- Progress is the nonnegative **net** sum of route-progress increments on
  adjacent valid sample intervals. Valid duration is the sum of those intervals.
  Never bridge an excluded interference interval.
- Invalid pre/post support or an all-interference response window raises an
  error, except for a rollout ending before window completion, which scores zero.
  Freeze schedules with sufficient pre-event context.
- No formation-valid frames makes SCS undefined and raises an error; no frozen
  response events makes TRS undefined and raises an error. Neither becomes an
  artificial perfect score.
- A logical case is the TRS evaluation unit: pool its events (both A/B rollouts
  for multi-person), average within each response class, then macro-average
  classes present. LSR/RF/SCS use the pair mean. Cells then average their logical
  cases. Changing the TRS evaluation unit would be a different aggregation
  convention and must not silently replace this one.
- Overall values require all six cells. The full paper suite additionally needs
  12 S cases and 12 A/B pairs per difficulty; validate the exact external frozen
  manifest, not just these counts, before claiming a full benchmark result.

The evaluator has no inputs for model-specific Pace. An external trace exporter
must satisfy this schema, preserve frozen masks and schedules, and apply the
documented coordinate and sampling conventions.
