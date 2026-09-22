# LeadVLA policy interface

## Input boundary (Appendix A.1)

`InputBuilder` accepts synchronized front/rear **RGB** uint8 observations at
10 Hz. Call `reset()` at each episode boundary. It selects five historical
observations over the preceding 31 frames and appends the current observation;
unavailable early history repeats the earliest frame. Images are resized to
384 × 384, ordered as six front images followed by six rear images.

The supplied local route has 24 `(x, y, heading)` knots, with x forward, y left,
and heading in radians. Relative route progress is nonnegative arc length from
the explicit route origin. Nominal spacing is approximately 0.15 m, giving
approximately 3.6 m of composition lookahead. The first 10 XY samples are
serialized into the language prompt, giving approximately 1.5 m of route context.
Provide the original nominal route, never an expert detour.

`PolicyInput` contains images, language prompt, the 24-point route/progress,
route origin and previously executed Pace. The backend receives only images and
prompt; the longer route and previous Pace are used by the deterministic
Composer. No actor identity, target box, depth, future event schedule or expert
action enters the backend through this API.

## Backend contract (Appendix A.2)

`predict(images, prompt)` performs one multimodal forward pass and returns:

| Field | Shape | Meaning |
| --- | --- | --- |
| `pace` | `[1]` | Target route-progress scale in `[0,1]` |
| `residual` | `[1,24,3]` | Local XY and heading adjustments |

The paper's model uses 24 shared action queries. Pace is decoded from their
original pooled representation. Six supervised objectives are target box,
lateral relation, distance relation, observation state, motion state and formation
state. Only **soft lateral and formation distributions** directly modulate the
Residual queries; Pace has no explicit semantic fusion. The modulation output
layer is initialized to zero. Other objectives shape the shared representation
through supervision, not an explicit follower-state controller bottleneck.

`leadvla.model` provides shared-query orchestration, the Pace branch, six
semantic heads and selective fusion. `leadvla.losses` and `leadvla.training`
provide supervised objectives and optimizer utilities. The application supplies
its backbone/tokenizer, Residual decoder and compatible weights. The CLI loads a
user-provided backend factory; only load modules and checkpoints you trust.
See [architecture.md](architecture.md).

## Deterministic Path Composer (Appendix A.3)

`ArcLengthPathComposer` uses the Table 7 defaults: H=24, Δt=0.2 s,
nominal speed=0.75 m/s, acceleration=0.5 m/s² and
deceleration=0.6 m/s². It starts from previously executed Pace, rate-limits future
speed, scales **arc-length increments**, interpolates the route, and adds the
Residual. Heading is unwrapped before interpolation and wrapped after composition.
Pace never simply scales robot-local XY coordinates.

The Composer exposes batched, differentiable tensor operations. Its returned
`scale_profile`, `query_progress`, `scaled_reference`, and `waypoint` enable
inspection and unit testing. Alternate Composer configurations are experimental
overrides, not the paper defaults. An explicit `scale_profile` bypasses rate
limiting and is provided only for controlled composition tests.

The CLI NPZ contains `front_rgb` and `rear_rgb` arrays `[N,H,W,3]` in chronological
order, plus `path_reference [24,3]`, `path_progress [24]`, `path_origin [3]`.
Do not include pickled objects. The current instruction is supplied separately.
The example CLI performs a single query over this causal history; it is not a
robot controller or an end-to-end real-world deployment.
