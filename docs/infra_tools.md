# Route, asset and recording tools

## Navigability and route construction

`GridMap` represents a boolean traversability grid with metric origin and cell
resolution. The infrastructure configuration uses 0.5 m cells. An NPZ grid
contains `planning_grid`, `world_min [2]` and scalar `resolution`; loading disables
pickled objects. World/cell conversion and clearance computation are explicit.

`plan_route(grid, anchors_cells, ...)` snaps supplied anchors to traversable
cells, uses clearance-aware A* connections, optionally smooths the path, resamples
it in arc length, and validates the result. Diagonal corner cutting is forbidden.
Smoothing that leaves navigable space or violates clearance raises an error;
the caller can reject that candidate rather than silently accepting it.

```python
import numpy as np
from leadinfra.routes import GridMap, plan_route
from leadinfra.validation import validate_route

# Illustrative empty grid, not a released scene asset.
grid = GridMap(np.ones((40, 40), dtype=bool), np.array([0., 0.]), 0.5)
route = plan_route(grid, [(5, 5), (25, 25)], min_clearance=1.0)
print(validate_route(route, grid, min_clearance=1.0))
print(route.sample(np.arange(1, 25) * 0.15))
```

`ArcLengthPath` supports sampling, nearest-segment projection, closed routes and
lap-progress unwrapping. Coordinates are in the grid/world frame. Transform them
to robot-local coordinates before constructing a policy input. At intersections,
an external active-route selector must disambiguate overlapping segments rather
than assuming nearest geometry alone identifies route progress.

`validate_route` checks dense sampled navigability, clearance and optional heading
curvature. These grid checks do not replace simulator physics qualification.
`route_distance` computes symmetric sampled Chamfer distance for redundancy
filtering; applications choose their own retention threshold.

## Asset catalog

`Asset` records a semantic ID, category and portable relative filename. `Catalog`
rejects duplicate IDs and resolves installed files only inside the caller's asset
root, including symlink resolution. The public configuration contains no absolute
machine paths. `inventory()` reports the installed catalog, not hypothetical assets.

`appearance_combinations` composes body, clothing, accessory and scale choices,
filtering each through a caller-supplied compatibility predicate. Actual meshes,
the 756 validated appearance configurations and their compatibility tables are
not included. A Cartesian product alone is not an appearance-validity guarantee.

## Physical trace recording

`EpisodeRecorder(output_root, episode_id)` creates a fresh episode directory and
appends validated 10 Hz `Frame` records to `frames.jsonl`. It flushes each frame,
rejects gaps and refuses to overwrite an existing episode directory. Use it as a
context manager to close the file on exceptions. These are evaluator-side physical
records, not the policy input stream.

RGB/video capture, hidden interaction executors, causal expert supervision and
training-dataset assembly are separate integrations. `leadvla.targets` provides
semantic encoding helpers once the observed labels are available; it does not
infer expert actions from a hidden behavior-program name.
