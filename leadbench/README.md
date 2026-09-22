# LeadBench — closed-loop evaluation tools

[Repository](../README.md) · [LeadVLA](../leadvla/README.md) · [LeadInfra](../leadinfra/README.md) · [Examples](../examples/README.md)

LeadBench measures whether a robot completes the route with its designated
follower while preserving route fidelity and responding to the interaction.
This package scores physical traces, validates frozen scoring metadata and
exports reports. It does not launch the simulator or provide the held-out suite.

## Code map

| File | Main interface | Responsibility |
| --- | --- | --- |
| [contract.py](contract.py) | `Contract` | Requested distance and bearing ranges |
| [metrics.py](metrics.py) | `Frame`, `Event`, `evaluate_episode` | Physical LSR/RF/TRS/SCS computation |
| [report.py](report.py) | `aggregate` | Counterfactual pairing and equal-cell aggregation |
| [evaluate.py](evaluate.py) | `python -m leadbench.evaluate` | Score an episode-list JSON file |
| [manifest.py](manifest.py) | `validate_manifest`, `digest` | Frozen scoring identity, membership and checksums |
| [replay.py](replay.py) | `replay`, `python -m leadbench.replay` | Evaluate files named by a frozen manifest |
| [termination.py](termination.py) | `TerminationMonitor` | Stateful physical episode termination |
| [export.py](export.py) | `render`, `python -m leadbench.export` | Markdown and CSV tables |

## Score the bundled example

Run from the **repository root**. Offline scoring uses the base package and
requires neither GPU inference nor simulator assets:

```bash
python -m pip install -e .
mkdir -p outputs
python -m examples.synthetic_trace > outputs/synthetic.json
python -m leadbench.evaluate outputs/synthetic.json > outputs/report.json
python -m leadbench.export outputs/report.json --format markdown
python -m leadbench.export outputs/report.json --format csv > outputs/report.csv
```

Expected synthetic-example scores:

| Condition | LSR | RF | TRS | SCS |
| --- | --- | --- | --- | --- |
| Core-S | 0.000 | 1.000 | 1.000 | 1.000 |
| Overall | — | — | — | — |

The synthetic robot progresses along a clean route with good target geometry but
does not reach the goal. Only Core-S is present, so `overall` is `null` rather
than an aggregate over a partial benchmark. This is a software example, not an
experimental result.

## What the metrics measure

| Metric | Question | Scored from |
| --- | --- | --- |
| **LSR** — Lead Success Rate | Did robot and target finish together? | Joint terminal geometry, separation, hold duration and failure status |
| **RF** — Route Fidelity | Did the robot preserve the nominal route? | Actual route deviation and projected forward progress |
| **TRS** — Target Response Score | Did it respond appropriately to the designated follower? | Realized progress and target distance around frozen lead/slow/wait/resume events |
| **SCS** — Social Contract Score | Did it maintain the requested distance and formation? | Visibility, distance and bearing on frozen formation-valid frames |

TRS does not use predicted Pace as a proxy for physical behavior. Distractor
events expect continued leading and are evaluated using the designated target.
Loss of target visibility remains an SCS failure; it does not remove the frame
from the denominator. See [metric definitions](../docs/metrics.md) for formulas,
sampling windows and edge cases.

## Supply your own traces

`leadbench.evaluate` accepts a JSON object with `schema_version: 1` and an
`episodes` list. Each episode contains its ID, logical case, cell, variant,
contract, frozen events, physical frames and termination label.

| Data | Required content |
| --- | --- |
| Identity | `episode_id`, `case_id`, `cell`, `variant` |
| Contract | Requested near/far distance and left/center/right bearing |
| Events | Source, expected response, onset and reaction grace |
| Frames | Contiguous 10 Hz robot/target geometry, visibility and frozen scoring masks |
| Termination | Success or a defined policy-failure termination label |

Use realized geometry, not predictions or hidden policy-specific scores.
Infrastructure failures must be retried outside this evaluator. The complete
field definitions are in the [trace schema](../docs/metrics.md#portable-json-interface).

## Replay a frozen manifest

**Requires your own manifest and recorded rollouts.** Unlike `evaluate`, which
reads a list of episodes from one JSON file, `replay` reads one episode JSON file
per manifest entry under a supplied trace root:

```bash
python -m leadbench.replay manifest.json traces \
  --expected-digest YOUR_FROZEN_DIGEST \
  --require-full-suite > outputs/full_report.json
python -m leadbench.export outputs/full_report.json --format markdown
```

`YOUR_FROZEN_DIGEST` denotes the actual SHA-256 recorded when freezing the
manifest; it is not a literal value to use. The replay command rejects missing
files, changed scoring identity, incomplete A/B pairs and mismatched digests.
The full-suite option requires 12 S and 24 M physical episodes per difficulty.
Omit it for a deliberately smaller development manifest.

The offline manifest freezes scoring membership, contracts and events. It is
not the full simulator manifest for scenes, instructions, actors and behaviors.
Those conditions must be frozen by the external runner. See the
[manifest format and scope](../docs/evaluation_tools.md).

## Aggregation and runtime integration

Reports contain **Core / Easy / Constrained × single-person / multi-person**.
Each A/B counterfactual pair forms one logical multi-person case. TRS balances
response classes within that logical case. Cases are averaged within cells;
overall values equally weight all six cells.

`TerminationMonitor` consumes physical `Frame` records, measured collision
penetration and frozen expected-wait flags. It checks joint completion, sustained
collision, terminal synchronization timeout, route stall and route timeout.
The calling runner supplies physics and handles simulator lifecycle/retries.
See [termination details](../docs/evaluation_tools.md#termination-monitoring).

## Tests and further reading

```bash
python -m unittest tests.test_metrics tests.test_bench_tools -v
```

See [evaluation protocol](../docs/protocol.md), [metric definitions](../docs/metrics.md),
[replay/report tooling](../docs/evaluation_tools.md), and the
[synthetic trace generator](../examples/synthetic_trace.py).
