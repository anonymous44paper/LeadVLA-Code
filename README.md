<h1 align="center">LeadVLA</h1>

<p align="center"><strong>Learning Follower-Aware Route Execution for Robot Leading</strong></p>

<p align="center">
  <a href="https://anonymous44paper.github.io/LeadVLA/">Project &amp; Demos</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="leadvla/README.md">LeadVLA</a> ·
  <a href="leadinfra/README.md">LeadInfra</a> ·
  <a href="leadbench/README.md">LeadBench</a> ·
  <a href="examples/README.md">Examples</a>
</p>

LeadVLA enables a robot to lead a language-designated follower along a planned
route while adapting its pace and local motion to the follower's behavior.
This repository brings together the policy components, interaction-construction
tools and closed-loop evaluation utilities.

<p align="center">
  <a href="docs/figures/method.webp"><img src="docs/figures/method.webp" width="100%" alt="LeadVLA overview: front and rear RGB history, language and route context feed shared action queries; Pace and selectively fused Residual are combined with the nominal route by a deterministic Path Composer."></a>
</p>
<p align="center"><em>Keep the route. Learn how to adapt.</em></p>

## Explore the repository

| Component | What it provides | Start here |
| --- | --- | --- |
| **LeadVLA** | Path–Pace–Residual composition, follower-semantic heads, selective fusion, training objectives and inference interfaces | [Model & inference guide](leadvla/README.md) |
| **LeadInfra** | Episode composition, asset interfaces, route planning, compatibility checks and physical trace recording | [Infrastructure guide](leadinfra/README.md) |
| **LeadBench** | Physical response metrics, frozen-manifest validation, termination monitoring, replay and reports | [Evaluation guide](leadbench/README.md) |

## Repository layout

```text
LeadVLA-Code/
├── leadvla/                  # Policy and training components
│   ├── README.md             # Interfaces, tensor shapes and inference usage
│   ├── model.py              # Shared action queries, semantic heads and fusion
│   ├── path_composer.py      # Deterministic Path + Pace + Residual composition
│   ├── inputs.py             # Front/rear RGB history and route-language prompt
│   ├── losses.py             # Masked action and semantic objectives
│   ├── targets.py            # Semantic labels and box encoding
│   ├── training.py           # Optimizer groups and stage-transition utilities
│   └── infer.py              # Inference CLI for a supplied model backend
├── leadinfra/                # Interaction-construction tools
│   ├── README.md             # Composition, route and asset workflows
│   ├── compose.py            # Seeded episode-specification composition
│   ├── catalog.py            # Asset IDs and appearance compatibility
│   ├── routes.py             # Grid planning and arc-length route geometry
│   ├── validation.py         # Route clearance, curvature and redundancy checks
│   └── recording.py          # Append-only physical frame recording
├── leadbench/                # Closed-loop evaluation tools
│   ├── README.md             # Scoring, replay and report workflows
│   ├── metrics.py            # LSR, RF, TRS and SCS
│   ├── manifest.py           # Frozen scoring metadata and identity checks
│   ├── termination.py        # Stateful episode termination monitoring
│   ├── replay.py             # Manifest-driven offline evaluation
│   └── export.py             # Markdown and CSV result tables
├── examples/                 # Small, runnable examples and configuration
│   ├── README.md             # Commands, outputs and example scope
│   ├── composition.json      # Illustrative scene/interaction configuration
│   ├── control_training.py   # Synthetic forward/backward exercise
│   └── synthetic_trace.py    # Synthetic physical trace generator
├── configs/training.json     # Model, loss and training settings
├── docs/                     # Architecture, schemas and protocol details
│   └── figures/              # Method and infrastructure figures
├── tests/                    # Numerical, gradient and integration tests
├── tools/check_release.py    # Working-tree disclosure checks
└── pyproject.toml            # Package metadata and optional dependencies
```

## Installation

Python **3.10+** is required. Run the following from a terminal:

```bash
git clone https://github.com/anonymous44paper/LeadVLA-Code.git
cd LeadVLA-Code
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The activation command above is for a POSIX shell. All subsequent commands in
these guides run from the **repository root**, not from a component subdirectory.

| What you want to use | Installation |
| --- | --- |
| Composition, offline metrics and report tools | `python -m pip install -e .` |
| Grid route planning and validation | `python -m pip install -e '.[infra]'` |
| Model, Composer and training utilities | `python -m pip install -e '.[policy]'` |
| All components and the full test suite | `python -m pip install -e '.[policy,infra]'` |

The policy extra requires PyTorch. For GPU use, install a PyTorch build compatible
with your environment before installing the extra. The bundled examples can run
on CPU. No model or simulator assets are downloaded by these examples.

## Quick start

### 1. Compose episode specifications

```bash
python -m leadinfra.compose examples/composition.json
```

Emits four illustrative episode specifications with a deterministic configuration
hash. This composes identifiers and interaction settings; it does not launch a
simulator. [Configuration and output guide →](leadinfra/README.md)

### 2. Exercise the policy components

With the policy extra installed:

```bash
python -m examples.control_training
```

Runs synthetic action queries through the control heads, Path Composer and joint
loss, then checks a backward pass. Prints named loss terms.
[Model and tensor interfaces →](leadvla/README.md)

### 3. Score a physical trace and export a table

```bash
mkdir -p outputs
python -m examples.synthetic_trace > outputs/synthetic.json
python -m leadbench.evaluate outputs/synthetic.json > outputs/report.json
python -m leadbench.export outputs/report.json --format markdown
```

The synthetic example scores **LSR=0, RF=1, TRS=1, SCS=1**. It contains one Core-S
case, so the six-cell overall score is `null`.
[Evaluation and replay guide →](leadbench/README.md)

## Examples, configuration and tests

- **[Examples](examples/README.md)** are small entry points for learning the APIs.
- **[Training configuration](configs/training.json)** records model, loss and
  optimization settings; it is not a standalone training launcher.
- **[Tests](tests/)** check masks, gradients, selective fusion, geometry,
  counterfactual aggregation and termination behavior.

```bash
# Base-package checks
python -m unittest tests.test_infra tests.test_metrics tests.test_bench_tools -v

# Full suite; requires policy and infra extras
python -m unittest discover -s tests -v
```

## Documentation

| Topic | Guide |
| --- | --- |
| Model structure and training objectives | [Architecture](docs/architecture.md) |
| RGB history, route coordinates and inference backend | [Policy interface](docs/policy.md) |
| Episode composition and information boundary | [LeadInfra concepts](docs/leadinfra.md) |
| Route geometry, asset catalogs and recording | [Infrastructure tools](docs/infra_tools.md) |
| Frozen conditions and reporting contract | [LeadBench protocol](docs/protocol.md) |
| Metric formulas and physical trace schema | [Metrics](docs/metrics.md) |
| Manifest validation, replay and exports | [Evaluation tools](docs/evaluation_tools.md) |

## Resource availability

The pre-built Unreal Engine (UE) infrastructure binaries and trained LeadVLA
checkpoints will be released after the paper review process concludes.

| Component | Resources required beyond this snapshot |
| --- | --- |
| LeadVLA end-to-end training/inference | Backbone/tokenizer integration, compatible Residual decoder, trained weights, dataset and complete training/deployment integration |
| LeadInfra simulation/data generation | Scene and actor assets, route catalog, behavior executors, simulator integration and causal expert |
| LeadBench full benchmark | Frozen 108-episode simulator manifest, scene assets, simulator runner and recorded experiment traces |

## License

Licensing is pending; this snapshot does not grant an open-source license.
No third-party assets, weights or simulator files are bundled. See [NOTICE](NOTICE.md).
