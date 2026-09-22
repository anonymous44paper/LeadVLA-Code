# Examples

[Repository](../README.md) · [LeadVLA](../leadvla/README.md) · [LeadInfra](../leadinfra/README.md) · [LeadBench](../leadbench/README.md)

These examples demonstrate the included APIs with illustrative configuration
and synthetic tensors/traces. They require no trained checkpoint, scene assets
or simulator connection. They are not recorded policy rollouts or paper results.
Run every command below from the **repository root** after installation.

## Choose an example

| File | Type | Demonstrates | Dependencies | Output |
| --- | --- | --- | --- | --- |
| [composition.json](composition.json) | Input configuration | Seeded combinations of route, robot, appearances and interactions | Base package | Four illustrative episode specifications via the composition CLI |
| [control_training.py](control_training.py) | Executable Python example | Control heads, Path Composer, joint objectives and backpropagation | `policy` extra | Named scalar loss values in JSON |
| [synthetic_trace.py](synthetic_trace.py) | Executable Python example | Physical trace format for offline evaluation | Base package | A JSON episode list containing one synthetic Core-S trace |

## Compose episodes

```bash
python -m leadinfra.compose examples/composition.json
```

`composition.json` is data read by the CLI, not a script to execute. Its seed
makes the generated specifications reproducible. Asset identifiers in this file
do not point to installed meshes or scenes. See [LeadInfra](../leadinfra/README.md)
for the fields and how to connect a catalog.

## Exercise control and training losses

```bash
python -m examples.control_training
```

The example creates synthetic shared queries, runs all six follower-related
heads and control branches, composes a trajectory, calculates the losses and
backpropagates the total. The small test decoder is not the paper's trained
Residual decoder. See [LeadVLA](../leadvla/README.md) for actual integration inputs.

## Generate, score and export a trace

```bash
mkdir -p outputs
python -m examples.synthetic_trace > outputs/synthetic.json
python -m leadbench.evaluate outputs/synthetic.json > outputs/report.json
python -m leadbench.export outputs/report.json --format markdown
```

Expected scores are LSR=0 and RF=TRS=SCS=1 for Core-S. The six-cell overall
aggregate is unavailable. `synthetic_trace.py` writes data to standard output;
the shell redirection saves it for the evaluator. These commands overwrite their
named example outputs if repeated, so use distinct output names for real work.
See [LeadBench](../leadbench/README.md) for manifest-driven evaluation.

## Code, configuration and integration templates

- **Library code** lives in `leadvla/`, `leadinfra/` and `leadbench/`.
- **Examples** here exercise that code on self-contained inputs.
- **Training settings** in [configs/training.json](../configs/training.json)
  describe the training configuration, not a distributed training launcher.
- **Inference/replay commands** in the component guides that reference
  `your_backend`, `checkpoints/model.pt` or `manifest.json` are integration
  templates. They require your own resources and are not additional bundled files.
