# Manifest validation and offline replay

## Manifest format

An offline manifest identifies the recorded rollouts and freezes their scoring
metadata. It contains `schema_version: 1` and an `episodes` list. Each entry has:

```json
{
  "episode_id": "example_001",
  "case_id": "example_case",
  "cell": "Core-S",
  "variant": "S",
  "contract": {"distance": "far", "bearing": "center"},
  "events": [{"source": "target", "response": "lead", "onset": 1.0, "grace": 0.0}],
  "trace_file": "example_001.json"
}
```

This example is illustrative, not an official benchmark episode. Trace files
contain one episode object using the [metric schema](metrics.md), including its
physical `frames` and `termination`. Paths are relative to the supplied trace
root; absolute paths, traversal and escaping symlinks are rejected.

This offline manifest does not replace the complete simulator manifest containing
route geometry, actors, natural-language instructions, behavior programs, layouts
and runtime settings. Those resources are not distributed here. Identity checking
of offline logs cannot certify that an external simulation used identical scenes
or instructions. That must be guaranteed by the simulator-side manifest.

## Reproducible selection

`validate_manifest()` checks semantic IDs, unique trace files, response events,
interaction contracts and complete S or A/B logical cases. Its SHA-256 uses sorted
canonical JSON, independent of the file location. Freeze and record that digest
before evaluating policies. `replay(..., expected_digest=...)` fails if the
manifest contents change, a file is missing or a trace differs from its specified
episode ID, case, cell, variant, contract or event schedule.

`--require-full-suite` additionally requires 12 single-person and 24 multi-person
physical episodes for each of Core, Easy and Constrained. Correct counts do not
by themselves prove use of the official held-out suite; the externally recorded
manifest digest must also be the intended one.

Replay processes exactly the manifest's files. Unlisted debug files are not
included. It does not retry failed policy runs, silently drop missing rollouts,
or synthesize missing metric values.

## Termination monitoring

`TerminationMonitor(timeout_seconds)` consumes one physical `Frame` every 0.1 s,
starting at episode time zero, plus penetration depth and a frozen expected-wait
flag. It implements joint success, sustained collision, synchronization timeout,
route stall and per-episode timeout. The waiting flag must not be derived from
the evaluated policy's stopping decision.

Sample intervals ending at a frozen waiting frame do not advance the stall
clock. The monitor applies collision, successful hold, synchronization timeout,
stall and episode timeout in that precedence order if conditions coincide. It
latches the first terminal result; further updates are errors. A runner supplies
penetration geometry and performs simulator lifecycle/retry handling.

The offline metric evaluator consumes the recorded terminal label and checks
success geometry/hold; it cannot reconstruct unrecorded collision penetration.

## Reports

`leadbench.export` renders Markdown or CSV with all six cells and an overall row.
Missing cells and unavailable overall values are marked with an em dash. Metrics
remain in [0,1]; rendering to three decimals does not change the stored values.
The aggregate first combines A/B rollouts into logical cases and then assigns
equal weight to the six conditions, as specified in the metric guide.
