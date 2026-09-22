# LeadBench evaluation protocol

## Frozen conditions

The paper evaluates 108 held-out physical episodes across Core, Easy and
Constrained difficulty. Each difficulty has 12 single-person episodes and 24
multi-person episodes, with the latter forming 12 counterfactual pairs.

Before any policy runs, freeze routes, robot/human initialization, actor
appearances, language instructions, requested formation, behavior/presence
programs, scene interference, event windows, formation-valid intervals, runtime
settings and terminal conditions. Held-out evaluation environments and scene
instantiations are separate from training. Do not alter membership or conditions
after observing an evaluated policy's results.

Each counterfactual pair preserves route, actors, scene and physical interaction
program. Only language-designated target and corresponding evaluator roles are
exchanged. Target-A and target-B rollouts form one logical multi-person case.
Actor IDs and ground-truth target state are evaluator metadata, never policy
inputs. A common policy-visible information boundary and execution environment
do not require identical internal model interfaces or inference frequencies.

## Terminal contract (Appendix C.4–C.5)

| Condition | Paper definition |
| --- | --- |
| Success | Robot and target both within 1.0 m of their frozen terminal regions, separation in [0.8,4.0] m, continuous joint hold for at least 0.5 s, no unrecovered hard failure |
| Sustained collision | Penetration ≥0.10 m for more than 20 consecutive frames at 10 Hz |
| Terminal synchronization timeout | Target fails terminal contract within 3.0 s after robot arrival |
| Route stall | Less than 0.25 m route progress over 20.0 s outside valid waiting periods |
| Route timeout | Frozen episode-specific duration exceeded |

Suspend the stall timer during frozen expected-waiting/target-absence periods,
not whenever the policy voluntarily stops. Target loss, rotation, reverse
progress, separation and overtaking remain diagnostics rather than independent
early termination rules. Retry infrastructure failures and exclude them from
policy evaluation; do not silently convert them into policy failures.

The included offline evaluator consumes termination labels produced by the
runner and independently checks the terminal geometry/hold for LSR. It does
**not** implement the simulator, detect collision penetration or certify that an
external runner followed these rules. `TerminationMonitor` provides stateful
checks using physical penetration and frozen waiting flags supplied by a runner.
LSR is not
gated by minimum RF, TRS or SCS scores.

## Reporting

Report all six difficulty × person-count cells. Combine A/B physical rollouts
into logical cases first. Within a logical case, TRS balances the response
classes present rather than letting frequent lead events dominate. Average cases
within cells, then equally average all six cells for an overall score. The
reference package's exact conventions are in [metrics.md](metrics.md).

Missing A/B partners are errors. A subset of cells can be inspected, but it has
no six-cell aggregate. A complete result export must also be checked against
the externally frozen manifest for missing or unexpected episode IDs; this
package does not ship that manifest and cannot certify suite completeness merely
from the presence of all six cell labels.
