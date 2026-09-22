# LeadInfra: composition interface

LeadInfra builds an episode, lets the interaction unfold in closed loop, and
generates supervision from what actually happens (paper Appendix B).

The full infrastructure spans 65,198 m² of navigable space, 3 robot embodiments,
756 compatibility-validated human appearances, 25 sensor blueprints, and 142
static objects. These describe the research infrastructure, **not the contents
of this code snapshot**. LeadVLA itself observes front/rear RGB rather than all
available sensor modalities.

## Included composition stage

`leadinfra.compose` combines environment, route, robot, actor, scene, sensing,
and explicitly compatible interaction choices. A seeded generator produces
portable episode specifications and a canonical configuration SHA-256. The hash
does not depend on the location of the input file. Identifiers are semantic keys,
not absolute file paths, simulator handles, or asset URLs.

`examples/composition.json` illustrates the interface. Its assets are fictitious;
the generated specifications are not the training dataset or benchmark manifest.
The composer does not perform physics, route clearance, appearance compatibility,
or scene feasibility validation. Users must supply valid catalogs and compatible
combinations; even a syntactically valid generated episode needs those checks.

The taxonomy follows Tables 11–13: 8 motion programs, 5 presence conditions and
6 distance/bearing formations. The 240 nominal combinations are **not** a claim
that every combination is valid or that all are exhaustively instantiated.
`leadinfra.quotas` provides exact integer allocation for caller-defined sampling
weights; the example uses simple seeded sampling rather than the full training
curriculum.

## Full generation pipeline and information boundary

1. Resolve assets; qualify a nominal route and scene configuration.
2. Select compatible follower/distractor appearances and interaction programs.
3. Render synchronized observations while robot–human interaction unfolds.
4. Let the causal expert react to currently realized follower/scene geometry.
5. Derive trajectory, Pace, and follower-related supervision from that execution.

The package includes portable composition, asset catalog interfaces, grid route
planning, post-planning validation, appearance combination filters and physical
trace recording. Simulator integration, motion and presence executors, the
causal expert and the full image/supervision recorder are not bundled.
The expert may use current realized simulator geometry; it must not read future
ground-truth states or hidden behavior schedules to decide its actions.

Policy input includes language-designated appearance, RGB history and the
**original nominal route**. Evaluator actor identities, expert detours, hidden
behavior states, future schedules and ground-truth target annotations are not
policy inputs. A referent switch requires an updated natural-language instruction,
not a hidden actor-ID change at the policy interface.
