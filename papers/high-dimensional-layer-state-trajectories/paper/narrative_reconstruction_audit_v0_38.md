# Narrative reconstruction audit v0.38

## Scope

`v0.38` is a manuscript-language reconstruction of `v0.37`. It retains the frozen results, figures, references, Methods and Supplementary Methods. Its title was also shortened to 73 characters to meet the Nature Article title-length guidance. It is not a new experiment package.

## One-sentence argument

For transformer inference, we show that treating a prompt-conditioned sequence of high-dimensional layer states as the empirical process object enables coordinated observation, order-sensitive comparison and bounded perturbation, supported by multi-checkpoint geometry controls and a declared relation-graph output-boundary test, without claiming a universal coordinate, dynamics law or correctness controller.

## Terminology ledger

| Canonical term | First-use explanation | Retained boundary |
|---|---|---|
| ordered high-dimensional layer-state trajectory | the sequence of hidden states visited by one prompt across realized layer order | layer depth is not physical time |
| implementation components | weights, heads, MLPs and blocks that implement the computation | complementary to, not replaced by, process analysis |
| clean and conflict candidates | provenance-defined candidates from the relation graph | not synonymous with correct and incorrect answers |
| prompt-condition classes | predefined stable, competition and closure task conditions | not inferred dynamical phases |
| \(\Delta U\) | task-built candidate-separation diagnostic and action-ranking scalar | not an energy, universal coordinate or generally superior readback |
| architecture-compatible propagation scaffold | recurrent order geometry compatible with tested architectures | not learned reasoning, a scale law or an identified architectural cause |
| confidence-gated local-transition actuator | declared gate plus local transition operator | token-boundary control only within the relation-graph task family |

## Narrative changes

1. The abstract now opens with the scientific problem, then gives the three editorial results: coordinate audit, architecture-training-execution separation and declared output-boundary control.
2. The Introduction gives a plain-language relation-graph overview before the mathematical trajectory definition, identifies existing component methods as complementary and states the three-result route map before Results.
3. The candidate-competition section explains \(\Delta U\) as an instrumental action-ranking scalar rather than a privileged representation.
4. The actuator section retains the failed additive interventions and original-budget shortfall, but gives the held-out token-boundary result a distinct paragraph before its action-specificity and transfer limits.
5. The Discussion now moves from central contribution to architecture-training separation, observation-to-intervention linkage, conditional interdisciplinary transfer, then a single compact boundary paragraph.
6. Figure legends 1-3 were clarified for readers who do not know the task vocabulary before seeing the figure.

## Evidence and boundary preservation

- Geometry: `1.60-2.24` fold real-versus-endpoint-preserving shuffle alignment and all listed controls are unchanged.
- Output boundary: `17/87`, `13/87`, `9/87` strict conflict-to-clean changes and `0/203`, `2/203`, `0/203` collateral changes are unchanged.
- Methods: byte-for-byte identical in the Markdown source between v0.37 and v0.38.
- Readbacks remain diagnostics; \(\Delta U\) remains a bounded observable; \(O_l^{\mathrm{cont}}\) remains descriptive when it includes realized transition information; chord-directed geometry remains non-geodesic; steering remains distinct from answer correctness and deployment control.
- The source contains no `ontology`, `semantic manifold` or `basin` claim in the main argument.

## Verification

- Manuscript delta: 27 insertions and 27 deletions relative to v0.37, restricted to the title, Abstract, Main text, Data/Code availability and Fig. 1-3 legends.
- `build_nature_initial_submission_v0_38.py` regenerated the TeX source from the edited Markdown.
- XeLaTeX compiled the PDF in two passes with no errors, undefined references or overfull boxes; three pre-existing underfull boxes remain in the long local-transition-operator Methods paragraph.
- The 37-page PDF was visually inspected at the title and abstract, the task overview and route map, the actuator and Discussion, and representative Extended Data caption pages. No clipping, overlap, blank page or figure-rendering defect was observed.

## Release status

The v0.58 release package is prepared to synchronize the v0.38 manuscript, its PDF and this audit with the public repository and the two existing Zenodo concept records. It retains the frozen v0.56 evidence package and the v0.20 Supplementary Methods. Do not cite v0.58 as publicly archived until both Zenodo version records are published.
