# Deep humanizer audit v0.40

## Scope and invariants

This is a second, full narrative pass from `manuscript_v0_39_humanized.md` to `manuscript_v0_40_deep_humanized.md`.

- Edited: Abstract, Introduction, Results, Discussion and all main/Extended Data figure legends.
- Frozen: title, authorship, references, Methods, source-data links, availability statements, declarations, numerical results, formulae, figure assets, experimental status and claim boundaries.
- Mode: Humanizer v3.0 academic and technical argument mode.

## Pattern classes audited

1. **Formulaic contrast.** Removed `not ... but ...`, `not only`, `rather than` and `instead of` constructions from the Abstract, main text and figure legends. Necessary negative claims remain as separate declarative sentences.
2. **Mechanical signposting.** Replaced sequential scaffolds such as “First, Second, Third”, “We next” and “We then” where the prose could name the analysis directly.
3. **Generic self-reference.** Replaced phrases such as “This study establishes”, “This design yields” and “The contribution is” with the observed object, analysis or result.
4. **Abstract-noun stacking.** Reduced labels such as “counterintuitive controls” and compressed constructions such as “candidate-coordinate compactness, predictive closure and cross-task actuator transfer” where a direct description was clearer.
5. **Ritualized conclusion.** Reworked the Discussion so that evidence, interpretation and scope appear in separate sentences rather than in a single conclusion-and-disclaimer pattern.
6. **Caption opacity.** Split multi-clause legends, used panel-specific subjects and moved necessary scope boundaries into short declarative sentences.
7. **Unsupported emphasis.** Removed or replaced generic emphasis and transition words where they added no evidence. Necessary causal and statistical language remains.

## Results of the pattern scan

Within the Abstract and main text, and separately within figure legends:

- `not ... but ...`: 0 occurrences.
- `not only`, `not merely`, `not just`: 0 occurrences.
- `rather than`, `instead of`: 0 occurrences.
- “This study”, “This framework”, “The contribution”, “This design yields”: 0 occurrences.
- Em and en dashes: 0 occurrences.

Residual negative language is retained only when it expresses a factual boundary, including that candidate labels do not denote correctness, \(\Delta U\) is not an energy or universal order parameter, and the output intervention does not establish answer-correctness, deployment or open-ended generation control.

## Integrity check

The Methods section from `## Methods` to `## Data availability` has the same SHA256 in v0.39 and v0.40:

`42CA080371BFCF037BA98627AC91E023E27B03A7C8C89C8C6735C532C76E00D2`

The v0.40 pass is language-only. It changes no evidence value, equation, figure, citation, dataset, outcome, statistical threshold, or scope ceiling.

The Nature-format PDF compiled twice with XeLaTeX to 37 pages. Its SHA256 is `1D6DE55461256EBE4E208BB9311FDC07E7F80839B2A0CE6F5F93978D51FBF34D`. The compile log contains no LaTeX errors, undefined control sequences, undefined references or overfull boxes. Visual checks covered the title and abstract, the Discussion and an Extended Data figure-legend page.
