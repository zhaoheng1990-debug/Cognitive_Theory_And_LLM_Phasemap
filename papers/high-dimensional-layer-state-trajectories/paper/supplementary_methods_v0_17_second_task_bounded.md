# Supplementary Methods: object-switch trajectory analysis and human-AI workflow

## 1. Empirical-object hierarchy

The study separates three levels that are easily conflated.

1. **Implementation objects:** model weights, attention heads, MLPs, residual blocks and layers. These are the engineered objects that realize inference.
2. **Empirical process object:** the ordered hidden-state sequence \(T_{m,p}=(h_{m,p,1},\ldots,h_{m,p,L_m})\). This is the object on which the present mechanism comparisons are defined.
3. **Coordinates and diagnostics:** direct hidden summaries, hidden-state flow, vocabulary-facing TopK/VIM readbacks, the decision-window clean-relative margin profile, \(\Delta U\), transition coordinates and residual diagnostics.

The methodological statement "empirical research object before coordinates" means choosing a testable process object before selecting measurements. It does not claim that engineering objects are unimportant or make a metaphysical claim about the layer-state trajectory.

H-shape denotes the ordered profile of clean-relative candidate margins across a specified layer window. It is a trajectory-shape summary, not a hidden-state tensor or an independent mechanism label. Earlier analyses also used related shorthand for window summaries and previous-transition history; those objects are kept distinct here. The decision-window H-shape is exactly \(dR\) over decision layers and is construction-aligned with \(\Delta U\). Legacy labels remain only in the Source Data manifest for provenance.

## 2. Intellectual provenance and operational boundaries

| Manuscript object | Mathematical or physical source idea | Operational use here | Identity explicitly not claimed |
|---|---|---|---|
| Layer-state trajectory | State trajectory of a discrete dynamical system | Ordered sequence of last-token hidden states across blocks | Layer depth is not physical time |
| Exploratory registration | Functional or phase registration | Data-selected windows provide model-specific indexing choices for downstream analyses | Prospective joint-role recurrence failed; windows are not confirmed phases or identical coordinates |
| \(\Delta U\) | Reaction coordinate and order-parameter compression | Oriented PC1 of a clean-relative decision-window margin matrix | Not energy, potential, thermodynamic order parameter or full mechanism |
| Transition closure | State-space and reduced-order closure | Test whether current-state and realized-transition coordinates compactly describe the observed update | Not an autonomous dynamical law when target-transition information enters |
| Chord-directed geometry | Finite-path alignment, detour and turning | Compare real order with endpoint-preserving shuffles under a declared cosine geometry | Not a learned metric, exact geodesic, least action or conservation law |
| Structured residual | Residual diagnostics in reduced-order modelling | Test whether deviation from the chord-directed reference retains information | Not a complete tangent-normal decomposition |
| Local-transition actuator | Feedback-control distinction between state estimation, abstention and actuator choice | Mechanism-conditioned transition operator plus confidence-gated bounded action | Crosses a declared emitted-token boundary in three checkpoints; not answer-correctness, safety or deployment control |

The source ideas were introduced after empirical problems required a compact description. They were not used as labels that made an observation true. Negative results narrowed each analogy.

## 3. Controlled prompt subset

The direct hidden-state analyses used eight complete graph groups selected with seed 20260610: group IDs 0, 2, 15, 23, 25, 26, 51 and 63. All 12 conditions per group were retained, giving 96 prompts. The same source rows were used for all three checkpoints. This design preserves within-group condition structure while making 50 layer-order shuffles tractable. It is not a first-row convenience sample.

The subset manifest contains prompt ID, group and graph IDs, raw condition, analysis label, task-defined clean and conflict candidates, source row and model-specific readback records. Final generated-answer correctness is not used in \(\Delta U\), coordinate-audit labels, intervention-class labels or policy labels.

Each prompt encodes a short relation chain from named entities to one of two declared answer candidates. The clean candidate is supported by the unperturbed chain. The conflict candidate is introduced by a distractor, competing statement, update or exception. Stable conditions retain a decisive clean chain, competition conditions support both candidates, and closure conditions redirect the chain. These labels are fixed by prompt construction before model fitting. They are analysis classes, not dynamical states discovered from the hidden trajectories. The task was selected to isolate evidence competition and updating under controlled counterfactual conditions; no claim is made that it represents open-ended reasoning.

**Supplementary Methods Table 1 | Analysis-label and information-boundary crosswalk.**

| Analysis family | Source object and raw conditions | Fitted target or endpoint | Inputs permitted | Information explicitly excluded |
|---|---|---|---|---|
| Coordinate audit and 96-prompt geometry subset | Twelve-condition panel. Stable: clean, rename, irrelevant, paraphrase and weak distractor (stored as stable-shift). Competition: balanced competition, direct conflict, equal evidence and source claim (stored as source-ambiguity). Closure: closure update, closure override and exception override. | Three-class prompt-condition label for proxy comparison or plot grouping | Prompt metadata for labels; fold-fitted predecision state and the audited proxy for prediction | \(\Delta U\), decision-window clean-relative profile, generated answer and correctness |
| Functional-window scan | Separate 12-condition discovery panel. Stable: clean, rename, permuted, redundant, irrelevant and paraphrase; weak distractor stored as stable-shift and collapsed with stable. Competition: balanced competition and direct conflict. Closure: closure update, closure override and exception override. | Exploratory preservation, class-information and downstream-coupling score | Candidate window measurements within graph-aware folds | Confirmatory interpretation; selection is not nested |
| Prospective functional-window boundary | Same 12 templates on 24 new graph groups with disjoint entities, relation phrases and candidate labels | Frozen-window role recurrence and rank | Discovery-only PCA and fitted models; confirmation labels only for final scoring | Window reselection or confirmatory claims after a failed gate |
| \(\Delta U\) construction | Declared clean and conflict candidate identities within each graph group | Oriented PC1 score of the clean-relative decision-window matrix | Candidate margins in the declared window; training-fold PCA where transfer is evaluated | Generated answer, final correctness and an analysis-independent mechanism label |
| Internal-control and confidence-gated intervention | Separate ten-condition panel. Stable: clean, redundant, irrelevant and paraphrase. Competition: weak distractor, balanced competition and direct conflict. Closure: closure update, closure override and exception override. | Three-class intervention label used by the mechanism classifier and gate | Training-graph prompt labels and pre-intervention trajectory features | Held-out outcomes, generated answer and correctness |
| Policy supervision | Eight-action internal library or expanded ten-action boundary library | Discrete \((\alpha,\beta)\) action maximizing the declared training utility; `none` is \((0,0)\) | Training-graph intervention responses and fitted mechanism features | Held-out action outcomes and correctness |
| Candidate-boundary endpoint | Paired no-intervention and controlled continuations | Conflict-to-clean crossing of the two declared candidates; in the expanded test, strict full-vocabulary top-1 crossing | Held-out outputs only for evaluation | Use as a fitting label; interpretation as correctness or open-ended answer control |

The coordinate-audit and intervention panels therefore do not share a universal latent class variable. In particular, weak distractor is stored as stable-shift and collapsed with stable in the coordinate audit, but is grouped with competition in the separately generated intervention panel. Both mappings were fixed in their corresponding data-generation and analysis scripts before model fitting. They are retained rather than harmonized after observing outcomes.

## 4. Direct hidden-state update summaries for Fig. 1

For normalized states \(x_l=h_l/\lVert h_l\rVert_2\), adjacent hidden-state change is

\[
c_l=1-\cos(x_l,x_{l+1}).
\]

Let \(s_l=x_{l+1}-x_l\) and hidden size \(d\). The normalized participation ratio is

\[
P_l=\frac{(\sum_{j=1}^{d}s_{l,j}^2)^2}{d\sum_{j=1}^{d}s_{l,j}^4}.
\]

\(P_l\) approaches one when step energy is evenly distributed and decreases when fewer dimensions dominate. Fig. 1 reports means and standard errors by collapsed coordinate-audit condition family and checkpoint. These families are predefined plot groups, not inferred trajectory phases. The calculations use native hidden states and no vocabulary projection. Executable filenames are listed in the Source Data manifest.

## 5. Readback and projection-selection controls

For clean candidate token \(c_p\), conflict candidate token \(e_p\) and output matrix \(W\),

\[
R_{p,l}=h_{p,l}^{\top}(W_{c_p}-W_{e_p}).
\]

TopK/VIM features summarize projected vocabulary neighbourhoods. The row-permuted control shuffles rows of \(W\), preserving matrix values but changing token correspondence. The Gaussian control uses a seeded random matrix of matched shape. Because these controls operate after hidden-state extraction, they audit coordinate dependence of the vocabulary-facing geometry without changing the native hidden-state trajectories.

For the independent prompt-class audit, every proxy augmented the same predecision base state. Proxy PCA was capped at 32 dimensions and fitted separately inside each training fold. Five-fold GroupKFold used relation graph as the split unit. In the 12-condition source panel, clean, rename, irrelevant and paraphrase were stored as stable; weak distractor as stable-shift; balanced competition, direct conflict and equal evidence as competition; source claim as source-ambiguity; and the three update or override conditions as closure. Stable and stable-shift then mapped to the stable classifier class; competition and source-ambiguity mapped to competition; closure mapped to closure. Macro F1 is the unweighted mean of class-specific F1 scores. The reported value is augmented minus base prompt-class macro F1. The label map is fixed by controlled prompt metadata and does not use \(\Delta U\), generated answers or the decision-window clean-relative margin profile.

**Algorithm 1: independent coordinate audit**

```text
for each checkpoint and proxy family:
    partition relation graphs into five folds
    for each held-out fold:
        fit all preprocessing, including proxy PCA, on training graphs only
        fit the base mechanism classifier on predecision state features
        fit the augmented classifier on the same base plus the proxy
        predict the held-out graphs
    concatenate held-out predictions across folds
    report macro-F1(augmented) - macro-F1(base)
```

## 6. Exploratory functional-window scan and prospective boundary test

Candidate windows span the first 45% of checkpoint depth. Window lengths are 2, 3, 4, 5, 6 and 8, step size is one layer, and TopK neighbourhood sizes are 100 and 500. For each window, the preservation score is one minus the mean clean-relative distance for relation-preserving conditions. The source field was named topology, but the quantity does not compare relation-preserving with structure-changing conditions and is not itself a topology-separation test. Prompt-condition-class information is GroupKFold macro F1, and downstream coupling is a GroupKFold correlation with \(\Delta U\). The objective is

\[
S=0.5S_{\mathrm{preservation}}+0.8\max(S_{\mathrm{class}},0)+0.8\max(S_{\mathrm{downstream}},0).
\]

The maximum-scoring interval is retained. Selection is non-nested because the same scan supplies candidate and selection score. GroupKFold prevents graph overlap in component estimates but does not remove selection optimism. The windows are exploratory and data-selected.

**Algorithm 2: exploratory functional registration**

```text
for each checkpoint:
    enumerate all permitted starts, lengths and neighbourhood sizes
    for each candidate window:
        calculate clean-relative preservation
        calculate group-held-out prompt-condition-class macro F1
        calculate group-held-out downstream correlation
        combine the three quantities using the stated score S
    retain the maximum-scoring window
    label the result exploratory because selection and evaluation share the scan
```

### 6.1 Prospective confirmation protocol and result

After the exploratory scan, the overall windows and neighbourhood sizes were frozen as Qwen layers 5-6 with \(K=500\), Llama layers 0-2 with \(K=500\), and Gemma layers 1-6 with \(K=100\). The prospective confirmation set comprised 24 new relation graphs and the same 12 perturbation templates. It replaced every discovery entity triple and relation pair and used 12 candidate labels drawn from an audited common one-token vocabulary that was disjoint from the discovery candidate labels. No confirmation prompt entered the original scan.

For each checkpoint, the discovery rows alone fitted the clean-relative decision-profile PCA, its sign orientation, feature scaling, a balanced three-class logistic classifier and ridge regression to \(\Delta U\). The fitted objects were then applied to confirmation rows. Confirmation labels were used only to score the predeclared endpoints. The full candidate pool was evaluated to obtain the rank of the already frozen window; no new winner replaced it. Two thousand graph-group bootstraps produced 95% intervals. Nine hundred and ninety-nine discovery-label permutations formed the class-information null, and 999 within-discovery-graph \(\Delta U\) permutations formed the downstream null.

The frozen per-checkpoint gate required: (i) a positive 95% bootstrap interval for the graph-paired difference between structure-changing and relation-preserving composite distances; (ii) macro F1 at least 0.55 and greater than the label-null 95th percentile; and (iii) positive downstream correlation greater than the target-null 95th percentile. Window-location recurrence was declared separately when the frozen window ranked in the top quartile of the unchanged candidate pool. The protocol and its SHA-256 hash were written before extraction. One initial run stopped after Qwen extraction because the imported feature function returned two objects rather than three; no endpoint was calculated. The interface correction and failed-run logs are retained.

**Supplementary Methods Table 2 | Prospective functional-window confirmation on 24 untouched graph groups.**

| Checkpoint | Topology separation (95% bootstrap interval) | Class macro F1 / label-null 95th | Downstream \(r\) / target-null 95th | Frozen rank | Joint role gate | Location gate |
|---|---:|---:|---:|---:|---|---|
| Qwen | -0.185 (-0.189, -0.181) | 0.739 / 0.521 | 0.659 / 0.623 | 58/112 | Fail | Fail |
| Llama | -0.092 (-0.092, -0.091) | 0.662 / 0.510 | 0.411 / 0.452 | 38/40 | Fail | Fail |
| Gemma | -0.090 (-0.091, -0.089) | 0.649 / 0.475 | 0.624 / 0.404 | 5/100 | Fail | Pass |

The class-information endpoint transferred in all checkpoints. Downstream coupling passed its frozen null in Qwen and Gemma but not Llama. Topology separation reversed in every checkpoint, and only Gemma retained top-quartile location. The three-checkpoint role-recurrence claim is therefore rejected. These results do not invalidate use of the discovery windows as explicitly exploratory analysis choices, but they prohibit treating the windows as confirmed functional phases or homologous cross-checkpoint locations.

## 7. Executable \(\Delta U\) definition

Within each graph group, the clean condition supplies a baseline. If a clean row is absent, the group mean is used. The clean-relative margin is

\[
dR_{p,l}=R_{p,l}-R_{p_0,l}.
\]

For mapped decision layers \(\mathcal L_{\mathrm{dec}}\), form

\[
D=[dR_{p,l}]_{p,l\in\mathcal L_{\mathrm{dec}}}.
\]

PCA is fitted to \(D\), and \(\Delta U_p\) is its PC1 score. The sign is oriented so that stable or stable-shift rows have greater mean scores than closure rows when those groups are available. Candidate identities from the controlled task enter \(R\); the model's generated answer and final correctness do not. \(\Delta U\) is therefore a task-constructed exploratory diagnostic of declared candidate-path separation. Its numerical axis is not transferred to a different prompt family. The controlled lexical and addition tests fit new task-specific coordinates with the same protocol and report held-out separation independently.

The decision-window clean-relative margin profile is \(D_{p,:}\). It cannot serve as independent evidence for \(\Delta U\) because both are the same construction at different dimensionalities. Correlation with a dominance readout derived from related decision-window information is a sanity check. Independent bounding analyses remove the direct margin, exclude decision and endpoint information, change the window, test residual signal or compare intervention nulls.

## 8. Transition-coordinate closure

Previous-transition history features comprise the preceding clean-relative margin change, preceding rank-gap change, reversals in both signals, and backtracking relative to the preceding and initial centres. The realized transition coordinate \(O_l^{\mathrm{cont}}\) comprises the first ten principal components of the observed transition representation. Closure models compare state-only, history-only, combined state-history, state plus \(O_l^{\mathrm{cont}}\), and bilinear state-transition feature sets.

If adjacent target-layer information enters \(O_l^{\mathrm{cont}}\), the analysis is descriptive factorization. Macro \(R^2\) is the unweighted mean of target-wise \(R^2\) values. The predictive audit estimates \(\hat O_l^{\mathrm{cont}}\) from current-state, previous-transition history and optional layer/condition context under graph-grouped cross-validation, excluding the realized target transition. Within each outer fold, the first-stage model predicts \(\hat O_l^{\mathrm{cont}}\) separately for training and test rows. The second-stage model then constructs state-coordinate products independently from each fold's own predictions; no observed \(O_l^{\mathrm{cont}}\) enters that stage. State-only closure reached macro \(R^2=0.822\), observed \(O_l^{\mathrm{cont}}\) reached 0.979, and the best target-excluded bilinear model using \(\hat O_l^{\mathrm{cont}}\) reached 0.862. Its increment over state only was 0.0394 (95% prompt-group bootstrap interval, 0.0368-0.0426), and it retained 25.1% of the observed-coordinate gain. Models predicting \(O_l^{\mathrm{cont}}\) itself ranged from macro \(R^2=0.41\) to 0.61. All four corrected bilinear feature sets completed five graph-grouped folds and improved over their matched additive versions by 0.0279-0.0318 macro \(R^2\). This is target-excluded incremental prediction, but remains exploratory and does not establish a task-general forward law. Metric-specific embedding diagnostics are not combined because distance and correlation scores have different scales and interpretations.

### Prospective final candidate-margin forecast

This separate analysis predicts the exact final clean-minus-conflict output-logit margin from partial decision-window histories. The frozen relation-graph split contains 67 training groups (670 prompts) and 29 held-out groups (290 prompts). At cutoff layer \(k\), the ordered-history predictor contains only the candidate-margin values observed at decision layers \(l\leq k\). It excludes all later states and transitions, the final generated token and answer-correctness labels. Ridge regression uses penalty 10. A balanced logistic regression with \(C=1\) predicts whether the final candidate margin is positive from the same features.

For the order null, the current-cutoff value is fixed and all earlier values are independently permuted within each prompt. This avoids the feature-renaming invariance of a common permutation followed by model refitting. Fifty null datasets are independently refitted and evaluated on the unchanged held-out graph groups. Confidence intervals use 1,000 relation-graph bootstrap replicates. The earliest real-history cutoff above the null \(R^2\) 95th percentile is Qwen layer 24 (\(R^2=0.574\), AUROC \(=0.857\)), Llama layer 14 (\(R^2=0.393\), AUROC \(=0.837\)) and Gemma layer 21 (\(R^2=0.714\), AUROC \(=0.984\)). This is a bounded prospective forecast of a declared candidate contrast. It is not an autonomous state equation or a general model of answer correctness.

## 9. Direct hidden-state chord geometry

For normalized hidden states, the endpoint chord is \(g=x_L-x_1\). Chord alignment is the mean cosine between each step \(s_l\) and \(g\). Path length is the sum of adjacent cosine distances, endpoint distance is the first-to-last cosine distance, detour is their ratio, and turning curvature is the mean angle between adjacent steps.

Each null retains \(x_1\), \(x_L\) and the complete intermediate state set but randomly permutes intermediate layer order. One permutation is shared across all 96 prompts within a checkpoint, and 50 permutations are evaluated. The 50 checkpoint-level means form the null distribution; prompt-level p values are not pooled. Alignment gain is real minus shuffle mean. Detour and curvature reductions are \(\log_2(\bar q_{\mathrm{shuffle}}/\bar q_{\mathrm{real}})\). Positive reductions favour observed order.

For a metric where larger values favour observed order, the null-standardized effect is

\[
Z_q=\frac{q_{\mathrm{real}}-\operatorname{mean}(q_{\mathrm{null}})}{\operatorname{sd}(q_{\mathrm{null}})}.
\]

For detour and curvature the numerator is reversed so that positive values again favour observed order. The plus-one empirical one-sided p value is \((1+n_{\mathrm{extreme}})/(50+1)\). All three observed checkpoint means were beyond all 50 corresponding alignment null means, giving the minimum attainable empirical value of \(1/51\); this finite resolution is reported rather than interpreted as a more precise p value. The standardized separation was 5.11, 5.57 and 5.63 null standard deviations for alignment; 11.87, 6.31 and 11.50 for detour; and 6.03, 4.20 and 6.42 for curvature in Qwen, Llama and Gemma, respectively.

**Algorithm 3: endpoint-preserving layer-order null**

```text
for each checkpoint:
    compute one real checkpoint mean from the 96-prompt controlled subset
    repeat 50 times:
        draw one permutation of intermediate layer indices
        apply that same permutation to every prompt
        retain the first and final layer states
        compute the checkpoint mean for each geometry metric
    compare the real mean with the 50 repeat-level means
```

The estimand is limited to normalized hidden states, cosine geometry and this endpoint-preserving null. It does not test an exact geodesic equation. A fixed-seed rerun reproduced the archived output hashes; the hashes and filenames are recorded in the Source Data manifest.

## 10. Independent task-family and random-initialization controls

### 10.1 Controlled lexical and addition tasks

The independent task family asks which of two natural words belongs to a declared semantic category. It contains 64 problem groups and seven category inventories: colours, directions, metals, shapes, landscape features, sky objects and fruits. Candidate words were drawn from a 28-word intersection that tokenized as one continuation token in Qwen, Llama and Gemma. Correct-candidate display position alternated by problem identifier. Two stable prompts, two prompts containing a competing annotation and two correction prompts were generated for each problem, giving 384 prompts per checkpoint. The prompts used each checkpoint's native chat template.

The external task inherited the frozen relation-graph decision windows: layers 20-25 in Qwen, 12-15 in Llama and 18-23 in Gemma. No layer scan or task-specific window selection was performed. Problem groups were divided once with seed 20260714 into 44 training and 20 held-out groups. For each problem, the clean prompt supplied the candidate-margin baseline. PC1 was fitted to the training-group clean-relative decision profiles and applied to the held-out groups. A balanced multinomial logistic classifier used only this one-dimensional coordinate. Its held-out macro F1 was compared with 200 fits in which training labels were permuted while held-out labels and the split were unchanged.

The direct hidden-state geometry analysis retained all 384 prompts and used 50 endpoint-preserving intermediate-layer permutations per checkpoint. Real-order alignment exceeded all 50 null means in each checkpoint. Held-out lexical-coordinate macro F1 exceeded the label-null 95th percentile in Qwen and Llama but not Gemma. Clean-condition candidate selection was 0.891, 0.547 and 0.953, respectively, so the Llama result is retained with a weak-behaviour boundary.

The controlled addition task contains 64 single-step addition problems and the same six-condition design. It uses the same frozen decision windows, 44/20 problem split, train-only PC1 construction, 200 label shuffles and 50 endpoint-preserving order shuffles. Geometry exceeded all order nulls in all checkpoints. Held-out coordinate macro F1 was 0.583, 0.456 and 0.570 in Qwen, Llama and Gemma, above the corresponding label-null 95th percentiles. Llama pair accuracy was 0.315, so the coordinate result does not imply strong arithmetic behaviour. An earlier mixed-operation version is retained as a coverage audit rather than a primary result.

### 10.2 Standard four-choice ARC control

The standard benchmark control used the allenai/ai2_arc ARC-Challenge validation split. The 96-item audit subset comprised the earliest validation rows in dataset order whose answer keys were exactly A-D. This deterministic eligibility rule was fixed without reference to checkpoint outputs, bounding the extraction budget while preventing outcome-based item selection. Every original question and all four answer choices were presented through the checkpoint-native chat template, and the model was instructed to return one option letter. No binary clean/conflict wrapper was introduced. Four-choice accuracy was 0.781, 0.333 and 0.771 in Qwen, Llama and Gemma.

Hidden states were extracted using the same last-token rule as the main analysis. For each checkpoint, 50 endpoint-preserving intermediate-layer shuffles supplied the null. Real-order alignment exceeded all 50 null means, with standardized separations of 3.21, 5.23 and 7.84. The deterministic subset is an audit set rather than a new benchmark estimate. This analysis tests the order geometry beyond binary prompt structure; it does not transfer \(\Delta U\), mechanism labels or the intervention policy.

### 10.3 Random-initialization control

The random control used the exact Qwen2.5-1.5B, Llama-3.2-1B and Gemma-2-2B model configurations and native tokenizers. Three models per configuration were instantiated with seeds 1701, 1702 and 1703; token embeddings, attention and MLP parameters, normalization parameters and output heads were random, and no trained tensor was loaded. The inputs were the same fixed 96 raw relation-graph prompts used in the trained geometry audit. Hidden states were extracted after every transformer block using the same last-token rule.

Each parameter state was compared with 50 endpoint-preserving layer-order permutations using the same metric definitions and null seed 20260717. Mean random alignment gains were 0.0768, 0.1028 and 0.0879 in Qwen, Llama and Gemma, versus 0.0317, 0.0403 and 0.0499 in the trained checkpoints. Fig. 4 used a separately seeded 50-shuffle null; its small gain differences arise from the sampled null mean, while the trained real-order alignment is identical. All nine random gains were positive and exceeded their matched trained checkpoint. The result establishes an architecture-compatible order effect across these configurations. It does not show that trained and random trajectories carry the same function or identify a universal architecture-training decomposition.

## 11. Auxiliary readback-centre geometry

This auxiliary analysis operates on normalized TopK/VIM centres \(C_l\), not raw hidden states. It uses the same broad family of endpoint-chord, cosine-path, detour and turning diagnostics and fixes endpoints while shuffling intermediate readback-centre layers. It is retained as a cross-observable diagnostic, and its estimand is never labelled hidden-state geometry.

## 12. Intervention labels, features and splits

Intervention-class labels are fixed by prompt condition in the separately generated ten-condition control panel. Stable contains clean, redundant, irrelevant and paraphrase; competition contains weak-distractor, balanced-competition and direct-conflict; closure contains closure-update, closure-override and exception-override. These are controlled intervention families rather than correctness labels or trajectory-derived dynamical phases. Their mapping is not substituted into the 12-condition coordinate audit.

Mechanism features are the predicted candidate-path separation and its magnitude, a companion dominance estimate, velocity magnitude and standardized velocity, normalized layer depth, a precursor-layer indicator, and baseline separation and dominance values. Policy features add mechanism probabilities and predicted-mechanism indicators. Training and test partitions are separated by relation graph, preventing rows from one graph from appearing in both partitions. Final correctness is not a target.

The first two intervention families used additive state directions, and the third used an additive flow direction. Their failure to yield robust mechanism-specific selectivity motivated the switch to a local mechanism-conditioned transition operator. At each intervention layer, principal-component analysis was fitted to the concatenated current and next hidden states from training graphs. Its dimension was \(r_l=\min(128,2n_{\mathrm{train}}-1,d_m)\). In the common PCA coordinates, separate stable and closure maps solved

\[
(A_l^{(k)},b_l^{(k)})=\underset{A,b}{\operatorname{argmin}}\;\sum_{i\in\mathcal T_k}\lVert z_{i,l+1}-Az_{i,l}-b\rVert_2^2+10\lVert A\rVert_F^2,
\]

for \(k\in\{\mathrm{stable},\mathrm{closure}\}\). The intercept was not penalized. The stable-row prediction was inverse-transformed to hidden-state space to define \(\hat F_l^{\mathrm{stable}}\). For realized next-layer state \(h_{p,l+1}\), the intervention is

\[
\tilde h_{p,l+1}=(1-\alpha_{p,l})h_{p,l+1}+\alpha_{p,l}\hat F_l^{\mathrm{stable}}(h_{p,l}),
\]

\[
h_{p,l+1}^{\mathrm{int}}=\tilde h_{p,l+1}+\beta_{p,l}\operatorname{sd}(\tilde h_{p,l+1})d_l,
\]

where \(d_l\) is the unit training-graph ridge direction for \(\Delta U\). The operator interpolation is applied first. The action grid is

\[
\mathcal A_0=\{(0,0),(0.1,0.1),(0.2,0.2),(0.3,0.3),
(0.3,0),(0,0.3),(0.3,0.2),(0.2,0.3)\},
\]

and `none` denotes \((0,0)\). For training prompts, closure labels maximize signed \(\Delta U\) shift; stable and competition labels minimize absolute shift. The final policy is compared with matched fixed combinations and 50 mechanism-label, policy-label and joint shuffles.

The mechanism classifier used the nine-dimensional vector

\[
x^{\mathrm{mech}}=(\widehat{\Delta U},|\widehat{\Delta U}|,\widehat D_B,\lVert v\rVert,z_v,\tilde l,I_{\mathrm{precursor}},\Delta U_{\mathrm{base}},D_{B,\mathrm{base}}).
\]

It was a 250-tree `RandomForestClassifier` with `max_depth=7`, `min_samples_leaf=6`, `class_weight=balanced_subsample` and seed \(42+s\), where \(s=0\) for the real fit and the shuffle index otherwise. The policy input appended the three predicted mechanism probabilities and three one-hot predicted-mechanism indicators, giving \(x^{\mathrm{policy}}\in\mathbb R^{15}\). The policy was a 200-tree random forest with `max_depth=6`, `min_samples_leaf=8`, the same class weighting and seed \(142+s\). Both used the scikit-learn default weighted-Gini split criterion. They contained no neural hidden layers and used neither gradient descent nor policy gradients.

For every training prompt, each action \(a=(\alpha,\beta)\) in the eight-element grid was executed. The discrete target was

\[
a_p^*=\begin{cases}
\operatorname*{argmax}_{a}\,[\Delta U_p^{(a)}-\Delta U_{p,\mathrm{base}}], & p\in S_{\mathrm{closure}},\\
\operatorname*{argmin}_{a}\,|\Delta U_p^{(a)}-\Delta U_{p,\mathrm{base}}|, & p\notin S_{\mathrm{closure}}.
\end{cases}
\]

The policy forest learned this discrete label on layer-expanded training rows and predicted one action per held-out prompt-layer row. Thus the policy is supervised action classification after an exhaustive training-grid evaluation, not reinforcement learning.

The mechanism-label null permutes training mechanism labels, refits the mechanism classifier and then refits the downstream policy. The policy-label null retains the real mechanism classifier but permutes training action labels before policy fitting. The joint null independently permutes both label sets and refits both classifiers. All nulls retain the graph split, test rows, transition operators, action grid and endpoint calculation.

**Executable procedure: held-out mechanism-conditioned intervention**

```text
split complete relation graphs 70:30 into training and held-out sets
fit transition operators, mechanism classifier and action policy on training graphs
for each held-out prompt and selected layer:
    predict mechanism probabilities
    select operator and precursor weights
    apply operator interpolation, then the scaled precursor update
    measure post-intervention Delta U and the declared candidate logits
compare the learned policy with matched fixed actions and refitted label-shuffle nulls
```

Specificity is

\[
s(m,c)=\operatorname{mean}_{i\in S_{\mathrm{closure}}}|\Delta U_i^{(c)}-\Delta U_{i,\mathrm{base}}|-\operatorname{mean}_{i\in S_{\mathrm{nonclosure}}}|\Delta U_i^{(c)}-\Delta U_{i,\mathrm{base}}|.
\]

The pseudo-log transform with \(\sigma=0.08\) is used only for Fig. 5 display. The primary endpoint of this original action-budget analysis is internal dominance specificity. Because the intervention is applied upstream of this held-out endpoint and is compared with fixed and refitted label-shuffle controls, it supports selective causal control of the measured internal answer-formation trajectory. Output-boundary control is tested separately below with an expanded, training-only action search and a strict full-vocabulary endpoint.

The candidate-choice boundary pairs no-intervention and real-policy outputs for each held-out prompt. It reports the change in final clean-minus-conflict candidate margin, the change in binary choice between those two candidates and their association with \(\Delta U\) shift. Intervals for these paired descriptive contrasts use 5,000 relation-graph bootstrap replicates with seed 20260714. In closure prompts, clean denotes the original unperturbed-chain candidate, whereas the updated or override-supported conflict candidate is generally the rule-consistent answer. A conflict-to-clean crossing is therefore an actuator-response endpoint, not an accuracy improvement.

### Out-of-fold margin-shift and boundary-crossing analysis

The quantitative bridge defines

\[
\delta M_p=M_{p,\mathrm{policy}}-M_{p,0},\qquad
\delta\Delta U_p=\Delta U_{p,\mathrm{policy}}-\Delta U_{p,0}.
\]

For checkpoint \(m\) and regime \(r\), an affine ridge model with penalty \(10^{-6}\) estimates

\[
\widehat{\delta M}_p=a_{m,r}+b_{m,r}\delta\Delta U_p.
\]

Five-fold `GroupKFold` holds out complete relation graphs, with the same graph held out across checkpoints. The predicted post-policy margin is \(\widehat M_{p,\mathrm{policy}}=M_{p,0}+\widehat{\delta M}_p\). For the at-risk population \(M_{p,0}<0\), the observed crossing label is \(I(M_{p,\mathrm{policy}}\geq0)\). Within each outer fold, an inner graph cross-fit supplies training predictions for a one-variable logistic calibrator, which maps predicted post-margin to crossing probability. Thus neither the margin-shift prediction nor its probability calibration evaluates a graph used for fitting.

The pooled closure margin-shift model reached \(R^2=0.850\) with a 95% graph-bootstrap interval of 0.811-0.882. Among 188 at-risk closure prompts, 15 crossed the candidate boundary and pooled AUROC was 0.976 (0.951-0.996). Under the original action budget, the observed counts were 0/87 in Qwen, 9/45 in Llama and 6/56 in Gemma. Qwen's mean shift was largest in absolute logit units, but its mean initial deficit was 5.03; 94.0% of its closure-layer actions used the maximum \((0.3,0.3)\) action. Continuous and crossing intervals use 1,000 graph-bootstrap replicates with seed 20260714. Internal trajectory control was observed in all three checkpoints, whereas candidate-boundary crossing under the original action budget was checkpoint-dependent. This endpoint is not open-ended generated-answer correctness.

### Confidence-gated emitted-token boundary control

The expanded output-boundary analysis was run independently in Qwen, Llama and Gemma with the same seed-42 split of complete relation graphs: 67 training graphs containing 670 prompts and 29 held-out graphs containing 290 prompts, of which 87 were closure prompts. Intervention layers were 15-19 in Qwen, 8-11 in Llama and 13-17 in Gemma. The principal-component direction defining \(\Delta U\), transition operators, precursor directions, mechanism classifier, action targets and boundary policy were fitted on training graphs only. Candidate token identities entered the paired-margin objective. Generated answer text, final-answer correctness and held-out candidate outcomes entered neither fitting nor gate construction.

The bounded action library was

\[
\mathcal A=\{(0,0),(0.3,0.3),(0.6,0.3),(0.6,0.6),(0.9,0.3),(0.9,0.6),(0.9,0.9),(1.0,0.6),(1.0,0.9),(1.0,1.2)\}.
\]

For action \(a=(\alpha,\beta)\), the state update retained the operator-then-precursor order defined above. Because \(0\leq\alpha\leq1\), operator interpolation could reach but not extrapolate beyond the fitted transition proposal. On a training closure prompt \(p\), action utility was

\[
u_p(a)=\delta M_p(a)+5I[M_{p,a}\geq0]-0.03(\alpha^2+\beta^2),
\]

where \(\delta M_p(a)\) is the change in clean-minus-conflict candidate margin and the indicator supplies a crossing bonus. For non-closure prompts,

\[
u_p(a)=-|\delta M_p(a)|-0.10(\alpha^2+\beta^2).
\]

The policy was trained to predict the utility-maximizing action from the same 15 pre-intervention features used above. This objective searches for a bounded action that crosses the declared candidate boundary in closure while penalizing movement and dose outside closure; it does not optimize correctness.

For prompt \(p\), let \(\pi_{p,l}(k)\) be the intervention-class posterior for class \(k\) at intervention layer \(l\in L_I\). Mean closure support and normalized posterior entropy were

\[
\bar\pi_p=|L_I|^{-1}\sum_{l\in L_I}\pi_{p,l}(\mathrm{closure}),\qquad
\bar H_p=|L_I|^{-1}\sum_{l\in L_I}\frac{-\sum_k\pi_{p,l}(k)\log\pi_{p,l}(k)}{\log3}.
\]

The gate was

\[
g_p=I[\bar\pi_p\geq0.50\;\land\;\bar H_p\leq0.90].
\]

The probability and entropy thresholds were declared in the experiment protocol before held-out execution. No grid search, threshold sweep or held-out outcome tuning was performed. The reported result therefore characterizes the operating point \((0.50,0.90)\), not an optimized gate or a threshold-robustness claim. When \(g_p=0\), every intervention-layer action was set to \((0,0)\). When \(g_p=1\), the predicted model-specific action vector was executed. The primary endpoint was a strict full-vocabulary top-1 continuation change from the declared conflict token to the declared clean token. The damage guards were any full-vocabulary top-1 change among the 203 non-closure prompts and intervention-layer state-norm ratios. Clean denotes the unperturbed-chain candidate and is generally not the rule-consistent answer in closure prompts.

The confidence-gated controller crossed the declared boundary in 17/87, 13/87 and 9/87 held-out closure prompts in Qwen, Llama and Gemma. Non-closure top-1 changed in 0/203, 2/203 and 0/203 prompts, respectively. Without the final abstention gate, strict crossings were 19/87, 13/87 and 9/87, while non-closure changes were 3.45%, 1.48% and 0.49%. These results support output-boundary causal leverage at one declared operating point across the three testbeds. They do not establish answer-correctness improvement, a safety guarantee or deployment control.

To test whether the effect required precise action-to-trajectory assignment, each safety-matched repeat permuted the complete model-specific action vector without fixed points among gate-open prompts within the same closure condition. This preserved the gate-open set, all gate-closed zero actions, total action budget and every condition-by-layer \((\alpha,\beta)\) multiset. Strict crossings were defined relative to each prompt's unperturbed top-1 output. Across 50 repeats, null crossing counts ranged from 15 to 20 in Qwen, 12 to 16 in Llama and 6 to 10 in Gemma. Their means were 17.64, 14.18 and 7.90. The one-sided empirical values were 0.804, 0.922 and 0.255, respectively.

The matched null therefore does not support fine-grained action-to-trajectory dose specificity in any checkpoint. A separately frozen component-ablation protocol retained the same gate-open sets and training fits. Operator-only actions kept each real gated \(\alpha\) and set \(\beta=0\); precursor-only actions kept \(\beta\) and set \(\alpha=0\); the uniform control assigned \((1.0,1.2)\), the maximum existing library action, to every gate-open prompt. Negative and positive references reproduced all previous top-1 token IDs exactly before the ablations were interpreted.

**Supplementary Methods Table 3 | Confidence-gated actuator component attribution.**

| Checkpoint | Gated policy: closure flips / non-closure changes | Operator only | Precursor only | Uniform maximum on same gate |
|---|---:|---:|---:|---:|
| Qwen | 17/87 / 0/203 | 17/87 / 0/203 | 0/87 / 0/203 | 22/87 / 0/203 |
| Llama | 13/87 / 2/203 | 14/87 / 2/203 | 1/87 / 0/203 | 15/87 / 2/203 |
| Gemma | 9/87 / 0/203 | 8/87 / 0/203 | 0/87 / 0/203 | 10/87 / 1/203 |

Across checkpoints, the gated policy and operator-only ablation each produced 39 strict crossings and 2 non-closure changes; precursor only produced 1 and 0, and the uniform maximum produced 47 and 3. All intervened states were finite and the largest recorded state-norm ratio was below 1.10. The supported algorithmic object is therefore more specific than the original two-component description: confidence-based abstention plus the bounded local transition operator carries the principal output-boundary leverage at the declared operating point. The precursor component alone is insufficient. The Llama and uniform-Gemma damage counts show that the same gate does not guarantee zero collateral output change across checkpoints. These comparisons do not identify a universal decomposition or establish correctness control.

**Executable procedure: confidence-gated local transition control**

```text
split complete relation graphs into training and held-out sets
fit the candidate-separation coordinate, transition operators and classifiers on training graphs
execute every bounded action on training prompts and assign the utility-maximizing action label
for each held-out prompt:
    compute layerwise mechanism posteriors from pre-intervention features
    open the gate only if mean closure support is at least 0.50 and mean entropy is at most 0.90
    if the gate is closed, apply the identity action at every intervention layer
    if the gate is open, apply the predicted bounded operator and precursor weights
    evaluate strict full-vocabulary top-1 candidate crossing and non-closure output change
compare the real assignment with safety-matched action-vector reassignments
```

### Frozen mixed-arithmetic actuator-transfer test

This experiment asked whether the already specified actuator procedure transferred to a second task after training-only refitting. It was not a zero-shot transfer of numerical operators or of the relation-graph \(\Delta U\) axis. The empirical object remained the ordered last-token layer-state process. The observable endpoint was the distractor-versus-arithmetic-consistent candidate margin and the full-vocabulary top-1 token before and after intervention.

The task contained 64 unique expressions sampled deterministically across addition, subtraction and multiplication. Candidate answers were number words from zero to twenty, counterbalanced by problem parity. Every problem supplied six conditions: clean and paraphrase mapped to stable; untrusted conflict and asserted conflict mapped to competition; explicit correction and final recheck mapped to the target correction class. A single group split with seed 20260714 assigned 44 complete problems to training and 20 to held-out evaluation. No held-out row selected a layer, action, threshold, regularization value or stopping rule.

The primary assay used a raw prompt ending immediately before the requested one-word continuation, matching the relation-graph actuator interface. For every checkpoint, tokenized prompt-plus-candidate text had to contain the complete prompt token sequence as an exact prefix, and the first appended token had to equal the audited single leading-space token for that candidate. All labels from zero to twenty passed this gate. An initial non-contextual token run failed this check and is excluded. A subsequent native-chat pilot passed contextual tokenization but is not pooled with the primary result: Llama began all 120 held-out continuations with free text rather than either candidate, so a candidate-token top-1 crossing was outside that interface's endpoint domain. These pilots diagnose interface validity and are not intervention evidence.

The following quantities were inherited without retuning from the relation-graph boundary experiment: Qwen, Llama and Gemma intervention and decision windows; the local transition-operator parameterization and ridge penalty; the ten-action \((\alpha,\beta)\) library; mechanism and action-classifier hyperparameters; closure-probability threshold 0.50; normalized-entropy threshold 0.90; and the 5% non-target collateral ceiling. Task-specific arithmetic coordinates, numerical transition maps, precursor directions and action labels were fitted only on the 44 training problems. Thus the frozen object was the intervention procedure and hyperparameter contract, whereas fitted numerical coordinates were allowed to adapt to the new task.

The primary control was the confidence-gated learned policy. Negative and attribution controls were no intervention, ungated policy, training-selected fixed global action, operator only, precursor only and uniform maximum action. For each checkpoint, 50 address-destruction nulls randomly reassigned complete action bundles across held-out prompts. Every repeat preserved the number of open gates and the exact multiset of layerwise \((\alpha,\beta)\) doses; 150 of 150 model-repeat dose-multiset checks passed. This null destroys prompt-to-action address while retaining actuator availability and aggregate budget. It does not test fine dose assignment among already addressed targets.

Strict eligibility required a held-out correction prompt whose unperturbed full-vocabulary top-1 token was the distractor candidate. A strict crossing required the intervened full-vocabulary top-1 to become the arithmetic-consistent candidate. Non-target damage was any top-1 change among stable and competition prompts. For checkpoint \(m\),

\[
s_m=\frac{N_{m,\mathrm{cross}}}{N_{m,\mathrm{eligible}}}-
\frac{N_{m,\mathrm{non-target\ change}}}{N_{m,\mathrm{non-target}}}.
\]

If \(N_{m,\mathrm{eligible}}<5\), the checkpoint was declared underpowered for a model-specific crossing verdict. Confirmation required: (i) at least two checkpoints with at least five eligible prompts, positive median eligible candidate-margin shift and one or more strict crossings; (ii) at least two checkpoints with collateral at or below 5%; (iii) pooled specificity above the 95th percentile of 50 pooled address-null repeats; and (iv) a non-zero pooled crossing count with no numerical-instability failure. Partial internal or pair-boundary transfer was the predeclared revised verdict when only part of this gate passed.

The primary results are in Supplementary Methods Table 6. Qwen was the only checkpoint to combine a powered denominator, positive median margin shift and a strict crossing. Llama crossed once but had a negative median shift; Gemma had a positive median shift but only four eligible prompts and no primary crossing. Pooled strict crossings were 2/45, non-target changes 1/240 and specificity 0.04028. The pooled null 95th percentile was -0.01111 and the plus-one one-sided empirical value was \(1/51=0.0196\). The address result therefore supports selective routing at the pooled level, but the two-checkpoint positive-direction gate failed. The frozen decision is partial protocol-level transfer, not confirmed cross-task output control.

The arithmetic PC1 coordinate did not provide a universal intervention axis. In Qwen, the median eligible candidate-margin shift was positive (0.5391) while the median task-coordinate shift was negative (-0.0206); Llama was negative on both, and underpowered Gemma was positive on both. These mixed directions are retained because strict output evidence must be evaluated at the declared token boundary rather than inferred from a compressed coordinate. Operator-only reproduced the two primary crossings, precursor-only produced none and uniform maximum produced two crossings plus one additional Gemma crossing with one Gemma non-target change (Supplementary Methods Table 7). These controls do not establish fine action assignment or a task-general executor.

For execution reproducibility, full-data extraction and held-out replay were allowed a maximum absolute FP16 candidate-margin discrepancy of 0.125, with any sign disagreement confined to that near-zero band. An independent repeat using the identical held-out batch partition had to reproduce every full-vocabulary top-1 token and candidate-pair sign exactly. All three checkpoints passed. An independent verifier recomputed the primary summaries, pooled null, decision gate and all action-dose multisets from retained row-level files without importing the execution code.

## 13. Human-AI hypothesis-to-audit workflow

This section records research provenance and responsibility. The workflow was not evaluated as a productivity intervention: the development interval records achieved throughput in this case but does not identify the counterfactual contribution of AI assistance. The collaboration used a repeated six-step loop.

1. **Human question or contradiction:** identify a mechanistic question, evidentiary gap or failed interpretation.
2. **Executable proposal:** use AI assistance to translate the question into candidate metrics, controls, scripts or consistency checks.
3. **Local execution:** run analyses against stated checkpoints and archived data.
4. **Evidence audit:** compare outputs with leakage restrictions, nulls, source files and rival explanations.
5. **Human decision:** retain, revise or reject the object, metric, interpretation or claim.
6. **Trace-preserving iteration:** archive scripts, outputs, negative results and the reason for the next test.

AI tools assisted with project-record retrieval, code scaffolding and debugging, batch planning, data-table organization, deterministic R plotting scripts, consistency checks and language editing. The human researcher selected the scientific problem, changed the empirical object, decided which controls were decisive, interpreted failures, approved claim boundaries and remained accountable for every reported output. AI suggestions were operational drafts or candidate hypotheses, not evidence.

The primary trajectory-mapping, cross-checkpoint, geometric and intervention code sequence was developed from 2 to 6 June 2026. A proxy audit on 10 June separated direct hidden-state, vocabulary-facing and construction-aligned objects and forced correction of earlier interpretations. These dates document development order and the achieved interval, but they do not quantify productivity relative to a no-AI process, establish scientific quality or mark completion of every final run. Final archive metadata should additionally preserve execution times, hashes and environments.

## 14. Responsibility matrix

| Function | Human authority | AI-assisted operation | Auditable artefact |
|---|---|---|---|
| Problem and object | Define question; select and revise process object | Retrieve prior discussions; compare formulations | Conversation archive and provenance ledger |
| Theory | Select source ideas and limits; reject misleading analogies | Organize concept mappings and draft formulas | Concept-provenance table |
| Experiment | Specify controls, nulls and acceptance criteria | Draft code scaffolds and parameter sweeps | Versioned scripts and configuration tables |
| Falsification | Interpret contradictions and decide theory change | Locate inconsistencies and propose follow-up checks | Negative-result and correction records |
| Data and figures | Select quantities; verify source mapping; approve claims | Assemble tables and deterministic R scripts | Source Data manifest and figure audit |
| Manuscript | Set evidence hierarchy, scope and final language | Terminology, formatting and language support | Versioned manuscript and audit report |
| Accountability | Approve and assume responsibility | No approval or authorship authority | Author Contributions and AI disclosure |

## 15. Research independence and disclosure completeness

The study was undertaken independently and was not an internal project of any listed organization. It used no employer data, computing infrastructure, internal systems, funding, research facilities or personnel support and did not involve organizational review or approval. H.Z. personally provided or paid for all local computing, storage, software subscriptions and other research expenses. Y.W., R.S. and W.C. contributed in their personal capacities as independent research collaborators. Affiliations identify the authors and do not imply employer sponsorship, endorsement or responsibility for the work.

The author names, affiliations, corresponding-author details, ORCID, acknowledgements, funding statement, narrative author contributions and competing-interests declaration were carried forward from the preprint record and checked against the main manuscript. The submitted scientific figures were generated from retained data and deterministic R code rather than an image-generation system.

AI product families and providers are disclosed in the main Methods. The consumer interfaces used during June and July 2026 did not consistently expose stable underlying model build identifiers, and interface-level task identifiers were not retained as scientific provenance; these fields are therefore not reconstructed. Scientific provenance instead rests on the executable scripts, fixed inputs, environment records, null specifications, retained outputs, source mappings and file-level hashes included in the archive. The public repository and Zenodo concept records are identified in the main Data and Code availability statements. Formal CRediT labels are supplied as submission metadata and do not replace the narrative author-contribution statement.

## 16. Additional architecture, order, metric and training controls

### 16.1 Random fixed-map relational continuity

Prompt states were propagated through fixed random mappings matched to the dimensional schedules of representative Qwen, Llama and Gemma configurations. The audit crossed three task families (lexical categories, addition and four-choice ARC-Challenge) with two independent seeds. The tested quantity was a cross-depth relation-preservation contrast computed from independently partitioned prompt sets and compared with lag-matched, block-donor and independent-map nulls. All 18 declared architecture-task-seed conditions passed the positive-contrast gate. The accepted claim is architecture-compatible relational continuity. Learned semantics, native-order privilege, endpoint direction and a least-action path law do not follow from this result.

### 16.2 Reordered execution and functional fidelity

The same trained Qwen checkpoint was executed in native, reversed and fixed-permutation block order. Geometry was compared with native full-vocabulary top-1 output, declared-candidate agreement and Jensen-Shannon divergence. Reversed and fixed-permutation executions retained positive chord contrasts but had zero full-vocabulary top-1 agreement with native execution; candidate agreement was 0.479 and mean Jensen-Shannon divergence was approximately 0.692. This falsified geometric straightness as a sufficient statistic for native computation. The retained claim is that realized order is functionally consequential even when a coarse geometric statistic remains positive.

### 16.3 Incremental path-history prediction

For each task-checkpoint condition, group-held-out current-state predictors were compared with otherwise matched predictors augmented by ordered-prefix summaries. Passing required a positive incremental held-out gain above the declared order-null threshold. Six of 18 initial conditions passed, all in addition; lexical prompts passed 0/9. An independently generated mixed-arithmetic replication passed 0/9. The relation-graph final-margin forecast remains valid within its declared construction, but the tested prefix summaries do not establish a task-general path-memory law. A compatible explanation is that the current high-dimensional state already compresses much of the usable history; this explanation has not yet been tested as an exact Markov property.

### 16.4 Radial sensitivity of the Llama statistic

Llama residual trajectories were decomposed into unit directions and state norms. Direction-only trajectories passed none of seven task gates, whereas norm-only fixed-direction trajectories passed all seven. A frozen radial exponent \(\gamma\in\{0,0.25,0.5,0.75,1,1.25\}\) produced monotonic metric changes in all seven tasks. The accepted claim is radial sensitivity under the declared metric and normalization. The statistic does not independently identify semantic direction, function or an architecture-wide law.

### 16.5 Pythia component compatibility

Embeddings (E), transformer blocks (B), final normalization (N) and output head (W) were swapped between initialized and trained Pythia-160m checkpoints in the complete \(2^4\) factorial design. All combinations were evaluated on the same 53 held-out lexical cloze prompts. Accuracy rose from 0.472 with all initialized components to 0.849 with all trained components; the best isolated trained component reached 0.604. The fully trained margin exceeded the additive component prediction by 2.621 (95% bootstrap confidence interval, 1.252-3.987). The supported claim is distributed compatibility among trained input, transition and output components in this checkpoint and task. Hybrid combinations are diagnostic interventions, not natural training states.

### 16.6 Training-time task coordinates

Seven Pythia checkpoints were evaluated with a model-native candidate coordinate and a coordinate fitted once at the final checkpoint. Model-native order gain was absent at initialization, appeared from step 1,000 and correlated with held-out accuracy (Spearman \(\rho=0.75\)). The fixed final coordinate transferred with macro F1 0.041 at steps 0 and 128, 0.701 at step 1,000 and 1.0 by step 16,000. A coordinate refitted within each checkpoint already separated categories perfectly at initialization and therefore failed the learning-evidence gate. These results support training-associated task organization in fixed coordinates while rejecting random-initialization linear separability as evidence of learned semantics.

**Supplementary Methods Table 4 | Claim-boundary audit.**

| Supported statement | Boundary |
|---|---|
| Ordered hidden states provide a common within-checkpoint process object | Engineering components still implement the process; cross-checkpoint coordinates are not assumed identical |
| Direct hidden and hidden-flow features provide stronger mechanism coordinates than readbacks in the tested audits | Readbacks remain useful diagnostics; the construction-aligned decision profile is not independent evidence |
| Exploratory functional windows retain prompt-condition-class information on new graph instantiations | The joint prospective recurrence gate failed: topology separation reversed in all checkpoints, downstream coupling missed its null in Llama and only Gemma retained top-quartile window location |
| \(\Delta U\) summarizes candidate-path separation | It is construction-defined, not an energy, final-answer label or universal order parameter |
| \(O_l^{\mathrm{cont}}\) improves closure | It is descriptive when adjacent target-layer information enters its construction; corrected target-excluded bilinear estimates retained 25.1% of the observed-coordinate gain and remain an exploratory predictive boundary |
| Hidden paths are chord-directed relative to endpoint-preserving shuffles | The result is metric-dependent and does not establish exact geodesics or least action |
| Chord-directed geometry recurs in lexical, addition and four-choice ARC tasks | Task-specific coordinate transfer is partial, and the numerical relation-graph \(\Delta U\) axis was not transferred |
| Ordered partial histories forecast the final candidate margin in held-out relation-graph groups | Cross-task ordered-prefix summaries did not generally add prediction beyond current state; no universal path-memory law is supported |
| Layer order yields chord-directed geometry in trained and randomly initialized Qwen, Llama and Gemma architectures | Random-initialization effects were larger; geometry alone is architecture-compatible and does not identify learned inference organization |
| Fixed random maps preserve prompt relations across depth under three stronger nulls | Relation preservation does not establish semantics, native-order privilege or task function |
| Reverse and permuted executions retain positive geometry but lose native outputs | Native order is functionally consequential; straightness is not sufficient for fidelity |
| Pythia task function depends on trained \(E/B/W\) compatibility and fixed task coordinates emerge during training | One small model, one training run and one lexical diagnostic do not establish a universal training law or complete representational geometry |
| Llama raw alignment is strongly radial-sensitive | Metric sensitivity does not make norm scale the mechanism or establish semantic direction |
| Local transition operators shift internal dominance beyond fixed and shuffled controls | Internal selectivity does not by itself establish an emitted-token change |
| Held-out internal displacement predicts candidate-margin movement and pooled crossing risk | Original-budget crossings were checkpoint-dependent; the map is task- and candidate-defined |
| Confidence-gated local transition control crosses the emitted-token candidate boundary in all three checkpoints | Strict flips were 17/87, 13/87 and 9/87; operator-only ablation preserved 17/87, 14/87 and 8/87 whereas precursor-only produced 0/87, 1/87 and 0/87; no checkpoint showed fine-grained assignment specificity; this is not answer correctness, safety or deployment control |
| The frozen actuator procedure shows pooled selectivity in mixed arithmetic | Only Qwen passed the model-level positive-direction gate; Llama's median target shift was negative and Gemma was underpowered. The two-model confirmation gate failed, so this is partial protocol transfer rather than cross-task output control |
| An AI-assisted workflow completed a compressed hypothesis-to-audit sequence | This single case has no no-AI counterfactual and does not identify a causal productivity gain; scientific object choice, falsification decisions, interpretation and accountability remained human |

**Supplementary Methods Table 5 | Held-out closure-prompt boundary outcomes under confidence-gated control.**

Strict-crossing eligibility required the baseline full-vocabulary top-1 token to be the conflict candidate. A pair-margin crossing means that the clean-minus-conflict margin became non-negative while another vocabulary token remained top-1. Categories are mutually exclusive and sum to 87 closure prompts per checkpoint.

| Checkpoint | Eligibility | Mutually exclusive gate-open eligible outcomes | Abstention or ineligibility |
|---|---|---|---|
| Qwen | 87/87 baseline-conflict; 84 gate-open | 17 strict crossing; 67 positive but below pair boundary; 0 pair crossing behind third token; 0 non-positive shift | 3 eligible gate-closed; 0 not baseline-conflict |
| Llama | 44/87 baseline-conflict; 40 gate-open | 13 strict crossing; 19 positive but below pair boundary; 6 pair crossings behind third token; 2 non-positive shifts | 4 eligible gate-closed; 43 not baseline-conflict |
| Gemma | 17/87 baseline-conflict; 17 gate-open | 9 strict crossing; 5 positive but below pair boundary; 3 pair crossings behind third token; 0 non-positive shifts | 0 eligible gate-closed; 70 not baseline-conflict |

For Qwen, positive-shift non-crossings began farther from the pair boundary than strict crossings (median baseline margin, -5.58 versus -3.67), although their median shifts were similar (3.98 versus 4.41). This pattern is consistent with insufficient bounded displacement at the tested operating point rather than a general failure to move the candidate margin. In Llama and Gemma, six and three eligible prompts crossed the pairwise candidate margin but remained behind a third full-vocabulary token. The decomposition therefore separates gate abstention, insufficient pairwise displacement and third-token competition; it does not establish that increasing the action budget would preserve collateral-output rates.

**Supplementary Methods Table 6 | Frozen mixed-arithmetic primary actuator result.**

Eligibility requires the unperturbed full-vocabulary top-1 token to be the distractor candidate on a held-out correction prompt. Specificity is target crossing rate minus non-target top-1-change rate. The arithmetic coordinate was fitted on training problems and is not the relation-graph \(\Delta U\) axis.

| Checkpoint | Eligible prompts | Strict crossings | Median candidate-margin shift | Median arithmetic-coordinate shift | Non-target changes | Specificity | Model-level gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen | 14 | 1 | 0.5391 | -0.0206 | 1/80 | 0.0589 | Pass |
| Llama | 27 | 1 | -0.0703 | -0.0414 | 0/80 | 0.0370 | Fail: median direction |
| Gemma | 4 | 0 | 0.0938 | 0.5538 | 0/80 | 0.0000 | Underpowered |
| Pooled | 45 | 2 | Not pooled | Not pooled | 1/240 | 0.0403 | Fail: only one model passes |

**Supplementary Methods Table 7 | Frozen mixed-arithmetic actuator-component controls.**

Values are strict crossings among the same model-specific eligible prompts followed by non-target top-1 changes among 80 stable or competition prompts. The maximum action is the largest action already present in the frozen library.

| Checkpoint | Primary guarded policy | Operator only | Precursor only | Uniform maximum |
|---|---:|---:|---:|---:|
| Qwen | 1/14; 1/80 | 1/14; 1/80 | 0/14; 0/80 | 0/14; 1/80 |
| Llama | 1/27; 0/80 | 1/27; 0/80 | 0/27; 0/80 | 1/27; 0/80 |
| Gemma | 0/4; 0/80 | 0/4; 0/80 | 0/4; 0/80 | 1/4; 1/80 |

**Supplementary Methods Table 8 | Experiment subset, window and null-control metadata.**

| Analysis | Subset / grouping | Window or ordering | Split / null | Primary estimand |
|---|---|---|---|---|
| Coordinate audit | 960 traces per checkpoint; graph groups | Precursor and decision windows mapped by depth | Real, row-permuted and Gaussian projections | Mechanism-coordinate score and preservation |
| Functional registration | Full functional-window scan; graph groups | First 45% depth; widths 2, 3, 4, 5, 6 and 8; step 1 | Group-aware folds inside non-nested exploratory selection | Preservation, prompt-condition-class and downstream objective |
| Prospective functional-window boundary | 24 new graph groups; disjoint entities, relation phrases and candidate labels | Discovery-selected window and K frozen per checkpoint | Discovery-only fits; 2,000 graph bootstraps; 999 label and target permutations | Joint role recurrence and frozen-window rank |
| \(\Delta U\) | Task-defined clean/conflict candidates | Decision window; predecision exclusion controls | Final correctness labels excluded | Oriented PC1 of clean-relative margin matrix |
| Direct hidden geometry | 96 prompts: 8 complete groups x 12 conditions | Full observed layer order | 50 fixed-endpoint intermediate-layer permutations | Alignment gain; log2 detour and curvature reduction |
| Mechanism-conditioned intervention | 670 policy-train and 290 held-out prompts | Layer-expanded rows | Group-aware split by graph group; 50 mechanism, policy and joint shuffles | Internal dominance specificity |
| Transition predictive boundary | 14,592 layer-transition rows; graph groups | Pre-transition state, history and context only | GroupKFold; target layer excluded | Retained fraction of observed-coordinate closure gain |
| Prospective final-margin forecast | 670 training and 290 held-out prompts; graph groups | Current and earlier decision-window margins only | 1,000 graph bootstraps; 50 independent within-prompt order nulls | Held-out final candidate-margin \(R^2\) and boundary AUROC |
| Candidate-choice boundary | 290 paired held-out prompts per checkpoint; 29 graph groups | No intervention versus mechanism-aware policy | 5,000 relation-graph bootstrap replicates | Candidate-margin shift and clean-candidate selection change |
| Independent controlled tasks | Lexical and addition: 64 problems x 6 conditions per checkpoint | Main-study windows frozen before coordinate evaluation | 44 train and 20 held-out problems; 200 training-label shuffles; 50 endpoint-preserving order shuffles | External chord alignment and held-out task-coordinate macro-F1 |
| Standard four-choice task | ARC-Challenge validation: 96 items per checkpoint | Full four-choice prompts; no binary reduction | Deterministic subset; 50 endpoint-preserving order shuffles | Chord alignment and four-choice accuracy |
| Random-initialization geometry | Three seeds per Qwen, Llama and Gemma configuration; same 96 relation-graph prompts | Full observed layer order | 50 endpoint-preserving intermediate-layer permutations per parameter state | Alignment gain relative to order-null mean |
| Held-out displacement-boundary bridge | 290 intervention-test prompts per checkpoint; 29 graph groups; 188 at-risk closure prompts | Policy-induced \(\Delta U\) and candidate-margin shifts | Out-of-fold graph-wise affine ridge fits; nested calibration; 1,000 graph bootstraps | Continuous margin-shift \(R^2\), crossing AUROC and observed crossing counts |
| Confidence-gated output boundary | Each checkpoint: 670 training and 290 held-out prompts; 87 held-out closure prompts | Model-specific intervention windows; shared expanded bounded action library | Train-only coordinate and fits; 50 safety-matched action-vector reassignments per checkpoint | Full-vocabulary conflict-to-clean top-1 flips and non-closure output change |
| Frozen mixed-arithmetic actuator transfer | 64 problems x 6 conditions; 44 train and 20 held-out problems per checkpoint | Relation-graph windows, action library, regularization and gate thresholds frozen; task-specific fits use training problems only | Raw-prompt token-boundary audit; 50 action-address reassignments per checkpoint; independent summary and dose-multiset verification | Strict distractor-to-arithmetic-consistent top-1 crossings, eligible margin shift, collateral change and predeclared cross-model verdict |
| Random-map relation continuity | 2 new seeds x 3 architectures x 3 tasks; 96 prompts each | Cross-half and first-to-last-quarter depth | 199 lag-1, contiguous-block and fingerprint-preserving donors | Unbiased linear CKA beyond all three nulls |
| Reordered execution | Trained Qwen; 48 prompts | Native, reverse and one frozen block permutation | 10,000 paired bootstraps; 100,000 sign flips | Geometry-function dissociation |
| Incremental path memory | Lexical, addition and independent mixed arithmetic; 3 checkpoints x 3 horizons | Ordered prefix versus current state and unordered history | Five-fold problem groups; 199 order shuffles; multiplicity correction | Incremental balanced-accuracy gain |
| Llama radial audit | 4 original and 3 independent tasks | Direction-only, norm-only and six frozen radial exponents | 199 layer-conditioned radial donors | Scale sensitivity of raw alignment statistic |
| Pythia component compatibility | 53 lexical prompts; step 0 and step 143,000 | All 16 \(E/B/N/W\) component combinations | 5,000 paired bootstraps; full \(2^4\) factorial contrasts | Pair accuracy and superadditive margin excess |
| Training-coordinate audit | 53 prompts x 7 Pythia checkpoints | Model-native candidate coordinate; held-out category coordinate | 999 layer-order shuffles; 199 label nulls | Order gain, fixed-coordinate transfer and macro F1 |
