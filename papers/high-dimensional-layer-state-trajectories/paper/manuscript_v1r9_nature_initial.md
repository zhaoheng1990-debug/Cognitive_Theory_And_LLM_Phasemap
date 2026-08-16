# Architecture, training and execution organize Transformer computation

Heng Zhao<sup>1,*</sup>, Yufei Wang<sup>2</sup>, Ruidian Song<sup>3</sup>, Weisheng Chen<sup>4</sup>

<sup>1</sup>Independent Researcher, Beijing 100071, China.

<sup>2</sup>China Mobile Communications Group Co., Ltd., Beijing 100054, China.

<sup>3</sup>Sinotrans Logistics Co., Ltd., Beijing 100054, China.

<sup>4</sup>Independent Researcher.

<sup>*</sup>Correspondence: Heng Zhao (zhaoheng1990@gmail.com). ORCID: 0009-0004-2395-9393.

## Summary

Transformer computation has three separable sources of organization: architecture constrains the transformations available to a network, training establishes learned compatibility among its components, and execution composes those components in the order of a forward pass. Existing mechanistic analyses rarely distinguish these contributions in one experiment<sup>1-8</sup>. We use the ordered layer-state record of individual forward passes to do so. Across several Transformer families, untrained models retain an architecture-compatible propagation scaffold; component swaps between untrained and trained states identify training-dependent functional compatibility; and direct block reordering shows that realized execution order is functionally consequential. A bounded local-transition intervention then provides causal leverage at a declared output boundary without implying general output control. This architecture-training-execution framework makes the organization of learned computation experimentally testable alongside the localization of internal representations. More broadly, it yields a separability principle for distributed learning systems: architectural constraints, learned relations among components and realized execution should be perturbed independently before they are assigned a common mechanistic role.

## Main text

Large language models compute through many distributed components, but their organization has three separable sources: architecture, training and execution (A-T-E)<sup>9-12</sup>. Architecture constrains the transformations a network can make; training establishes learned compatibility among components; and execution specifies the order in which a particular forward pass composes them. These contributions are usually studied through different experimental objects, making it difficult to determine whether they can be separated within one account of Transformer computation. The resulting separability principle is experimental: each contribution should be perturbed independently before it is assigned a role in a system-level mechanism.

Component-level methods have made important parts of this problem accessible. Probes measure information available in activations<sup>1,13</sup>; circuit and head analyses identify local motifs<sup>2,3,14,15</sup>; causal tracing and editing test localized associations<sup>4</sup>; vocabulary-facing readbacks expose output-facing tendencies<sup>5,16,17</sup>; and sparse-feature methods seek interpretable activation units<sup>6,18,19</sup>. Recent reviews place these approaches in the broader mechanistic-interpretability literature<sup>20,21</sup>. Each approach addresses a genuine implementation question. Their results alone do not distinguish an architectural propagation constraint from learned compatibility among components or from the consequences of executing trained blocks in their native order.

Recent trajectory studies provide complementary evidence at a different level. Damirchi *et al.* use layerwise activation displacements to train classifiers of reasoning validity, while Pandey *et al.* quantify geometric features of layer-wise representations and report a phase structure<sup>7,8</sup>. Representation-comparison and path studies likewise show that conclusions depend on the object, similarity statistic and temporal ordering under study<sup>22-24</sup>. Here, the trajectory is held as a common record while component states, execution order and local transitions are perturbed, allowing architecture, training and realized execution to be separated experimentally. Depth-indexed decodability and realized execution also require separate treatment: intermediate entities can become decodable out of an assumed serial order<sup>25</sup>, and input-dependent programs can skip or loop pretrained layers<sup>26</sup>.

We use the ordered sequence of high-dimensional layer states visited by one prompt as an empirical process object for that separation. For checkpoint \(m\), prompt \(p\) and layer \(l\), the last-token state \(h_{m,p,l}\in\mathbb{R}^{d_m}\) defines \(T_{m,p}=(h_{m,p,1},\ldots,h_{m,p,L_m})\). This representation retains both the state at each depth and the sequence in which states occur. It permits an order-sensitive null, a comparison between initialized and trained component configurations, and a localized perturbation at an adjacent transition. Implementation objects remain essential for extraction and intervention; the trajectory supplies a common record of the realized computation (Fig. 1).

We made the intervention test concrete with controlled relation-graph prompts. A chain of stated relations supports one candidate continuation, while an additional statement can create a competing candidate. We refer to the candidates as clean and conflict according to their prompt provenance. These labels identify a declared experimental boundary and do not denote correct and incorrect answers. Stable, competition and closure conditions were set from prompt metadata before modelling. The study combines this controlled assay with lexical, addition and multiple-choice geometry tests, exact-initial training audits and direct order interventions across representative Transformer checkpoints.

### A process-level trajectory separates organizational contributions

The trajectory object first provides coordinates for comparing a prompt across depth. Direct hidden decision states, precursor states and hidden flow contained more information about predefined prompt conditions than the vocabulary-facing readbacks tested in grouped audits (Fig. 1b). Projection controls showed that a readback's geometry depended on the output matrix and projection rule. We therefore treat it as an output-facing diagnostic, not as a primary coordinate of the process. The comparison is not a claim for a universally superior coordinate. In predecision tests, complete hidden-state and margin histories added information beyond matched recent logit margins, whereas the compact candidate coordinate used below did not improve the simple margin baseline.

The candidate coordinate \(\Delta U\) is therefore used instrumentally. For each relation-graph prompt, clean-relative candidate margins across a decision window form a matrix \(D\), and the oriented first principal component defines \(\Delta U\). It summarizes separation between the two declared candidate supports and provides a continuous target for a finite intervention action library. The decision-window profile from which \(\Delta U\) is constructed is retained as a construction-aligned reference, not as independent support. Predecision, candidate-margin exclusion and window-displacement controls define the diagnostic's inferential boundary (Extended Data Figs. 2 and 5).

This process-level representation makes the later experiments commensurable. The same ordered state record supports a geometry test for architecture, component swaps for training, direct block reordering for execution and a localized transition intervention. It is an enabling measurement choice. The scientific result concerns the separable organization revealed by those tests.

### Architecture supplies a recurrent propagation scaffold

We first asked whether native layer order carries a reproducible geometric signature. In Qwen, Llama and Gemma, normalized hidden-state paths were more chord-directed than 50 endpoint-preserving shuffles of the intermediate layers. Native trajectories exceeded the corresponding null means by 1.60-2.24-fold, with null-standardized effects of 5.11-5.63; detour and turning curvature were reduced in the same comparisons (Fig. 2b,c). The signature recurred in controlled lexical and addition tasks and in a four-choice ARC-Challenge audit without reducing the task to a binary candidate pair.

A paired multi-scale confirmation then tested the same geometry protocol on 24 new graph groups (240 prompts) in Qwen2.5-1.5B and a later-generation Gemma-3-12B QAT checkpoint, without selecting a layer window. In Gemma-3-12B, native chord alignment was 0.0864 versus 0.0240 for the shared endpoint-preserving null, an alignment gain of 0.0624 and a null-standardized effect of 3.39; all 50 shuffle values were lower (empirical (P=1/51)). The paired Qwen run gave a gain of 0.0374 and a null-standardized effect of 5.63 (Fig. 2b,c). This is a checkpoint-specific larger-model confirmation of the order-null signature. Differences in model family, quantization and capture runtime mean that it is not a pure within-family scale law.

The random-initialization control changed the interpretation of this geometry. Random models from the same checkpoint configurations showed larger alignment gains, and relational-continuity controls passed across the tested architecture-task-seed combinations (Fig. 2a; Extended Data Fig. 5). Chord-directed propagation alone is therefore insufficient evidence for a learned task or a recovered reasoning trajectory. Instead, it marks a scaffold compatible with the tested architectures. The experiments do not identify the responsible architectural feature, establish function in random models or imply a scale law.

Execution experiments supplied the complementary functional test. Reversed and fixed-permutation Qwen executions retained positive chord contrasts while losing native top-1 agreement; Llama controls separated direction and radial scale; and direct order perturbations changed outputs in Qwen, Llama and Gemma (Fig. 4). Geometry and native function consequently have different invariance boundaries. The recurring observation is an architecture-compatible propagation scaffold, while the function of a trained model depends on how its blocks are sequentially executed.

### Training organizes compatible function across components and coordinates

Training added a second organizational signature that is distinct from the scaffold. In an exact-initial Pythia-160m \(2^4\) factorial, embeddings, blocks, final normalization and output head were independently swapped between step 0 and the trained checkpoint. Across 53 held-out lexical cloze prompts, all-trained Pythia reached candidate-pair accuracy 0.849 from 0.472 at initialization; no isolated trained component exceeded 0.604, and the superadditive candidate-margin excess was 2.621 (95% confidence interval 1.252-3.987). The result shows that the tested function depended on compatible trained components rather than a single trained component.

We prospectively tested that component-compatibility object in a second exact-initial architecture. OLMo supplied a documented step-0 checkpoint and final checkpoint with independently swappable input embedding (E), Transformer blocks (B) and output head (W); its final normalization had no state tensors and was excluded before outcomes. The registered \(2^3\) E/B/W factorial used 96 tokenizer-qualified prompts from 12 held-out lexical categories with no prompt, category or candidate-label overlap with the predecessor OLMo component-audit input sets, as recorded in the archived disjointness audit. All-trained OLMo reached candidate-pair accuracy 0.990, compared with 0.521 for all-initial OLMo. Its standardized candidate margin increased by 2.606 (95% confidence interval 2.276-2.938); all-trained OLMo exceeded every isolated component condition, all isolated recovery fractions remained below the frozen 0.80 threshold, and the E:B and E:B:W interaction contrasts were positive (Fig. 3a-c). All 12 categories satisfied the predeclared aggregate robustness criterion. Complementary training-line diagnostics are shown in Extended Data Fig. 6.

The factorials provide a bounded cross-architecture recurrence for distributed component compatibility. They are lexical-category and candidate-pair assays, and they do not establish a universal training law. A complementary training audit asks a different question: how task-compatible coordinates emerge across training. In Pythia, model-native candidate-coordinate order gain emerged with training and tracked accuracy; a final coordinate transferred poorly to early checkpoints. In the separate TinyLlama training line, coordinate organization recurred under its own bounded assay. These two branches, component compatibility and task-coordinate organization, converge on training-associated organization without claiming that one second architecture independently replicated both objects (Fig. 3c).

### Realized trained order is functionally consequential

The training factorials establish compatibility among trained components. We next asked whether their realized sequence matters after training. Direct execution-order perturbations in Qwen, Llama and Gemma changed model outputs relative to native execution (Fig. 4a). In the prospective cross-checkpoint replication, reverse and frozen-permutation executions in both Llama and Gemma produced zero native full-vocabulary top-1 agreement across 96 prompts per condition; candidate-choice agreement remained partial, so the result reflects broad output disruption rather than a simple relabelling of the two declared candidates. In Qwen, reverse and fixed-permutation executions likewise produced zero native full-vocabulary top-1 agreement, candidate agreement of 0.479 and a mean Jensen-Shannon divergence of 0.692, even though their chord contrasts remained positive. The same dissociation motivated scale and direction controls in Llama and heterogeneous geometry analyses in Gemma.

These experiments support a functional role for realized trained block order across the tested families. They do not support a common geometric response to non-native execution. In some settings, structured geometry persisted when output fidelity was lost; in others, the geometric expression changed with the metric or residual scale. The cross-model regularity is functional order dependence under the frozen reverse/permutation interventions. This does not imply that the native order is the only valid execution program. Chord-directed geometry remains a separate, metric-dependent observation. Reconstruction and geometry diagnostics are provided in Extended Data Fig. 7.

### Local transition operators provide bounded causal leverage

We used intervention to test whether a local part of the ordered process could be perturbed selectively. Additive state and flow directions did not yield robust mechanism-specific control. We therefore changed the actuator. A prompt-condition classifier selected a local transition operator, and a policy chose discrete operator and order-precursor weights using training-graph data only. The classifier labels remained predefined prompt conditions rather than trajectory-derived phases.

The gated local-transition actuator increased internal candidate-dominance specificity beyond fixed-combination and joint-shuffle controls in all three checkpoints (Fig. 5a-c). A declared confidence-and-entropy gate, inherited from the Qwen configuration and frozen before held-out execution, identified closure prompts for the expanded finite action library. At the full-vocabulary top-1 boundary, conflict-to-clean crossings occurred in 17/87 Qwen, 13/87 Llama and 9/87 Gemma closure prompts. Non-closure changes were 0/203, 2/203 and 0/203, respectively (Fig. 5d,e). Operator-only ablations retained most crossings, whereas the precursor alone was insufficient.

The result establishes bounded output-boundary leverage within the relation-graph task family. It does not measure answer correctness, deployment safety or open-ended generation control. Matched direction controls failed the three-checkpoint correction, so the experiments do not identify a checkpoint-general perturbation direction (Extended Data Fig. 8). A frozen mixed-arithmetic extension produced partial transfer and did not meet the confirmation gate for general cross-task control (Extended Data Fig. 9).

### Boundaries of the organizational separation

Several negative results delimit the synthesis. Functional windows differed by checkpoint and failed the joint recurrence gate, excluding a common phase claim (Extended Data Fig. 3). Pairwise direction isomorphism was weak, and ordered-prefix tests did not establish a general path-memory law (Extended Data Fig. 4). Descriptive transition closure exceeded target-excluded prediction. The geometry findings are metric-dependent, and the prospective OLMo factorial is bounded to lexical-category candidate pairs. These results leave open a single architecture-independent law of Transformer dynamics. They instead motivate the separability principle: architectural constraints, learned component relations and realized execution should be tested independently before a common mechanism is inferred.

### Discussion

The experiments distinguish three contributions to Transformer computation. Architecture supplies a recurrent propagation scaffold. Training organizes compatible function across components and task coordinates. Realized block order executes a functionally consequential composition. The trajectory representation provided a common experimental record for comparing them. Each contribution has a different invariance boundary.

This separation adds an organizational level to mechanistic evidence. Parameters, heads, MLPs and blocks remain the implementation substrate, and component-level methods remain indispensable. Their contributions to a trained computation depend on relations among components and on the order in which the resulting transformations are applied. The Pythia and OLMo factorials make this point directly. In Pythia, no isolated trained component recovered the all-trained performance; in OLMo, none of the isolated E, B or W substitutions recovered the all-trained held-out lexical signature. The OLMo confirmation was prospectively registered on a new lexical surface, strengthening the component result without turning it into a general account of learning.

These experiments broaden the empirical objects available to mechanistic analysis. A component or representation can be informative without independently specifying the function of the system in which it participates. The factorial and order interventions expose learned relations among components and the realized program that composes them. They shift a mechanistic question from where a feature is represented to how a trained computation is organized, yielding experimentally separable objects for component compatibility, execution-program equivalence and local transition causality. This is a methodological parallel, not a claim of shared mechanism, with population-dynamics studies that treat an evolving neural population as the object of analysis<sup>27</sup>.

The execution result also has a defined scope. The frozen reverse and permutation schedules test native-output fidelity after a fixed trained component set is re-executed in a non-native order. Recent Program-of-Layers work shows that input-dependent programs that skip or loop pretrained layers can preserve or improve performance under other objectives<sup>26</sup>. That finding is compatible with the present result: the interventions establish that the native sequence is functionally consequential under the schedules tested here, without establishing that it is the only valid execution program.

The random-initialization control clarifies what the geometric result means. Random initializations could show stronger chord alignment than trained checkpoints, so training cannot be inferred from this geometry alone. Its role is exposed by the factorial and coordinate-emergence tests, while native-order interventions show that efficient finite-path geometry is not a sufficient surrogate for function. These dissociations keep a compact geometric metric from absorbing distinct architectural, training and execution effects.

The local-transition experiments connect the organizational account to causal leverage. A mechanism-conditioned operator crossed a declared output boundary in a subset of held-out prompts while preserving most non-target outputs. The actuator acts on a provenance-defined candidate competition and a fixed relation-graph family. It leaves open how far analogous interventions can transfer to tasks with different output structures, and the failed direction-specificity correction leaves the active transition remapping unresolved.

The same distinction creates testable engineering questions. Component replacement, model merging, adapters, pruning and distillation alter trained parts of a system, whereas dynamic-depth and routing methods alter how those parts are executed. The present experiments do not evaluate these applications, but they identify two quantities such interventions may need to preserve: compatibility among trained components and the functional equivalence of the resulting execution program. A tensor-compatible replacement or an efficient alternative schedule is therefore not, by itself, evidence of functional equivalence.

Together, the results establish a bounded experimental route from locating internal states to testing how learned computation is organized across components, execution and transitions. They also state a broader separability principle: architecture, learned compatibility and realized execution should be distinguished experimentally before a distributed computation is assigned a common mechanism. That principle is transferable as a test design, not as a claim that other systems share the Transformer mechanisms observed here.

## References
1. Rogers, A., Kovaleva, O. & Rumshisky, A. A primer in BERTology: what we know about how BERT works. *Transactions of the Association for Computational Linguistics* **8**, 842-866 (2020).
2. Elhage, N. et al. A mathematical framework for transformer circuits. *Transformer Circuits Thread* https://transformer-circuits.pub/2021/framework/index.html (2021).
3. Olsson, C. et al. In-context learning and induction heads. *Transformer Circuits Thread* https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html (2022).
4. Meng, K. et al. Locating and editing factual associations in GPT. *Advances in Neural Information Processing Systems* **35**, 17359-17372 (2022).
5. Belrose, N. et al. Eliciting latent predictions from transformers with the tuned lens. Preprint at https://doi.org/10.48550/arXiv.2303.08112 (2023).
6. Bricken, T. et al. Towards monosemanticity: decomposing language models with dictionary learning. *Transformer Circuits Thread* https://transformer-circuits.pub/2023/monosemantic-features/index.html (2023).
7. Damirchi, H., Meza De la Jara, I., Abbasnejad, E., Shamsi, A., Zhang, Z. & Shi, J. Truth as a trajectory: what internal representations reveal about large language model reasoning. Preprint at https://doi.org/10.48550/arXiv.2603.01326 (2026).
8. Pandey, V., Singh, G. & Mahdid, Y. Trajectory geometry of Transformer representations across layers. Preprint at https://doi.org/10.48550/arXiv.2606.09287 (2026).
9. Vaswani, A. et al. Attention is all you need. *Advances in Neural Information Processing Systems* **30**, 5998-6008 (2017).
10. Brown, T. B. et al. Language models are few-shot learners. *Advances in Neural Information Processing Systems* **33**, 1877-1901 (2020).
11. Wei, J. et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems* **35**, 24824-24837 (2022).
12. Bubeck, S. et al. Sparks of artificial general intelligence: early experiments with GPT-4. Preprint at https://doi.org/10.48550/arXiv.2303.12712 (2023).
13. Hewitt, J. & Liang, P. Designing and interpreting probes with control tasks. In *Proceedings of EMNLP-IJCNLP* 2733-2743 (2019).
14. Michel, P., Levy, O. & Neubig, G. Are sixteen heads really better than one? In *Advances in Neural Information Processing Systems* **32** (2019).
15. Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S. & Garriga-Alonso, A. Towards automated circuit discovery for mechanistic interpretability. *Advances in Neural Information Processing Systems* **36**, 16318-16352 (2023).
16. Geva, M., Schuster, R., Berant, J. & Levy, O. Transformer feed-forward layers are key-value memories. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* 5484-5495 (2021).
17. Chuang, Y.-S. et al. DoLa: decoding by contrasting layers improves factuality in large language models. In *The Twelfth International Conference on Learning Representations* (2024).
18. Huben, R., Cunningham, H., Riggs Smith, L., Ewart, A. & Sharkey, L. Sparse autoencoders find highly interpretable features in language models. In *The Twelfth International Conference on Learning Representations* (2024).
19. Gao, L. et al. Scaling and evaluating sparse autoencoders. In *The Thirteenth International Conference on Learning Representations* (2025).
20. Zhao, H. et al. Explainability for large language models: a survey. *ACM Transactions on Intelligent Systems and Technology* **15**, Article 20 (2024).
21. Rai, D. et al. A practical review of mechanistic interpretability for transformer-based language models. Preprint at https://doi.org/10.48550/arXiv.2407.02646 (2024).
22. Raghu, M., Gilmer, J., Yosinski, J. & Sohl-Dickstein, J. SVCCA: singular vector canonical correlation analysis for deep learning dynamics and interpretability. *Advances in Neural Information Processing Systems* **30**, 6076-6085 (2017).
23. Kornblith, S., Norouzi, M., Lee, H. & Hinton, G. Similarity of neural network representations revisited. In *Proceedings of the 36th International Conference on Machine Learning* 3519-3529 (2019).
24. Lange, R. D., Kwok, D., Matelsky, J. K., Wang, X., Rolnick, D. & Kording, K. Deep networks as paths on the manifold of neural representations. *Proceedings of Machine Learning Research* **221**, 102-133 (2023).
25. Liu, X., Liu, Y., Zhang, J., Zhang, Y., Zhang, K. & Liu, Q. Layer-order inversion: rethinking latent multi-hop reasoning in large language models. Preprint at https://doi.org/10.48550/arXiv.2601.03542 (2026).
26. Li, Z., Li, Y. & Zhou, T. Skip a layer or loop it? Learning program-of-layers in LLMs. Preprint at https://doi.org/10.48550/arXiv.2606.06574 (2026).
27. Churchland, M. M. et al. Neural population dynamics during reaching. *Nature* **487**, 51-56 (2012).
28. Chen, B., Huang, K., Raghupathi, S., Chandratreya, I., Du, Q. & Lipson, H. Automated discovery of fundamental variables hidden in experimental data. *Nature Computational Science* **2**, 433-442 (2022).
29. Chen, S. et al. Data-driven reaction coordinate discovery in overdamped and non-conservative systems: application to optical matter structural isomerization. *Nature Communications* **12**, 2548 (2021).
30. Clark, P. et al. Think you have solved question answering? Try ARC, the AI2 Reasoning Challenge. Preprint at https://doi.org/10.48550/arXiv.1803.05457 (2018).

## Methods

### Checkpoints, prompts and hidden-state extraction

We analysed representative checkpoints from the Qwen, Llama and Gemma families: Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct and Gemma-2-2B-it (exact revisions in Extended Data Table 1 and extraction identities in Extended Data Fig. 1). The principal coordinate-audit dataset contained 960 prompt-condition traces per checkpoint. Analyses requiring identical cross-checkpoint conditions and repeated geometry nulls used a 96-prompt controlled subset. Eight complete graph groups were sampled with seed 20260610, and all 12 conditions in each group were retained. This preserved complete within-group panels and equal condition counts while keeping 50 matched shuffles computationally auditable. The subset was not the first 96 rows.

For each prompt, we extracted the last non-padding-token state from each transformer block output and excluded the embedding layer. Qwen, Llama and Gemma contributed 28, 16 and 26 layer states with hidden dimensions 1,536, 2,048 and 2,304, respectively. An inference pass was represented as \(T_{m,p}=(h_{m,p,1},\ldots,h_{m,p,L_m})\). Layer depth is an ordered evolution coordinate. It is not interpreted as physical time.

### Readback diagnostics and projection controls

The relation-graph task defined two candidate labels for every prompt. The clean candidate followed the unperturbed relation chain, whereas the conflict candidate was introduced by a distractor, competing statement, update or exception. In the 12-condition coordinate-audit panel, the stable class contained clean, rename, irrelevant, paraphrase and weak-distractor rows, with the last stored as stable-shift; the competition class contained balanced competition, direct conflict, equal evidence and source claim, with the last stored as source-ambiguity; and the closure class contained closure update, closure override and exception override. Labels came from controlled prompt metadata, not from \(\Delta U\), a fitted classifier or the generated answer. The intervention panel used its own predeclared mapping (Supplementary Methods Table 1). Relation graph was the grouping unit for every split.

Vocabulary-facing margins were calculated from the output matrix \(W_m\). For the task-defined clean candidate token \(c_p\) and conflict candidate token \(e_p\),

\[
R_{p,l}=h_{p,l}^{\top}(W_{c_p}-W_{e_p}).
\]

Vocabulary-neighbourhood features summarized the top-ranked vocabulary items under this projection at each layer. Projection-selection geometry controls replaced the real output matrix by either a row-permuted matrix or a seeded Gaussian matrix while retaining the hidden states and analysis pipeline. These analyses test coordinate dependence of readback geometry. They do not test whether readbacks contain any task information and do not make a readback the primary process object.

The decision-window clean-relative margin profile is defined as the vector of clean-relative margins over decision layers. Because \(\Delta U\) is derived from the same matrix, this quantity is a construction-aligned reference. The Source Data manifest maps repository fields to these reader-facing definitions.

The independent coordinate audit augmented the same predecision base state with one proxy family at a time. Proxy dimensionality was capped at 32; when reduction was required, principal-component analysis was fitted only in each training fold and applied to its held-out fold. Five-fold group cross-validation treated each relation graph as indivisible, so no prompts from one graph occurred in both training and test data and each graph was held out once. Mechanism macro F1 was the unweighted mean of class-specific F1 scores; its reported gain was the augmented-model score minus the base-state score. These labels were fixed before model fitting and did not use \(\Delta U\) or generated answers.

### Exploratory functional-window scan and prospective boundary test

Cross-checkpoint windows were selected by a sliding scan over the first 45% of model depth. Candidate window lengths were 2, 3, 4, 5, 6 and 8 layers with a one-layer step; readback neighbourhood sizes \(K=100\) and \(K=500\) were evaluated. For each window, a preservation score was calculated as one minus the mean clean-relative distance for relation-preserving conditions. The source code stored this quantity under a topology field, but it is not itself a topology-separation test. Prompt-condition-class information was measured by group-aware cross-validated macro F1, and downstream coupling by a group-aware cross-validated correlation with \(\Delta U\). The exploratory objective was

\[
S=0.5S_{\mathrm{preservation}}+0.8\max(S_{\mathrm{class}},0)+0.8\max(S_{\mathrm{downstream}},0).
\]

The maximum-scoring interval was retained for each checkpoint. Selection and evaluation used the same scan. Group-aware folds prevent graph-group overlap within the component estimates but do not correct selection optimism because window selection was not nested.

For the prospective boundary test, the selected window and \(K\) were frozen before extraction of 24 new graph groups. The confirmation set retained the 12 condition templates but used six new entity triples, four new relation-phrase pairs and 12 candidate labels disjoint from discovery. The discovery rows alone fitted the decision-profile PCA, feature scaling, three-class logistic classifier and ridge model. Confirmation labels were used only for scoring. Two thousand graph bootstraps and 999 discovery-label or within-graph target permutations supplied intervals and nulls. Per checkpoint, recurrence required a positive topology-separation interval, macro F1 at least 0.55 and above the label-null 95th percentile, and positive downstream correlation above its target-null 95th percentile. Frozen-window location was tested separately against the unchanged candidate pool, with top quartile declared in advance. No checkpoint passed the joint role gate because topology separation was negative in all three; Qwen and Gemma passed the two predictive endpoints, whereas Llama passed class information only. Frozen-window ranks were 58/112, 38/40 and 5/100. We therefore retain the intervals as exploratory indexing choices and do not interpret them as confirmed or universal phases.

### Construction and boundary of \(\Delta U\)

Within each graph group, the clean condition supplied a layerwise baseline \(R_{p_0,l}\). The construction is a task-specific diagnostic and is not treated as a physical energy or general reaction coordinate, despite related coordinate-discovery work in physical systems<sup>28,29</sup>. For every condition \(p\), the clean-relative margin was

\[
dR_{p,l}=R_{p,l}-R_{p_0,l}.
\]

If a group lacked a clean row, the implementation used the group mean as the baseline. Let \(\mathcal{L}_{\mathrm{dec}}\) be the mapped decision-layer set and

\[
D=\left[dR_{p,l}\right]_{p,\,l\in\mathcal{L}_{\mathrm{dec}}}.
\]

The task metadata specify a clean candidate and a competing conflict candidate, and these candidate identities enter \(R\) and \(D\). The model's final generated answer and its correctness label do not. Principal-component analysis was fitted to \(D\), and \(\Delta U_p\) was the score on PC1. Its sign was oriented, where possible, so that stable or stable-shift rows had larger mean scores than closure rows. The decision-window clean-relative margin profile is the row \(D_{p,:}\) itself.

Predecision controls rebuilt the predictors after excluding decision layers and endpoint-derived features. Candidate-margin-excluded controls used neighbourhood-flow variables that did not include the direct clean-versus-conflict margin; topology and auxiliary-band controls changed the analysed layer interval. The correlation of \(\Delta U\) with a dominance readout built from related decision-window information is treated only as construction sanity. \(\Delta U\) is a bounded candidate-path separation observable, not an energy, a final-answer label or independent proof of mechanism.

### State-transition closure

Closure analyses compared feature families for describing the observed next-layer state structure. State-only features contained the selected current-state variables. Previous-transition history features encoded prior clean-relative margin changes, prior rank-gap changes, reversals in each signal and two backtracking indicators. The realized continuous transition coordinate \(O_l^{\mathrm{cont}}\) comprised the first ten principal components of the observed transition representation. State plus \(O_l^{\mathrm{cont}}\) and bilinear models used

\[
\hat y_{p,l+1}=f(z_{p,l},O_l^{\mathrm{cont}})
\]

and

\[
\hat y_{p,l+1}=f(z_{p,l},O_l^{\mathrm{cont}},z_{p,l}\otimes O_l^{\mathrm{cont}}),
\]

respectively. Closure was summarized by macro \(R^2\), the unweighted mean of target-wise \(R^2\) values, together with variance-weighted \(R^2\), RMSE and MAE where available. When \(O_l^{\mathrm{cont}}\) was calculated from adjacent states, the analysis is descriptive factorization because target-transition information enters the coordinate.

For the predictive boundary test, \(\hat O_l^{\mathrm{cont}}\) was estimated under graph-grouped cross-validation from current-state features, previous-transition history and, in the richest model, layer and prompt-condition context. The target layer and realized transition were excluded from these predictors. Within each outer fold, independently predicted training- and test-fold coordinates were combined with current-state variables and their bilinear products. All preprocessing and both stages were fitted on training groups only. The corrected branch produced complete out-of-fold predictions for all four feature sets; unaffected state-only and realized-coordinate branches were numerically invariant to the correction. Supplementary Methods Table 1 maps the information boundary for each analysis family.

### Prospective final-margin forecasting

We separately forecast the exact final clean-minus-conflict output-logit margin from partial decision-window histories. The frozen relation-graph split contained 67 training groups (670 prompts) and 29 held-out groups (290 prompts). At cutoff \(k\), the ordered-history ridge model used \(R_{p,l}\) only for decision layers \(l\leq k\); no later layer state, realized transition, final generated token or correctness label entered its predictors. Ridge penalty was 10. A balanced logistic model with \(C=1\) predicted the sign of the final candidate margin from the same features. We report held-out \(R^2\), mean absolute error and boundary AUROC, with 1,000 graph bootstrap replicates.

The order null retained the current-cutoff value and independently permuted all preceding values within each prompt. Fifty such null datasets were refitted and evaluated on the unchanged held-out graph groups. This destroys common history order while retaining each prompt's observed values and current state. The earliest cutoff whose real held-out \(R^2\) exceeded the null 95th percentile was layer 24 in Qwen, layer 14 in Llama and layer 21 in Gemma. This analysis tests prospective information in a task-defined reduced trajectory. It is not a closed-form state equation, next-token correctness model or universal forecasting law.

### Direct hidden-state chord geometry

The direct geometry audit used the 96-prompt controlled subset. States were normalized as \(x_l=h_l/\lVert h_l\rVert_2\), step vectors were \(s_l=x_{l+1}-x_l\), and the endpoint chord was \(g=x_L-x_1\). Chord alignment was the mean cosine between \(s_l\) and \(g\). Path length was \(\sum_l[1-\cos(x_l,x_{l+1})]\), endpoint distance was \(1-\cos(x_1,x_L)\), and detour was their ratio. Turning curvature was the mean angle between adjacent step vectors.

For Fig. 1, adjacent hidden-state change was \(1-\cos(x_l,x_{l+1})\). The dimensional participation fraction of a step was \((\sum_j s_{l,j}^2)^2/[d_m\sum_j s_{l,j}^4]\). A value near one indicates broadly distributed step energy, whereas a small value indicates concentration in fewer hidden dimensions. Means and standard errors were calculated within collapsed coordinate-audit condition family and checkpoint on the 96-prompt controlled subset.

For each checkpoint and each of 50 repeats, the null held the first and final states fixed and applied one random permutation to the intermediate layer order. The same permutation was applied to all 96 prompts within that checkpoint. This preserves the state set and endpoints while destroying observed layer order. The inferential unit was the checkpoint-level mean for one shared permutation; no prompt-wise p values were aggregated. Alignment gain was the real mean minus the shuffle mean. Detour and curvature reductions were

\[
G_q=\log_2\left(\frac{\operatorname{mean}(q_{\mathrm{shuffle}})}{\operatorname{mean}(q_{\mathrm{real}})}\right),
\]

for \(q\in\{\mathrm{detour},\mathrm{curvature}\}\). For each metric, a null-standardized effect was the signed real-minus-null-mean difference divided by the standard deviation of the 50 checkpoint-level null means, oriented so that positive values favour observed order. A plus-one empirical one-sided p value was \((1+n_{\mathrm{null\ as\ or\ more\ extreme}})/(50+1)\); its minimum resolution was therefore \(1/51\). The reported z values are standardized null separations, not Gaussian p values. These diagnostics are metric-specific finite-path comparisons; they do not fit a continuous geometric evolution equation.

### Independent task and random-initialization controls

The controlled lexical task contained 64 problems with candidate words drawn from seven categories. The controlled addition task contained 64 single-step problems. In each task, candidate order was counterbalanced and every problem supplied six matched prompt conditions. A fixed split assigned 44 problems to coordinate fitting and 20 to held-out evaluation. The task-specific coordinate was PC1 of training-problem clean-relative margin profiles and was applied without refitting; a one-coordinate logistic classifier was compared with 200 training-label shuffles. The numerical relation-graph \(\Delta U\) axis was not reused.

For a standard non-binary benchmark, we used a deterministic 96-item ARC-Challenge validation audit subset whose answer labels were exactly A-D<sup>30</sup>. The subset comprised the earliest eligible rows in dataset order, a rule fixed without reference to model outputs to bound computation and preclude outcome-based selection. Original questions and all four choices were presented through each checkpoint's native chat template; no binary wrapper was added. This is a fixed audit set, not a new benchmark estimate. Lexical, addition and ARC geometry used the same normalized-state definitions and 50 endpoint-preserving shuffles as the relation-graph analysis; no model-specific window search was performed.

The random-initialization control used the exact Qwen2.5-1.5B, Llama-3.2-1B and Gemma-2-2B configurations and native tokenizers. Three models per architecture were instantiated with seeds 1701, 1702 and 1703. Embeddings, transformer blocks, normalization parameters and output heads were newly initialized; no trained tensor was loaded. Each random state and its matched trained checkpoint used the same 96 raw relation-graph prompts and 50 endpoint-preserving shuffles with null seed 20260717. The separately seeded Fig. 4 null gives slightly different trained gain estimates although the real trajectories are identical. This within-architecture comparison separates architecture-compatible order effects from training-specific organization for the tested configurations and prompt subset; it does not identify a universal Transformer law.

### Frozen closure audits

The closure protocol fixed the predecision cutoffs, five-fold graph partition, metrics, nulls and multiplicity families before execution. For the coordinate competition, \(R_l\) was the clean-minus-conflict output-logit margin at predecision layer \(l\), and \(dR_l\) was its clean-condition-matched version in the same graph group. Each training fold fitted scaling, PCA and a class-balanced multinomial logistic classifier. The compared features were the current or recent \(R_l\) and \(dR_l\), a one-dimensional PC1 score \(\Delta U_{\mathrm{pre}}\) from the complete training-fold \(dR\) history, the complete standardized \(dR\) history, and a 32-component training-fold PCA of the cutoff hidden state. No post-cutoff state, final token, correctness label or held-out statistic entered the construction. The primary contrasts used 100,000 graph-level sign flips and Benjamini-Hochberg adjustment across nine checkpoint-by-contrast tests (Supplementary Methods Table 9).

For the direction null, the released held-out gate, layer schedule, action weights, local operator and precursor components were retained exactly. A random direction was sampled at each intervention layer from the empirical training-update subspace, projected orthogonally to the structured component and normalized to the same component norm. Fifty-one seeds per checkpoint preserved the complete gate pattern, action schedule, layer placement and full-vocabulary endpoint. The primary specificity was closure strict conflict-to-clean crossings minus non-closure full-vocabulary top-1 changes. The three checkpoint tests form one multiplicity family (Supplementary Methods Table 10).

For new-graph geometry, 24 disjoint graph groups and ten fixed conditions per group supplied the same 240 raw prompt strings to Qwen2.5-1.5B and Gemma-3-12B-it-QAT-Q4_0. For both checkpoints, the extracted record was the last non-padding token from every native block output, with the embedding state excluded. Fifty shared permutations retained the first and last block states and permuted only the intermediate order for every prompt in a repeat. Chord alignment was the primary outcome; detour and turning curvature were supporting metrics. The two checkpoint-level chord tests form a separate multiplicity family (Supplementary Methods Table 11). The larger Gemma checkpoint was loaded through the pinned native capture runtime; its GGUF QAT loading path, quantization and model digest are in Extended Data Table 1 and Source Data, but the state-extraction rule was unchanged.

### Falsification-first scaffold, order and metric controls

The cross-task random-map test instantiated two additional seeds (1801 and 1802) for each architecture and reused each fixed map across 96 lexical, 96 addition and 96 ARC prompts. Cross-half and first-to-last-quarter unbiased linear CKA were compared with 199 condition- and answer-stratified lag-1 donor, contiguous four-step donor and prompt-fingerprint-preserving donor nulls. A condition passed only when both CKA measures exceeded the 95th percentile of all three nulls at empirical \(P\leq0.05\). This tests relation preservation by a fixed random architecture, not semantics, native-order privilege or function.

The order-function dissociation used a trained Qwen residual-execution cache for 48 prompts under native, reverse and one frozen permuted block order. An eight-prompt reconstruction audit reproduced native full-vocabulary top-1 exactly and candidate margins within 0.05 before the cache was used. Geometry was the paired change in unit-state future-chord alignment. Functional endpoints were full-vocabulary top-1 agreement, candidate-choice agreement, Jensen-Shannon divergence and native-token rank; 10,000 paired bootstraps and 100,000 group sign flips quantified uncertainty. Passing means only that greater alignment is insufficient for native functional fidelity.

The incremental path-memory audit compared ordered candidate-margin prefixes with the current margin, a fold-fitted 32-component current hidden state, unordered prefix summaries and 199 within-prompt order shuffles. Five-fold problem-group splits covered lexical and addition tasks at 25%, 50% and 75% depth in each checkpoint; a separate mixed-arithmetic task repeated the same nine model-by-horizon tests. Benjamini-Hochberg correction was applied within contrast family. A condition passed only if ordered history exceeded all four controls at \(q\leq0.05\). This audit is separate from the relation-graph forecast, which retains a positive but task-bounded result (Extended Data Fig. 4).

The Llama radial audit used consistent block-output residual states from four original and three independent tasks. For \(h_l=r_lu_l\), direction-only paths set \(r_l\) constant and norm-only paths held \(u_l\) fixed. A frozen dose series used \(h_l(\gamma)=(r_l/r_0)^\gamma u_l\) for \(\gamma\in\{0,0.25,0.5,0.75,1,1.25\}\), with 199 layer-conditioned recipient-norm donor walks at each value. The statistic was strongly radial-sensitive, but the deliberately strict binary sufficiency gate failed; no functional or semantic conclusion follows.

### Training-time component and coordinate audits

We used one documented EleutherAI Pythia-160m training run at step 0 and step 143,000. On a frozen 53-prompt single-token lexical cloze set, all 16 combinations of step-0 and trained input embeddings \(E\), transformer blocks \(B\), final normalization \(N\) and untied output head \(W\) were evaluated. Primary endpoints were clean-versus-conflict pair accuracy and mean candidate margin. Prompt-paired uncertainty used 5,000 bootstraps; factorial contrasts used the complete orthogonal \(2^4\) design. Component boundaries and prompts were fixed before outcomes were inspected. Hybrid states are perturbational controls, not natural training checkpoints.

### Prospective exact-initial OLMo E/B/W factorial

The second-architecture component audit followed a configuration audit establishing that OLMo final normalization, `model.norm`, has zero state tensors and cannot enter an independently swappable factorial. Exploratory rows from that configuration audit were protocol-ineligible and do not enter the present statistics. We used `allenai/OLMo-1B-0724-hf` at documented `step0-tokens0B` (commit `f4198b342c80f0ec85b2d7caedec977124a8ffb8`) and final `main` (commit `d7cbab742d80589e714b1a2d7f838dcd21cbe143`). Before task forwards, SHA-256 values for both weight shards and the shared configuration, tokenizer and safetensors-index files were verified. The checkpoints shared the OLMo configuration and tokenizer; E (`model.embed_tokens`), B (`model.layers`) and W (`lm_head`) had 1, 112 and 1 changed state tensors, respectively.

The prospective component order was E, B and W, with zero denoting the exact-initial state and one the final trained state. All eight conditions from `000` through `111` were evaluated on the same manifest. The manifest was created with the step-0 tokenizer alone before formal model loading. It retained 96 one-token lexical continuations from 12 categories, with eight prompts per category. No prompt string, category or clean/conflict candidate label overlapped earlier component-audit inputs. Exploratory rows from the preceding configuration audit were excluded from formal statistics and confidence intervals.

Co-primary endpoints were candidate-pair accuracy, \(1[\operatorname{logit}(\mathrm{clean})>\operatorname{logit}(\mathrm{conflict})]\), and vocabulary-standardized candidate margin, \([\operatorname{logit}(\mathrm{clean})-\operatorname{logit}(\mathrm{conflict})]/\operatorname{sd}(\mathrm{full\ vocabulary\ logits})\), at the final next-token position. Paired prompt bootstraps used 10,000 replicates. TAR0 required all-trained gain over all-initial for both endpoints. TAR1 required all-trained performance above every isolated component condition. TAR2 required each isolated standardized-gain recovery fraction to remain below 0.80. TAR3 required a positive E:B or E:B:W standardized-margin interaction confidence interval. TAR4 required a positive standardized all-trained superadditive excess. Category robustness required at least nine of twelve categories to have positive mean all-trained-minus-best-isolated and all-trained-minus-all-initial standardized margins. A 199-permutation clean/conflict-label null and exact repeated forwards for eight `000` and eight `111` prompts were supporting controls.

The completed factorial produced 768 prompt-hybrid rows. Candidate-pair accuracy was 0.5208 for `000` and 0.9896 for `111`; their paired difference was 0.46875 (95% confidence interval 0.375 to 0.573). The all-trained standardized-margin gain was 2.6061 (95% confidence interval 2.276 to 2.938). All five TAR gates and category robustness passed. An independent post-execution audit reproduced the gates from raw rows. It corrected only a category metadata label that had counted 64 prompt-hybrid rows rather than eight unique prompts per category; no statistic or verdict changed.

The learned-coordinate audit used the same 53 prompts at training steps 0, 128, 1,000, 4,000, 16,000, 64,000 and 143,000. The model-native candidate coordinate was \(c_{p,l}=\cos(h_{p,l},W_{c_p})-\cos(h_{p,l},W_{e_p})\); mean prompt-wise Spearman correlation with depth was compared with 999 endpoint-preserving interior-layer shuffles. For the category coordinate, clean answer words rather than prompt rows were divided into three deterministic folds. A 16-component training-fold PCA and fixed-regularization multinomial logistic regression were applied to every layer of held-out words. We compared a coordinate refitted within each checkpoint with the final-checkpoint coordinate transferred without refitting to earlier checkpoints, and used 199 category-label permutations at step 0 and the final checkpoint. The explicit category word made within-checkpoint random features linearly decodable at initialization; that negative control is retained rather than interpreted as learned semantics.

### Readback-centre geometry and residual diagnostics

The auxiliary readback-centre analysis used normalized vocabulary-neighbourhood centre trajectories rather than raw hidden states. Its endpoint chord, cosine-distance path, detour and turning metrics used an endpoint-preserving permutation of intermediate readback-centre layers. This analysis is therefore reported in Extended Data as a cross-observable diagnostic and is not merged with the direct hidden-state estimand.

Residual diagnostics compared each observed update with its chord-directed reference and summarized components parallel and perpendicular to the reference direction. These are reduced-order residual proxies. They test whether the remainder retains classification or \(\Delta U\)-prediction information, but are not an exhaustive tangent-normal decomposition.

### Mechanism-conditioned intervention

Three additive-state controls used prototype- or linear-coefficient-derived directions; a subsequent additive-flow control used a flow direction. Because these controls did not provide robust mechanism-specific selectivity, the final intervention changed the actuator to a mechanism-conditioned local transition operator. At intervention layer \(l\), a common principal-component basis was fitted to the concatenated current and next hidden states from training graphs only, with dimension \(r_l=\min(128,2n_{\mathrm{train}}-1,d_m)\). For stable and closure rows separately, ridge regression estimated \((A_l^{(k)},b_l^{(k)})\) by

\[
\underset{A,b}{\operatorname{argmin}}\;\sum_{i\in\mathcal T_k}\lVert z_{i,l+1}-Az_{i,l}-b\rVert_2^2+10\lVert A\rVert_F^2,
\]

where \(z\) denotes the shared PCA coordinate and \(k\in\{\mathrm{stable},\mathrm{closure}\}\). No held-out graph entered either fit. The stable-row prediction was inverse-transformed to hidden-state space to define \(\hat F_l^{\mathrm{stable}}\). For realized block output \(h_{p,l+1}\), its proposal was applied as

\[
\tilde h_{p,l+1}=(1-\alpha_{p,l})h_{p,l+1}+\alpha_{p,l}\hat F_l^{\mathrm{stable}}(h_{p,l}),
\]

followed by the order-precursor update

\[
h_{p,l+1}^{\mathrm{int}}=\tilde h_{p,l+1}+\beta_{p,l}\operatorname{sd}(\tilde h_{p,l+1})d_l.
\]

Here \(d_l\) is the unit ridge-coefficient direction for \(\Delta U\), fitted on training graphs; \(\alpha\) controls operator interpolation and \(\beta\) controls the scaled precursor update. The operator step was applied before the precursor step. The action grid was \(\{(0,0),(0.1,0.1),(0.2,0.2),(0.3,0.3),(0.3,0),(0,0.3),(0.3,0.2),(0.2,0.3)\}\). The fixed-combination baseline used \((0.3,0.3)\).

The prompt-condition classifier, labelled "mechanism classifier" in archived code and Fig. 5d, used three prompt-condition labels fixed in the ten-condition control panel. Stable comprised clean, redundant, irrelevant and paraphrase conditions; competition comprised weak-distractor, balanced-competition and direct-conflict conditions; closure comprised closure-update, closure-override and exception-override conditions. These are controlled intervention families, not correctness labels or trajectory-derived dynamical phases. Its nine input features were predicted candidate-path separation and its magnitude, a companion dominance estimate, velocity magnitude and standardized velocity, normalized layer depth, a precursor-layer indicator, and baseline separation and dominance values. A 250-tree random forest used maximum depth 7, minimum leaf size 6 and class-balanced bootstrap weights. The policy classifier added three prompt-condition probabilities and three predicted-class indicators, giving 15 inputs, and used a 200-tree random forest with maximum depth 6, minimum leaf size 8 and the same weighting rule. Both used standard weighted-Gini tree splits. There were no neural hidden layers, gradient optimizer or policy-gradient objective.

Policy supervision was obtained by evaluating all eight \((\alpha,\beta)\) actions on training graphs. For closure prompt \(p\), the target action maximized \(\Delta U_p^{(a)}-\Delta U_{p,\mathrm{base}}\); for stable and competition prompts, it minimized the absolute shift. The policy forest then classified the 15-dimensional held-out layer features into the resulting discrete action labels. Policy labels encode \((\alpha,\beta)\); none denotes \((0,0)\).

Classifiers used a 70:30 group-aware split by relation graph on layer-expanded rows. Qwen and Gemma used 3,350 training rows from 670 policy-training prompts and Llama used 2,680 rows because of its shorter layer stack; 290 prompts formed the held-out test partition. Rows from one graph could not occur in both partitions. In a prompt-condition-label null, training prompt-condition labels were permuted before refitting the prompt-condition classifier and downstream policy. In a policy-label null, the action labels were permuted before refitting the policy while retaining the real prompt-condition classifier. The joint null independently permuted both label sets and refitted both classifiers; test features, graph split, transition operators and evaluation code were unchanged. Each null family was repeated 50 times. Final-answer correctness was not a classifier target.

For checkpoint \(m\) and condition \(c\), specificity was

\[
s(m,c)=\operatorname{mean}_{i\in S_{\mathrm{closure}}}|\Delta U_i^{(c)}-\Delta U_{i,\mathrm{base}}|-\operatorname{mean}_{i\in S_{\mathrm{nonclosure}}}|\Delta U_i^{(c)}-\Delta U_{i,\mathrm{base}}|.
\]

Fig. 5 reports real and matched fixed-combination specificity; their difference is the real-versus-fixed gain. Shuffle-null separation is reported against the joint shuffle distribution. For visualization only, the pseudo-log transform is \(f(x)=\operatorname{asinh}[x/(2\sigma)]/\ln 10\), with \(\sigma=0.08\); it is approximately linear near zero and logarithmic at larger magnitudes. It does not change the underlying specificity calculation.

The candidate-choice boundary analysis paired the no-intervention and mechanism-aware-policy rows for each of 290 held-out prompts per checkpoint. The clean-candidate margin was the final clean-candidate logit minus the final conflict-candidate logit. Changes in margin, \(\Delta U\) and the binary choice between these two candidates were calculated within prompt. Ninety-five per cent intervals for the paired descriptive contrasts resampled the 29 held-out relation graphs with replacement for 5,000 replicates using seed 20260714. Exact paired-flip tests compared conflict-to-clean with clean-to-conflict flips.

For the displacement-to-boundary bridge, a five-fold graph split shared across checkpoints cross-fitted model- and regime-specific affine maps \(\delta M=a_{m,r}+b_{m,r}\delta\Delta U\), where \(\delta M\) is the policy-induced candidate-margin change. The predicted post-policy margin was \(M_{0}+\widehat{\delta M}\). On prompts with \(M_0<0\), crossing meant that the observed policy margin was non-negative. Nested graph-held-out logistic calibration converted predicted post-margin to crossing probability. Continuous prediction and crossing metrics used 1,000 graph-bootstrap replicates. Final correctness labels did not enter either model. Clean denotes the unperturbed-chain candidate and is not necessarily the rule-consistent answer under closure; this analysis measures crossing between the two declared candidates, not unrestricted generated-answer correctness.

### Confidence-gated boundary control

The output-boundary extension was run independently in Qwen, Llama and Gemma on the same seed-42 graph split: 67 training graphs (670 prompts) and 29 held-out graphs (290 prompts, including 87 closure prompts). Intervention layers were 15-19, 8-11 and 13-17, respectively. In every checkpoint, the principal-component direction defining \(\Delta U\), local transition operators, precursor directions, prompt-condition classifier, action labels and boundary policy were fitted using training graphs only. Candidate token identities entered the paired-margin objective; the model's generated answer did not enter fitting or gate construction.

The expanded action library was

\[
\begin{aligned}
\mathcal A=\{&(0,0),(0.3,0.3),(0.6,0.3),(0.6,0.6),(0.9,0.3),\\
&(0.9,0.6),(0.9,0.9),(1.0,0.6),(1.0,0.9),(1.0,1.2)\}.
\end{aligned}
\]

For a training closure prompt, action utility was the clean-minus-conflict margin shift plus a 5-unit crossing bonus minus \(0.03(\alpha^2+\beta^2)\); for non-closure prompts it was the negative absolute margin shift minus \(0.10(\alpha^2+\beta^2)\). Thus \(\alpha\leq1\) interpolated to, but did not extrapolate beyond, the fitted transition proposal. The policy used the same 15 pre-intervention features as the internal-control analysis.

For prompt \(p\), let \(\pi_{p,l}(k)\) be the mechanism posterior at intervention layer \(l\). We defined

\[
\bar\pi_p=|L_I|^{-1}\sum_{l\in L_I}\pi_{p,l}(\mathrm{closure}),\qquad
\bar H_p=|L_I|^{-1}\sum_{l\in L_I}\frac{-\sum_k\pi_{p,l}(k)\log\pi_{p,l}(k)}{\log 3},
\]

and opened the gate only when \(\bar\pi_p\geq0.50\) and \(\bar H_p\leq0.90\). These thresholds were fixed in the experiment protocol before held-out execution; no threshold sweep or held-out outcome tuning was performed. They should therefore be read as one declared operating point, not an estimated optimum. Gate-closed actions were set to \((0,0)\). The primary behavioural endpoint was a strict change from the conflict token to the clean token as the full-vocabulary top-1 continuation. Non-closure top-1 change and intervention-layer norm ratios were damage guards.

The gate values were inherited from the Qwen configuration. The archive establishes their use as a fixed operating point before held-out execution, but does not establish the original Qwen values as model-free a priori constants or as independently optimized train-only thresholds.

For the safety-matched null, the complete model-specific action vector was permuted without fixed points among gate-open prompts within each closure condition. Each of 50 permutations preserved the gate-open set, zero actions outside the gate, and every condition-by-layer \((\alpha,\beta)\) multiset. Strict conflict-to-clean top-1 changes were counted relative to each prompt's unperturbed baseline. The one-sided empirical value was \((1+n_{\mathrm{null}\geq\mathrm{real}})/(50+1)\). The real counts did not exceed the matched-null 95th percentile in any checkpoint. This null tests action-to-trajectory assignment, not whether the gated actuator family can cross the candidate boundary.

For component attribution, the gate-open set, thresholds, graph split, operators, precursor directions and all training fits were frozen before execution. The operator-only control retained each gated policy value of \(\alpha\) and set \(\beta=0\); the precursor-only control retained \(\beta\) and set \(\alpha=0\). A uniform control applied the pre-existing maximum library action \((1.0,1.2)\) to every gate-open prompt and zero elsewhere. Negative and positive references were rerun and reproduced all 290 previous top-1 token IDs exactly in every checkpoint. Operator-only strict crossings were 17/87, 14/87 and 8/87; precursor-only crossings were 0/87, 1/87 and 0/87; and uniform-maximum crossings were 22/87, 15/87 and 10/87. Non-closure top-1 changes were 0/203, 2/203 and 0/203 for operator only, and 0/203, 2/203 and 1/203 for the uniform control. All intervened states were finite and the maximum recorded state-norm ratio was below 1.10. These ablations attribute most leverage to the gated local transition operator within the tested actuator; they do not establish a universal decomposition.

### Frozen mixed-arithmetic actuator-transfer test

The transfer test contained 64 unique addition, subtraction and multiplication expressions with answer labels from zero to twenty. Each problem supplied clean, paraphrase, untrusted-conflict, asserted-conflict, explicit-correction and final-recheck prompts. The first two were non-target stable prompts, the middle two non-target competition prompts and the final two target correction prompts. All six rows from a problem were assigned together by a frozen split (44 training and 20 held-out problems; seed 20260714). The arithmetic-consistent candidate is correct only within this exact one-word task contract.

The relation-graph intervention windows, ten-action library, operator regularization, classifier settings, gate thresholds and 5% collateral ceiling were frozen. Task-specific coordinates, numerical transition operators, precursor directions and action policies were fitted on training problems only; this tests protocol transfer after training-only refitting, not zero-shot transfer of numerical parameters. The primary interface ended a raw prompt immediately before its one-word continuation, matching the original actuator assay. All candidate labels passed an exact-prefix and single-token continuation audit in all three tokenizers. A native-chat pilot was excluded from the primary endpoint because Llama began all 120 held-out responses with free text rather than either candidate, demonstrating that a candidate-token boundary must be valid for the evaluated interface.

Strict eligibility required the unperturbed full-vocabulary top-1 token to equal the distractor candidate on a held-out correction prompt. The primary guarded actuator was compared with no intervention, its ungated form, a training-selected fixed action, operator-only, precursor-only, uniform-maximum and 50 address-reassignment nulls per checkpoint. Each address null reassigned complete action bundles across held-out prompts while preserving the model-specific action-dose multiset and number of open gates. Specificity was the strict target-crossing rate minus the non-target top-1-change rate. Confirmation required at least two checkpoints with at least five eligible prompts, positive median candidate-margin movement and at least one strict crossing; at least two checkpoints below the collateral ceiling; and pooled specificity above the address-null 95th percentile. Qwen yielded 1/14 crossings, median margin shift 0.539 and 1/80 non-target changes; Llama yielded 1/27, -0.070 and 0/80; Gemma yielded 0/4, 0.094 and 0/80. Pooled specificity was 0.0403 versus a null 95th percentile of -0.0111 (one-sided empirical \(P=1/51\)). Independent recomputation reproduced every primary summary and all 150 null action-dose multisets. The two-checkpoint positive-direction gate failed, fixing the verdict as partial transfer rather than cross-task output control.

### Statistics and reproducibility

The analysis units, group splits, prompt subsets, windows and nulls are summarized in Supplementary Methods Tables 8-11. The claim-family register separates heterogeneous tests by their statistical unit and null; no single study-wide adjusted P value is claimed. Main comparisons are descriptive estimates and matched-null z scores; no single background citation is used to support a core mechanistic result. Processed panel tables are mapped by the global Source Data manifest. Figures were rendered deterministically in R using `ggplot2`, `patchwork` and related packages and exported as vector PDF/SVG and 600-dpi TIFF. Scientific panels were generated from retained data and code, not by a generative image system.

The direct hidden-state geometry analysis used seed 20260610. A fixed-seed rerun reproduced the hashes of the controlled-subset manifest, prompt-level metrics and summary tables. Exact checkpoint revisions and extraction rules are recorded in Extended Data Table 1; classifier features, subsets and null construction are recorded in Supplementary Methods Table 8, and the claim-boundary audit is Supplementary Methods Table 4. Repository accessions and source mappings are reported below.

### Research independence and resource provenance

This study was conducted independently and was not an internal project of any listed organization. No employer data, computing infrastructure, internal systems, funding, research facilities or personnel support were used, and the work did not involve organizational review or approval. All local computing, storage, software subscriptions and other research expenses were provided or paid for personally by H.Z.; public model checkpoints and software were accessed under their stated licences. Y.W., R.S. and W.C. participated in their personal capacities as independent research collaborators. Affiliations identify the authors and do not imply employer sponsorship, endorsement or responsibility for the work.

### AI-assisted research workflow and human oversight

Generative AI tools, including GPT (OpenAI), Gemini (Google), Doubao (ByteDance) and Kimi (Moonshot AI), were used as interactive software for archival retrieval, code scaffolding and debugging, batch-analysis planning, data organization, deterministic R plotting scripts, consistency audits and language editing. H.Z. formulated the scientific questions, selected and revised the empirical process object, specified controls and acceptance criteria, interpreted positive and negative results, set claim boundaries and approved all reported analyses, figures and text. Y.W., R.S. and W.C. provided the domain and methodological support described in the Author contributions statement, and all authors reviewed the manuscript. Analyses were executed locally against the stated checkpoints and source data; generated proposals and code were checked against retained outputs, null controls and source mappings. AI outputs were treated as unverified candidate hypotheses or operational drafts, not as evidence. AI systems were not authors and made no autonomous scientific or editorial decisions. No scientific conclusion was accepted solely from AI output, and no publication figure was produced by an image-generation system. Timestamped records document the compressed N=1 workflow, but without a no-AI counterfactual they support workflow provenance rather than a causal estimate of productivity. The task-level workflow and responsibility matrix are reported in Supplementary Methods.

## Data availability

Processed panel- and row-level tables are archived under the Zenodo Source Data concept DOI https://doi.org/10.5281/zenodo.21317355 and mirrored in the paper-specific public branch at https://github.com/zhaoheng1990-debug/Cognitive_Theory_And_LLM_Phasemap/tree/high-dimensional-layer-state-trajectories. The currently public repository release is v0.57 and retains the frozen empirical evidence package, the archived Article v0.37 and Supplementary Methods v0.20. Prompt and checkpoint metadata, random seeds, split records, null specifications, figure contracts and file-level checksums are included. The display and format updates in the present manuscript do not alter those empirical files. Raw hidden-state arrays are large derivatives of third-party checkpoints and are not redistributed; prompts, checkpoint revisions and extraction scripts required to regenerate them are provided subject to checkpoint access, storage and licensing constraints. Model weights are not redistributed.

## Code availability

Analysis scripts, deterministic R figure scripts, environment records, smoke tests and portable path templates are archived under the Zenodo Code concept DOI https://doi.org/10.5281/zenodo.21317440 and mirrored in the paper-specific public branch at https://github.com/zhaoheng1990-debug/Cognitive_Theory_And_LLM_Phasemap/tree/high-dimensional-layer-state-trajectories. The currently public repository release is v0.57 and includes the frozen claim-family verification matrix, independent read-only verifiers and the mixed-arithmetic transfer runner. Workstation-specific paths and internal development identifiers are removed or replaced with repository-relative arguments while provenance is retained in the private audit record. Full model reruns require the listed checkpoints and any upstream datasets that cannot be redistributed.

## Acknowledgements

H.Z. thanks Youguang Zhang, Weisheng Zhao and Hao Sheng for comments on the manuscript, and acknowledges formative research training at Beihang University. H.Z. also thanks family members for personal support during the independent research period.

## Funding

No external funding was received for this study. All computing, storage, software subscriptions and other research expenses were provided or paid for personally by H.Z.; no resources from the authors' employers were used.

## Author contributions

H.Z. conceived the research, defined the questions, formulated the hypotheses developed during the investigation, designed the experiments, established the evidential tests and revised the interpretations in response to the results. Y.W., an artificial-intelligence specialist, contributed domain knowledge and methodological guidance throughout the research. R.S., a researcher in operations research and mathematics, contributed mathematical-method support to parts of the study. W.C. participated in discussions of the research questions and provided advice and conceptual inspiration from medical, neuroscience and cognitive-science perspectives. Y.W., R.S. and W.C. contributed in their personal capacities as independent research collaborators and did not provide employer data or resources. All authors reviewed the manuscript and approved the submitted version. AI-assisted operations are disclosed in Methods and do not constitute authorship.

## Competing interests

The authors declare no competing interests.

## Additional information

Supplementary information is available for this paper. Correspondence and requests for materials should be addressed to Heng Zhao (zhaoheng1990@gmail.com).

## Figure legends

**Fig. 1 | Experimental separation of Transformer organizational contributions.** a, A visual guide depicts an ordered, high-dimensional layer-state record, a local transition and competing candidate supports; reported tests retain the full hidden state rather than the drawing. b, Direct hidden-state and hidden-flow coordinates retain more predefined prompt-condition information than the tested vocabulary-facing readbacks in grouped audits. c, A controlled relation graph defines chain-supported and conflict candidates for the bounded intervention assay. These labels are provenance-defined and do not mean correct and incorrect. See Source Data Fig. 1 and the Source Data manifest.

**Fig. 2 | Architecture-compatible propagation scaffold and multi-scale confirmation.** a, Random initializations show larger alignment gains than their trained checkpoint counterparts for the tested architectures. b, The native-order alignment contrast recurs across controlled addition, lexical and ARC-Challenge audits and in a paired multi-scale, 12B new-graph confirmation. c, Every displayed task-model pair exceeds its 50 endpoint-preserving shuffles on its own null-standardized scale. The 12B result is a checkpoint-specific confirmation, not a pure scale law because family, quantization and capture runtime differ. Strong alignment alone is not evidence of learned reasoning. All finite-path metrics are defined against the stated endpoint-preserving null. See Source Data Fig. 2 and the Source Data manifest.

**Fig. 3 | Training organizes compatible function across components and coordinates.** a, An exact-initial Pythia \(2^4\) factorial varies input embeddings (E), blocks (B), final normalization (N) and output head (W) on its registered candidate-margin scale. b, A prospective exact-initial OLMo \(2^3\) E/B/W factorial uses a vocabulary-standardized candidate-margin scale on 96 held-out prompts in 12 categories. `111` and `000` denote all-trained and all-initial component states, respectively. c, Interaction estimates remain on their registered experiment-specific scales and are not raw cross-experiment comparisons. d, The Pythia training-associated coordinate line is complementary to the factorial tests; TinyLlama diagnostics are in Extended Data Fig. 6. See Source Data Fig. 3 and the Source Data manifest.

**Fig. 4 | Realized trained order is functionally consequential.** a, Native, reverse and frozen-permutation schedules visibly differ in the block identities executed at early, middle and late positions. b, Frozen reverse or permutation schedules yield zero native full-vocabulary top-1 agreement at every displayed endpoint. c, Candidate-choice agreement can remain partial even when full-vocabulary fidelity is lost. d, Future-chord alignment changes heterogeneously across Llama and Gemma conditions; metric-specific values are not directly comparable across metric types. The frozen schedules test native-output fidelity after training and do not establish that native order is the only valid execution program. See Source Data Fig. 4 and the Source Data manifest.

**Fig. 5 | A bounded local-transition actuator changes a declared output boundary.** a, A visual guide depicts the gate and local transition operator moving an eligible closure trajectory across the declared candidate boundary while a non-target path is retained. b, Held-out conflict-to-clean full-vocabulary top-1 crossings occur in 9/87 Gemma, 13/87 Llama and 17/87 Qwen closure prompts; non-target changes are 0/203, 2/203 and 0/203. c, Operator-only ablations retain most strict crossings, whereas the precursor-only control is insufficient. d, A matched direction null fails the corrected three-checkpoint test; diamonds show the real actuator and horizontal bars the model-specific 95th null percentile. e, A mixed-arithmetic extension remains partial and fails its confirmation gate. The endpoint is a provenance-defined candidate crossing, not answer correctness or general deployment control. See Source Data Fig. 5 and the Source Data manifest.

## Extended Data figure legends

**Extended Data Fig. 1 | Checkpoint and extraction identities.** Model checkpoint configurations and the layer-state extraction rule used across the main analyses. Exact revisions are retained in Extended Data Table 1 and the Source Data manifest.

**Extended Data Fig. 2 | Coordinate and readback qualification.** The grouped-audit comparison of direct hidden-state coordinates and vocabulary-facing readbacks. Readbacks remain diagnostics rather than the empirical process object. See Source Data manifest.

**Extended Data Fig. 3 | Frozen functional-window confirmation did not establish a common phase.** The registered statistics are shown on their respective scales; the joint recurrence criterion failed. See Source Data manifest.

**Extended Data Fig. 4 | Incremental ordered-prefix gate.** Ordered-prefix comparisons at the registered depth horizons. The result does not establish a general path-memory law. See Source Data manifest.

**Extended Data Fig. 5 | Random-initialization geometry control.** Alignment gains for trained checkpoints and matched random initializations. Greater random alignment does not indicate superior computation. See Source Data manifest.

**Extended Data Fig. 6 | Complementary training-line diagnostics.** Pythia and TinyLlama training-line diagnostics support a complementary evidence branch and do not replicate the factorial estimand. See Source Data manifest.

**Extended Data Fig. 7 | Execution-order reconstruction and geometry diagnostics.** Reconstruction and metric-specific geometry diagnostics for non-native execution. See Source Data manifest.

**Extended Data Fig. 8 | Direction-specificity null.** No checkpoint passes the corrected three-checkpoint direction-specificity threshold. See Source Data manifest.

**Extended Data Fig. 9 | Mixed-arithmetic actuator transfer.** The observed transfer is partial and does not pass its confirmation gate for general cross-task output control. See Source Data manifest.

## Extended Data tables

**Extended Data Table 1 | Model, checkpoint and hidden-state extraction metadata.**

| Family | Exact checkpoint and revision | Layers / hidden size | Extraction |
|---|---|---|---|
| Qwen | Qwen/Qwen2.5-1.5B-Instruct, 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | 28 / 1,536 | Last non-padding token from block output \(l+1\); embedding excluded |
| Llama | meta-llama/Llama-3.2-1B-Instruct, 9213176726f574b556790deb65791e0c5aa438b6 | 16 / 2,048 | Last non-padding token from block output \(l+1\); embedding excluded |
| Gemma | google/gemma-2-2b-it, 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8 | 26 / 2,304 | Last non-padding token from block output \(l+1\); embedding excluded |
| Gemma new-graph confirmation | Gemma-3-12B-it QAT Q4_0 GGUF; full model digest in Source Data run metadata | 48 / 3,840 | Same last non-padding token from each block output as above; embedding excluded. Pinned native capture source with GGUF QAT Q4_0 loading |
