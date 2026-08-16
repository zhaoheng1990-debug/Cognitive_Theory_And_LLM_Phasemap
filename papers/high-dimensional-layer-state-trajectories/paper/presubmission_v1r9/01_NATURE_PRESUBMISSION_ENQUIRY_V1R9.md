# Nature Presubmission Enquiry V1R9

**Target journal:** Nature
**Manuscript type:** Physical Sciences - Presubmission Enquiry
**Title:** Architecture, training and execution organize Transformer computation

## System abstract box

Large language models are built from distributed components, but component-level measurements rarely distinguish what the architecture imposes, what training establishes between components and what executed layer order contributes. We tested these sources separately across Qwen, Llama and Gemma checkpoints. Ordered layer-state records enabled matched perturbations of architecture, trained components and block order. Untrained networks retained an architecture-compatible propagation scaffold; swaps of initialized and trained components isolated training-dependent functional compatibility; and reordering trained blocks disrupted native outputs. A confidence-gated local-transition operator crossed a predeclared candidate boundary in 17/87 Qwen, 13/87 Llama and 9/87 Gemma held-out prompts, with 0/203, 2/203 and 0/203 non-target changes. A paired 240-prompt all-layer confirmation in Qwen2.5-1.5B and Gemma-3-12B reproduced the endpoint-preserving order-null signature in the 12B checkpoint (native alignment 0.0864 versus null mean 0.0240; all 50 null values lower). The evidence supports a separability principle: architectural constraints, learned component relations and realized execution should be tested independently rather than assigned a shared mechanism. This is an experimental design principle, not a claim of common mechanisms across systems (refs. 1-8).

**Paste note:** the Nature submission form treats this as a plain-text field. The reference callout is plain text rather than HTML or Markdown.

## Manuscript comments box

Dear Editors,

V1R9 adds a paired 240-prompt, all-layer confirmation in Qwen2.5-1.5B and Gemma-3-12B. In the larger checkpoint, native chord alignment was 0.0864 versus a null mean of 0.0240 (gain 0.0624; null-standardized effect 3.39), with all 50 endpoint-preserving null values lower. We present this as a checkpoint-specific confirmation of the order-null signature, rather than a pure scale law.

We seek advice on whether our Article, Architecture, training and execution organize Transformer computation, would be suitable for Nature. The paper addresses a general problem in the study of distributed learned systems: component-level observations do not by themselves distinguish architectural constraints, learned relations among components and the execution sequence that composes those relations into a computation.

Our central contribution is an experimental principle of separability. We hold an individual forward pass as an ordered layer-state record and intervene separately on architecture, trained component configurations and realized block order. The experiments distinguish an architecture-compatible propagation scaffold, training-dependent component compatibility and functionally consequential execution order. A local transition operator additionally crosses a declared full-vocabulary candidate boundary in held-out prompts while preserving most non-target outputs. The paper reports the associated failures and limits, including failed additive controls, non-confirmatory mixed-arithmetic transfer and the absence of a corrected cross-checkpoint direction effect.

The broader claim is methodological. The same experimental logic can be used wherever a distributed learning system has measurable states and legitimate interventions: independently test architectural constraints, learned component relations and realized execution before assigning them a common mechanism. It does not assume that different artificial, biological or physical systems share mechanisms. Processed Source Data, extraction metadata and verification code are archived at the Zenodo concept records listed below and mirrored in the paper-specific GitHub branch. This manuscript is not under consideration elsewhere; no version is publicly posted as a preprint at the date of this enquiry. All authors have approved the enquiry and declare no competing interests.

We would value your assessment of the Article's suitability for Nature.

Sincerely,

Heng Zhao
Corresponding author
Independent Researcher
zhaoheng1990@gmail.com
ORCID: 0009-0004-2395-9393

## Reference list for the manuscript-comments box

1. Rogers, A., Kovaleva, O. & Rumshisky, A. A primer in BERTology: what we know about how BERT works. *Transactions of the Association for Computational Linguistics* **8**, 842-866 (2020).
2. Elhage, N. et al. A mathematical framework for transformer circuits. *Transformer Circuits Thread* https://transformer-circuits.pub/2021/framework/index.html (2021).
3. Olsson, C. et al. In-context learning and induction heads. *Transformer Circuits Thread* https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html (2022).
4. Meng, K. et al. Locating and editing factual associations in GPT. *Advances in Neural Information Processing Systems* **35**, 17359-17372 (2022).
5. Belrose, N. et al. Eliciting latent predictions from transformers with the tuned lens. Preprint at https://doi.org/10.48550/arXiv.2303.08112 (2023).
6. Bricken, T. et al. Towards monosemanticity: decomposing language models with dictionary learning. *Transformer Circuits Thread* https://transformer-circuits.pub/2023/monosemantic-features/index.html (2023).
7. Damirchi, H., Meza De la Jara, I., Abbasnejad, E., Shamsi, A., Zhang, Z. & Shi, J. Truth as a trajectory: what internal representations reveal about large language model reasoning. Preprint at https://doi.org/10.48550/arXiv.2603.01326 (2026).
8. Pandey, V., Singh, G. & Mahdid, Y. Trajectory geometry of Transformer representations across layers. Preprint at https://doi.org/10.48550/arXiv.2606.09287 (2026).

## Do not upload for this enquiry

- Do not upload the complete manuscript PDF.
- Do not attach the formal cover letter as a separate file.
- Do not provide a preprint DOI or URL while the deposit remains on hold.
