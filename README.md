# Cognitive Theory and LLM PhaseMap

This repository documents an independent research program on **Cognitive Theory** and **LLM PhaseMap**, aiming to study cognition, memory, reasoning, hallucination, and controllable trajectory dynamics in large language models.

The central hypothesis is that cognition can be modeled as **effective compression of a possibility space under constraint fields**.

In this framework, an LLM is not treated merely as a text generator, but as a dynamical cognitive system whose internal trajectories evolve over a learned semantic / relational manifold.

---

## Core Research Directions

This project currently focuses on four connected research layers.

### 1. Cognitive Theory

Cognition is modeled as a coupled structure:

`Pc = (Omega_c, F_c)`

where:

* `Omega_c` is the possibility space.
* `F_c` is the constraint field.
* Cognitive progress is measured by effective structural compression.
* The basic unit of cognitive compression is called `Cbit`.

In short:

`Cognition = effective compression of possibility space under constraint fields`

---

### 2. LLM PhaseMap

LLM PhaseMap studies how language models move from prompt-induced candidate trajectories toward dominant closure trajectories.

Key objects include:

* `Delta U`: trajectory advantage observable
* `O_l_cont`: continuous trajectory advantage flow
* `B_m`: model-specific competition band
* `gamma_i`: candidate closure trajectory
* `gamma_star`: dominant trajectory
* structured residual dynamics

A simplified process is:

`Prompt -> candidate trajectories -> trajectory advantage -> competition band -> dominant trajectory -> answer`

---

### 3. Semantic / Cognitive Manifold Hypothesis

The model vocabulary projection space is treated as a learned semantic / relational manifold:

`W_m = M_m`

where:

* `W_m` is the model-specific vocabulary projection geometry.
* `M_m` is the semantic / relational manifold induced by training.
* Prompt inputs act as seed carriers.
* A prompt induces an initial semantic position and direction spectrum.

A simplified chain is:

`Prompt -> Seed -> (x0, Sigma0) -> local readback geometry -> candidate trajectories`

---

### 4. Memory, Operators, and Control

Long-term memory is not treated as stored prompt text.

Instead, memory is modeled as internal trajectory geometry and reusable operator structure.

Important objects include:

* `MemoryUnit`
* `SeedAnchor`
* `ExpansionMap`
* `OperatorRoutingMap`
* `OperatorMemory`
* `PolicyMemory`
* `StructuralResolutionOperator`

The project also studies trajectory-level control, wrong-closure auditing, and closed-loop steering of internal trajectory advantage.

---

## Important Methodological Position

This repository strictly distinguishes:

`ontology != observable != proxy`

For example, TopK token sets are no longer treated as semantic neighborhoods themselves.

Instead, they are treated as **PSG-corrected vocabulary readback observables**.

That means:

* TopK is useful.
* TopK can preserve hidden-state geometry.
* TopK can support downstream dynamical analysis.
* But TopK is not the semantic neighborhood itself.
* TopK is not direct proof of hidden semantic ontology.

A safer formulation is:

`TopK = PSG-corrected vocabulary readback observable`

or:

`TopK = vocabulary-projected readback section of local semantic geometry`

This distinction is central to the project.

Useful observables must be calibrated, audited, and separated from the theoretical objects they approximate.

---

## Current Status

This repository is an evolving research archive.

It includes theoretical notes, experiment designs, audit protocols, empirical summaries, and code related to:

* Cognitive Theory
* LLM PhaseMap
* Cbit and cognitive compression
* PSG-corrected readback observables
* Semantic manifold and near-geodesic trajectory competition
* SEM long-term memory experiments
* ASA trajectory-level steering and control
* Structural resolution and audited reuse
* Wrong-closure and hallucination auditing
* Cognitive theory applications beyond LLMs

The goal is not to present a finished theory.

The goal is to build a reproducible and self-correcting research path toward a mathematical theory of cognitive systems and mechanistic LLM dynamics.

---

## Research Principle

A guiding rule of this project is:

`Delta Cbit_eff_new > Cost_complexity_new`

New concepts, variables, operators, or experiments are only valuable if they produce positive effective compression of the research possibility space.

When additional complexity no longer yields Cbit gain, the framework must be audited, simplified, or rolled back.

In this sense, the project treats theory-building itself as a cognitive process.

---

## Repository Philosophy

This project follows four methodological principles:

1. **Object First**
   Define the research object before defining metrics.

2. **Observable Calibration**
   Separate the theoretical object from the observable and from the proxy.

3. **Cbit-Gain Audit**
   New theoretical additions must reduce the effective possibility space more than they increase complexity.

4. **Theory Retraction as Progress**
   A theory is allowed, and expected, to downgrade its own claims when better audits reveal that an observable was over-interpreted.

---

## Disclaimer

This repository represents an ongoing independent research program.

The terminology, theory, and experiments are actively evolving.

Some claims are established within the project evidence chain, while others remain hypotheses, proposed research directions, or pending validations.
