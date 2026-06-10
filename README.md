# Cognitive Theory and LLM PhaseMap

This repository documents an independent research program on **Cognitive Theory** and **LLM PhaseMap**, aiming to study cognition, memory, reasoning, hallucination, and controllable trajectory dynamics in large language models.

The central hypothesis is that cognition can be modeled as **effective compression of a possibility space under constraint fields**. In this framework, an LLM is not treated merely as a text generator, but as a dynamical cognitive system whose internal trajectories evolve over a learned semantic / relational manifold.

## Core Research Directions

This project currently focuses on four connected research layers:

1. **Cognitive Theory**
   A general theoretical framework in which cognition is formulated as:

   [
   \mathcal P_c=(\Omega_c,\mathcal F_c)
   ]

   where (\Omega_c) denotes the possibility space and (\mathcal F_c) denotes the constraint field. Cognitive progress is measured by effective structural compression, or **Cbit**.

2. **LLM PhaseMap**
   A mechanistic framework for analyzing how language models move from prompt-induced candidate trajectories toward dominant closure trajectories. Key objects include trajectory advantage (\Delta U), continuous advantage flow (O_l^{cont}), competition bands, and structured residual dynamics.

3. **Semantic / Cognitive Manifold Hypothesis**
   The model’s vocabulary projection space (W_m) is treated as a learned semantic / relational manifold:

   [
   W_m=\mathcal M_m
   ]

   Prompt inputs are interpreted as seed carriers that induce initial semantic positions, direction spectra, and candidate trajectory families.

4. **Memory, Operators, and Control**
   Long-term memory is modeled not as stored prompt text, but as internal trajectory geometry and reusable operator structures. The project also explores trajectory-level closed-loop control, operator memory, structural resolution, and wrong-closure auditing.

## Important Methodological Position

This repository distinguishes carefully between:

[
\text{ontology} \neq \text{observable} \neq \text{proxy}
]

For example, TopK token sets are no longer treated as semantic neighborhoods themselves. They are instead treated as **PSG-corrected vocabulary readback observables**: useful projection-based readouts of local semantic geometry, but not direct proof of hidden ontology.

This distinction is central to the project’s methodology: useful observables must be calibrated, audited, and separated from the theoretical objects they approximate.

## Current Status

The project is an evolving research archive. It includes theoretical notes, experiment designs, audit protocols, empirical summaries, and code related to:

* PhaseMap dynamics
* Cbit and cognitive compression
* PSG-corrected readback observables
* Semantic manifold and near-geodesic trajectory competition
* SEM long-term memory experiments
* ASA trajectory-level steering and control
* Structural resolution and audited reuse
* Cognitive theory applications beyond LLMs

The goal is not to present a finished theory, but to build a reproducible and self-correcting research path toward a mathematical theory of cognitive systems and mechanistic LLM dynamics.

## Research Principle

A guiding rule of this project is:

[
\Delta Cbit_{eff}^{new} > Cost_{complexity}^{new}
]

New concepts, variables, operators, or experiments are only valuable if they produce positive effective compression of the research possibility space. When additional complexity no longer yields Cbit gain, the framework must be audited, simplified, or rolled back.

In this sense, the project treats theory-building itself as a cognitive process.

