# Pythia training-coordinate component-swap protocol v0.1

## Question

The random-initialization results separate an architecture-compatible trajectory
scaffold from learned task organization. This experiment asks where the learned
organization resides: in semantic coordinates, in the layer computation, or in
their trained compatibility.

## Frozen design

- Model: `EleutherAI/pythia-160m` from one documented training run.
- Checkpoints: `step0` and `step143000`.
- Task: the previously frozen 53-item, single-token, lexical base-cloze subset.
- Components: input embedding (`E`), all transformer blocks (`B`), final layer
  normalization (`N`) and output head (`W`). Pythia does not tie `E` and `W`.
- Intervention: evaluate all 16 binary combinations of step-0 and trained
  components. No component boundary or task item is selected after seeing the
  outcomes.
- Primary outcome: clean-versus-conflict candidate margin and pair accuracy.
- Uncertainty: paired prompt bootstrap. Factorial effects are computed with the
  complete orthogonal 2^4 design.

## Falsification-first gates

1. **Anchor reproduction.** The all-step-0 and all-trained combinations must
   reproduce the previously extracted accuracies to within one prompt and their
   mean margins to within 0.02.
2. **Training-function gate.** The all-trained condition must exceed the
   all-step-0 condition by at least 0.20 accuracy and reach at least 0.75.
3. **No-single-component gate.** No single trained component on the step-0
   background may lie within 0.10 accuracy of the all-trained condition.
4. **Blocks-necessity gate.** The best condition with step-0 blocks must remain
   at least 0.10 below the all-trained condition.
5. **Output-coordinate-necessity gate.** The best condition with the step-0
   output head must remain at least 0.10 below the all-trained condition.
6. **Superadditive-compatibility gate.** The all-trained gain over step 0 must
   exceed the sum of the four isolated single-component gains, with a positive
   paired-bootstrap 95% confidence interval.

Gates 1-2 establish only training-associated function. Gates 3-5 support a
distributed compatibility interpretation. Gate 6 is deliberately stronger and
may fail even when several components are necessary.

## Decision tree

- If only gates 1-2 pass, retain the conservative conclusion that training
  changes task function but do not localize it by component exchange.
- If gates 3-5 also pass, treat trained blocks and output coordinates as jointly
  necessary in this diagnostic.
- If gate 6 passes, support superadditive trained compatibility; otherwise retain
  necessity without claiming superadditivity.
- If an isolated component approaches the fully trained result, localize the
  follow-up to that component and reject a broadly distributed account.

## Claim boundary

This is a same-architecture, same-vocabulary, one-run component-compatibility
test on a bounded lexical cloze diagnostic. It does not identify a universal
semantic manifold, a training law, causal mediation through individual weights,
or general language understanding. Component swaps can also create
out-of-distribution hybrids; their value here is perturbational, not as natural
models.
