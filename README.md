# Transformer-research-Dynamics-of-Interpretation-Basin-Selection
Transformer hidden states are generally treated as the primary internal representation of large language models. The standard assumption is that hidden geometry is encoded in high-dimensional continuous vectors, while vocabulary projections are merely output interfaces.
In this work, we test an extreme compression hypothesis.

Instead of using hidden-state coordinates, logits, probabilities, or token ranks, we retain only the identity of tokens appearing in the Top-K vocabulary projection of each hidden state.

Surprisingly, we find that a large fraction of hidden-state topology remains recoverable from Top-K token identity alone.

Across layers and multiple neighborhood sizes, discrete vocabulary identity preserves substantial geometric structure, suggesting that hidden states may admit an alternative description as vocabulary-induced semantic neighborhoods.

We do not claim that hidden states can be replaced by Top-K identity. Rather, we report a robust empirical phenomenon: hidden geometry appears far more recoverable from discrete vocabulary structure than conventional intuition would suggest.

This observation raises new questions about the nature of internal state representations in large language models.
