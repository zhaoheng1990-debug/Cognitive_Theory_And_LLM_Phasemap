# Reproducibility release V1R9

V1R9 adds a paired larger-checkpoint geometry confirmation to the Article
**Architecture, training and execution organize Transformer computation**.

## Added evidence

- A fixed 24-group, 240-prompt relation-graph set was run in Qwen2.5-1.5B and
  Gemma-3-12B-it-QAT-Q4_0.
- All native layers were retained without functional-window selection.
- Each checkpoint was tested against the same 50 endpoint-preserving order
  shuffles.
- Gemma-3-12B had chord alignment 0.0864 against null mean 0.0240 (gain
  0.0624; null-standardized effect 3.39; 50 of 50 null values lower).

## Release boundary

The addition is a checkpoint-specific larger-model confirmation of the
order-null signature. It is not a pure within-family scale law, a
quantization-invariance test or a universal Transformer result.

## Verification

Run `code/release_v1r9/run_v1r9_reproduction.ps1`. It first verifies the
inherited V1R4 endpoints and then recomputes the V1R9 paired confirmation from
processed prompt and null records. No model weights, raw hidden-state arrays or
GPU are required for this route.
