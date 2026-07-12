# Python GPU environment manifest V0.1

## Scope

This manifest records the Python environment used for GPU/model-side analysis scripts that require PyTorch, Transformers and local Hugging Face model loading.

## Environment

| Field | Value |
|---|---|
| Conda environment | `dhrf_4080s` |
| Python | `3.11.15` |
| Platform | `Windows-10-10.0.26200-SP0` |
| GPU | `NVIDIA GeForce RTX 4080 SUPER` |
| CUDA available | `True` |
| CUDA version reported by PyTorch | `12.1` |
| Device count | `1` |

## Core packages

| Package | Version |
|---|---|
| `torch` | `2.5.1+cu121` |
| `transformers` | `5.9.0` |
| `numpy` | `2.4.4` |
| `pandas` | `3.0.3` |
| `scikit-learn` | `1.8.0` |
| `scipy` | `1.17.1` |

## Execution note

Use the GPU environment for scripts that load local transformer checkpoints or compute hidden states:

```powershell
conda run -n dhrf_4080s python <script.py>
```

The default base Python environment is not sufficient for these scripts because it does not include PyTorch or Transformers.
