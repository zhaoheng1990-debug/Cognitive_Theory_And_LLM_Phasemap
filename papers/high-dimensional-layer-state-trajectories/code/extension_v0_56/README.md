# v0.56 closure audits

This extension closes three prespecified reviewer-facing questions without reopening functional-window selection, intervention-policy search or prompt-outcome selection.

## What is included

| Audit | Question | Primary result | Boundary |
|---|---|---|---|
| Predecision coordinate competition | Does the one-dimensional `DeltaU_pre` coordinate add information beyond a cheap, matched recent output-logit margin? | No. `DeltaU_pre` was lower in held-out macro F1 in Qwen, Llama and Gemma. Complete predecision history and the cutoff hidden state improved on the recent readback. | This does not isolate an order-only effect because an unordered-prefix comparator was not tested. `DeltaU` remains a compact task construction, not a necessary or superior coordinate. |
| Empirical-subspace random direction | Does the structured actuator exceed directions sampled from the empirical update subspace at identical layers, norms, gates and action schedules? | The nominal Gemma result exceeded its 95th percentile; Qwen and Llama did not. No checkpoint passed BH correction across the three-model family. | The relation-graph gate/executor actuator still crosses declared boundaries, but this test does not establish checkpoint-general direction specificity. |
| New-graph geometry confirmation | Does endpoint-preserving order geometry recur on new graph groups in Qwen-1.5B and a larger independent Gemma checkpoint? | Yes. Both checkpoints exceeded all 50 shared endpoint-preserving shuffles in chord alignment on 240 fixed new prompts. | This is checkpoint-specific geometry evidence, not a pure scale law, quantization-invariant statement or functional-control result. |

`PROTOCOL.md` fixes the inputs, splits, metrics and decision rules before execution. The multiplicity register separates these heterogeneous evidence families rather than applying one invalid study-wide correction.

## Fast verification

Run each verifier from this directory after setting the repository root as the working directory:

```bash
python verify_predecision_coordinate_audit.py ../../source_data/source_data/extension_v0_56/coordinate_audit
python verify_empirical_direction_null.py ../../source_data/source_data/extension_v0_56/direction_null
python verify_newgraph_geometry_confirmation.py --output-root ../../source_data/source_data/extension_v0_56/newgraph_geometry
```

The verifiers are read-only unless `--report` is supplied. The coordinate and direction scripts recompute released summaries from retained processed rows. The geometry script recomputes all metrics when raw state intermediates are available; in the public mirror it verifies released prompt metrics, shared-null summaries, dimensions and integrity metadata because the large raw arrays are excluded.

## Full reruns

`run_predecision_coordinate_audit.py` and `run_empirical_direction_null.py` reuse the released controlled-task inputs. Set `QWEN_MODEL_PATH`, `LLAMA_MODEL_PATH` and `GEMMA_MODEL_PATH` explicitly before a full rerun; no workstation-specific path is embedded in the runners. `run_newgraph_geometry_confirmation.py` requires explicit checkpoint and runtime arguments and includes the frozen prompt construction and native Transformer extraction for Qwen.

The larger-checkpoint capture route is implemented in `native_capture/gemma3_layer_capture.cpp`. It requires the pinned `llama.cpp` source revision `1a064ab0921238c1daa397d6f4a900ef33884de2`, a compatible Vulkan runtime and the licensed Gemma-3-12B-it-QAT-Q4_0 GGUF checkpoint. Model weights, runtime binaries and large raw state arrays are intentionally not redistributed. `newgraph_geometry/run_metadata.json` records model identifiers, the checkpoint hash and extraction qualification. The released capture manifest uses relative prompt paths as an archive record; the runner writes its executable local-path manifest at runtime.

## Integrity and scope

Each audit has a runner, independent verifier, frozen protocol, processed Source Data and an entry in `claim_family_verification_matrix_v0_56.md`. The human numerical-review checklist is deliberately unsigned; it must not be cited as a completed author review until H.Z. has personally performed and signed it.

Internal experiment identifiers, workstation paths and failed engineering trials are omitted from this public package. The full evidence boundaries are in the manuscript, Supplementary Methods and `multiplicity_claim_family_register_v0_56.md`.
