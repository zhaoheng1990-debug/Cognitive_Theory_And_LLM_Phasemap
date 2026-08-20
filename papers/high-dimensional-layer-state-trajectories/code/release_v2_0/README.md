# v2.0.0 processed-data verification

Run the repository-level verifier from the repository root:

```text
python code/release_v2_0/run_v2_0_reproduction.py
```

It recomputes eight endpoint families from the processed Source Data and writes
reports to `verification/v2_0/`. It does not load model weights or rerun model
inference. Install the dependencies listed in the repository-root
`requirements.txt` before execution.
