# Code entry points

`verify_v1r4_source_data.py` is the primary portable entry point. It uses only
the Python standard library and checks 20 reported endpoints against the
processed V1R4 Source Data.

`run_v1r4_reproduction.ps1` locates the package layout and runs that verifier.
Use `-Mode full-model` only as a reminder to follow the documented full-model
runners in the public repository; model weights and raw hidden-state arrays are
not part of this archive.

For the v2.0.0 eight-family processed-data verification, use
`../release_v2_0/run_v2_0_reproduction.py` from the repository root.
