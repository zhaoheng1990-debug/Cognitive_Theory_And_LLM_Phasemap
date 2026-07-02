# Productization Dev-RC Run Command Patch Spec 6901A-7000Z

## Objective

Close the last known productization paper-cut before dev-RC freeze:

\[
\boxed{
Standalone\ portable\ dev\text{-}RC\ source\ tree
+
external\ project\ preflight
+
correct\ install/run\ surface
}
\]

This is not a new AgentOS cognitive capability. It is a productization patch.

## Required patch

The standalone source tree README / install guide currently suggests:

```bash
python examples/run_portable_smoke.py
```

In the PM audit environment this failed with:

```text
ModuleNotFoundError: No module named 'agentos_core_slim_portable'
```

because the project root was not on `PYTHONPATH`.

Patch the install/run surface so the default documented command is one of:

```bash
PYTHONPATH=. python examples/run_portable_smoke.py
```

or a robust module entrypoint, for example:

```bash
python -m agentos_core_slim_portable.smoke
```

If adding a module entrypoint, it must be minimal and must not introduce new runtime capability.

## Required validation

1. Build a fresh standalone source tree.
2. Build a fresh source bundle zip.
3. Extract the bundle into a clean temp directory.
4. Run:
   - `python -m pytest tests -q`
   - documented smoke command exactly as written in README / install guide.
5. Confirm smoke output includes:
   - `kernel_owner = AgentOSKernel.ICMMetabolismPolicy`
   - `advisory_only = True`
   - `harness_no_external_effect = True`
   - `replay_verified = True`
   - `production_release = False`
6. Produce final dev-RC freeze packet with bounded claims.

## Non-goals

Do not add:
- new Harness type;
- new AgentProfile;
- new ICM policy;
- production write;
- external API;
- network call;
- new cognitive module;
- new RC claim beyond dev-RC.
