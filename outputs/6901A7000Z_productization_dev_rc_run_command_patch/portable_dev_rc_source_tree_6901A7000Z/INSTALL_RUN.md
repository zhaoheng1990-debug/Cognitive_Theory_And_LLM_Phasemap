# Install / Run Standalone Portable Dev-RC 6901A-7000Z

From a fresh extraction:

```powershell
cd portable_dev_rc_source_tree_6901A7000Z
python -m pytest tests -q --basetemp .pytest_tmp
python -m agentos_core_slim_portable.smoke
```

The explicit `--basetemp .pytest_tmp` keeps pytest scratch files inside the extracted package and avoids reliance on user-level temp directories.

The smoke command is a module entrypoint so it runs from a clean extracted source tree without requiring external `PYTHONPATH` setup.

No external API, network, browser, connector, production memory, user memory, or theory baseline mutation is required.
