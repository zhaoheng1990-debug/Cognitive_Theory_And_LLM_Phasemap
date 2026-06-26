# HPM-RT1 Portable Install Guide

Version: v1.1

Positioning: black-box LLM external runtime warning and audit layer. The tool only reads visible input, visible output, evidence items, and page/API metadata; it does not read model-internal reasoning, logits, hidden states, attention, or provider-side retrieval traces.

## 1. Environment

Python 3.10+ is recommended. On Windows migration machines, Python should be available through `py -3` or `python`.

## 2. Recommended Windows three-exe flow

After extracting the portable package, run these files from the project root:

```powershell
.\HPM_RT1_Install_Dependencies.exe
.\HPM_RT1_Deploy_Modules.exe
.\HPM_RT1_Launcher.exe
```

The three executables have separate responsibilities:

- `HPM_RT1_Install_Dependencies.exe`: creates local `.venv`, installs dependencies from `wheelhouse/` when available, then installs the HPM-RT1 package.
- `HPM_RT1_Deploy_Modules.exe`: validates core modules, API, and frontend files, then creates output folders, runtime config, and a desktop shortcut.
- `HPM_RT1_Launcher.exe`: starts the FastAPI gateway and Streamlit dashboard.

Open:

- API: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:8501`
- Gateway PDF report: `Desktop\HPM_RT1_Gateway_Audit_Reports\hpm_rt1_gateway_audit_report.pdf`

## 3. Manual Install

From the project root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

For offline environments, install from the bundled `wheelhouse/`:

```bash
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
python -m pip install -e .
```

Or install the full development extras:

```bash
python -m pip install -e ".[dev]"
```

## 4. Start API

```bash
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## 5. Start Frontend

```bash
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

## 6. Verify

```bash
python -m pytest
python -m hpm_rt1_beta.cli --input examples/hpm_rt1_external_testset_min32.jsonl --output-dir outputs/cli_external_min32_output
```

## 7. Boundary

HPM-RT1 is an advisory-only risk governance module:

```text
no automatic answer rewriting
no automatic truth correction
no model weight update
no active HPM-12 repair
```

Every audit output includes `audit_confidence_scores`; all scores are in the 0-1 range.

Realtime gateway audit also generates a desktop PDF report:

```text
Desktop\HPM_RT1_Gateway_Audit_Reports\hpm_rt1_gateway_audit_report.pdf
```
