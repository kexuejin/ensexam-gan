# Environment

The fork currently uses the validated torch/MPS Python environment from the previous clean-doc workspace:

```bash
/Volumes/Tool/source/clean-doc/.venv-torch310-mps-stable/bin/python
```

For local commands, copy `.env.example` to `.env` and source it:

```bash
cp .env.example .env
source .env
$ENSEXAM_PYTHON -m py_compile scripts/run_second_stage_residual_repair.py
```

This keeps command examples stable while the environment is still shared.

Do not commit `.env`. It is intentionally ignored.

