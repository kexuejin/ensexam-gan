#!/usr/bin/env python3
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "eval" / "batch_eval_hardcase_checkpoints.py"), run_name="__main__")
