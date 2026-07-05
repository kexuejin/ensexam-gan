#!/usr/bin/env python3
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "infer" / "run_second_stage_residual_repair.py"), run_name="__main__")
