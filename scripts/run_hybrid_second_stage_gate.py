#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "infer" / "run_hybrid_second_stage_gate.py"), run_name="__main__")
