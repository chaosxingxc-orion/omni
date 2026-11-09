"""Load the original binding sidecar with explicit N preflight, no lease acquire."""
import runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
import run_n
run_n.original.preflight=run_n.preflight
runpy.run_path(str(ROOT/'delivery/s-single-intent-fewshot-v1/code/bind_execution.py'),run_name='__main__')
