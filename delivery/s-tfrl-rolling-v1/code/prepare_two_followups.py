"""Compatibility entry point; tasks are defined in the shared CPU plan."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
subprocess.run([sys.executable,'-B',str(OUT/'code/prepare_plan.py'),'--plan',str(OUT/'config/successor-cpu-plan.json')],check=True)
