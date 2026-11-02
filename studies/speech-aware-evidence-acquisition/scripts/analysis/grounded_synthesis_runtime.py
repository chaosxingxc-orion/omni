"""Compatibility entry for the registered grounded FlightRuntime bridge."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from speech_aware_evidence_acquisition.rails import grounded_synthesis_runtime as _implementation

globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith('__')})
if __name__ == '__main__':
    raise SystemExit('Use the registered FlightRuntime job to dispatch grounded shards.')
else:
    sys.modules[__name__] = _implementation
