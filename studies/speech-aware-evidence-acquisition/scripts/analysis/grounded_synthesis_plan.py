"""Compatibility entry for the shared grounded synthesis plan module."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from speech_aware_evidence_acquisition.rails import grounded_synthesis_plan as _implementation
# Preserve existing analysis imports while the shared flight uses the package.
globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith('__')})
if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
