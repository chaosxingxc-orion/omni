"""Compatibility entry for the shared grounded synthesis execution module."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from speech_aware_evidence_acquisition.rails import grounded_synthesis_execution as _implementation
# Preserve existing analysis imports while the shared flight uses the package.
globals().update({k: v for k, v in vars(_implementation).items() if not k.startswith('__')})
if __name__ == '__main__':
    raise SystemExit("Import this module through the registered runtime.")
else:
    sys.modules[__name__] = _implementation
