"""CLI compatibility entry for configured external exploration."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from speech_aware_evidence_acquisition.rails.external_exploration import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'collect'])
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    result = {'prepare': prepare, 'run': run, 'collect': collect}[args.action](args.path)
    print(json.dumps({'schema': result['schema'], 'candidate_accepted': False}))
