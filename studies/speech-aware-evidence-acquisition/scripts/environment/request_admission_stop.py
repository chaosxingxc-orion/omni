"""Publish an attempt-bound durable stop request; no signals or model calls."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from speech_aware_evidence_acquisition.core.admission_control import request_stop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--request-id', required=True)
    args = parser.parse_args()
    print(json.dumps(request_stop(args.config, request_id=args.request_id)))
