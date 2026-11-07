"""Score isolated P records; gold is read only by this offline process."""
import argparse
import json
from pathlib import Path

from speech_aware_evidence_acquisition.analysis.perception_node_score import score, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('roster', 'outputs', 'carrier', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--maximum', type=int, default=3)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    result = score(read(args.roster), read(args.outputs), maximum=args.maximum,
                   legal_ids=set(read(args.carrier)))
    if args.control:
        result['paired'] = paired(read(args.control), result)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
