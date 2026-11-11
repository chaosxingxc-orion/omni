"""Collect atomically persisted per-sample S1 aggregates and score the frozen set.

This is a one-shot CPU consumer, never a dispatcher or polling daemon. Supply
archived runtime row directories after a terminal event or existing 600s tick.
Same-byte duplicate replicas are reused; conflicting samples stop collection.
"""
import argparse
import json
import hashlib
from pathlib import Path
if __package__:
    from .score import lines, save, score
else:
    from score import lines, save, score


def collect(paths):
    seen = {}
    origins = {}
    duplicates = 0
    for directory in paths:
        path = Path(directory)
        files = sorted(path.glob('*.json')) if path.is_dir() else [path]
        for file in files:
            raw = file.read_bytes()
            # Runtime writes one JSON object per row; .jsonl is explicit aggregate input.
            objects = [json.loads(line) for line in raw.splitlines() if line.strip()] if file.suffix == '.jsonl' else [json.loads(raw)]
            for obj in objects:
                if not isinstance(obj, dict) or 'sample_id' not in obj:
                    raise ValueError('not a persisted sample aggregate: ' + str(file))
                sid = obj['sample_id']
                digest = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                if sid in seen:
                    if seen[sid][0] != digest:
                        raise ValueError('conflicting sample result, review custody: ' + sid)
                    duplicates += 1
                    continue
                seen[sid] = (digest, obj)
                origins[sid] = {'path': str(file), 'sha256': hashlib.sha256(raw).hexdigest()}
    return [seen[s][1] for s in sorted(seen)], {'identical_replicas_reused': duplicates, 'origins': origins}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rows', action='append', default=[])
    p.add_argument('--gold', required=True)
    p.add_argument('--folds', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--runtime-cost')
    args = p.parse_args()
    predictions, custody = collect(args.rows)
    runtime = json.loads(Path(args.runtime_cost).read_bytes()) if args.runtime_cost else None
    summary, detail = score(lines(args.gold), predictions, json.loads(Path(args.folds).read_bytes()), runtime)
    out = Path(args.out)
    summary['input_pins'] = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [args.gold, args.folds]}
    summary['collection'] = {'samples': len(predictions), 'identical_replicas_reused': custody['identical_replicas_reused']}
    save(out / 'predictions.private.jsonl', predictions, True)
    save(out / 'CUSTODY.private.json', custody)
    save(out / 'SUMMARY.json', summary)
    save(out / 'scores.private.jsonl', detail, True)
    print(json.dumps({k: summary[k] for k in ['status', 'denominator', 'counts', 'metrics', 'fold_status_counts']}, indent=2))


if __name__ == '__main__':
    main()
