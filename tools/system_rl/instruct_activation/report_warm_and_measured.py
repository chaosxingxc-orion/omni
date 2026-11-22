"""Count warmup wall time as well as measured wall time on the same service."""
import argparse
from pathlib import Path

from io_utils import doc, pin, require, save


def read_phase(config, lane):
    cfg = doc(pin(config))
    root = Path(cfg['output'])
    closed = doc(pin(root / 'RESULT.json'))
    require(closed['passed'] and doc(pin(root / 'DRAINED.json'))['guards_releasable'],
            'phase is still open')
    ref, = closed['rounds']
    data = doc(ref)
    width = data['concurrency_per_service'][lane]
    requests = [row for row in data['requests'] if row['service'] == lane]
    require(data['admission'] == 'group_drained' and width > 0, 'unexpected admission')
    batches = {}
    for row in requests:
        batches.setdefault(row['group'], []).append(row)
    elapsed = []
    for group in sorted(batches):
        batch = batches[group]
        release, = {row['batch_release_from_round_seconds'] for row in batch}
        require(sorted(row['replicate'] for row in batch) == list(range(width)), 'incomplete batch')
        elapsed.append(max(row['finished_from_round_seconds'] for row in batch) - release)
    require(elapsed and min(elapsed) > 0, 'invalid duration')
    return (cfg['start_receipts'][lane], [wire['sha256'] for wire in data['input_refs']],
            dict(round=ref['sha256'], requests=len(requests), width=width, batch_s=elapsed,
                 total_active_batch_s=sum(elapsed)))


def report(studies, lane):
    result = {}
    panel = None
    for study in studies:
        phases = [read_phase(study.with_name(study.name + suffix) / 'CONFIG.local.json', lane)
                  for suffix in ('-warm', '-latency')]
        (first, warm, w), (second, measured, m) = phases
        require(first == second and w['width'] == m['width'] and not set(warm).intersection(measured)
                and len(set(warm + measured)) == len(warm + measured), 'incompatible phases')
        identity = warm, measured, w['width']
        require(panel is None or panel == identity, 'different inputs or width across studies')
        panel = identity
        result[study.name] = dict(warm=w, measured=m, total_active_batch_s=
                                  w['total_active_batch_s'] + m['total_active_batch_s'])
    return dict(studies=result, scope='Release to last reply for each batch, summed. Excludes service load and gaps between batches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lane')
    parser.add_argument('output', type=Path)
    parser.add_argument('studies', nargs='+', type=Path)
    args = parser.parse_args()
    save(args.output, report(args.studies, args.lane))
