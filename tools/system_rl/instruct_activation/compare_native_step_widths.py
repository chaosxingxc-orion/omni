"""Compare pure decode at each observed width using the two engine result timers."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from io_utils import doc, pin, require, save
from report_batch_stage import report


def window(config, lane):
    settings = doc(pin(config))
    root = Path(settings['output'])
    require(doc(pin(root/'DRAINED.json'))['guards_releasable'], 'window still active')
    result = doc(pin(root/'RESULT.json'))
    ref, = result['rounds']
    require(result['passed'], 'window failed')
    run = doc(ref)
    selected = sorted((r for r in run['requests'] if r['service'] == lane),
                      key=lambda r:(r['group'],r['replicate']))
    outputs = [json.loads(doc(r['receipt_ref'])['raw']) for r in selected]
    stats = run['metrics'][lane]
    require(run['admission'] == 'group_drained' and stats['mm_hits'] == 0
            and stats['mm_queries'] == len(selected), 'unmatched cache or admission')
    key = ([r['sha256'] for r in run['input_refs']], [r['prompt_token_ids'] for r in outputs],
           run['concurrency_per_service'][lane], stats['prefix_hits'], stats['prefix_queries'])
    return ref['sha256'], key, int(stats['itl_intervals'])


def compare(spec):
    native = doc(spec['native_events'])
    native_id, native_inputs, count = window(spec['native_config'], spec['native_lane'])
    custom_id, custom_inputs, _ = window(spec['custom_config'], spec['custom_lane'])
    require(native['round_sha256'] == native_id and count == native['intervals']
            and native_inputs == custom_inputs, 'unpaired measurement')
    ledger = report(Path(spec['custom_config']), Path(spec['custom_state']), spec['custom_lane'])
    require(ledger['round_sha256'] == custom_id, 'custom trace differs')
    samples = {}
    for label, data in [('native', native['rows']), ('custom', ledger['steps'])]:
        values = defaultdict(list)
        for row in data:
            if label == 'native' and row['kind'] == 'decode':
                values[row['generation_tokens']].append(row['elapsed_ms'])
            elif label == 'custom' and len(row['decode']) == len(row['alloc']):
                require(all(v == 1 for v in row['alloc'].values()), 'not one-token decode')
                # Native timer covers future.result() and conditional sample_tokens().
                values[len(row['decode'])].append(row['ms']['wait'] + row['ms'].get('sample', 0))
        require(values, 'missing pure decode')
        samples[label] = values
    widths = sorted(set(samples['native']) | set(samples['custom']))
    rows = {}
    for width in widths:
        row = {}
        for label, values in samples.items():
            times = values.get(width, [])
            row[label] = dict(steps=len(times), median_ms=median(times) if times else None)
        if all(x['steps'] for x in row.values()):
            row['difference_ms'] = row['custom']['median_ms'] - row['native']['median_ms']
        rows[str(width)] = row
    return dict(native_round=native_id, custom_round=custom_id, widths=rows,
                scope='Separate cold natural windows. All pure decode steps at each width. '
                      'Native engine iteration timer vs. custom wait plus sample host intervals. '
                      'Includes stalls and instrumentation. No device-kernel attribution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    save(args.output, compare(doc(pin(args.config))))
