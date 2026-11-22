"""Bind newly admitted TRAIN recordings to exact v2 ranks and independent card evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

from dataset_encoder import save, sha, put
from p_screen_inputs import Inputs, checked, doc, lines


def load_bank(config, repair=None):
    module = Path(config['source_root'])/'full-train-retrieval-producer/representative-top5'
    sys.path.insert(0, str(module))
    from consume_bank import RepresentativeSupply
    return RepresentativeSupply(Path(config['recall_delta']['bank']), repair)


def prepare(config):
    delta = config['recall_delta']
    mode = delta['mode']
    if mode not in ('new_gate_admissions', 'previously_screened_v2_admissions', 'old_pool_v2_admissions'):
        raise ValueError('unknown recall delta scope')
    queue = Path(delta['queue'])
    trace = Path(delta['trace'])
    bank = load_bank(config)
    profile = bank.manifest['profile']
    receipt = doc(Path(delta['bank'])/'READBACK.json')
    assert receipt['manifest_sha256'] == sha(Path(delta['bank'])/'MANIFEST.json')
    assert receipt['results_sha256'] == sha(trace/'eight-card-results.json')
    assert sha(trace/'banks.json') == bank.manifest['files']['banks.json']
    assert sha(trace/'keys.npy') == bank.manifest['files']['keys.npy']
    original = Path(config['recall'])
    old = doc(original/'RECALL-COMPLETE.json')
    prior = {r['reference_id'] for r in lines(checked(old['p_gate_roster']))}
    assert len(prior) == old['p_gate_hits']
    prior_calls = None
    if mode == 'previously_screened_v2_admissions':
        receipt_path = checked(delta['prior_terminal'])
        terminal = doc(receipt_path)
        if terminal['outcome'] != 'candidate_pool_exhausted':
            raise ValueError('previous screening did not reach its terminal population boundary')
        db_path = receipt_path.parent/'p-screen.sqlite'
        with sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True) as db:
            calls = db.execute('select rid,status,finished from calls order by rid').fetchall()
        if (Counter(status for _,status,_ in calls) != terminal['status']
                or any(status not in ('valid','invalid') or not finished or rid not in prior
                       for rid,status,finished in calls)):
            raise ValueError('previous screening ledger differs from terminal receipt')
        prior_calls = {rid for rid,_,_ in calls}

    source = doc(checked(doc(config['program'])['inputs']['raw_dataset']))
    recordings = {}
    for domain in source['domains']:
        name = domain['domain']
        for line, row in enumerate(lines(checked(domain['metadata'])), 1):
            gold = row['scenario']+'_'+row['action']
            for i, rec in enumerate(row['recordings']):
                recordings[f'{name}:{line:06d}:{i:04d}'] = (str(row['slurp_id']), rec['file'], gold)
    rows = {s:list(lines(original/s/'rows.jsonl')) for s in doc(original/'profile.json')['hosts']}
    meta = {s:doc(trace/(s+'.json')) for s in rows}
    seen = set()
    totals = Counter()
    selected = []
    for lane in range(bank.bank['config']['workers']):
        for shard, i, labels, witnesses in lines(trace/f'ranks-{lane}.jsonl'):
            r = rows[shard][i]
            rid = r['reference_id']
            source_id, filename, gold = recordings[rid]
            expected, groups, component, *_ = meta[shard][i]
            assert (expected, groups, component) == (gold, r['excluded'], r['component'])
            assert rid not in seen and len(labels) == len(witnesses) == len(set(labels))
            seen.add(rid)
            totals['hits'] += gold in labels
            eligible = rid not in prior if mode == 'new_gate_admissions' else rid in (
                prior if prior_calls is None else prior_calls)
            if gold not in labels or not eligible:
                continue
            selected.append(dict(reference_id=rid, source_id=source_id, recording=filename,
                                 audio_sha256=r['audio_sha256'], rank=labels.index(gold)+1,
                                 gold=gold, labels=labels, witnesses=witnesses,
                                 component=component, excluded=groups))
    result = doc(trace/'eight-card-results.json')['counts'][profile+':all']
    assert (len(seen), totals['hits']) == (result['rows'], result['hits'])
    assert selected
    selected.sort(key=lambda row:row['reference_id'])
    put(queue, b''.join((json.dumps(r,sort_keys=True)+'\n').encode() for r in selected))
    scope = {} if mode == 'new_gate_admissions' else dict(mode=mode)
    if prior_calls is not None:
        scope.update(prior_terminal=delta['prior_terminal'], prior_screened=len(prior_calls))
    save(queue.parent/'DELTA.json', dict(
        queue_sha256=sha(queue), knowledge_manifest_sha256=receipt['manifest_sha256'],
        recall_results_sha256=receipt['results_sha256'], prior_gate_sha256=old['p_gate_roster']['sha256'],
        prior_gate=old['p_gate_hits'], rows=len(selected),
        by_domain=Counter(r['reference_id'].split(':',1)[0] for r in selected),
        by_intent=Counter(r['gold'] for r in selected), **scope))
    print(f'Frozen {len(selected)} recordings for {mode}', flush=True)


class DeltaInputs(Inputs):
    def __init__(self, config):
        self.load_renderer(config)
        self.bank = load_bank(config, self.repair)
        queue = Path(config['recall_delta']['queue'])
        self.receipt = doc(queue.parent/'DELTA.json')
        assert sha(queue) == self.receipt['queue_sha256']
        assert sha(Path(config['recall_delta']['bank'])/'MANIFEST.json') == self.receipt['knowledge_manifest_sha256']
        self.gate = {r['reference_id']:r for r in lines(queue)}
        assert len(self.gate) == self.receipt['rows']
        self.gold = {rid:r['gold'] for rid,r in self.gate.items()}
        source = doc(checked(doc(config['program'])['inputs']['raw_dataset']))
        self.audio_roots = {d['domain']:Path(d['audio_root']) for d in source['domains']}
        self.permanent = tuple(self.repair.cfg['policy']['permanent_labels']) if 'policy' in self.repair.cfg and 'permanent_labels' in self.repair.cfg['policy'] else tuple(
            doc(checked(doc(config['business'])['encoder']['p_materializer_config']))['policy']['permanent_labels']
        )

    def supply(self, rid):
        r = self.gate[rid]
        query = dict(identity=dict(component_id=r['component']), excluded_source_groups=r['excluded'])
        supply = self.bank.supply(r['labels'], r['witnesses'], query, self.permanent)
        if any(c['provenance'].get('gaps') for c in supply['cards'].values()):
            raise ValueError('P_FIELD_SUPPORT_UNAVAILABLE')
        return supply

    def packet(self, rid):
        return self.render_audio(rid, self.supply(rid))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config',type=Path)
    p.add_argument('--check-supply',action='store_true')
    a = p.parse_args()
    config = doc(a.config)
    prepare(config)
    if a.check_supply:
        x = DeltaInputs(config)
        failures = Counter()
        for rid in x.gate:
            try:
                x.supply(rid)
            except (AssertionError, ValueError) as e:
                failures[str(e)] += 1
        if failures:
            print(failures)
            raise SystemExit(1)
        print(f'Source checks passed for {len(x.gate)} packets', flush=True)
