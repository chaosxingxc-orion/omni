"""Score common natural histories against previously closed train and eval forwards."""
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile

from decode_action_replay import choice
from io_utils import digest, doc, pin, require, save


def compare(config):
    proof = doc(config['training_proof'])
    train = doc(config['training_config'])
    require(proof['passed'] and proof['optimizer_updates'] == 0, 'training forward not verified')
    panel_ref = pin(Path(train['hot']).parent / train['updated_http_parity'])
    panel = doc(panel_ref)
    require(panel_ref['sha256'] == proof['hf_panel'] and panel['adapter_sha'] == proof['adapter']
            and panel['checkpoint_sha'] == proof['checkpoint'], 'training reference drift')
    cfg = doc(pin(config['candidate_config']))
    lane = config['lane']
    rounds = Path(cfg['output'])
    result = doc(pin(rounds / 'RESULT.json'))
    require(result['passed'] and doc(pin(rounds / 'DRAINED.json'))['guards_releasable'], 'open window')
    round_ref, = result['rounds']
    run = doc(round_ref)
    rows = sorted((r for r in run['requests'] if r['service'] == lane),
                  key=lambda r: (r['group'], r['replicate']))
    width = panel['width']
    require(run['admission'] == 'group_drained' and cfg['grouped_concurrency'] == [width]
            and run['concurrency_per_service'][lane] == width, 'unexpected batch configuration')
    started = doc(pin(cfg['start_receipts'][lane]))
    container = started['container_id']
    service, = [s for s in cfg['services'] if s['name'] == lane]
    stream = subprocess.check_output(['docker', 'cp', container + ':/runtime/state/policy/ACTIVATION.json', '-'])
    with tarfile.open(fileobj=io.BytesIO(stream)) as archive:
        item, = archive.getmembers()
        active = json.load(archive.extractfile(item))
    require(active['service_identity']['container_name'] == service['container']
            and subprocess.check_output(['docker','inspect','--format','{{.Id}}',service['container']],
                                        text=True).strip() == container
            and active['service_identity']['source_boot'] == cfg['source_boot']
            and active['service_identity']['physical_devices'] == started['physical_devices']
            and active['alias'] == active['policy_version'] == 'checkpoint-sha256:' + proof['adapter']
            and active['weights_ref']['sha256'] == proof['adapter']
            and sorted(w['rank'] for w in active['workers'] if w['sha256'] == proof['adapter']) == list(range(4)),
            'candidate checkpoint identity differs')
    scores = {}
    for rank in train['devices']:
        part = doc(pin(Path(train['hot']) / f'rank-{rank}/RESULT.json'))
        require(part['panel'] == panel_ref['sha256'] and part['rank'] == rank and part['step'] == proof['step']
                and part['checkpoint'] == proof['checkpoint'] and part['adapter'] == proof['adapter']
                and part['optimizer_updates'] == 0, 'unrelated training forward')
        for row in part['rows']:
            i = row['index']
            require(i not in scores and i % len(train['devices']) == rank, 'unexpected training rank')
            scores[i] = row
    require(len(rows) == len(scores) == proof['requests'], 'incomplete population')
    modes = {m: dict(max_action_abs=0., max_complete_sequence_abs=None, over_token=0, over_sequence=0)
             for m in ('train', 'eval')}
    salt = digest([cfg['measurement_id'], run['block']])
    common = complete = actions = 0
    for i, (row, wire_ref, ref) in enumerate(zip(rows, run['input_refs'], panel['rows'], strict=True)):
        wire = doc(wire_ref)
        original = dict(ref['packet']['wire'])
        original.pop('cache_salt')
        intent = doc(pin(Path(row['receipt_ref']['path']).with_name('INTENT.json')))
        response = json.loads(doc(row['receipt_ref'])['raw'])
        require(wire == original and digest(dict(wire, cache_salt=salt)) == intent['wire_sha256']
                and (row['group'], row['replicate']) == divmod(i, width)
                and response['prompt_token_ids'] == ref['prompt'] and response['model'] == wire['model']
                and scores[i]['response'] == ref['response'], 'sample, prompt or response changed')
        ids, content = choice(row['receipt_ref'])
        n = next((j for j, (a,b) in enumerate(zip(ids,ref['tokens'])) if a != b),
                 min(len(ids),len(ref['tokens'])))
        entire = ids == ref['tokens']
        common += n
        actions += len(ids)
        complete += int(entire)
        for mode, field in [('train','train_scores'), ('eval','scores')]:
            vals = scores[i][field]
            require(len(vals) == len(ref['tokens']), 'incomplete training scores')
            ds = [t['logprob'] - v for t,v in zip(content[:n], vals[:n], strict=True)]
            require(all(map(math.isfinite, ds)), 'non-finite probability')
            m = modes[mode]
            m['max_action_abs'] = max(m['max_action_abs'], max(map(abs,ds), default=0.))
            m['over_token'] += sum(abs(d) > proof['limits']['token'] for d in ds)
            if entire:
                err = abs(math.fsum(ds))
                m['max_complete_sequence_abs'] = max(err, m['max_complete_sequence_abs'] or 0.)
                m['over_sequence'] += int(err > proof['limits']['sequence'])
    return dict(candidate_round=round_ref['sha256'], reference_hf_panel=panel_ref['sha256'],
                reference_proof=config['training_proof']['sha256'], step=proof['step'], requests=len(rows),
                actions=actions, common_actions=common, unscored_actions=actions-common, complete_sequences=complete,
                limits=proof['limits'], modes=modes, candidate_top1_observed=False, training_admission=False)


if __name__ == '__main__':
    cfg = doc(pin(sys.argv[1]))
    save(cfg['output'], compare(cfg))
