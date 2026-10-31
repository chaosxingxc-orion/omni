"""Validate legacy lock prompt limits and state namespaces on a pinned checkpoint.

Reads recorded requests only, without model, tokenizer, scorer or gold contacts.
This proves the legacy side of compatibility, not fresh actual-data rendering.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def objects(value):
    if isinstance(value, dict):
        yield value
        for v in value.values(): yield from objects(v)
    elif isinstance(value, list):
        for v in value: yield from objects(v)
    elif isinstance(value, str):
        try: parsed = json.loads(value)
        except (ValueError, TypeError):
            marker = 'Native episode limits: '
            if marker in value:
                yield json.JSONDecoder().raw_decode(value.split(marker, 1)[1])[0]
        else:
            if parsed != value: yield from objects(parsed)


def audit(checkpoint, expected_sha):
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    doc = read_checkpoint(checkpoint, expected_sha256=expected_sha)
    if not doc.rows: raise ValueError('empty checkpoint')
    results = []
    for row in doc.rows:
        if row.keep_defect() is not None: raise ValueError('incomplete checkpoint row')
        output, = row.output_rows
        episode = output['native_episode_receipt']
        branch, = [b for b in episode['perception_branches']['branches'] if b['branch_id'] == 'lock']
        contacts = {c['agent_contact_id']: c for v in episode['role_visits'] for c in v['execution_receipt']['contacts']}
        traces = {json.loads(line)['record_id']: json.loads(line)['payload'] for line in row.trace_lines}
        row_proof = []
        for phase, field, index, remaining in [('perception', 'contact_id', 7, 17), ('solver', 'solver_contact_id', 8, 9)]:
            contact_id = f'phase-native-module-v1-c17:{phase}:{index}:{row.sample_id}'
            if branch[field] != contact_id: raise ValueError('legacy contact namespace differs')
            contact = contacts[contact_id]
            request = traces[contact['request_id'] + '/request']
            if request['payload_sha256'] != contact['request_sha256']: raise ValueError('request custody differs')
            limits = [v for v in objects(request['payload']) if 'remaining_model_contacts' in v]
            if len(limits) != 1 or limits[0]['remaining_model_contacts'] != remaining:
                raise ValueError('legacy prompt budget differs')
            if phase == 'solver':
                state_id = f'phase-native-module-v1-c17:perception:7:{row.sample_id}:perception:{row.sample_id}'
                if state_id not in json.dumps(request['payload']): raise ValueError('legacy state namespace absent')
            row_proof.append({'phase': phase, 'request_sha256': contact['request_sha256']})
        results.append({'sample_id': row.sample_id, 'contacts': row_proof})
    return {'schema': 'saea-legacy-lock-prompt-audit-v1', 'checkpoint_sha256': expected_sha,
            'rows': len(results), 'verified_contacts': 2 * len(results),
            'row_proofs_sha256': hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest(),
            'prompt_remaining_contacts': {'perception': 17, 'solver': 9},
            'contact_indices': {'perception': 7, 'solver': 8}, 'new_model_calls': 0,
            'limitation': 'Legacy side verified; fresh actual-data request equivalence and registered migration still required.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); result = audit(a.checkpoint, a.sha256)
    with a.out.open('x') as f: json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
