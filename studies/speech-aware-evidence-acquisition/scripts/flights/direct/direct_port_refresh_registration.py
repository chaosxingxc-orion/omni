"""Refresh an unspent Direct package after an accepted delivery-assets speech proof.

Archive predecessor bytes before publishing staged changes. Never mint, flip,
rebuild supply, score or call a model. A partial publish must be reconciled from
its archive; the same archive path cannot be reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for directory in (HERE, REPO / 'src', REPO / 'scripts/analysis', REPO / 'scripts/flights/batch'):
    sys.path.insert(0, str(directory))
import direct_port_family as fam
import direct_port_write_registration as writer
import direct_port_quality_read_registration as quality
import direct_port_read_manifest as evidence
from l338_port_deployment_binding import validate_binding
import smoke_signoff

PKG = quality.PKG
PROOF = 'docs/readiness/2026-10-30-slurp-delivery-assets-proof-v1'
AUTH = quality.AUTH


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def ledger_rows(text):
    rows = {}
    for line in text.splitlines():
        if not line.startswith('| '):
            continue
        fields = [s.strip() for s in line.strip().strip('|').split('|')]
        if len(fields) == 15:
            rows.setdefault(fields[1], []).append((line, fields))
    return rows


def require_unspent(repo, data):
    facts = read(repo / PKG / 'registered-facts-v1.json')
    draft = read(repo / PKG / 'root-admission-draft-v1.json')
    decision = data / 'coordination/slurp-direct-only-fold-b-closemarker-npu-v1/root-admission-v1.json'
    if decision.exists():
        raise ValueError('Direct receipt already minted; refresh refused')
    rows = ledger_rows((repo / 'docs/exposure-ledger.md').read_text())
    for spec in facts['identity'].values():
        doc = read(repo / 'configs/experiment' / (spec['cell'] + '.json'))
        cell, = doc['cells']
        if cell['readiness']['state'] != 'held':
            raise ValueError('Direct refresh requires held cells')
        if (cell['run_id'], cell['attempt_id']) != (spec['run_id'], spec['attempt_id']):
            raise ValueError('Direct cell identity differs')
        for run in (spec['run_id'], spec['fit_run_id']):
            if any((data / base / run).exists() for base in ('runs', 'attempts', 'prepared')):
                raise ValueError('Direct generation/FIT already started or prepared')
            entries = rows.get(run, [])
            if len(entries) != 1 or entries[0][1][8] != 'no':
                raise ValueError('Direct ledger must have exactly one unconsumed row per identity')
    for name in ('analysis-recipe-v1.json', 'quality-analysis-recipe-v1.json'):
        path = repo / PKG / name
        recipe = read(path)
        entries = rows.get(recipe['read_run_id'], [])
        if len(entries) != 1 or entries[0][1][8] != 'no' or sha(path) not in entries[0][1][-1]:
            raise ValueError('READ recipe/ledger pin is missing, consumed or ambiguous')
        if (data / 'analysis' / recipe['read_run_id']).exists():
            raise ValueError('Direct READ already started')
    if draft['status'] != 'DRAFT-NOT-MINTED':
        raise ValueError('Direct draft is no longer unminted')
    return facts


def accepted_proof(repo, data, binding):
    spec = read(repo / PROOF / 'diagnostic-draft-v1.json')
    conditions_path = repo / PROOF / 'signoff-conditions-v1.json'
    conditions = read(conditions_path)
    table_path = data / conditions['table_relpath']
    table = read(table_path)
    marker_path = data / 'work/flight_state/delivery-assets-proof-v1/signoff.marker'
    marker = json.loads(marker_path.read_text().split('\n', 1)[1])
    if (marker['verdict'] != 'PASS' or marker['conditions_sha256'] != sha(conditions_path)
            or marker['table_sha256'] != sha(table_path)
            or not all(c['ok'] for c in smoke_signoff.evaluate(conditions, table))):
        raise ValueError('delivery-assets proof sign-off is not accepted')
    audit_path = table_path.with_name('strict-evidence-audit.json')
    audit = read(audit_path)
    if audit['status'] != 'PASS' or audit['read_run_id'] != spec['read_run_id'] or audit['roster_rows'] != 4:
        raise ValueError('delivery-assets strict audit is not accepted')
    for pin in audit['inputs']:
        path = Path(pin['path']).resolve()
        if not path.is_relative_to(data.resolve()) or sha(path) != pin['sha256']:
            raise ValueError('strict proof input changed')
    run = evidence.run_evidence(data, spec['run_id'], spec['attempt_id'], repo)
    if (run['identity']['runtime_receipt_sha256'] != binding['runtime_receipt']['sha256']
            or run['identity']['profile_sha256'] != binding['profile_sha256']
            or spec['service_session'] != binding['session_id']):
        raise ValueError('proof and new binding use different deployments')
    if table['diagnostic_identity'] != {'run_id': spec['run_id'], 'attempt_id': spec['attempt_id'], 'read_run_id': spec['read_run_id']}:
        raise ValueError('proof table identity differs')
    if table['sealed_hashes'] != {k: sha(run['paths'][k]) for k in ('outputs', 'trace')}:
        raise ValueError('proof table sealed hashes differ')
    return spec, {str(p.relative_to(data)): sha(p) for p in (marker_path, table_path, audit_path)}


def refresh(repo, data, binding_path, archive):
    repo, data, archive = repo.resolve(), data.resolve(), archive.resolve()
    if archive.exists() or not archive.is_relative_to(data) or archive.is_relative_to(repo):
        raise ValueError('use a fresh archive under data root outside repository')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip():
        raise ValueError('refresh requires a clean repository')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    old = require_unspent(repo, data)
    binding = validate_binding(read(binding_path), repo, data)
    spec, proof_pins = accepted_proof(repo, data, binding)
    package = repo / PKG
    structural = read(package / 'analysis-recipe-v1.json')
    quality_raw = (package / 'quality-analysis-recipe-v1.json').read_bytes()
    old_ledger = (repo / 'docs/exposure-ledger.md').read_text()
    fam.select_closemarker_successor()
    fam.bind_measured_deployment(binding)
    writer.PKG = fam.PACKAGE
    at = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix='direct-refresh-') as tmp:
        stage = Path(tmp)
        shutil.copytree(repo / 'configs', stage / 'configs')
        shutil.copytree(repo / 'docs/receipts', stage / 'docs/receipts')
        shutil.copytree(package, stage / PKG)
        writer.write_package(stage, head, at, data, verify_audio=True)
        pkg = stage / PKG
        facts = read(pkg / 'registered-facts-v1.json')
        for key, before in old['identity'].items():
            for field in ('run_id', 'attempt_id', 'fit_run_id', 'fit_attempt_id', 'rows', 'split_identity_hash'):
                if facts['identity'][key][field] != before[field]:
                    raise ValueError('refresh changed frozen Direct identity or population')
        smoke_doc = read(stage / 'configs/experiment' / (facts['identity']['smoke']['cell'] + '.json'))
        agent_doc = read(repo / 'configs/experiment' / (spec['cell'] + '.json'))
        if set(smoke_doc['populations'][0]['ids']) != set(agent_doc['populations'][0]['ids']):
            raise ValueError('Direct and proof rosters differ')
        structural.update(agent_run=spec['run_id'], agent_attempt=spec['attempt_id'],
                          agent_reference=f"analysis/{spec['read_run_id']}/smoke-decode-cost.json")
        dump(pkg / 'analysis-recipe-v1.json', structural)
        # Preserve the already registered Direct structural sign-off semantics.
        shutil.copyfile(package / 'signoff-conditions-v1.json', pkg / 'signoff-conditions-v1.json')
        facts['pins']['signoff_conditions']['sha256'] = sha(pkg / 'signoff-conditions-v1.json')
        dump(pkg / 'registered-facts-v1.json', facts)
        draft = read(pkg / 'root-admission-draft-v1.json')
        for key, name in [('registered_facts', 'registered-facts-v1.json'),
                          ('structural_read', 'analysis-recipe-v1.json'),
                          ('signoff_conditions', 'signoff-conditions-v1.json')]:
            draft['antecedents'][key] = {'path': PKG + '/' + name, 'sha256': sha(pkg / name)}
        dump(pkg / 'root-admission-draft-v1.json', draft)
        (pkg / 'README.md').write_text(
            '# Direct Fold B: unspent deployment refresh\n\n'
            f'Refreshed at {at} from commit `{head}` after the accepted delivery-assets proof. '
            'Four cells remain held and unminted: a matched four-row structural smoke, '
            'then lock/aa/ab full 1,080-row TRAIN Fold B passes.\n\n'
            'The shared campaign performs one complete official scoring job per pass. '
            'Read its saved scores with `scripts/analysis/direct_port_score_evidence.py` '
            'under `quality-analysis-recipe-v1.json`; do not run Agent-specific scoring. '
            'The quality recipe and ledger row retain their original bytes.\n\n'
            'The structural recipe now references the accepted delivery-assets speech proof '
            'on the same resident deployment and four-row roster. Sign-off conditions '
            'remain unchanged. Archive intent records previous bytes and every changed hash. '
            'Review, commit, mint, pin readiness, and pass real preflight before contact. '
            'This refresh performs no model calls, scoring, supply preparation or readiness flip.\n')
        if (pkg / 'quality-analysis-recipe-v1.json').read_bytes() != quality_raw:
            raise ValueError('quality READ changed')
        replacements = {}
        for run in fam.eligible_run_ids():
            entries = ledger_rows((pkg / 'draft-ledger-rows.md').read_text())[run]
            _, fields = entries[0]
            fields[3] = fields[3].replace('DRAFT -- REGISTERED.', 'Registered before contact.')
            fields[-1] = fields[-1].replace('DRAFT concrete measured registration', 'Refreshed unspent registration')
            fields[-1] += f' REGISTERED BEFORE CONTACT; root countersigned under {AUTH}; recorded by Codex. Prior bytes preserved in refresh archive; mint and preflight required.'
            replacements[run] = writer.reg.table_row(fields)
        row, fields = ledger_rows(old_ledger)[structural['read_run_id']][0]
        fields[-1], n = re.subn(r'Plan SHA256 [0-9a-f]{64}', 'Plan SHA256 ' + sha(pkg / 'analysis-recipe-v1.json'), fields[-1])
        if n != 1:
            raise ValueError('expected one structural READ digest')
        replacements[structural['read_run_id']] = writer.reg.table_row(fields)
        new_ledger = old_ledger
        for run, replacement in replacements.items():
            line, = [entry[0] for entry in ledger_rows(old_ledger)[run]]
            new_ledger = new_ledger.replace(line, replacement, 1)
        (stage / 'docs/exposure-ledger.md').write_text(new_ledger)
        if ledger_rows(new_ledger)[quality.READ] != ledger_rows(old_ledger)[quality.READ]:
            raise ValueError('quality READ ledger row changed')
        outputs = [p for p in stage.rglob('*') if p.is_file()
                   and (not (repo / p.relative_to(stage)).exists() or p.read_bytes() != (repo / p.relative_to(stage)).read_bytes())]
        allowed = [PKG, PKG + '-preregistration.md', 'docs/exposure-ledger.md',
                   f"configs/campaign/{fam.CELLS['lock']['cell']}.json"]
        allowed += [f'configs/{kind}/{s[field]}.json' for s in fam.CELLS.values()
                    for kind, field in [('baseline', 'baseline'), ('experiment', 'cell')]]
        if any(not any(str(p.relative_to(stage)) == a or str(p.relative_to(stage)).startswith(a + '/') for a in allowed) for p in outputs):
            raise ValueError('unexpected refresh output')
        require_unspent(repo, data)
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip():
            raise ValueError('repository changed during staging')
        archive.mkdir()
        pins = {}
        for p in outputs:
            rel = p.relative_to(stage)
            original = repo / rel
            if original.exists():
                backup = archive / 'before' / rel
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, backup)
            pins[str(rel)] = {'before': sha(original) if original.exists() else None, 'after': sha(p)}
        dump(archive / 'intent.json', {'commit': head, 'binding_sha256': sha(binding_path), 'proof': proof_pins,
                                     'files': pins, 'model_calls': 0, 'scorer_calls': 0, 'ready': False})
        for p in outputs:
            target = repo / p.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
        dump(archive / 'complete.json', {'files': len(outputs), 'model_calls': 0, 'ready': False})
    return {'archive': str(archive), 'files': len(outputs), 'model_calls': 0, 'ready': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refresh(args.repo_root, args.data_root, args.binding, args.archive)))


if __name__ == '__main__':
    main()
