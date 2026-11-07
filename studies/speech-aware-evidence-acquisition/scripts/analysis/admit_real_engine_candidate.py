"""Bind a registered real workload to an observed resident and refresh shared preparation.

Owner authorization is already recorded. This command makes no inference calls;
the normal clean-tree/preflight/lease/FIT checks still precede dispatch.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/flights/direct'),str(ROOT/'scripts/flights')]
from register_real_engine_candidate import read,sha,write


def validate_predecessor_decision(data, spec, original, pin, new_relpath):
    """Bind a fresh receipt to the exact existing original authority."""
    if not isinstance(pin, dict) or set(pin) != {'relpath', 'sha256'}:
        raise ValueError('exact predecessor decision pin required')
    old_relpath = pin['relpath']
    for rel in (old_relpath, new_relpath):
        if (not isinstance(rel, str) or not rel.startswith('coordination/')
                or '\\' in rel or '..' in Path(rel).parts or Path(rel).is_absolute()):
            raise ValueError('decision path must be confined under coordination')
        if not (data/rel).resolve().is_relative_to(data.resolve()/'coordination'):
            raise ValueError('decision path escapes data root')
    if old_relpath != original['decision_relpath'] or new_relpath == old_relpath:
        raise ValueError('predecessor and fresh decision paths differ from registration')
    if (data/new_relpath).exists() or (data/new_relpath).is_symlink():
        raise ValueError('fresh decision already exists')
    path = data/old_relpath
    if path.is_symlink() or not path.is_file() or sha(path) != pin['sha256']:
        raise ValueError('predecessor decision pin differs')
    decision = read(path)
    if (decision.get('schema') != 'saea-campaign-decision-v1'
            or decision.get('status') != 'decided'
            or decision.get('decision_id') != original['decision_id']
            or spec.get('decision_id') != original['decision_id']
            or spec.get('decision_relpath') != original['decision_relpath']
            or decision.get('authorization_basis') != original['authority']
            or original['run_id'] not in decision.get('eligible_run_ids', [])
            or original['fit_run_id'] not in decision.get('eligible_run_ids', [])
            or original['attempt_id'] not in decision.get('eligible_attempt_ids', [])
            or original['fit_attempt_id'] not in decision.get('eligible_attempt_ids', [])):
        raise ValueError('predecessor decision does not authorize original registration')
    budgets = decision.get('budget')
    if not isinstance(budgets, dict) or not all(isinstance(budgets.get(k), dict) for k in ('generation','fit')):
        raise ValueError('predecessor decision lacks registered budgets')
    return decision


def successor_ledger(text, old_generation, old_fit, new_fit, new_plan_path, new_plan_sha):
    """Retain original rows byte-for-byte and append one fresh FIT registration."""
    from speech_aware_evidence_acquisition.contracts.ledger import _parse_ledger_tables
    rows = _parse_ledger_tables(text)
    def one(run):
        found = [row for row in rows if row.get('run/experiment id') == run]
        if len(found) != 1:
            raise ValueError('expected exactly one predecessor exposure row')
        return found[0]
    generation, fit = one(old_generation), one(old_fit)
    if any(row.get('run/experiment id') == new_fit for row in rows):
        raise ValueError('fresh FIT exposure already registered')
    if generation.get('consumed', '').lower() != 'no' or fit.get('consumed', '').lower() != 'no':
        raise ValueError('predecessor exposure row consumed')
    keys = ('date','run/experiment id','execution profile','scope','model/tool','speech carrier + split',
            'split role','split identity hash','consumed','calls','speech audio seconds','gpu hours',
            'inherited exposure','artifacts','notes')
    if any(key not in fit for key in keys):
        raise ValueError('predecessor FIT exposure row incomplete')
    copied = dict(fit)
    copied.update({'run/experiment id':new_fit,'artifacts':new_plan_path,
                   'inherited exposure':'All predecessor generation and FIT costs retained',
                   'notes':f'Plan SHA256 {new_plan_sha}; fresh FIT successor'})
    if any('|' in copied[key] or '\n' in copied[key] for key in keys):
        raise ValueError('unsafe exposure row')
    return text + '\n## Fresh successor FIT\n\n| ' + ' | '.join(keys) + ' |\n| ' + ' | '.join(['---']*len(keys)) + ' |\n| ' + ' | '.join(copied[key] for key in keys) + ' |\n'


def validate_successor_plans(repo, spec, experiment, pins, *, fresh=False):
    """Validate the separately pinned successor documents before any receipt is made."""
    from dataclasses import fields
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset

    if not isinstance(pins, dict) or set(pins) != {'generation', 'fit', 'protocol'}:
        raise ValueError('three explicit successor document pins required')
    docs = {}
    paths = set()
    for kind, pin in pins.items():
        if not isinstance(pin, dict) or set(pin) != {'relpath', 'sha256'}:
            raise ValueError('exact successor document pin required')
        rel = pin['relpath']
        if (not isinstance(rel, str) or not rel.startswith('docs/')
                or '..' in Path(rel).parts
                or (not fresh and rel.endswith('/generation-plan-registered-v1.json')) or rel in paths):
            raise ValueError('successor documents must have distinct new paths')
        if fresh and rel != f"docs/readiness/{spec['cell']}/{kind}-{'plan-' if kind != 'protocol' else ''}registered-v1.json":
            raise ValueError('fresh document path differs from registration')
        paths.add(rel)
        path = resolve_asset(repo, rel)
        if sha(path) != pin['sha256']:
            raise ValueError('successor document pin differs: ' + kind)
        docs[kind] = read(path)
    allowed = {item.name for item in fields(ExecutionPlan)}
    cell, = experiment['cells']
    for kind, rid, aid in (('generation', spec['run_id'], spec['attempt_id']),
                           ('fit', spec['fit_run_id'], spec['fit_attempt_id'])):
        doc = docs[kind]
        ep = doc.get('execution_plan')
        if not isinstance(ep, dict) or set(ep) - allowed:
            raise ValueError('invalid successor execution plan')
        args = dict(ep)
        if 'owner_allowance' in doc:
            args['owner_allowance'] = doc['owner_allowance']
        plan = ExecutionPlan(**args).validate(source_root=repo)
        if (doc.get('schema') != f'saea-registered-{kind}-plan-v1'
                or doc.get('plan_kind') != kind
                or doc.get('cell') != spec['cell'] or doc.get('experiment') != spec['experiment']
                or doc.get('composed_config_hash') != spec['composed_config_hash']
                or doc.get('expected_session_id') != spec['service_session']
                or doc.get('protocol_document') != pins['protocol']
                or plan.run_id != rid or plan.attempt_id != aid
                or plan.protocol_hash != spec['protocol_hash']
                or plan.split_identity_hash != spec['split_identity_hash']):
            raise ValueError('successor registered plan identity differs: ' + kind)
        expected = experiment['execution_plan']
        for key in ('execution_profile', 'split_role', 'planned_gpu_hours', 'planned_speech_audio_seconds'):
            if getattr(plan, key) != expected[key]:
                raise ValueError('successor registered plan budget differs: ' + kind)
        expected_calls = expected['planned_model_calls']
        if kind == 'fit':
            baseline = read(resolve_asset(repo, 'configs/baseline/' + cell['baseline_fragment'] + '.json'))
            if (baseline['native_contact_fit']['run_id'] != rid
                    or baseline['native_contact_fit']['attempt_id'] != aid):
                raise ValueError('successor FIT baseline identity differs')
            expected_calls = baseline['native_contact_fit']['contact_cap']
        if plan.planned_model_calls != expected_calls:
            raise ValueError('successor registered plan calls differ: ' + kind)
    return docs


def fresh_ledger(text, run, fit, plans, original):
    """Update only the two unconsumed fresh rows' final plan pins."""
    from speech_aware_evidence_acquisition.contracts.ledger import _parse_ledger_tables
    import re
    rows = _parse_ledger_tables(text)
    for kind, rid in (('generation', run), ('fit', fit)):
        matches = [row for row in rows if row.get('run/experiment id') == rid]
        if len(matches) != 1 or matches[0].get('consumed', '').lower() != 'no':
            raise ValueError('one unconsumed fresh exposure row required')
        if (matches[0].get('artifacts') != original[kind]['relpath'] or
                re.findall(r'Plan SHA256 ([0-9a-f]{64})', matches[0].get('notes', '')) != [original[kind]['sha256']]):
            raise ValueError('fresh exposure registration pin differs')
    lines = text.splitlines(keepends=True)
    for kind, rid in (('generation', run), ('fit', fit)):
        indices = [i for i, line in enumerate(lines) if '| '+rid+' |' in line]
        if len(indices) != 1 or len(re.findall(r'Plan SHA256 [0-9a-f]{64}', lines[indices[0]])) != 1:
            raise ValueError('fresh exposure plan hash missing or ambiguous')
        lines[indices[0]] = re.sub(r'Plan SHA256 [0-9a-f]{64}', 'Plan SHA256 '+plans[kind], lines[indices[0]])
    return ''.join(lines)


def verify_registration_identity(repo, name, spec, cell, data=None):
    """Reject changed configuration before minting an irreversible receipt."""
    from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
    args = (repo, cell['model_fragment'], cell['dataset_fragment'],
            cell['baseline_fragment'], name)
    cfg = compose(*args, data_root=data) if data is not None else compose(*args)
    expected_protocol = protocol_hash(cfg.config_hash, cfg.values['arm'],
                                      spec['split_identity_hash'])
    if (spec.get('composed_config_hash') != cfg.config_hash
            or spec.get('protocol_hash') != expected_protocol):
        raise ValueError('registered configuration changed; rebuild registration before admission')


def pinned_runtime_config(pin):
    """Require the same immutable runtime overlay the worker will receive."""
    import os
    from speech_aware_evidence_acquisition.core.flight.factory import load_external_runtime_config
    if (not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}
            or not isinstance(pin['path'], str) or not Path(pin['path']).is_absolute()
            or str(Path(pin['path']).resolve()) != pin['path']
            or not isinstance(pin['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', pin['sha256'])
            or Path(pin['path']).is_symlink() or sha(Path(pin['path'])) != pin['sha256']
            or os.environ.get('SAEA_RUNTIME_CONFIG_SHA256') != pin['sha256']):
        raise ValueError('exact pinned runtime config and environment required')
    path = Path(pin['path'])
    return path, load_external_runtime_config(path)


def test_contact_authorized(repo, data, name, spec, exp):
    """Only an independently validated full-test registration gains this flag."""
    if spec.get('workload_kind') != 'final-report-method':
        return False
    from final_report_workload import validate, DATASET
    view = read(repo / f'configs/dataset/{DATASET}.json')
    validate(repo, data, name, spec, exp, view)
    return True


def _exclusive_same(path, raw):
    import os
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    path = Path(path)
    try:
        publish_new(path, raw)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise ValueError('existing publication differs: ' + str(path))
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def resume_external_completion(workspace, data):
    """Finish an already validated publication without minting again."""
    import base64
    workspace, data = Path(workspace).resolve(), Path(data).resolve()
    intent = read(workspace/'publication-intent.json')
    if set(intent) != {'schema', 'completion', 'receipt_base64'} or intent['schema'] != 'saea-admission-publication-intent-v1':
        raise ValueError('invalid publication intent')
    completion = intent['completion']
    if (not isinstance(completion, dict) or completion.get('schema') != 'saea-external-admission-completion-v1'
            or set(completion) != {'schema','candidate','receipt','workspace_manifest','binding'}):
        raise ValueError('invalid publication completion')
    decision = base64.b64decode(intent['receipt_base64'], validate=True)
    pin = completion['receipt']; rel = Path(pin['relpath'])
    if (rel.is_absolute() or '..' in rel.parts or not str(rel).startswith('coordination/')
            or not (data/rel).resolve().is_relative_to(data/'coordination')
            or hashlib.sha256(decision).hexdigest() != pin['sha256']):
        raise ValueError('publication receipt pin differs')
    manifest = completion['workspace_manifest']; mp = Path(manifest['path'])
    if not mp.resolve().is_relative_to(workspace/'manifests') or mp.is_symlink() or sha(mp) != manifest['sha256']:
        raise ValueError('publication workspace manifest differs')
    inventory = read(mp)
    if inventory.get('schema') != 'saea-experiment-workspace-v1' or not isinstance(inventory.get('files'), dict):
        raise ValueError('invalid publication workspace inventory')
    for name, digest in inventory['files'].items():
        relfile = Path(name); path = workspace/relfile
        if (relfile.is_absolute() or '..' in relfile.parts or not name.startswith(('configs/', 'docs/'))
                or not path.resolve().is_relative_to(workspace) or path.is_symlink()
                or not path.is_file() or sha(path) != digest):
            raise ValueError('publication workspace asset differs: ' + name)
    raw = (json.dumps(completion, sort_keys=True, indent=2) + '\n').encode()
    _exclusive_same(data/rel, decision)
    _exclusive_same(workspace/'completion.json', raw)
    return completion


def publish_external_completion(workspace, data, completion, decision_bytes):
    """Persist staged bytes in the intent before publishing receipt and marker."""
    import base64
    if hashlib.sha256(decision_bytes).hexdigest() != completion['receipt']['sha256']:
        raise ValueError('receipt differs from publication intent')
    intent = {'schema':'saea-admission-publication-intent-v1', 'completion':completion,
              'receipt_base64':base64.b64encode(decision_bytes).decode('ascii')}
    raw = (json.dumps(intent, sort_keys=True, indent=2) + '\n').encode()
    _exclusive_same(Path(workspace)/'publication-intent.json', raw)
    return resume_external_completion(workspace, data)


def admit_external(repo, data, config):
    """Validate and publish while restoring the caller's workspace environment."""
    import os
    keys = ('SAEA_EXPERIMENT_WORKSPACE', 'SAEA_EXPERIMENT_WORKSPACE_MANIFEST',
            'SAEA_EXPERIMENT_WORKSPACE_SHA256')
    saved = {key: os.environ.get(key) for key in keys}
    try:
        return _admit_external(repo, data, config)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _admit_external(repo, data, config):
    """Validate a pinned raw-speech successor before publishing its decision."""
    import os
    from dataclasses import fields, replace
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan, FrozenCoreGate, assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
    from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset, seal_workspace
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell, declared_readiness
    from speech_aware_evidence_acquisition.core.flight.cli import assert_prepared_artifacts_current
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.flight.runtime_overlay import apply_runtime_overlay
    from speech_aware_evidence_acquisition.core.flight.repin import _upsert_decision_requirement, _set_ready
    from speech_aware_evidence_acquisition.core.session.serving import find_resident_session, assert_identity_matches
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session
    from registered_workload import validate_workload, assert_execution_binding
    from run_real_engine_candidate import recovery_options
    from thinking_system_mint_receipt import mint

    required = {'schema','candidate','candidate_pin','plans','decision_relpath','workspace'}
    fresh = isinstance(config,dict) and config.get('schema') == 'saea-external-fresh-admission-v1'
    expected = required if fresh else required | {'predecessor_decision'}
    if not isinstance(config,dict) or set(config) not in (expected, expected | {'runtime_config'}) or config['schema'] not in ('saea-external-successor-admission-v1','saea-external-fresh-admission-v1'):
        raise ValueError('exact external admission configuration required')
    runtime_path = None
    overlay = None
    if 'runtime_config' in config:
        runtime_path, overlay = pinned_runtime_config(config['runtime_config'])
    repo, data = Path(repo).resolve(), Path(data).resolve()
    workspace = Path(config['workspace'])
    if (not workspace.is_absolute() or workspace.exists() or workspace.resolve().is_relative_to(repo)
            or workspace.resolve().is_relative_to(data)):
        raise ValueError('fresh isolated workspace required')
    for key in ('SAEA_EXPERIMENT_WORKSPACE','SAEA_EXPERIMENT_WORKSPACE_MANIFEST','SAEA_EXPERIMENT_WORKSPACE_SHA256'):
        if os.environ.get(key): raise ValueError('pre-existing workspace forbidden')
    if not os.environ.get('SAEA_EXPERIMENT_MANIFEST') or not os.environ.get('SAEA_EXPERIMENT_MANIFEST_SHA256'):
        raise ValueError('pinned external asset manifest required')
    assert_execution_environment(); require_clean_study_repo(repo)
    name = config['candidate']
    if not isinstance(name,str) or not re.fullmatch('[a-z0-9][a-z0-9-]*',name):
        raise ValueError('candidate name invalid')
    candidate_rel = f'docs/readiness/{name}/candidate.json'
    if (not isinstance(config['candidate_pin'],dict) or set(config['candidate_pin']) != {'relpath','sha256'}
            or config['candidate_pin']['relpath'] != candidate_rel):
        raise ValueError('candidate pin invalid')
    candidate_path = resolve_asset(repo,candidate_rel)
    if sha(candidate_path) != config['candidate_pin']['sha256']: raise ValueError('candidate changed')
    spec = read(candidate_path)
    supported = {'gepa-solver-template-pair', 'gepa-perception-template-pair',
                 'intent-slot-node-ablation', 'perception-only-native'}
    if spec.get('workload_kind') not in supported:
        raise ValueError('only registered raw-speech successor workloads are supported')
    exp_rel = f'configs/experiment/{name}.json'
    exp = read(resolve_asset(repo,exp_rel)); cell, = exp['cells']
    if cell['readiness']['state'] != 'held': raise ValueError('held registration required')
    if fresh:
        if cell.get('resume_from') is not None or cell.get('checkpoint_compatibility') is not None or 'predecessor_registration' in spec:
            raise ValueError('fresh registration cannot have predecessor custody')
        rel = config['decision_relpath']
        if (not isinstance(rel,str) or not rel.startswith('coordination/') or '..' in Path(rel).parts
                or not (data/rel).resolve().is_relative_to(data/'coordination')
                or rel != spec['decision_relpath'] or (data/rel).exists() or (data/rel).is_symlink()):
            raise ValueError('fresh decision destination differs or exists')
    else:
        if not cell.get('resume_from') or not cell.get('checkpoint_compatibility'):
            raise ValueError('held checkpoint successor required')
        ancestor = spec['predecessor_registration']
        old_path = resolve_asset(repo,ancestor['relpath'])
        if sha(old_path) != ancestor['sha256']: raise ValueError('predecessor registration changed')
        old = read(old_path)
        original_plan_rel = f"docs/readiness/{old['cell']}/generation-plan-registered-v1.json"
        original_plan_bytes = resolve_asset(repo,original_plan_rel).read_bytes()
        old_receipt = validate_predecessor_decision(data,spec,old,config['predecessor_decision'],config['decision_relpath'])
    verify_registration_identity(repo,name,spec,cell,data)
    plans = validate_successor_plans(repo,spec,exp,config['plans'],fresh=fresh)
    if not fresh:
        for kind in ('generation','fit'):
            for field in ('execution_profile','split_role','split_identity_hash','planned_model_calls','planned_gpu_hours','planned_speech_audio_seconds'):
                if old_receipt['budget'][kind].get(field) != plans[kind]['execution_plan'].get(field):
                    raise ValueError('successor budget differs from predecessor authority')
    rows = validate_workload(repo,data,name,spec)
    assert_execution_binding(spec,repo=repo)
    if 'stage3_algorithm' in exp:
        from speech_aware_evidence_acquisition.core.stage3_algorithms import validate_stage3_algorithm_label
        validate_stage3_algorithm_label(exp['stage3_algorithm'],cell=name)
    receipt_rel = 'docs/'+spec['runtime_receipt']
    runtime_doc = read(resolve_asset(repo,receipt_rel))
    def build(cell_doc):
        parsed = CampaignCell.from_document(cell_doc)
        return build_flight_runtime(repo,data,model=cell_doc['model_fragment'],dataset=cell_doc['dataset_fragment'],
            baseline=cell_doc['baseline_fragment'],experiment=name,run_id=parsed.run_id,attempt_id=parsed.attempt_id,
            runtime_root=data/'work/scratch/runtime-root',lock_path=repo.parents[1]/'docs/datasets.lock.json',
            runtime_config_path=runtime_path, **recovery_options(parsed))
    runtime = build(cell)
    resident = find_resident_session(data)
    if resident is None or resident[1]['session_id'] != spec['service_session']:
        raise ValueError('expected resident absent')
    assert_identity_matches(resident[1],runtime.session_required_identity())
    for instance in runtime_doc['deployment']['instances']:
        instance_session = SessionReceipt.from_json(read(data/'serving'/instance['instance_id']/'session-receipt.json'))
        verify_vllm_session(instance_session,runtime_doc)
    for rid, aid in ((spec['run_id'],spec['attempt_id']),(spec['fit_run_id'],spec['fit_attempt_id'])):
        if fresh and ((data/'attempts'/rid).exists() or (data/'runs'/rid).exists()):
            raise ValueError('fresh run already has attempt or run custody')
        if (data/'attempts'/rid/(aid+'.json')).exists() or (data/'runs'/rid/aid).exists():
            raise ValueError('successor attempt already exists')
    rels = [config['plans'][kind]['relpath'] for kind in ('generation','fit','protocol')]
    snapshot_rel = f'docs/readiness/{name}/admission-protocol-snapshot.json'
    if snapshot_rel in rels or snapshot_rel == candidate_rel: raise ValueError('snapshot path collision')
    old_ledger = resolve_asset(repo,'docs/exposure-ledger.md').read_text()
    # Validate the predecessor ledger and fresh identity before creating any output.
    if fresh:
        fresh_ledger(old_ledger,spec['run_id'],spec['fit_run_id'],{'generation':'0'*64,'fit':'0'*64},config['plans'])
    else:
        successor_ledger(old_ledger,spec['run_id'],old['fit_run_id'],spec['fit_run_id'],rels[1],'0'*64)
    fragment_rels = [f"configs/{kind}/{cell[key]}.json" for kind,key in
                     (('model','model_fragment'),('dataset','dataset_fragment'),('baseline','baseline_fragment'))]
    for rel in [candidate_rel,exp_rel,'docs/exposure-ledger.md',*fragment_rels,*rels]:
        publish_new(workspace/rel,resolve_asset(repo,rel).read_bytes())
    if not fresh: publish_new(workspace/original_plan_rel,original_plan_bytes)
    if fresh and 'prepared_phase_supply' not in plans['protocol']:
        plans['protocol']['prepared_phase_supply'] = read(resolve_asset(repo,fragment_rels[2]))['prepared_phase_supply']
        publish_new(workspace/snapshot_rel,(json.dumps(plans['protocol'],indent=2,ensure_ascii=False)+'\n').encode())
    else:
        publish_new(workspace/snapshot_rel,resolve_asset(repo,rels[2]).read_bytes())
    os.environ['SAEA_EXPERIMENT_WORKSPACE'] = str(workspace)
    def seal():
        result=seal_workspace(repo)
        os.environ['SAEA_EXPERIMENT_WORKSPACE_MANIFEST']=result['path']
        os.environ['SAEA_EXPERIMENT_WORKSPACE_SHA256']=result['sha256']
        return result
    seal()
    # Check the compiler's supply under the actual successor configuration.
    # A stale artifact is rebuilt with the native CPU producer and must select
    # exactly the predecessor output before its new immutable pin is used.
    from speech_aware_evidence_acquisition.core.flight.repin import (
        ArtifactPin, _compare_artifact_output, _set_preparation_requirement,
    )
    from speech_aware_evidence_acquisition.core.preparation import PreparationError
    initial_runtime = build(cell)
    supply_pin = read(workspace/fragment_rels[2]).get('prepared_phase_supply')
    source_supply = supply_pin
    try:
        assert_prepared_artifacts_current([initial_runtime])
    except PreparationError:
        baseline_path = workspace / fragment_rels[2]
        baseline = read(baseline_path)
        old_pin = baseline.pop('prepared_phase_supply', None)
        if not isinstance(old_pin, dict):
            raise
        old_path = data / old_pin['relpath']
        if not old_path.is_file() or sha(old_path) != old_pin['sha256']:
            raise ValueError('predecessor prepared supply absent or changed')
        baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False)+'\n')
        seal()
        target_rel = f"prepared/{spec['run_id']}/phase-supply-admission-{hashlib.sha256(str(workspace).encode()).hexdigest()[:16]}.json"
        target = data / target_rel
        if not target.resolve().is_relative_to((data/'prepared').resolve()):
            raise ValueError('prepared supply path escapes data root')
        if target.exists() or target.is_symlink():
            raise ValueError('fresh prepared supply already exists')
        prepared = build(cell).prepare_phase_supply(target)
        _compare_artifact_output(old_path=old_path, new_path=target, allow_output_change=False)
        new_pin = ArtifactPin(target_rel, prepared['sha256'])
        baseline['prepared_phase_supply'] = new_pin.as_dict()
        supply_pin = new_pin.as_dict()
        baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False)+'\n')
        _set_preparation_requirement(workspace/exp_rel, cell_id=cell['cell_id'],
            run_id=spec['run_id'], previous_pin=ArtifactPin.from_mapping(old_pin), pin=new_pin)
        exp = read(workspace/exp_rel); cell, = exp['cells']
        seal()
    snapshot_sha=sha(resolve_asset(repo,snapshot_rel))
    draft={'schema':'saea-campaign-decision-v1','status':'DRAFT-NOT-MINTED',
        'decision_id':spec['decision_id'],'eligible_run_ids':[spec['run_id'],spec['fit_run_id']],
        'eligible_attempt_ids':[spec['attempt_id'],spec['fit_attempt_id']],
        'authorization_basis':spec['authority'],'owner':'root',
        'decision':'Execute the fresh registered workload.' if fresh else 'Continue the registered workload with all predecessor costs retained.',
        **({} if fresh else {'predecessor_decision':config['predecessor_decision']}),
        'antecedents':{'owner':{'path':spec['authority'],'sha256':sha(resolve_asset(repo,spec['authority']))},
            'protocol':{'path':snapshot_rel,'sha256':snapshot_sha},
            'runtime':{'path':receipt_rel,'sha256':sha(resolve_asset(repo,receipt_rel))}},
        'protocol_binding':{'admission_snapshot':{'path':snapshot_rel,'sha256':snapshot_sha},
            'relation':'replace only exact composed_config_hash and protocol_hash identities',
            'source_identity':{'composed_config_hash':spec['composed_config_hash'],
                               'protocol_hash':spec['protocol_hash']}},
        'budget':{k:plans[k]['execution_plan'] for k in ('generation','fit')},
        'population':{'count':rows,'split_identity_hash':spec['split_identity_hash'],'test_contact_authorized':False},
        'service':{'expected_session_id':spec['service_session'],'profile':spec['profile'],
                   'session_sha256':sha(resident[0]/'session.json')}}
    draft_path=workspace/f'docs/readiness/{name}/root-admission-draft-v1.json'
    write(draft_path,draft); seal()
    family=SimpleNamespace(PACKAGE=f'docs/readiness/{name}',DECISION_ID=spec['decision_id'],
        DECISION_RELPATH=config['decision_relpath'],eligible_run_ids=lambda:[spec['run_id'],spec['fit_run_id']])
    staged=mint(str(repo),str(data),'Codex',True,family=family,draft_input=draft_path,
        resolve_path=lambda rel:resolve_asset(repo,rel),tool='scripts/analysis/admit_real_engine_candidate.py')
    decision_bytes=staged.pop('text').encode()
    decision_sha=hashlib.sha256(decision_bytes).hexdigest()
    _upsert_decision_requirement(cell['readiness'],decision_id=spec['decision_id'],
        relpath=config['decision_relpath'],sha256=decision_sha)
    _set_ready(cell['readiness'])
    (workspace/exp_rel).write_text(json.dumps(exp,indent=2,ensure_ascii=False)+'\n')
    seal()
    composed=compose(repo,cell['model_fragment'],cell['dataset_fragment'],cell['baseline_fragment'],name,data_root=data)
    if overlay:
        effective=apply_runtime_overlay(composed.values,overlay)
        composed=type(composed)(values=effective, config_hash=hashlib.sha256(
            json.dumps(effective,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            fragments=composed.fragments)
    new_hash=composed.config_hash; new_protocol=protocol_hash(new_hash,composed.values['arm'],spec['split_identity_hash'])
    old_hash,old_protocol=spec['composed_config_hash'],spec['protocol_hash']
    def rewrite(value):
        if isinstance(value,dict): return {key:rewrite(item) for key,item in value.items()}
        if isinstance(value,list): return [rewrite(item) for item in value]
        if isinstance(value,str): return {old_hash:new_hash,old_protocol:new_protocol}.get(value,value)
        return value
    for rel in (candidate_rel,rels[2]):
        path=workspace/rel; path.write_text(json.dumps(rewrite(read(path)),indent=2,ensure_ascii=False)+'\n')
    final_protocol_doc = rewrite(read(workspace/rels[2]))
    if fresh or 'prepared_phase_supply' in final_protocol_doc:
        final_protocol_doc['prepared_phase_supply'] = supply_pin
    final_protocol_doc['admission_snapshot'] = {'relpath':snapshot_rel,'sha256':snapshot_sha}
    binding = {**draft['protocol_binding'], 'final_identity':{'composed_config_hash':new_hash, 'protocol_hash':new_protocol}}
    final_protocol_doc['protocol_binding'] = binding
    (workspace/rels[2]).write_text(json.dumps(final_protocol_doc,indent=2,ensure_ascii=False)+'\n')
    protocol_pin={'relpath':rels[2],'sha256':sha(workspace/rels[2])}
    for kind in ('generation','fit'):
        path=workspace/config['plans'][kind]['relpath']; doc=rewrite(read(path));doc['protocol_document']=protocol_pin
        if 'prepared_phase_supply' in doc:
            doc['prepared_phase_supply'] = supply_pin
        path.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+'\n')
    ledger=(fresh_ledger(old_ledger,spec['run_id'],spec['fit_run_id'],{k:sha(workspace/config['plans'][k]['relpath']) for k in ('generation','fit')},config['plans'])
            if fresh else successor_ledger(old_ledger,spec['run_id'],old['fit_run_id'],spec['fit_run_id'],rels[1],sha(workspace/rels[1])))
    (workspace/'docs/exposure-ledger.md').write_text(ledger)
    seal()
    final_spec=read(resolve_asset(repo,candidate_rel))
    final_exp=read(resolve_asset(repo,exp_rel)); final_cell,=final_exp['cells']
    validate_workload(repo,data,name,final_spec)
    final_pins={kind:{'relpath':config['plans'][kind]['relpath'],'sha256':sha(resolve_asset(repo,config['plans'][kind]['relpath']))} for kind in ('generation','fit','protocol')}
    final_plans=validate_successor_plans(repo,final_spec,final_exp,final_pins,fresh=fresh)
    for kind in ('generation', 'fit'):
        if 'prepared_phase_supply' in final_plans[kind] and final_plans[kind]['prepared_phase_supply'] != supply_pin:
            raise ValueError('registered plan prepared supply differs')
    if read(resolve_asset(repo,snapshot_rel)) != plans['protocol']:
        raise ValueError('admission protocol snapshot changed')
    expected_protocol= rewrite(plans['protocol'])
    if 'prepared_phase_supply' in expected_protocol:
        expected_protocol['prepared_phase_supply'] = supply_pin
    expected_protocol['admission_snapshot']={'relpath':snapshot_rel,'sha256':snapshot_sha}
    expected_protocol['protocol_binding']=binding
    if expected_protocol != read(resolve_asset(repo,rels[2])):
        raise ValueError('execution protocol differs from admission snapshot')
    if not fresh and resolve_asset(repo,original_plan_rel).read_bytes() != original_plan_bytes:
        raise ValueError('original generation plan changed')
    if declared_readiness(CampaignCell.from_document(final_cell),data,decision_documents={config['decision_relpath']:decision_bytes}).state != 'ready':
        raise ValueError('staged decision does not satisfy readiness')
    runtime=build(final_cell)
    assert_prepared_artifacts_current([runtime])
    values, carrier, _, identity=runtime._prepare()
    generated=runtime._build_execution_plan(values,carrier,identity)
    for kind, runtime_plan in (
            ('generation', generated),
            ('fit', replace(generated, run_id=spec['fit_run_id'],
                            attempt_id=spec['fit_attempt_id'],
                            planned_model_calls=final_plans['fit']['execution_plan']['planned_model_calls']))):
        document = final_plans[kind]
        args = dict(document['execution_plan'])
        if 'owner_allowance' in document:
            args['owner_allowance'] = document['owner_allowance']
        registered = ExecutionPlan(**args).validate(source_root=repo)
        if any(getattr(runtime_plan, f.name) != getattr(registered, f.name) for f in fields(ExecutionPlan)):
            raise ValueError('runtime ' + kind + ' plan differs')
    gate=FrozenCoreGate.for_study_repo(repo,data,data/'work/scratch/runtime-root')
    for kind, plan in [('generation',generated),('fit',replace(generated,run_id=spec['fit_run_id'],attempt_id=spec['fit_attempt_id'],planned_model_calls=final_plans['fit']['execution_plan']['planned_model_calls']))]:
        gate._require_exposure_preregistration(plan,runtime._lock)
        runtime._require_exposure_row_unconsumed(plan.run_id)
    for pin in draft['antecedents'].values():
        if sha(resolve_asset(repo,pin['path'])) != pin['sha256']: raise ValueError('admission antecedent changed')
    supply_evidence_rel = f'docs/readiness/{name}/admission-supply-equivalence.json'
    if sha(data/source_supply['relpath']) != source_supply['sha256'] or sha(data/supply_pin['relpath']) != supply_pin['sha256']:
        raise ValueError('prepared supply pin changed')
    source_output = read(data/source_supply['relpath'])['output']
    final_output = read(data/supply_pin['relpath'])['output']
    if source_output != final_output:
        raise ValueError('prepared supply output changed')
    output_sha = hashlib.sha256(json.dumps(source_output, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    evidence = {'schema':'saea-admission-supply-equivalence-v1', 'source':source_supply,
                'final':supply_pin, 'output_sha256':output_sha}
    write(workspace/supply_evidence_rel, evidence)
    manifest=seal()
    # Keep the exact pending receipt outside the scientific inventory.  If the
    # process stops after receipt publication, this intent identifies the only
    # receipt that may be completed without another mint.
    completion = {'schema':'saea-external-admission-completion-v1', 'candidate':name,
                  'receipt':{'relpath':config['decision_relpath'],'sha256':decision_sha},
                  'workspace_manifest':manifest,
                  'binding':{'protocol':rels[2], 'snapshot':snapshot_rel,
                             'generation':rels[0], 'fit':rels[1],
                             'candidate':candidate_rel, 'baseline':fragment_rels[2],
                             'supply_evidence':supply_evidence_rel}}
    publish_external_completion(workspace, data, completion, decision_bytes)
    return {'candidate':name,'admission_minted':True,'receipt_sha256':decision_sha,
            'completion_marker':str(workspace/'completion.json'),
            'workspace_manifest':manifest,'new_model_calls':0,'preflight_pending':True}


def admit(repo,data,name):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.session.serving import find_resident_session,assert_identity_matches
    from speech_aware_evidence_acquisition.core.flight.repin import repin_package,rewrite_dependent_hashes
    from refresh_registered_plan_pins import refresh_package
    from thinking_system_mint_receipt import mint
    assert_execution_environment()
    pkg=repo/'docs/readiness'/name;spec=read(pkg/'candidate.json')
    exp=read(repo/'configs/experiment'/f'{name}.json');cell,=exp['cells']
    if cell['readiness']['state']!='held':raise ValueError('candidate not held')
    verify_registration_identity(repo, name, spec, cell, data)
    # Labels are used by the official reader; reject invalid declared metadata
    # before any draft/receipt or model work, including non-paired workloads.
    if 'stage3_algorithm' in exp:
        from speech_aware_evidence_acquisition.core.stage3_algorithms import validate_stage3_algorithm_label
        validate_stage3_algorithm_label(exp['stage3_algorithm'], cell=name)
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from registered_workload import validate_workload, assert_execution_binding
    rows=validate_workload(repo,data,name,spec)
    assert_execution_binding(spec, repo=repo)
    receipt=repo/'docs'/spec['runtime_receipt']
    if not receipt.is_file():raise ValueError('observed runtime receipt missing')
    lock=repo.parents[1]/'docs/datasets.lock.json'
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell
    from run_real_engine_candidate import recovery_options
    runtime=build_flight_runtime(repo,data,model=cell['model_fragment'],dataset=cell['dataset_fragment'],baseline=cell['baseline_fragment'],experiment=name,run_id=cell['run_id'],attempt_id=cell['attempt_id'],runtime_root=data/'work/scratch/runtime-root',lock_path=lock,**recovery_options(CampaignCell.from_document(cell)))
    resident=find_resident_session(data)
    if resident is None or resident[1]['session_id']!=spec['service_session']:raise ValueError('expected resident absent')
    assert_identity_matches(resident[1],runtime.session_required_identity())
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session
    doc=read(receipt)
    for instance in doc['deployment']['instances']:
        s=SessionReceipt.from_json(read(data/'serving'/instance['instance_id']/'session-receipt.json'))
        verify_vllm_session(s,doc)
    runs=[spec['run_id'],spec['fit_run_id']]
    draft={'schema':'saea-campaign-decision-v1','status':'DRAFT-NOT-MINTED','decision_id':spec['decision_id'],'eligible_run_ids':runs,'eligible_attempt_ids':[spec['attempt_id'],spec['fit_attempt_id']],'authorization_basis':spec['authority'],'owner':'root','decision':'Execute the exact registered real population through the registered shared runtime under recorded owner authority; no synthetic replay or test tuning.','antecedents':{'owner':{'path':spec['authority'],'sha256':sha(repo/spec['authority'])},'protocol':{'path':str((pkg/'protocol-registered-v1.json').relative_to(repo)),'sha256':sha(pkg/'protocol-registered-v1.json')},'runtime':{'path':str(receipt.relative_to(repo)),'sha256':sha(receipt)}},'service':{'expected_session_id':spec['service_session'],'profile':spec['profile'],'session_sha256':sha(resident[0]/'session.json')},'population':{'count':rows,'split_identity_hash':spec['split_identity_hash'],'test_contact_authorized':test_contact_authorized(repo,data,name,spec,exp)},'budget':{kind:read(pkg/f'{kind}-plan-registered-v1.json')['execution_plan'] for kind in ('generation','fit')}}
    write(pkg/'root-admission-draft-v1.json',draft)
    family=SimpleNamespace(PACKAGE=str(pkg.relative_to(repo)),DECISION_ID=spec['decision_id'],DECISION_RELPATH=spec['decision_relpath'],eligible_run_ids=lambda:runs)
    result=mint(str(repo),str(data),'Codex',False,family=family,tool='scripts/analysis/admit_real_engine_candidate.py')
    before=(data/spec['decision_relpath']).read_bytes()
    repin=repin_package(repo_root=repo,data_root=data,runtime_root=data/'work/scratch/runtime-root',lock_path=lock,experiments=[(name,name)],readiness_dir=pkg,preregistration_path=repo/'docs/readiness'/f'{name}-preregistration.md',decision=(spec['decision_relpath'],spec['decision_id']),set_ready=True,allow_output_change=False,run_check_fn=None)
    refresh_package(repo,pkg)
    rewrite_dependent_hashes(repo_root=repo,readiness_dir=pkg,preregistration_path=repo/'docs/readiness'/f'{name}-preregistration.md',exposure_ledger_path=repo/'docs/exposure-ledger.md',run_ids=set(runs),mapping={})
    if before!=(data/spec['decision_relpath']).read_bytes():raise ValueError('admission mutated')
    write(pkg/'admission-preparation.json',{'new_model_calls':0,'repin':repin,'admission_sha256':hashlib.sha256(before).hexdigest(),'remaining':['commit exact runtime configuration','shared preflight','guarded detached dispatch']})
    return {'candidate':name,'admission_minted':True,'new_model_calls':0,'preflight_pending':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--candidate');group.add_argument('--config',type=Path);group.add_argument('--resume-workspace',type=Path)
    a=p.parse_args();print(json.dumps(resume_external_completion(a.resume_workspace,a.data) if a.resume_workspace else admit_external(a.repo,a.data,read(a.config)) if a.config else admit(a.repo,a.data,a.candidate)))
