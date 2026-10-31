"""Full native FIT joins real boundaries and accepts only receipted fallback pairs."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/analysis'))
import agent_port_fit_evidence as audit
from speech_aware_evidence_acquisition.core.preparation import _artifact_bytes


def payload(request='r1'):
    return {'request_id': request, 'sample_id': 'sample', 'media_relpath': 'speech.wav',
            'raw_speech_query': {'media_relpath': 'speech.wav', 'sha256': 'a' * 64},
            'history': [{'role': 'control', 'contact_id': 'contact'}],
            'decoding_params': {'max_tokens': 4096, 'enable_thinking': True}}


def trace(path, *, fallback=False, fault=None):
    first = payload()
    second = payload('r2')
    second['decoding_params'] = {'max_tokens': 1024, 'enable_thinking': False}
    rows = []
    for p in ([first, second] if fallback else [first]):
        key = p['request_id']
        rows += [{'channel': 'MODEL_REQUEST', 'record_id': key + '/request',
                  'payload': {'payload': p, 'payload_sha256': audit.digest(p)}},
                 {'channel': 'MODEL_WIRE_RESPONSE', 'record_id': key + '/wire-response',
                  'payload': {'request_id': key, 'sample_id': p['sample_id']}}]
    if fallback:
        rows.append({'channel': 'TOOL', 'record_id': 'contact/fallback', 'payload': {
            'schema': 'saea-thinking-answer-fallback-v1', 'agent_contact_id': 'contact',
            'attempt': 1, 'first_request_id': 'r1', 'reask_request_id': 'r2',
            'reask_enable_thinking': False, 'reask_max_tokens': 1024, 'first_finish_reason': 'length'}})
    if fault == 'missing-response': rows.pop(1)
    elif fault == 'duplicate-request': rows.append(copy.deepcopy(rows[0]))
    elif fault == 'wrong-sample': rows[1]['payload']['sample_id'] = 'other'
    elif fault == 'missing-fallback': rows.pop()
    elif fault == 'wrong-fallback': rows[-1]['payload']['reask_request_id'] = 'other'
    elif fault == 'hash': rows[0]['payload']['payload_sha256'] = 'other'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.mark.parametrize('fallback', [False, True])
def test_trace_joins_logical_contacts_and_fallback(tmp_path, fallback):
    path = tmp_path / 'trace.jsonl'
    trace(path, fallback=fallback)
    primary, count, retries = audit.trace_contacts(path, ['sample'])
    assert list(primary) == [('sample', 'contact')]
    assert count == 1 + int(fallback) and retries == int(fallback)
    assert primary[('sample', 'contact')]['boundary'] == audit.boundary_identity((payload(),))


@pytest.mark.parametrize('fault', ['missing-response', 'duplicate-request', 'wrong-sample',
                                  'missing-fallback', 'wrong-fallback', 'hash'])
def test_bad_contact_custody_refused(tmp_path, fault):
    path = tmp_path / 'trace.jsonl'
    trace(path, fallback=True, fault=fault)
    with pytest.raises(ValueError): audit.trace_contacts(path, ['sample'])


def fit_fixture(tmp_path):
    identity = {'model_fragment': 'model', 'model_fragment_sha256': 'model-hash',
                'runtime_receipt_sha256': 'runtime-hash', 'binary_sha256': 'binary', 'model_files': []}
    spec = {'run_id': 'generation', 'fit_run_id': 'fit', 'fit_attempt_id': 'attempt', 'fit_cap': 17, 'fit_session': 'i0'}
    contact = {'sample_id': 'sample', 'agent_contact_id': 'contact'}
    digest = audit.digest(contact)
    boundary = audit.boundary_identity((payload(),))
    inputs = {'native_contact': {**contact, 'contact_identity_sha256': digest, 'raw_speech_query': payload()['raw_speech_query']},
              'fit_plan': {'run_id': 'fit', 'attempt_id': 'attempt', 'planned_model_calls': 17},
              'sources': {'required_identity': identity, 'boundary_sha256': boundary}}
    output = {**contact, 'contact_identity_sha256': digest, 'session_id': 'i0',
              'session_identity_sha256': audit.digest(identity), 'generation_contacts': 0,
              'every_request_fits': True, 'contacts': 1, 'cumulative_contacts': 1, 'sample_count': 1,
              'contact_counts': {'POST /tokenize': 1}, 'boundary_sha256': boundary,
              'rows': [{'sample_id': 'sample', 'fits': True, 'margin_tokens': 100, 'measured_prompt_tokens': 20}]}
    journal = {**contact, 'contact_identity_sha256': digest, 'method': 'POST', 'route': '/tokenize',
               'run_id': 'fit', 'attempt_id': 'attempt', 'generation_run_id': 'generation',
               'session_id': 'i0', 'contact': 1}
    return tmp_path / (digest + '.request-fit.json'), spec, identity, inputs, output, journal


@pytest.mark.parametrize('fault', [None, 'runtime', 'session', 'unfit', 'boundary', 'endpoint', 'count'])
def test_native_fit_artifact_and_journal(tmp_path, fault):
    path, spec, identity, inputs, output, journal = fit_fixture(tmp_path)
    identity = copy.deepcopy(identity)
    if fault == 'runtime': identity['runtime_receipt_sha256'] = 'other'
    elif fault == 'session': output['session_id'] = 'parent'
    elif fault == 'unfit': output['rows'][0]['fits'] = False
    elif fault == 'boundary': output['boundary_sha256'] = 'other'
    elif fault == 'endpoint': journal['route'] = '/completion'
    elif fault == 'count': journal['contact'] = 2
    path.write_bytes(_artifact_bytes(kind='request-fit', inputs=inputs, output=output, contact_class='tokenizer'))
    path.with_name(path.name + '.contacts.jsonl').write_text(json.dumps(journal) + '\n')
    if fault:
        with pytest.raises(ValueError): audit.check_fit(path, spec, ['sample'], identity)
    else:
        pair, value, count, pins = audit.check_fit(path, spec, ['sample'], identity)
        assert pair == ('sample', 'contact') and count == 1 and len(pins) == 2


def test_predecessor_fit_keeps_unfinished_costs_and_scopes_attempts(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    fit = {'run_id': 'fit-old', 'attempt_id': 'old', 'contact_cap': 4}
    fit_dir = data / 'prepared' / 'fit-old' / 'old'
    fit_dir.mkdir(parents=True)
    for name in ('one', 'two'):
        (fit_dir / (name + '.request-fit.json')).write_text('{}')
        (fit_dir / (name + '.request-fit.json.contacts.jsonl')).write_text('{}')
    path = tmp_path / 'trace.jsonl'
    trace(path)
    primary = {'boundary': audit.boundary_identity((payload(),)), 'raw': payload()['raw_speech_query']}
    def check(path, spec, expected, identity):
        sample = 'sample' if path.name.startswith('one') else 'unfinished'
        assert sample in expected
        return (sample, 'contact'), primary, 1 if sample == 'sample' else 2, {}
    monkeypatch.setattr(audit, 'check_fit', check)
    sessions = {'primary_fit_session': 'i0', 'members': {'i0': {}}, 'inputs': {}}
    result = audit.audit_source(data, fit, 'generation', {'sample'}, ['sample', 'unfinished'],
                                path, sessions, predecessor=True)
    assert result['fit_contacts'] == 2
    assert result['kept_fit_contacts'] == result['generation_contacts'] == 1
    assert result['non_kept_fit_contacts'] == 1
    # Current attempts cannot hide a foreign FIT contact as an interrupted predecessor.
    with pytest.raises(AssertionError):
        audit.audit_source(data, fit, 'generation', {'sample'}, ['sample', 'unfinished'], path, sessions)


def test_predecessor_fit_still_rejects_kept_boundary_mismatch(tmp_path, monkeypatch):
    fit = {'run_id': 'fit', 'attempt_id': 'old', 'contact_cap': 2}
    directory = tmp_path / 'prepared/fit/old'
    directory.mkdir(parents=True)
    (directory / 'one.request-fit.json').write_text('{}')
    (directory / 'one.contacts.jsonl').write_text('{}')
    path = tmp_path / 'trace.jsonl'; trace(path)
    monkeypatch.setattr(audit, 'check_fit', lambda *args: (('sample', 'contact'),
                        {'boundary': 'wrong', 'raw': payload()['raw_speech_query']}, 1, {}))
    sessions = {'primary_fit_session': 'i0', 'members': {'i0': {}}, 'inputs': {}}
    with pytest.raises(ValueError, match='actual FIT boundary'):
        audit.audit_source(tmp_path, fit, 'generation', {'sample'}, ['sample'], path, sessions, predecessor=True)


def test_request_ids_are_local_to_each_attempt(tmp_path):
    old, new = tmp_path / 'old.jsonl', tmp_path / 'new.jsonl'
    trace(old); trace(new)
    # Both attempts legitimately start at r1; no cross-attempt dictionary merge.
    left = audit.trace_contacts(old, ['sample'])
    right = audit.trace_contacts(new, ['sample'])
    assert left == right and left[1] + right[1] == 2


def test_unregistered_predecessor_is_rejected(tmp_path):
    from types import SimpleNamespace
    bundle = SimpleNamespace(manifest={'predecessor_attempt': {'attempt_id': 'old'}})
    with pytest.raises(ValueError):
        audit.predecessor_source(tmp_path, tmp_path, {'attempt_id': 'new'}, {}, bundle)


@pytest.mark.parametrize('cost_only', [False, True])
@pytest.mark.parametrize('changed_serving', [False, True])
def test_build_combines_attempts_and_preserves_noncheckpoint_generation_cost(tmp_path, monkeypatch, cost_only, changed_serving):
    from types import SimpleNamespace
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    cell = {'run_id': 'generation', 'attempt_id': 'new', 'model_fragment': 'model', 'baseline_fragment': 'base'}
    doc = {'cells': [cell], 'populations': [{'ids': ['old-row', 'new-row']}], 'population': {'count': 2},
           'service_session': 'parent', 'family_budget': {'generation_calls_cap': 20}}
    base = {'arm': 'raw-speech-agent-native-tools', 'native_episode': {'enabled': True},
            'native_contact_fit': {'run_id': 'fit-new', 'attempt_id': 'new', 'contact_cap': 4}}
    for kind, name, value in [('experiment', 'experiment', doc), ('baseline', 'base', base)]:
        path = repo / 'configs' / kind / (name + '.json'); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    directory = data / 'runs/generation/new'; directory.mkdir(parents=True)
    trace = directory / 'trace.jsonl'; trace.write_text('trace')
    manifest = directory / 'generation.run-manifest.json'; manifest.write_text('{}')
    old_record = data / 'old.json'; old_record.write_text('{}')
    bundle = SimpleNamespace(split_ids={'old-row', 'new-row'}, manifest={
        'status': 'COMPLETED', 'raw_trace': {'path': str(trace), 'sha256': audit.serving.sha(trace)}})
    monkeypatch.setattr(audit, 'require_read', lambda *a: None)
    monkeypatch.setattr(audit.scored, 'verify_bundle', lambda *a: bundle)
    sessions = {'primary_fit_session': 'i0', 'members': {}, 'inputs': {}}
    monkeypatch.setattr(audit.serving, 'sessions', lambda *a, **k: sessions)
    sources = [{
        'doc': doc, 'cell': {**cell, 'attempt_id': 'old'}, 'fit': {'run_id':'fit-old','attempt_id':'old'}, 'expected': ['old-row'],
        'trace': trace, 'attempt_record': old_record, 'predecessor_commit': 'a' * 40,
        'actual': {'calls': 7}, 'pins': {}}]
    if cost_only:
        sources.append({**sources[0], 'cell': {**cell, 'attempt_id': 'middle'},
                        'expected': [], 'actual': {'calls': 4}, 'trace': None})
    monkeypatch.setattr(audit, 'predecessor_sources', lambda *a: sources)
    historical_map = None
    historical_sha = None
    if changed_serving:
        from test_historical_fit_serving import fixture
        snapshot, _, identity = fixture(tmp_path)
        sessions['members'] = {'i0': {**identity, 'runtime_receipt_sha256': 'new-runtime'}}
        for item in sources:
            item['doc'] = {**doc, 'service_session': 'old-parent'}
            item['fit'] = {'run_id': 'fit-' + item['cell']['attempt_id'], 'attempt_id': item['cell']['attempt_id']}
            fit_directory = data / 'prepared' / item['fit']['run_id'] / item['fit']['attempt_id']
            fit_directory.mkdir(parents=True)
            (fit_directory / 'one.request-fit.json').write_text(json.dumps({
                'output': {'session_id': 'old-i0'},
                'inputs': {'sources': {'required_identity': identity}}}))
        historical_map = tmp_path / 'historical-map.json'
        historical_map.write_text(json.dumps({
            'schema': audit.historical.MAP_SCHEMA, 'read_id': 'read', 'run_id': 'generation',
            'attempt_id': 'new', 'predecessors': {item['cell']['attempt_id']: {
                'predecessor_commit': item['predecessor_commit'], 'fit_run_id': item['fit']['run_id'],
                'fit_attempt_id': item['fit']['attempt_id'],
                'snapshot': {'relpath': snapshot.name, 'sha256': audit.serving.sha(snapshot)}} for item in sources}}))
        historical_sha = audit.serving.sha(historical_map)
    seen = []
    def source(data, fit, run, expected, full, trace, sessions, predecessor=False):
        assert sessions['primary_fit_session'] == ('old-i0' if predecessor and changed_serving else 'i0')
        seen.append((expected, predecessor))
        if not expected:
            return {'generation_contacts': 0, 'logical_contacts': 0, 'fit_contacts': 4,
                    'kept_fit_contacts': 0, 'non_kept_fit_contacts': 4, 'fallback_contacts': 0,
                    'primary_fit_session': 'i0', 'inputs': {}}
        return {'generation_contacts': 3 if predecessor else 2, 'logical_contacts': 1,
                'fit_contacts': 7 if predecessor else 2, 'kept_fit_contacts': 3 if predecessor else 2,
                'non_kept_fit_contacts': 4 if predecessor else 0, 'fallback_contacts': 0,
                'primary_fit_session': 'i0', 'inputs': {}}
    monkeypatch.setattr(audit, 'audit_source', source)
    result = audit.build(repo, data, 'experiment', 'read',
                         historical_map=historical_map, historical_map_sha256=historical_sha)
    assert seen == [({'old-row'}, True)] + ([(set(), True)] if cost_only else []) + [({'new-row'}, False)]
    assert result['generation_contacts'] == 5
    assert result['accounted_generation_contacts'] == 9 + 4 * cost_only
    assert result['non_kept_fit_contacts'] == 4 + 4 * cost_only and result['rows'] == 2
    if changed_serving:
        assert result['inputs'][str(historical_map)] == historical_sha
        with pytest.raises(ValueError, match='changed service requires'):
            audit.build(repo, data, 'experiment', 'read')


def test_cost_only_ancestor_retains_fit_contacts_without_complete_rows(tmp_path, monkeypatch):
    fit = {'run_id': 'fit', 'attempt_id': 'empty', 'contact_cap': 4}
    directory = tmp_path / 'prepared/fit/empty'; directory.mkdir(parents=True)
    (directory / 'one.request-fit.json').write_text('{}')
    (directory / 'one.contacts.jsonl').write_text('{}')
    monkeypatch.setattr(audit, 'check_fit', lambda *args: (('sample', 'contact'), {}, 1, {}))
    sessions = {'primary_fit_session': 'i0', 'members': {'i0': {}}, 'inputs': {}}
    result = audit.audit_source(tmp_path, fit, 'generation', set(), ['sample'], None, sessions, predecessor=True)
    assert result['generation_contacts'] == result['kept_fit_contacts'] == 0
    assert result['fit_contacts'] == result['non_kept_fit_contacts'] == 1
    with pytest.raises(ValueError, match='missing generation trace'):
        audit.audit_source(tmp_path, fit, 'generation', {'sample'}, ['sample'], None, sessions, predecessor=True)


def test_empty_trace_allowed_only_with_empty_expected_roster(tmp_path):
    path = tmp_path / 'empty.jsonl'; path.write_text('')
    assert audit.trace_contacts(path, set()) == ({}, 0, 0)
    with pytest.raises(ValueError):
        audit.trace_contacts(path, {'sample'})


def test_no_ancestry_is_supported_without_compatibility(tmp_path):
    from types import SimpleNamespace
    bundle = SimpleNamespace(manifest={})
    assert audit.predecessor_sources(tmp_path, tmp_path, {}, {}, bundle) == []


@pytest.mark.parametrize('fault', [None, 'missing-proof', 'wrong-protocol', 'wrong-cost', 'missing-retained-block'])
def test_cost_only_source_requires_pinned_checkpoint_and_executed_proof(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    import speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility as compat
    from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter, sha256_file
    cell = {'run_id': 'run', 'attempt_id': 'new', 'model_fragment': 'model'}
    old_cell = {**cell, 'attempt_id': 'old'}
    doc = {'populations': [{'ids': ['sample']}], 'service_session': 'parent'}
    old_doc = {**doc, 'cells': [old_cell]}
    base = {'native_contact_fit': {'run_id': 'fit-old', 'attempt_id': 'old', 'contact_cap': 4}}
    monkeypatch.setattr(compat, 'git', lambda repo, *a: json.dumps(old_doc if '/experiment/' in a[-1] else base).encode())
    identity = {'config_hash': 'a'*64, 'protocol_hash': 'b'*64, 'split_identity_hash': 'c'*64, 'implementation': {'runtime': 'd'*64}}
    writer = RowCheckpointWriter(tmp_path/'runs/run/old', 'run', 'old', identity=identity)
    if fault == 'missing-retained-block':
        rows = [json.dumps({'record_id': 'r1/'+kind, 'channel': channel, 'payload': {}})
                for kind, channel in [('request','MODEL_REQUEST'),('response','MODEL_RESPONSE')]]
        writer.append_row(sample_id='sample', complete=True, output_rows=[{'sample_id':'sample','request_id':'r1','text':'answer'}], trace_lines=rows)
    writer.close()
    declaration = SimpleNamespace(attempt_id='old', checkpoint_sha256=sha256_file(writer.path))
    bridge = SimpleNamespace(predecessor_commit='e'*40, predecessor_fragments={'experiment':'old','baseline':'base'}, reviewed_source_changes={}, change='contact-executor-width-v1')
    proof = {'predecessor_commit':bridge.predecessor_commit,'reviewed_source_changes':{},'change':bridge.change,
             'protocol_hash':identity['protocol_hash'],'implementation':identity['implementation']}
    if fault == 'missing-proof':proof = {}
    if fault == 'wrong-protocol':proof['protocol_hash']='f'*64
    record = tmp_path/'attempts/run/old.json';record.parent.mkdir(parents=True)
    record.write_text(json.dumps({'run_id':'run','attempt_id':'old','state':'ABORTED','actual':{'calls':-1 if fault=='wrong-cost' else 599,'audio_seconds':1569.6}}))
    bundle = SimpleNamespace(manifest_path=tmp_path/'new/run.run-manifest.json')
    if fault:
        with pytest.raises(ValueError):audit._predecessor_source(tmp_path,tmp_path,cell,doc,bundle,declaration,bridge,proof,None)
    else:
        source=audit._predecessor_source(tmp_path,tmp_path,cell,doc,bundle,declaration,bridge,proof,None)
        assert source['expected']==[] and source['trace'] is None and source['actual']['calls']==599
        assert str(writer.path) in source['pins'] and str(record) in source['pins']


@pytest.mark.parametrize('fault', [None, 'missing-proof', 'duplicate-block', 'foreign-block', 'overlap', 'duplicate-fit'])
def test_v2_resolution_checks_all_proofs_and_attempt_cost_namespaces(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core.flight import checkpoint_compatibility as compat
    from speech_aware_evidence_acquisition.core import config
    bridge = {'schema': compat.SCHEMA, 'predecessor_commit': 'a'*40,
              'predecessor_fragments': dict(model='m', dataset='d', baseline='b', experiment='e'),
              'successor_attempt_id': 'next', 'change':'contact-executor-width-v1', 'reviewed_source_changes': {}}
    declarations = [{'attempt_id': name, 'checkpoint_sha256': 'b'*64} for name in ['old','empty']]
    cell = {'attempt_id':'next', 'resume_from': declarations,
            'checkpoint_compatibility': {'schema':compat.MULTI_SCHEMA,'successor_attempt_id':'next',
                                         'bridges':{'old':bridge,'empty':bridge}}}
    blocks = [{'attempt_id':'old'}]
    if fault == 'duplicate-block':blocks.append(dict(blocks[0]))
    if fault == 'foreign-block':blocks.append({'attempt_id':'foreign'})
    bundle = SimpleNamespace(manifest={'predecessor_attempts':blocks,
                             'config':{'fragments':{},'config_hash':'config'}},
                             manifest_path=tmp_path/'manifest.json')
    (tmp_path/'flight-receipts.json').write_text('{}')
    monkeypatch.setattr(config,'compose',lambda *a,**k:SimpleNamespace(config_hash='config',values={'cells':[cell]}))
    monkeypatch.setattr(compat.MultiCheckpointCompatibility,'validate_ancestry',lambda *a:None)
    proofs = {'old':{'proof':'old'},'empty':{'proof':'empty'}}
    if fault == 'missing-proof':proofs.pop('empty')
    monkeypatch.setattr(audit.scored,'read_flight_receipts',lambda *a:{'stages':{'preflight':{
        'checkpoint_compatibility':{'schema':'saea-checkpoint-compatibility-proofs-v2','predecessors':proofs}}}})
    seen=[]
    def source(repo,data,cell,doc,bundle,declaration,item,proof,block):
        seen.append((declaration.attempt_id,proof,block))
        return {'expected':['row'] if declaration.attempt_id=='old' or fault=='overlap' else [],
                'fit':{'run_id':'fit','attempt_id':'same' if fault=='duplicate-fit' else declaration.attempt_id},'pins':{}}
    monkeypatch.setattr(audit,'_predecessor_source',source)
    if fault:
        with pytest.raises(ValueError):audit.predecessor_sources(tmp_path,tmp_path,cell,{},bundle)
    else:
        result=audit.predecessor_sources(tmp_path,tmp_path,cell,{},bundle)
        assert len(result)==2 and result[1]['expected']==[]
        assert seen==[('old',{'proof':'old'},{'attempt_id':'old'}),('empty',{'proof':'empty'},None)]


def test_actual_build_enforces_read_registration_before_loading_files(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    seen=[]
    def require(repo, read_id):
        seen.append((repo,read_id));raise ValueError('not registered')
    monkeypatch.setitem(sys.modules,'direct_port_read_manifest',SimpleNamespace(require_read=require))
    with pytest.raises(ValueError,match='not registered'):
        audit.build(tmp_path,tmp_path,'absent','read-id')
    assert seen==[(tmp_path,'read-id')]
