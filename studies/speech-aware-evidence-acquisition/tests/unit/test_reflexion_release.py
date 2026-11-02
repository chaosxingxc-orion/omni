"""Reflexion release reconstructs actual native responses, FIT and memory events."""
import json
from types import SimpleNamespace
import pytest
from test_learning_candidate_release import reviewed, release  # noqa: F401
from speech_aware_evidence_acquisition.rails import learning_candidate_release as m
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical
@pytest.mark.parametrize('fault', [None, 'cumulative', 'native', 'fit', 'parent', 'order', 'event', 'claim'])
def test_actual_finalized_native_candidate_source(tmp_path, monkeypatch, fault):
    """Real adapter/journals/FIT; only endpoint, source and compose are doubles."""
    import importlib
    import shutil
    from dataclasses import asdict
    from test_reflexion_runtime import setup
    from speech_aware_evidence_acquisition.core import tfgrpo_fit, config
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
    runtime, endpoint, _, _, _, _ = setup(tmp_path, monkeypatch)
    kwargs = runtime.frozen_core_client_kwargs()
    runtime._data_root = kwargs['binding'].data_root
    runtime._explicit_run_dir = runtime._data_root / 'runs' / runtime.plan.run_id / runtime.plan.attempt_id
    shutil.copyfile(tmp_path / 'method.json', runtime._data_root / 'method.json')
    calls = []
    def transport(raw):
        calls.append(raw); body = json.loads(endpoint(raw)); body['finish_reason'] = 'stop'
        return canonical(body)
    kwargs.update(run_dir=runtime._run_dir(), transport=transport)
    runtime.frozen_core_client_kwargs = lambda: kwargs
    original_fit = importlib.reload(tfgrpo_fit).build_tfgrpo_fit_gate
    runtime._composed.values.update(transport='vllm', tfgrpo_contact_fit={
        'run_id':'REFLEXION-RELEASE-FIT', 'attempt_id':'fit-v1', 'contact_cap':1})
    runtime._session = SimpleNamespace(host='127.0.0.1', port=8200)
    runtime._gate = object(); runtime._adapter = SimpleNamespace(serving_geometry=lambda:{'slot_tokens':49152})
    runtime._transport = SimpleNamespace(render_request=lambda raw:{'model':'frozen-model',
        'messages':[{'role':'user','content':json.loads(raw)['history'][0]['content']}], 'max_tokens':100})
    runtime._gate_touch = lambda *a:None; runtime._require_exposure_row_unconsumed = lambda *a:None
    fits = []
    class Tokenizer:
        def __init__(self, **kwargs): pass
        def request_json(self, method, route, payload):
            fits.append(payload); return {'count':12, 'tokens':list(range(12))}
    monkeypatch.setattr(tfgrpo_fit, 'build_tfgrpo_fit_gate',
        lambda r, **kw: original_fit(r, client_factory=Tokenizer, **kw))
    runtime.run_phases(); runtime._client.close()
    runtime._client._adapter._attempt_handle.mark_finalized()
    data = runtime._data_root; base = runtime._run_dir()
    candidate_path = base / 'tfgrpo/learner/candidate.json'
    candidate_doc = json.loads(candidate_path.read_bytes())
    runtime._composed.config_hash = 'a' * 64
    monkeypatch.setattr(config, 'compose', lambda *a: runtime._composed)
    bundle = SimpleNamespace(manifest={'plan':asdict(runtime.plan), 'config':{
        'config_hash':'a' * 64, 'fragments':dict.fromkeys(('model','dataset','baseline','experiment'), 'fixture')}})
    if fault == 'cumulative': candidate_doc['cumulative_known_usage']['calls'] += 1
    elif fault == 'parent': candidate_doc['parent_sha256'] = '0' * 64
    elif fault == 'order': candidate_doc['memory_order'] = []
    elif fault == 'event': (base/'tfgrpo/learner/updates/0000.json').write_text('{}')
    elif fault == 'claim': candidate_doc['semantic_support_verified'] = True
    elif fault == 'native':
        journal = ContactJournal(data, runtime.plan, 'prepared-context-text')
        next(journal.root.glob('*.response.json')).write_text('{}')
    elif fault == 'fit':
        next((data / 'prepared/REFLEXION-RELEASE-FIT/fit-v1').glob('*.request-fit.json')).write_text('{}')
    args = dict(repo=tmp_path, data=data, candidate=candidate_doc,
        candidate_pin={'relpath':str(candidate_path.relative_to(data)), 'sha256':consumption.digest(candidate_path)},
        source_manifest={'relpath':str((base / (runtime.plan.run_id + '.run-manifest.json')).relative_to(data)), 'sha256':'b' * 64},
        bundle=bundle)
    if fault:
        with pytest.raises(ValueError): m.verify_reflexion_source(**args)
    else:
        result = m.verify_reflexion_source(**args)
        assert result['known_usage']['calls'] == 1 and result['historical_usage']['calls'] == 312
    assert len(calls) == len(fits) == 1  # Verification never opens a model or FIT client.


def test_reflexion_review_routes_through_existing_h2_consumption(reviewed,monkeypatch):
    from test_learning_candidate_release import put, refresh
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
    from speech_aware_evidence_acquisition.core.bundle import verify as bv
    repo,data,params,c,release,review,document=reviewed
    c.pop('sha256');c['strategy']=m.REFLEXION_STRATEGY;c['sha256']=sha(c)
    params['candidate']=put(data,'candidate.json',c)
    proposal=bv.verify_bundle(data/'manifest.json').trace_records['proposal']['org_learning_proposal']
    proposal['candidate'].update(sha256=params['candidate']['sha256'],experience_sha256=c['sha256'])
    release.update(strategy=m.REFLEXION_STRATEGY,candidate=params['candidate'])
    review['candidate']=params['candidate']
    refresh(data,params,release,review)
    seen=[]
    monkeypatch.setattr(m,'verify_reflexion_source',lambda **kwargs:seen.append(kwargs) or {'document':document})
    assert consumption.verify(params,repo=repo,data=data,model_id='model',consumer_ids=params['consumer_ids'])==c['sha256']
    assert len(seen)==1 and len(params['consumer_ids'])==312
    review['decision']='reject';refresh(data,params,release,review)
    with pytest.raises(ValueError,match='review'):
        consumption.verify(params,repo=repo,data=data,model_id='model',consumer_ids=params['consumer_ids'])


def test_reflexion_cannot_remove_review_flag_to_bypass_release():
    candidate = {'strategy': m.REFLEXION_STRATEGY}
    with pytest.raises(ValueError,match='independent semantic release'):
        consumption.verify_release({},repo=None,data=None,candidate=candidate,bundle=None)
