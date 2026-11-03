"""Release tests use explicit source/transport doubles, never production candidates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from speech_aware_evidence_acquisition.rails import learning_candidate_release as m
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha
from test_tfgrpo_consumption import release, candidate, SOURCE

ROSTER = 'docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json'
DATASET = 'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'


def put(root, rel, doc):
    p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canonical(doc))
    return {'relpath':rel,'sha256':consumption.digest(p)}


@pytest.fixture
def reviewed(release, monkeypatch):
    from speech_aware_evidence_acquisition.core.bundle import verify as bv
    repo,data,params=release
    c=candidate();c.pop('sha256');c.update(strategy=m.STRATEGY,semantic_review_required=True,
        historical_source_custody={'fixture':'verified-source-double'},parent_sha256='c'*64,
        cumulative_known_usage={'calls':3,'prompt_tokens':30,'completion_tokens':15},unresolved_conflicts=[])
    c['sha256']=sha(c);params['candidate']=put(data,'candidate.json',c)
    proposal=bv.verify_bundle(data/'manifest.json').trace_records['proposal']['org_learning_proposal']
    proposal['candidate'].update(sha256=params['candidate']['sha256'],experience_sha256=c['sha256'])
    ids=[f'slurp:train:{i:06d}:real' for i in range(1000,1312)];params['consumer_ids']=ids
    rows=[{'sample_id':sid,'audio_sha256':sha(sid)} for sid in ids+[SOURCE]+[f'slurp:train:{i:06d}:real' for i in range(2000,2312)]]
    rel={'schema':m.SCHEMA,'strategy':m.STRATEGY,'candidate':params['candidate'],
        'source_manifest':params['source_manifest'],'consumer_roster':put(repo,ROSTER,{'items':[{'sample_id':sid} for sid in ids]}),
        'learning_dataset':put(repo,DATASET,{'count':625,'items':rows})}
    document={'groups':[{'query_id':SOURCE,'group_sha256':'d'*64}]}
    review={'schema':m.REVIEW_SCHEMA,'decision':'eligible-for-development-evaluation','reviewer':'independent-test-reviewer',
        'procedure':'explicit fixture assessment, not model evidence','candidate':rel['candidate'],'source_manifest':rel['source_manifest'],
        'consumer_roster':rel['consumer_roster'],'batch_sha256':c['batch_sha256'],'parent_sha256':c['parent_sha256'],
        'source_sha256':sha(c['historical_source_custody']),
        'items':{k:{'content_sha256':sha(v),'decision':'supported','evidence':[{'query_id':SOURCE,'group_sha256':'d'*64}],
                    'reason':'fixture support'} for k,v in c['items'].items()},'conflicts':{}}
    rel['review']=put(data,'review.json',review);params['candidate_release']=put(data,'release.json',rel)
    monkeypatch.setattr(m,'verify_ace_source',lambda **kw:{'document':document})
    return repo,data,params,c,rel,review,document


def refresh(data,params,rel,review):
    rel['review']=put(data,'review.json',review);params['candidate_release']=put(data,'release.json',rel)


def test_reviewed_candidate_consumed_without_mutating_candidate(reviewed):
    repo,data,params,c,*_=reviewed;before=(data/'candidate.json').read_bytes()
    assert consumption.verify(params,repo=repo,data=data,model_id='model',consumer_ids=params['consumer_ids'])==c['sha256']
    assert (data/'candidate.json').read_bytes()==before and not c['deployment_accepted']


def test_runtime_cannot_skip_review_by_bypassing_stage(reviewed):
    repo,data,params,*_=reviewed;params.pop('candidate_release')
    with pytest.raises(ValueError,match='requires independent semantic release'):
        consumption.verify(params,repo=repo,data=data,model_id='model',consumer_ids=params['consumer_ids'])


@pytest.mark.parametrize('fault',['reject','missing_item','content','foreign_evidence','parent','source','roster','empty_reviewer','empty_reason','conflict'])
def test_invalid_review_rejected_even_when_repinned(reviewed,fault):
    repo,data,params,c,rel,review,document=reviewed
    item=next(iter(review['items'].values()))
    if fault=='reject':review['decision']='reject'
    elif fault=='missing_item':review['items']={}
    elif fault=='content':item['content_sha256']='0'*64
    elif fault=='foreign_evidence':item['evidence'][0]['query_id']='slurp:train:999999:real'
    elif fault=='parent':review['parent_sha256']='0'*64
    elif fault=='source':review['source_sha256']='0'*64
    elif fault=='roster':review['consumer_roster']['sha256']='0'*64
    elif fault=='empty_reviewer':review['reviewer']=''
    elif fault=='empty_reason':item['reason']=' '
    else:review['conflicts']={'foreign':{'retained_items_affected':False,'reason':'fixture'}}
    refresh(data,params,rel,review)
    with pytest.raises(ValueError):consumption.verify(params,repo=repo,data=data,model_id='model',consumer_ids=params['consumer_ids'])


@pytest.mark.parametrize('fault',['utterance','audio','subset','order','strategy'])
def test_source_eval_isolation_and_complete_roster(reviewed,fault):
    repo,data,params,c,rel,review,document=reviewed
    view=json.loads((repo/DATASET).read_bytes());roster=json.loads((repo/ROSTER).read_bytes())
    if fault=='utterance':
        replacement=SOURCE.rsplit(':',1)[0]+':headset';old=params['consumer_ids'][0]
        params['consumer_ids'][0]=replacement;roster['items'][0]['sample_id']=replacement
        view['items'][0]['sample_id']=replacement
    elif fault=='audio':view['items'][0]['audio_sha256']=next(r['audio_sha256'] for r in view['items'] if r['sample_id']==SOURCE)
    elif fault=='subset':params['consumer_ids'].pop();roster['items'].pop()
    elif fault=='order':params['consumer_ids'].reverse()
    else:rel['strategy']='reflexion_not_implemented'
    rel['learning_dataset']=put(repo,DATASET,view);rel['consumer_roster']=put(repo,ROSTER,roster)
    review['consumer_roster']=copy.deepcopy(rel['consumer_roster']);refresh(data,params,rel,review)
    with pytest.raises(ValueError):consumption.verify(params,repo=repo,data=data,model_id='model',consumer_ids=params['consumer_ids'])


def test_stale_review_pin_rejected(reviewed):
    repo,data,params,_,_,review,_=reviewed;review['decision']='reject';put(data,'review.json',review)
    with pytest.raises(ValueError,match='changed'):consumption.verify(params,repo=repo,data=data,model_id='model',consumer_ids=params['consumer_ids'])


def test_native_source_failure_reaches_real_consumption_boundary(reviewed,monkeypatch):
    repo,data,params,*_=reviewed
    def fail(**kw):raise ValueError('native custody or cumulative costs differ')
    monkeypatch.setattr(m,'verify_ace_source',fail)
    with pytest.raises(ValueError,match='native custody'):
        consumption.verify(params,repo=repo,data=data,model_id='model',consumer_ids=params['consumer_ids'])


def test_conflict_cannot_be_claimed_resolved_by_changing_candidate(reviewed):
    _,_,_,c,rel,review,doc=reviewed
    c['unresolved_conflicts']=[{'query_id':SOURCE,'reason':'contradiction'}]
    with pytest.raises(ValueError,match='all unresolved'):m.verify_review(review,release=rel,candidate=c,document=doc)
    review['conflicts']={sha(c['unresolved_conflicts'][0]):{'retained_items_affected':True,'reason':'unresolved'}}
    with pytest.raises(ValueError,match='affects retained'):m.verify_review(review,release=rel,candidate=c,document=doc)


def test_compact_arm_cannot_drop_release(reviewed):
    params=copy.deepcopy(reviewed[2]);params['target_arm']='learned'
    compact=copy.deepcopy(reviewed[2]);compact.pop('candidate_release');compact.update(rendering='short-display-id-v1',render_manifest={'relpath':'probes/map','sha256':'d'*64})
    params['compact_params']=compact
    with pytest.raises(ValueError,match='same candidate'):consumption.validate_params(params)


@pytest.mark.parametrize('fault', [None, 'compact-v3', 'cumulative', 'native', 'fit', 'parent'])
def test_actual_finalized_native_candidate_source(tmp_path, monkeypatch, fault):
    """Real adapter/journals/FIT; only endpoint, source and compose are doubles."""
    import importlib
    import shutil
    from dataclasses import asdict
    from test_ace_runtime import setup
    from speech_aware_evidence_acquisition.core import tfgrpo_fit, config
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
    from speech_aware_evidence_acquisition.rails import ace_incremental as ace
    limits = {'reflector_contract': ace.COMPACT_CONTRACT_V3} if fault == 'compact-v3' else {}
    runtime, endpoint, _, _, _, _ = setup(tmp_path, monkeypatch, **limits)
    if fault == 'compact-v3':
        from test_ace_compact_protocol import compact_reflection, compact_delta
        def answer(payload):
            stage = json.loads(payload['history'][0]['content'])['stage']
            return json.dumps(compact_reflection() if stage == ace.REFLECT else compact_delta())
        endpoint._responder = answer
    kwargs = runtime.frozen_core_client_kwargs()
    runtime._data_root = kwargs['binding'].data_root
    runtime._explicit_run_dir = runtime._data_root / 'runs' / runtime.plan.run_id / runtime.plan.attempt_id
    shutil.copyfile(tmp_path / 'ace.json', runtime._data_root / 'ace.json')
    calls = []
    def transport(raw):
        calls.append(raw); body = json.loads(endpoint(raw)); body['finish_reason'] = 'stop'
        return canonical(body)
    kwargs.update(run_dir=runtime._run_dir(), transport=transport)
    runtime.frozen_core_client_kwargs = lambda: kwargs
    original_fit = importlib.reload(tfgrpo_fit).build_tfgrpo_fit_gate
    runtime._composed.values.update(transport='vllm', tfgrpo_contact_fit={
        'run_id':'ACE-RELEASE-FIT', 'attempt_id':'fit-v1', 'contact_cap':2})
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
    elif fault == 'native':
        journal = ContactJournal(data, runtime.plan, 'prepared-context-text')
        next(journal.root.glob('*.response.json')).write_text('{}')
    elif fault == 'fit':
        next((data / 'prepared/ACE-RELEASE-FIT/fit-v1').glob('*.request-fit.json')).write_text('{}')
    args = dict(repo=tmp_path, data=data, candidate=candidate_doc,
        candidate_pin={'relpath':str(candidate_path.relative_to(data)), 'sha256':consumption.digest(candidate_path)},
        source_manifest={'relpath':str((base / (runtime.plan.run_id + '.run-manifest.json')).relative_to(data)), 'sha256':'b' * 64},
        bundle=bundle)
    if fault not in (None, 'compact-v3'):
        with pytest.raises(ValueError): m.verify_ace_source(**args)
    else:
        result = m.verify_ace_source(**args)
        assert result['known_usage']['calls'] == 2 and result['historical_usage']['calls'] == 312
    assert len(calls) == len(fits) == 2  # Verification never opens a model or FIT client.
