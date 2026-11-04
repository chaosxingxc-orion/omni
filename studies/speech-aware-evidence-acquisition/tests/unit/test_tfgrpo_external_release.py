"""Real external transport receipt replay reaches the shared consumer without native seal."""
import copy
import json
from pathlib import Path
import shutil

import pytest
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
from speech_aware_evidence_acquisition.rails import tfgrpo_external_release as release
from speech_aware_evidence_acquisition.rails.tfgrpo_external_compile import compile_preview
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
import test_tfgrpo_external_proposals as proposal_tests

ROOT = Path(__file__).parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'relpath': path.name, 'sha256': consumption.digest(path)}


@pytest.fixture
def material(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'; data = tmp_path / 'data'; data.mkdir()
    for rel in (release.ROSTER, release.DATASET):
        dest = repo / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / rel, dest)
    rows = json.loads((repo / release.DATASET).read_text())['items']
    roster = json.loads((repo / release.ROSTER).read_text())['items']
    target = roster[0]['sample_id']; source = rows[0]['sample_id']; selected = roster[1]['sample_id']
    old = proposal_tests.fixture
    def fixture():
        groups, context, candidate = old()
        groups[0]['query_id'] = source
        candidate['evidence_query_ids'] = [source]
        candidate['relative_feedback'][0]['query_id'] = source
        return groups, context, candidate
    monkeypatch.setattr(proposal_tests, 'fixture', fixture)
    original_job = proposal_tests.job
    def job(path):
        config_path, root = original_job(path)
        cfg = json.loads(config_path.read_text()); cfg['query_ids'] = [source]
        config_path.write_text(json.dumps(cfg))
        return config_path, root
    monkeypatch.setattr(proposal_tests, 'job', job)
    spec, _ = proposal_tests.prepared_material(data, monkeypatch)
    candidate = compile_preview(spec)
    cp = write(data / 'candidate.json', candidate); sp = write(data / 'spec.json', spec)
    def repo_pin(rel): return {'relpath': rel, 'sha256': consumption.digest(repo / rel)}
    doc = {'schema': release.SCHEMA, 'candidate': cp, 'source_manifest': sp,
           'consumer_roster': repo_pin(release.ROSTER), 'learning_dataset': repo_pin(release.DATASET),
           'consumer_ids': [target], 'selection_ids': [selected]}
    review = {'schema': release.REVIEW_SCHEMA, 'decision': 'eligible-for-development-evaluation',
              'reviewer': 'independent-unit-reviewer', 'procedure': 'test-only explicit semantic assessment',
              'bindings': {k: v for k, v in doc.items() if k != 'schema'},
              'items': {k: {'content_sha256': sha(v), 'decision': 'supported',
                  'evidence': [{'query_id': source, 'group_sha256': 'a'*64}], 'reason': 'conditional evidence rule'}
                  for k, v in candidate['items'].items()}}
    doc['review'] = write(data / 'review.json', review)
    write(data / 'release.json', doc)
    base = repo / 'configs/probes/base.txt'; base.parent.mkdir(parents=True, exist_ok=True); base.write_text('Solver.')
    kwargs = dict(candidate_path=data/'candidate.json', source_manifest=data/'spec.json',
                  candidate_release=data/'release.json', repo=repo, data=data, base_template='probes/base.txt',
                  template='probes/external.txt', consumer_ids=[target], model_id='frozen-actor')
    return kwargs, candidate, doc, review, source, selected


def test_external_release_reaches_shared_flight_without_native_seal(material, monkeypatch):
    kwargs, candidate, doc, review, source, selected = material
    def forbidden(*args, **kwargs): raise AssertionError('native seal or model dispatch called')
    monkeypatch.setattr(consumption, 'verify_learning_seal', forbidden)
    from speech_aware_evidence_acquisition.rails.external_teacher import ExternalTeacher
    monkeypatch.setattr(ExternalTeacher, '__call__', forbidden)
    params = release.prepare(**kwargs)
    values = {'native_episode': {'consumption_module': {'components': [
        {'component_id': consumption.COMPONENT, 'state': 'on', 'params': params}]}},
        'phase_material': {'solver': {'instruction_template': params['template']['relpath']}}}
    assert consumption.verify_for_flight(values, repo=kwargs['repo'], data=kwargs['data'],
        model_id=kwargs['model_id'], consumer_ids=kwargs['consumer_ids']) == candidate['sha256']
    text = (kwargs['repo']/'configs/probes/external.txt').read_text()
    assert list(candidate['items'].values())[0] in text and source not in text
    params['target_arm'] = 'learned'
    values['phase_material']['solver']['instruction_template'] = params['base_template']['relpath']
    values['native_episode']['roles'] = {'solver': {'template_variants': {'node': 'solver', 'arms': [
        {'arm_id': 'learned', 'instruction_template': {'relpath': 'configs/'+params['template']['relpath'],
          'sha256': params['template']['sha256']}, 'deliver': False}, {'arm_id': 'lock', 'deliver': True}]}}}
    assert consumption.verify_for_flight(values, repo=kwargs['repo'], data=kwargs['data'],
        model_id=kwargs['model_id'], consumer_ids=kwargs['consumer_ids']) == candidate['sha256']


@pytest.mark.parametrize('fault', [None, 'promotion', 'benefit', 'missing_rejection',
                                 'empty_uncertainty', 'unsupported', 'supported_alias', 'foreign_evidence'])
def test_diagnostic_hypothesis_is_explicit_nonpromoting_and_source_bound(material, fault):
    kwargs, candidate, doc, review, source, selected = material
    review['decision'] = 'eligible-for-hypothesis-diagnostic'
    review['diagnostic'] = {'purpose': 'falsify-source-bound-hypothesis',
        'candidate_text_unchanged': True, 'claims_benefit': False, 'automatic_promotion': False,
        'comparison': 'shared-perception-paired-unchanged-lock',
        'metrics': ['official SLU', 'slot recall'],
        'rejection_conditions': ['Reject when entity recall regresses.']}
    for item in review['items'].values():
        item.update(decision='hypothesis-for-diagnostic', uncertainties=['Audio not inspected.'],
                    competing_explanations=['Perception paraphrase may cause the false positive.'])
    item = next(iter(review['items'].values()))
    if fault == 'promotion': review['diagnostic']['automatic_promotion'] = True
    if fault == 'benefit': review['diagnostic']['claims_benefit'] = True
    if fault == 'missing_rejection': review['diagnostic']['rejection_conditions'] = []
    if fault == 'empty_uncertainty': item['uncertainties'] = []
    if fault == 'unsupported': item['decision'] = 'unsupported'
    if fault == 'supported_alias': item['decision'] = 'supported'
    if fault == 'foreign_evidence': item['evidence'][0]['query_id'] = selected
    doc['review'] = write(kwargs['data'] / 'review.json', review)
    write(kwargs['data'] / 'release.json', doc)
    if fault:
        with pytest.raises(ValueError): release.prepare(**kwargs)
        assert not (kwargs['repo'] / 'configs/probes/external.txt').exists()
    else:
        params = release.prepare(**kwargs)
        assert params['purpose'] == 'development-evaluation'
        assert consumption.verify(params, repo=kwargs['repo'], data=kwargs['data'],
            model_id=kwargs['model_id'], consumer_ids=kwargs['consumer_ids']) == candidate['sha256']
        assert candidate['deployment_accepted'] is False
        assert list(candidate['items'].values())[0] in (kwargs['repo'] / 'configs/probes/external.txt').read_text()


@pytest.mark.parametrize('fault', ['candidate', 'wire', 'missing_review', 'review_binding', 'item', 'foreign_evidence',
    'model', 'test', 'learned', 'utterance', 'audio', 'selection', 'template', 'downgrade'])
def test_external_release_rejects_changed_or_leaking_material(material, fault):
    kwargs, candidate, doc, review, source, selected = material
    data = kwargs['data']; repo = kwargs['repo']
    if fault == 'candidate':
        candidate['items'][next(iter(candidate['items']))] = 'Changed'
        candidate['sha256'] = sha({k:v for k,v in candidate.items() if k!='sha256'})
        write(data/'candidate.json', candidate)
    elif fault == 'wire':
        prepared = Path(candidate['prepared']['path'])
        (prepared.parent/'request.json').write_text('{}')
    elif fault == 'missing_review': kwargs.pop('candidate_release')
    elif fault in ('review_binding', 'item', 'foreign_evidence'):
        if fault=='review_binding': review['bindings']['consumer_ids'] = [selected]
        elif fault=='item': review['items'][next(iter(review['items']))]['decision'] = 'unsupported'
        else: review['items'][next(iter(review['items']))]['evidence'][0]['query_id'] = selected
        doc['review'] = write(data/'review.json', review); write(data/'release.json', doc)
    elif fault=='model': kwargs['model_id']='other'
    elif fault in ('test','learned','selection'):
        target = {'test':'slurp:test:0001:real', 'learned':source, 'selection':selected}[fault]
        kwargs['consumer_ids']=[target]; doc['consumer_ids']=[target]
        review['bindings']['consumer_ids']=[target]
        doc['review']=write(data/'review.json',review);write(data/'release.json',doc)
    elif fault in ('utterance','audio'):
        dataset=json.loads((repo/release.DATASET).read_text()); roster=json.loads((repo/release.ROSTER).read_text())
        target=kwargs['consumer_ids'][0]; row=next(r for r in dataset['items'] if r['sample_id']==target)
        if fault=='audio': row['audio_sha256']=next(r for r in dataset['items'] if r['sample_id']==source)['audio_sha256']
        else:
            replacement=':'.join(source.split(':')[:3])+':another-recording'
            row['sample_id']=replacement
            next(r for r in roster['items'] if r['sample_id']==target)['sample_id']=replacement
            kwargs['consumer_ids']=[replacement];doc['consumer_ids']=[replacement]
        write(repo/release.DATASET,dataset);write(repo/release.ROSTER,roster)
        for key in ('consumer_roster','learning_dataset'):
            doc[key]['sha256']=consumption.digest(repo/doc[key]['relpath'])
        review['bindings']={k:v for k,v in doc.items() if k not in ('schema','review')}
        doc['review']=write(data/'review.json',review);write(data/'release.json',doc)
    if fault in ('template','downgrade'):
        params=release.prepare(**kwargs)
        if fault=='template':(repo/'configs/probes/external.txt').write_text('Changed')
        else:params.pop('source_kind')
        with pytest.raises(ValueError):consumption.verify(params,repo=repo,data=data,model_id=kwargs['model_id'],consumer_ids=kwargs['consumer_ids'])
    else:
        with pytest.raises((ValueError,TypeError)):release.prepare(**kwargs)

@pytest.mark.parametrize('fault', [None, 'weights', 'seed', 'unbound', 'runtime'])
def test_runtime_alias_preserves_source_actor_and_rejects_science_drift(material, fault):
    kwargs,candidate,doc,review,_,_=material;repo=kwargs['repo'];data=kwargs['data']
    actor={'model_lock_key':'same-weights','model_materialization_root':'models/frozen','server_args':['--max-model-len','65536','--dtype','bfloat16'],'decoding_params':{'temperature':0,'seed':42,'max_tokens':16384},'base_url':'http://local-a','runtime_receipt':'a.json'}
    runtime=copy.deepcopy(actor);runtime.update(base_url='http://local-b',runtime_receipt='b.json');runtime['server_args'][1]='49152';runtime['decoding_params']['max_tokens']=4096
    if fault=='weights':runtime['model_lock_key']='different'
    if fault=='seed':runtime['decoding_params']['seed']=43
    pins={}
    for key,name,value in [('actor_model','frozen-actor',actor),('runtime_model','tp4-runtime',runtime)]:
        rel='configs/model/'+name+'.json';write(repo/rel,value);pins[key]={'relpath':rel,'sha256':consumption.digest(repo/rel)}
    binding=write(data/'deployment.json',{'schema':'saea-external-actor-deployment-v1',**pins})
    doc.update(runtime_model_id='tp4-runtime',deployment_binding=binding)
    review['bindings']={k:v for k,v in doc.items() if k not in ('schema','review')}
    if fault=='unbound':review['bindings'].pop('runtime_model_id')
    doc['review']=write(data/'review.json',review);write(data/'release.json',doc)
    kwargs.update(runtime_model_id='tp4-runtime',deployment_binding=data/'deployment.json')
    if fault in ('weights','seed','unbound'):
        with pytest.raises(ValueError):release.prepare(**kwargs)
        return
    params=release.prepare(**kwargs)
    assert params['model_id']==candidate['model_id']=='frozen-actor'
    assert params['runtime_model_id']=='tp4-runtime'
    if fault=='runtime':
        with pytest.raises(ValueError,match='consumer model'):
            consumption.verify(params,repo=repo,data=data,model_id='unbound-runtime',consumer_ids=kwargs['consumer_ids'])
    else:
        assert consumption.verify(params,repo=repo,data=data,model_id='tp4-runtime',consumer_ids=kwargs['consumer_ids'])==candidate['sha256']


def test_contract_control_preserves_lock_and_exact_candidate_rendering(material):
    kwargs, candidate, doc, review, *_ = material
    control = kwargs['repo']/'configs/probes/control.txt'
    control.write_text('Solver with candidate hypotheses.')
    doc.update(base_template={'relpath':'probes/base.txt','sha256':consumption.digest(kwargs['repo']/'configs/probes/base.txt')},
               contract_control={'relpath':'probes/control.txt','sha256':consumption.digest(control)})
    review['bindings']={k:v for k,v in doc.items() if k not in ('schema','review')}
    doc['review']=write(kwargs['data']/'review.json',review);write(kwargs['data']/'release.json',doc)
    params = release.prepare(**kwargs, contract_control='probes/control.txt')
    assert params['base_template']['relpath'] == 'probes/base.txt'
    assert (kwargs['repo']/'configs/probes/external.txt').read_text() == release.render(candidate, control.read_text(), kwargs['consumer_ids'])
    params['target_arm'] = 'learned'
    arms = [{'arm_id': name, 'instruction_template': {'relpath':'configs/'+params[key]['relpath'], 'sha256':params[key]['sha256']}, 'deliver':False}
            for name,key in [('learned','template'),('control','contract_control')]] + [{'arm_id':'lock','deliver':True}]
    values = {'phase_material':{'solver':{'instruction_template':'probes/base.txt'}},
              'native_episode':{'consumption_module':{'components':[{'component_id':consumption.COMPONENT,'state':'on','params':params}]},
                                'roles':{'solver':{'template_variants':{'node':'solver','arms':arms}}}}}
    assert consumption.verify_for_flight(values, repo=kwargs['repo'], data=kwargs['data'],model_id=kwargs['model_id'],consumer_ids=kwargs['consumer_ids']) == candidate['sha256']
    bad = copy.deepcopy(values); bad['native_episode']['roles']['solver']['template_variants']['arms'][1]['arm_id']='compact'
    with pytest.raises(ValueError, match='source-bound Solver arms'): consumption.verify_for_flight(bad,repo=kwargs['repo'],data=kwargs['data'],model_id=kwargs['model_id'],consumer_ids=kwargs['consumer_ids'])
    control.write_text('Changed after preparation')
    with pytest.raises(ValueError, match='artifact changed'): consumption.verify(params,repo=kwargs['repo'],data=kwargs['data'],model_id=kwargs['model_id'],consumer_ids=kwargs['consumer_ids'])
