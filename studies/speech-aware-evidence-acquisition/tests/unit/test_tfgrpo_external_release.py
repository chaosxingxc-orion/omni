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
