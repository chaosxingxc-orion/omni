import copy
import hashlib
import json
from pathlib import Path

import pytest
import gepa_v2_node_adapter as adapter
import intent_slot_workload as workload_module
from test_intent_slot_workload import ROOT, config
from speech_aware_evidence_acquisition.rails.node_prompt_candidate import sha, for_contract
from speech_aware_evidence_acquisition.rails.phase_material import load_phase_material_specs, render_static_instruction
from speech_aware_evidence_acquisition.rails.controllers import default_phase_program


@pytest.mark.parametrize('node,role', [('slot', 'slot'), ('decision', 'solver')])
@pytest.mark.parametrize('complete_coverage', [False, True])
def test_external_candidate_reaches_real_workload_and_keeps_handoff(tmp_path, monkeypatch, node, role, complete_coverage):
    original_read = workload_module.read_asset
    monkeypatch.setattr(workload_module, 'read_asset', lambda repo, rel: original_read(ROOT, rel))
    monkeypatch.setattr(workload_module.NodeWorkload, 'material_checked', lambda *args: (
        {'source_roster': {}}, {'roster': {'roster_id': 'fixture', 'items': [{'audio_seconds': 1}],
                                          'split_identity_hash': 'a'*64}, 'view': {}}))
    monkeypatch.setattr(workload_module.NodeWorkload, 'runtime_binding', lambda *args: {
        'model_fragment': 'bf16-thinking-tfgrpo-tp4-ctx48k-v1', 'service_session': 'fixture'})
    # NodeWorkload resolves templates in the target tree.
    (tmp_path/'configs/probes').mkdir(parents=True)
    for path in (ROOT/'configs/probes').glob('slurp_native_candidate_*'):
        (tmp_path/'configs/probes'/path.name).write_bytes(path.read_bytes())
    real_confined = adapter.confined
    def fixture_confined(root, rel):
        path = real_confined(root, rel)
        if path.exists():
            return path
        root = Path(root)
        return real_confined(ROOT/'configs' if root.name == 'configs' else ROOT, rel)
    monkeypatch.setattr(adapter, 'confined', fixture_confined)
    c = config(); c['node_arm'] = 'enhanced-p-slot-decision'; c['pipeline'] = adapter.PIPELINE.copy()
    if complete_coverage:
        c['pipeline'].update({
            'coverage_recovery': {'strategy': 'complete-original-candidate-set-v1', 'max_extra_contacts': 3},
            'perception_length_recovery': 'compact-reask-v1',
            'perception_repair_template': {'relpath': 'configs/probes/slurp_perception_length_repair_v1.txt',
                                           'sha256': 'e9baa301ffa49bfc067d9571e517e7228d2ea48171a9c18448184c4f79a74507'},
            'reference_normalizer': 'quarantine-unsupplied-v1',
            'slot_length_recovery': {'strategy': 'answer-only-compact-v1', 'answer_tokens': 4096,
                'template': {'relpath': 'configs/probes/slurp_slot_length_repair_v1.txt',
                             'sha256': '29fe78371df29113791a5ebb0373ea0d8c799456a0d154a00f23b2346b0eaba0'}}})
        for name in ('slurp_perception_length_repair_v1.txt', 'slurp_slot_length_repair_v1.txt'):
            (tmp_path/'configs/probes'/name).write_bytes((ROOT/'configs/probes'/name).read_bytes())
    thinking = {'enable_thinking': False, 'answer_reserve_tokens': 4096}
    for n in ('slot', 'decision'):
        path = tmp_path/'configs/probes'/('slurp_native_candidate_slot_v1.txt' if n == 'slot' else 'slurp_native_candidate_decision_v2.txt')
        c[n+'_policy'] = {'thinking': copy.deepcopy(thinking), 'instruction_template': {
            'relpath': path.relative_to(tmp_path).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    tokenizer = {'rendering': {'mode':'text', 'add_generation_prompt':False,
                    'add_special_tokens':False, 'truncation':False}, 'fixture': 'words'}
    count = lambda text: len(text.split())
    engine, contract, baseline = adapter.prepare(tmp_path, tmp_path, c, node, tokenizer, ['slurp:train:1'], [])
    assert for_contract(contract).node == node
    parent = engine.seed_candidate(contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
    original = (tmp_path/'configs'/baseline['phase_material'][role]['instruction_template']).read_bytes()
    seed = engine.executable_template(parent, contract=contract,
        expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
    assert seed.encode('utf-8') == original
    source_specs = load_phase_material_specs(baseline, default_phase_program(baseline['policy']))
    assert render_static_instruction(source_specs[role], tmp_path, {}, tmp_path) == seed
    assert baseline['thinking_policy'][role] == c[node+'_policy']['thinking']
    instruction = {'slot': 'Check every candidate for speech supported entities.',
                   'decision': 'Compare evidence before selecting an existing candidate ID.'}[node]
    mutation = {'schema': engine.MUTATION_SCHEMA, 'module': engine.MODULE,
        'parent_sha256': parent['candidate_sha256'], 'evidence_query_ids': ['slurp:train:1'],
        'diagnosis': 'Compare the supplied evidence.', node+'_instruction': instruction}
    text = json.dumps(mutation)
    # Exercise prepare -> recorded fake wire -> collect -> actual custody verifier.
    import external_exploration as exploration
    from test_external_exploration import job, write
    from test_external_teacher import network
    (tmp_path/'external').mkdir()
    job_path, external_root = job(tmp_path/'external')
    cfg = exploration.read(job_path)
    source_path = exploration.checked(cfg['source'])
    source = exploration.read(source_path)
    source['document']['groups'][0]['query_id'] = 'slurp:train:1'
    cfg['source'] = write(source_path, source)
    cfg['query_ids'] = ['slurp:train:1']
    cfg['context'] = write(tmp_path/'context.json', {'contract': contract, 'parent': parent})
    # The profile's template is fixed; the wire fixture never calls a teacher.
    profile = exploration.read(exploration.ROOT/'configs/learners/tfrl/gepa.json')
    cfg['profile'] = write(tmp_path/'profile.json', profile)
    # Existing profile pin uses relative paths; resolve it from its original directory.
    profile['template'] = exploration.pin(exploration.checked(profile['template'], exploration.ROOT/'configs/learners/tfrl'))
    cfg['profile'] = write(tmp_path/'profile.json', profile)
    write(job_path, cfg)
    exploration.prepare(job_path)
    _, _, caller, request = exploration.load_prepared(external_root)
    second = dict(mutation, **{node+'_instruction': instruction + ' Review the evidence.'})
    response = {'id':'fixture', 'model':caller.model_id, 'choices':[{'finish_reason':'stop',
        'message':{'content':json.dumps({'algorithm':'gepa', 'candidates':[mutation, second]})}}],
        'usage':{'prompt_tokens':100, 'completion_tokens':30}}
    network(caller, monkeypatch, [response]); caller(request)
    exploration.collect(external_root)
    source_spec = {'schema':'saea-gepa-external-mutation-v1',
        'prepared':exploration.pin(external_root/'prepared.json'),
        'collected':exploration.pin(external_root/'collected.json'), 'candidate_index':0}
    result = adapter.compile_node(tmp_path, tmp_path, c, node, contract=contract,
        expected_contract_sha256=contract['contract_sha256'], parent=parent,
        expected_parent_sha256=parent['candidate_sha256'], external_mutation=source_spec, count_tokens=count, tokenizer=tokenizer)
    derived = workload_module.NodeWorkload(result['workload']).derive(tmp_path, tmp_path, {})[0]
    expected = copy.deepcopy(baseline)
    expected['phase_material'][role]['instruction_template'] = result['publication']['instruction_template']['relpath']
    assert derived == expected
    specs = load_phase_material_specs(derived, default_phase_program(derived['policy']))
    rendered = render_static_instruction(specs[role], tmp_path, {}, tmp_path)
    assert rendered == instruction + original.decode('utf-8')[len(contract['original_instruction']):]
    assert result['workload'][node+'_policy']['thinking'] == c[node+'_policy']['thinking']
    assert result['workload'][('decision' if node == 'slot' else 'slot')+'_policy'] == c[('decision' if node == 'slot' else 'slot')+'_policy']
    assert result['workload']['pipeline'] == c['pipeline']
    assert derived['native_episode']['intent_slot_pipeline'] == baseline['native_episode']['intent_slot_pipeline']
    assert ('candidate_id' in rendered) if node == 'decision' else ('every requested candidate' in rendered)
    other = 'decision' if node == 'slot' else 'slot'
    foreign = copy.deepcopy(mutation); foreign['schema'] = adapter.PromptCandidateContract(other).MUTATION_SCHEMA
    with pytest.raises(ValueError):
        engine.validate_mutation(json.dumps(foreign), contract=contract,
            expected_contract_sha256=contract['contract_sha256'], expected_parent_sha256=parent['candidate_sha256'])
    bad = copy.deepcopy(c); bad.pop('pipeline')
    with pytest.raises(ValueError, match='explicit compatible v2'):
        adapter.prepare(tmp_path, tmp_path, bad, node, tokenizer, ['slurp:train:1'], [])
    if complete_coverage:
        bad = copy.deepcopy(c); bad['pipeline']['coverage_recovery']['max_extra_contacts'] = 4
        with pytest.raises(ValueError, match='invalid bounded Slot coverage recovery policy'):
            adapter.prepare(tmp_path, tmp_path, bad, node, tokenizer, ['slurp:train:1'], [])
        changed = copy.deepcopy(c); changed['pipeline']['coverage_recovery']['max_extra_contacts'] = 2
        with pytest.raises(ValueError, match='contract differs'):
            adapter.compile_node(tmp_path, tmp_path, changed, node, contract=contract,
                expected_contract_sha256=contract['contract_sha256'], parent=parent,
                expected_parent_sha256=parent['candidate_sha256'], external_mutation={}, count_tokens=count, tokenizer=tokenizer)
    bad_contract = copy.deepcopy(contract); bad_contract['frozen']['knowledge_sha256'] = '0' * 64
    bad_contract['contract_sha256'] = sha({k:v for k,v in bad_contract.items() if k != 'contract_sha256'})
    with pytest.raises(ValueError, match='contract differs'):
        adapter.compile_node(tmp_path, tmp_path, c, node, contract=bad_contract,
            expected_contract_sha256=bad_contract['contract_sha256'], parent=parent,
            expected_parent_sha256=parent['candidate_sha256'], external_mutation={}, count_tokens=count, tokenizer=tokenizer)


def test_full_fold_source_fields_derive_without_material_or_binding_mocks(tmp_path, monkeypatch):
    """Real Fold A 1080 roster/source shape, with isolated fake vector and runtime assets."""
    import shutil
    from tfrl_hardset_a0_registration import pin
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical
    repo, data = tmp_path/'repo', tmp_path/'data'
    repo.mkdir(); data.mkdir()
    c = json.loads((ROOT/'configs/evaluation/fold-baseline-v1/fold-a-workload.json').read_text())
    # Reuse the checked-in population, dataset and source fields; external vector bytes
    # and deployment receipt are offline fixtures with newly computed pins.
    for rel in (c['population']['roster']['relpath'],
                'configs/dataset/'+c['population']['dataset']+'.json',
                workload_module.observation.SOURCE_EXP):
        (repo/rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, repo/rel)
    shutil.copytree(ROOT/'configs/probes', repo/'configs/probes')
    source = json.loads((ROOT/c['population']['source_baseline']['relpath']).read_text())
    view = json.loads((repo/('configs/dataset/'+c['population']['dataset']+'.json')).read_text())
    for store, role in [('intent-boundaries', 'perception'), ('train-exemplars', 'solver')]:
        query = source['knowledge_stores'][store]['phase_supply'][role]['query_source']
        rel = 'vectors/'+store+'.jsonl'; path = data/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as stream:
            for row in view['items']:
                record = {'id':row['sample_id'], 'audio_sha256':row['audio_sha256'], 'source_sample_rate_hz':row['sample_rate_hz']}
                stream.write(json.dumps(record)+'\n')
        query.update(relpath=rel, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    source_path = repo/'configs/evaluation/fold-baseline-v1/source-fixture.json'
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps(source))
    c['population']['source_baseline'] = pin(repo, str(source_path.relative_to(repo)))
    material = {'schema':'saea-intent-slot-fold-material-v1',
        'source_baseline':c['population']['source_baseline'], 'source_roster':c['population']['roster'],
        'dataset':{'relpath':'configs/dataset/'+c['population']['dataset']+'.json',
                   'sha256':c['population']['dataset_sha256']}}
    material_path = repo/'configs/evaluation/fold-baseline-v1/material-fixture.json'
    material_path.write_text(json.dumps(material)); c['material'] = pin(repo, str(material_path.relative_to(repo)))
    model_name = 'bf16-thinking-tfgrpo-tp4-ctx48k-v1'
    model_path = repo/('configs/model/'+model_name+'.json'); model_path.parent.mkdir(parents=True)
    model = json.loads((ROOT/('configs/model/'+model_name+'.json')).read_text())
    model['runtime_receipt'] = 'receipts/fixture.json'; model_path.write_text(json.dumps(model))
    profile_path = repo/'configs/deployment/fixture.json'; profile_path.parent.mkdir(parents=True)
    profile_path.write_text(json.dumps({'model_fragment':model_name, 'admitted_level':32,
                                        'instances':[{'name':'fixture','cards':[0,1,2,3]}]}))
    receipt = repo/'docs/receipts/fixture.json'; receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({'deployment':{'instances':[{'instance_id':'fixture'}]}}))
    c['runtime_binding'] = {'model_fragment':model_name, 'profile':'fixture',
                            'service_session':'fixture', 'runtime_receipt':'receipts/fixture.json'}
    c['pipeline'] = adapter.PIPELINE.copy()
    for node, filename in [('slot','slurp_native_candidate_slot_v1.txt'),
                           ('decision','slurp_native_candidate_decision_v2.txt')]:
        path = repo/'configs/probes'/filename
        c[node+'_policy'] = {'thinking':{'enable_thinking':False,'answer_reserve_tokens':4096},
                             'instruction_template':pin(repo, str(path.relative_to(repo)))}
    job = workload_module.NodeWorkload(c)
    base, exp, roster, derived_view, _ = job.derive(repo, data, c['material'])
    assert job.ROWS == len(roster['items']) == derived_view['count'] == 1080
    assert exp['execution_plan']['planned_model_calls'] == 6480
    tokenizer = {'rendering':{'mode':'text','add_generation_prompt':False,
                              'add_special_tokens':False,'truncation':False}, 'fixture':'words'}
    count = lambda value: len(value.split())
    for node, role in [('slot','slot'), ('decision','solver')]:
        engine, contract, seed_base = adapter.prepare(repo, data, c, node, tokenizer, [c['sample_ids'][0]], [])
        parent = engine.seed_candidate(contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
        assert engine.executable_template(parent, contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count).encode() == (repo/'configs'/base['phase_material'][role]['instruction_template']).read_bytes()
        mutation = {'schema':engine.MUTATION_SCHEMA,'module':engine.MODULE,
                    'parent_sha256':parent['candidate_sha256'], 'evidence_query_ids':[c['sample_ids'][0]],
                    'diagnosis':'Fixture.', node+'_instruction':'Examine all provided evidence.'}
        monkeypatch.setattr(adapter, 'verify_external_mutation', lambda *args, **kwargs: {'mutation_text':canonical(mutation).decode()})
        result = adapter.compile_node(repo, data, c, node, contract=contract,
            expected_contract_sha256=contract['contract_sha256'], parent=parent,
            expected_parent_sha256=parent['candidate_sha256'], external_mutation={}, count_tokens=count, tokenizer=tokenizer)
        compiled = workload_module.NodeWorkload(result['workload']).derive(repo, data, c['material'])[0]
        expected = copy.deepcopy(seed_base)
        expected['phase_material'][role]['instruction_template'] = result['publication']['instruction_template']['relpath']
        assert compiled == expected
        assert result['workload'][node+'_policy']['thinking'] == c[node+'_policy']['thinking']
