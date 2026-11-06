import copy
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
def test_external_candidate_reaches_real_workload_and_keeps_handoff(tmp_path, monkeypatch, node, role):
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
    tokenizer = {'rendering': {'mode':'text', 'add_generation_prompt':False,
                    'add_special_tokens':False, 'truncation':False}, 'fixture': 'words'}
    count = lambda text: len(text.split())
    engine, contract, baseline = adapter.prepare(tmp_path, tmp_path, c, node, tokenizer, ['slurp:train:1'], [])
    assert for_contract(contract).node == node
    parent = engine.seed_candidate(contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
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
    assert rendered.startswith(instruction + '\n\n')
    original = (tmp_path/'configs'/baseline['phase_material'][role]['instruction_template']).read_text()
    assert rendered.endswith(original)
    assert ('candidate_id' in rendered) if node == 'decision' else ('every requested candidate' in rendered)
    other = 'decision' if node == 'slot' else 'slot'
    foreign = copy.deepcopy(mutation); foreign['schema'] = adapter.PromptCandidateContract(other).MUTATION_SCHEMA
    with pytest.raises(ValueError):
        engine.validate_mutation(json.dumps(foreign), contract=contract,
            expected_contract_sha256=contract['contract_sha256'], expected_parent_sha256=parent['candidate_sha256'])
    bad = copy.deepcopy(c); bad.pop('pipeline')
    with pytest.raises(ValueError, match='explicit v2'):
        adapter.prepare(tmp_path, tmp_path, bad, node, tokenizer, ['slurp:train:1'], [])
    bad_contract = copy.deepcopy(contract); bad_contract['frozen']['knowledge_sha256'] = '0' * 64
    bad_contract['contract_sha256'] = sha({k:v for k,v in bad_contract.items() if k != 'contract_sha256'})
    with pytest.raises(ValueError, match='contract differs'):
        adapter.compile_node(tmp_path, tmp_path, c, node, contract=bad_contract,
            expected_contract_sha256=bad_contract['contract_sha256'], parent=parent,
            expected_parent_sha256=parent['candidate_sha256'], external_mutation={}, count_tokens=count, tokenizer=tokenizer)
