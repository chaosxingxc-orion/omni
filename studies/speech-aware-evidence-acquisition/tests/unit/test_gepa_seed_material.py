import copy
import json
from pathlib import Path
import shutil
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
import gepa_seed_material as m

ROOT = Path(__file__).resolve().parents[2]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.raw(value))
    return {'root': 'repo', 'relpath': str(path), 'sha256': m.sha(path)}


def test_rooted_pin_refuses_drift_and_escape(tmp_path):
    path = tmp_path/'value.json'; put(path, {'v': 1})
    pin = {'root': 'repo', 'relpath': 'value.json', 'sha256': m.sha(path)}
    assert m.read_pin(tmp_path, tmp_path, pin)[1] == {'v': 1}
    for bad in ({**pin, 'sha256': '0'*64}, {**pin, 'relpath': '../outside'}, {**pin, 'root': 'unknown'}):
        with pytest.raises(ValueError): m.read_pin(tmp_path, tmp_path, bad)


def test_partition_keeps_original_order_and_rejects_selection_leak(tmp_path, monkeypatch):
    ids = [f'slurp:train:{n:06d}:audio-{n}' for n in range(313)]
    h1 = {'items': [{'sample_id': s} for s in ids]}
    monkeypatch.setattr(m, 'population', lambda _: (h1, {}))
    for path, doc in [(m.H1, h1), (m.H2, {'items': [{'sample_id':'slurp:train:other'}]}),
                      (f'configs/dataset/{m.DATASET}.json', {}),
                      ('configs/dataset/slurp-speech-native-test-full.json', {'items':[{'sample_id':'slurp:test:1'}]})]:
        put(tmp_path/path, doc)
    body = {'feedback_ids': ids[:233], 'selection_ids': ids[233:]}
    partition = {**body, 'partition_sha256': m.prompt.sha(body), 'input_files_sha256': {
        p: m.sha(tmp_path/p) for p in (m.H1, m.H2, f'configs/dataset/{m.DATASET}.json', 'configs/dataset/slurp-speech-native-test-full.json')}}
    chosen, excluded = m.fixed_partition(tmp_path, partition)
    assert chosen == ids[:16] and ids[233] in excluded and 'slurp:test:1' in excluded
    bad = copy.deepcopy(partition); bad['feedback_ids'][0] = ids[233]
    bad['partition_sha256'] = m.prompt.sha({k:v for k,v in bad.items() if k not in ('partition_sha256','input_files_sha256')})
    with pytest.raises(ValueError, match='partition'): m.fixed_partition(tmp_path, bad)
    partition['input_files_sha256'][m.H1] = '0'*64
    with pytest.raises(ValueError, match='identity'): m.fixed_partition(tmp_path, partition)


def prepare(tmp_path, monkeypatch):
    repo = tmp_path/'repo'; data = tmp_path/'data'; repo.mkdir(); data.mkdir()
    shutil.copytree(ROOT/'configs', repo/'configs')
    base = json.loads((ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_text())
    for component in base['native_episode']['consumption_module']['components']:
        if component['state'] == 'on':
            source = ROOT/component['record']; target = repo/component['record']
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    for relative in ('scripts/flights/nx/l338_port_pins.json', 'docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars-minus-r01.json'):
        target = repo/relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/relative, target)
    base['thinking_policy'] = {r: copy.deepcopy(m.THINKING_ARMS['baseline']) for r in ('perception', 'solver')}
    base['serving_slot_tokens'] = 49152
    model = json.loads((ROOT/'configs/model/bf16-thinking-tfgrpo-tp4-ctx48k-v1.json').read_text())
    tokenizer = data/'tokenizer'; tokenizer.mkdir()
    binding = {'runtime_receipt':'receipts/runtime.json'}
    put(repo/'docs/receipts/runtime.json', {'model':{'model_dir':str(tokenizer)}})
    docs = {'source_baseline':base, 'model':model, 'runtime_binding':binding, 'partition':{'feedback_ids':['slurp:train:1']}}
    pins = {}
    for name, doc in docs.items():
        path = repo/'configs/input'/f'{name}.json'; put(path, doc); pins[name] = m.pin(repo,path)
    spec = {'schema':'saea-gepa-seed-build-v1', **pins,
        'runtime_receipt': m.pin(repo, repo/'docs/receipts/runtime.json'), 'tokenizer_path':str(tokenizer),
        'fit_estimate':{'slot_tokens':49152,'completion_ceiling':4096,'base_prompt_tokens':9600,'tokens_per_supplied_item':120}}
    path = tmp_path/'spec.json'; put(path,spec)
    monkeypatch.setattr(m, 'runtime_binding', lambda *_:model)
    monkeypatch.setattr(m, 'fixed_partition', lambda *_:([f'slurp:train:{i}' for i in range(16)], []))
    identity = {'rendering':{'mode':'text','add_generation_prompt':False,'add_special_tokens':False,'truncation':False},'fixture':'words'}
    class Counter:
        closed = False
        def __call__(self, text): return len(text.split())
        def close(self): self.closed = True
    count = Counter(); monkeypatch.setattr(m,'load_text_token_counter',lambda _:(count,identity))
    return repo,data,path,base,model,count


def test_seed_compiles_real_native_layout_and_preserves_bytes(tmp_path, monkeypatch):
    repo,data,spec,base,model,count = prepare(tmp_path,monkeypatch)
    out = repo/'configs/gepa/seed-v1'
    result = m.build(repo,data,spec,out)
    compiled = m.read_pin(repo,data,result['compiled_baseline'])[1]
    contract = m.read_pin(repo,data,result['contract'])[1]
    material = m.read_pin(repo,data,result['template_material'])[1]
    original = repo/'configs'/base['phase_material']['solver']['instruction_template']
    assert (repo/'configs'/material['instruction_template']['relpath']).read_bytes() == original.read_bytes()
    assert compiled['policy'] == 'phase-native-module-v1-c3'
    assert compiled['knowledge_stores'] == base['knowledge_stores']
    assert compiled['thinking_policy'] == base['thinking_policy']
    assert result['feedback_probe_ids'] == [f'slurp:train:{i}' for i in range(16)]
    assert not result['native_fit_completed'] and not result['seed_is_learned']
    assert count.closed
    verified = m.verify(repo, data, result)
    assert verified['compiled_baseline'] == compiled
    bad = copy.deepcopy(result); bad['feedback_probe_ids'] = []
    with pytest.raises(ValueError, match='feedback'): m.verify(repo, data, bad)
    target = repo/result['compiled_baseline']['relpath']
    corrupted = copy.deepcopy(compiled); corrupted['knowledge_stores'] = {}
    put(target, corrupted)
    bad = copy.deepcopy(result); bad['compiled_baseline'] = m.pin(repo, target)
    with pytest.raises(ValueError, match='compiled seed'): m.verify(repo, data, bad)
    put(target, compiled)
    assert contract['frozen']['model_sha256'] == m.prompt.sha(model)
    changed = copy.deepcopy(base); changed['knowledge_stores']['extra'] = {}
    assert m.frozen_fields(repo,changed,model,{'real':'identity'})['knowledge_sha256'] != contract['frozen']['knowledge_sha256']
    with pytest.raises(ValueError, match='fresh owned'): m.build(repo,data,spec,out)


@pytest.mark.parametrize('fault',['tokenizer','thinking','geometry'])
def test_seed_rejects_changed_source_before_publish(tmp_path,monkeypatch,fault):
    repo,data,path,base,model,count = prepare(tmp_path,monkeypatch)
    spec = m.read(path)
    if fault == 'tokenizer':
        other = data/'other'; other.mkdir(); spec['tokenizer_path'] = str(other)
    elif fault == 'geometry': spec['fit_estimate']['slot_tokens'] = 24576
    else:
        base['thinking_policy']['perception']['enable_thinking'] = False
        source = repo/spec['source_baseline']['relpath']; put(source,base)
        spec['source_baseline'] = m.pin(repo,source)
    put(path,spec); out = repo/'configs/gepa/seed-v1'
    with pytest.raises(ValueError): m.build(repo,data,path,out)
    assert not out.exists()
