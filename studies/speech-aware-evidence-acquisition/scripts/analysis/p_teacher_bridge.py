"""Compile a pinned GEPA wire response into a P-only template and patch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import external_exploration as external
from gepa_external_mutation import verify_external_mutation
from speech_aware_evidence_acquisition.rails.node_prompt_candidate import for_contract, sha
from speech_aware_evidence_acquisition.rails.gepa_prompt_material import publish_template


def compile_config(cfg):
    if set(cfg) != {'schema', 'source', 'tokenizer_identity', 'tokenizer_dir', 'output_dir', 'parent_template'} or cfg['schema'] != 'saea-p-teacher-bridge-v1':
        raise ValueError('exact bridge configuration required')
    prepared = external.checked(cfg['source']['prepared'])
    context = external.read(prepared.parent / 'context.json')
    contract, parent = context['contract'], context['parent']
    receipt = verify_external_mutation(cfg['source'], contract=contract, parent=parent)
    identity = external.read(external.checked(cfg['tokenizer_identity']))
    tokenizer_dir = Path(cfg['tokenizer_dir'])
    for name, digest in identity['files'].items():
        if hashlib.sha256((tokenizer_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError('tokenizer bytes changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    engine = for_contract(contract)
    child = engine.derive_candidate(receipt['mutation_text'], parent=parent, expected_parent_sha256=parent['candidate_sha256'], contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
    out = Path(cfg['output_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    material = publish_template(out, candidate=child, contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count, tokenizer_identity=identity, expected_tokenizer_sha256=contract['frozen']['tokenizer_sha256'], parent=parent, expected_parent_sha256=parent['candidate_sha256'], mutation_text=receipt['mutation_text'])
    old = cfg['parent_template']
    parent_text = engine.executable_template(parent, contract=contract, expected_contract_sha256=contract['contract_sha256'], count_tokens=count)
    if set(old) != {'relpath', 'sha256'} or old['sha256'] != hashlib.sha256(parent_text.encode()).hexdigest():
        raise ValueError('parent template pin differs')
    pin = dict(material['instruction_template'])
    pin['relpath'] = 'configs/' + pin['relpath']
    patch = [{'op': 'test', 'path': '/perception_policy/instruction_template', 'value': old}, {'op': 'replace', 'path': '/perception_policy/instruction_template', 'value': pin}]
    result = {'schema': 'saea-p-teacher-bridge-result-v1', 'source_receipt': receipt, 'material': material, 'patch': patch, 'candidate': child, 'new_api_calls': 0, 'new_npu_calls': 0}
    for name, value in [('result.json', result), ('patch.json', patch)]:
        path = out / name
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
        if path.exists() and path.read_bytes() != data:
            raise ValueError('output already exists with different content')
        if not path.exists():
            from speech_aware_evidence_acquisition.core.atomic_file import publish_new
            publish_new(path, data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    result = compile_config(external.read(args.config))
    print(json.dumps({'candidate_sha256': result['candidate']['candidate_sha256'], 'material': result['material'], 'new_api_calls': 0}))

if __name__ == '__main__':
    main()
