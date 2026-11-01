"""CPU-only token audit of immutable experience rendering; does not release a policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import validate_experience
from speech_aware_evidence_acquisition.rails.tfgrpo_consumption import SEPARATOR


def audit(candidate,base,tokenizer):
    validate_experience(candidate,candidate['model_id'])
    items=sorted(candidate['items'].items())
    if not items:raise ValueError('nonempty candidate required')
    full='\n'.join(f'[{key}]. {value}' for key,value in items)
    short='\n'.join(f'[{i}]. {value}' for i,(_,value) in enumerate(items,1))
    body='\n'.join(value for _,value in items)
    count=lambda text:len(tokenizer.encode(text,add_special_tokens=False))
    original=base+SEPARATOR+full+'\n';compact=base+SEPARATOR+short+'\n'
    full_delta=count(original)-count(base);short_delta=count(compact)-count(base)
    referenced=[key for key,_ in items if any(key in value for _,value in items)]
    return {'schema':'saea-tfgrpo-render-cost-audit-v1','candidate_content_sha256':candidate['sha256'],
        'items':len(items),'original_template_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'short_id_template_sha256':hashlib.sha256(compact.encode()).hexdigest(),
        'value_bytes_sha256':hashlib.sha256(body.encode()).hexdigest(),
        'original_template_tokens':count(original),'base_template_tokens':count(base),
        'original_added_tokens':full_delta,'short_id_added_tokens':short_delta,
        'saved_tokens':full_delta-short_delta,'added_token_reduction_fraction':(full_delta-short_delta)/full_delta,
        'key_characters':sum(len(key) for key,_ in items),'value_characters':sum(len(value) for _,value in items),
        'id_references_inside_values':len(referenced),'values_and_order_unchanged':True,
        'new_model_calls':0,'new_tokenizer_endpoint_calls':0,'candidate_or_policy_written':False,
        'limits':['Token counts are local tokenizer counts, not complete multimodal request FIT.',
                  'Changing rendered identifiers can change model outputs; semantic or accuracy equivalence is not established.',
                  'No evidence that context shortening fixes the failed candidate or covers all-wrong groups.']}


if __name__=='__main__':
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from transformers import AutoTokenizer
    assert_execution_environment()
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','base-template','tokenizer','runtime-receipt'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();receipt=json.loads(a.runtime_receipt.read_bytes())
    if Path(receipt['model']['model_dir']).resolve()!=a.tokenizer.resolve():raise ValueError('tokenizer root differs from receipt')
    expected={entry['path']:entry['sha256'] for entry in receipt['model']['files']}
    pins={}
    for name in ('config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','added_tokens.json','special_tokens_map.json'):
        path=a.tokenizer/name
        if not path.exists():continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        matches=[key for key,value in expected.items() if key.endswith('/'+a.tokenizer.name+'/'+name) and value==digest]
        if len(matches)!=1:raise ValueError('tokenizer identity differs: '+name)
        pins[name]=digest
    if not {'tokenizer_config.json','vocab.json','merges.txt'}<=pins.keys():raise ValueError('incomplete tokenizer identity')
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    result=audit(json.loads(a.candidate.read_bytes()),a.base_template.read_text(),tokenizer)
    result.update(candidate_file_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
        base_template_sha256=hashlib.sha256(a.base_template.read_bytes()).hexdigest(),
        runtime_receipt_sha256=hashlib.sha256(a.runtime_receipt.read_bytes()).hexdigest(),tokenizer_files=pins)
    print(json.dumps(result,ensure_ascii=False,indent=2))
