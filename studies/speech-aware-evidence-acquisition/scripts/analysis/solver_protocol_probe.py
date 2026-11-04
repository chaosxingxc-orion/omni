"""CPU-only structural output probe; never loads model weights or sends requests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from speech_aware_evidence_acquisition.rails.slurp_solver_protocol import response_format


def probe(inventory_path, output, responses=None):
    import xgrammar
    from xgrammar.testing import _is_grammar_accept_string
    from jsonschema import Draft202012Validator
    inv=json.loads(Path(inventory_path).read_text())
    rf=response_format(inv['actions_by_scenario'],inv['entity_types']);schema=rf['json_schema']['schema']
    grammar=xgrammar.Grammar.from_json_schema(json.dumps(schema))
    validator=Draft202012Validator(schema)
    cases={
        'legal_pair':{'scenario':'play','action':'music','entities':[]},
        'illegal_cross_pair':{'scenario':'music','action':'play_music','entities':[]},
        'illegal_action':{'scenario':'play','action':'play_music','entities':[]},
        'open_filler':{'scenario':'audio','action':'volume_up','entities':[{'type':'person','filler':'任意 unseen filler'}]},
        'missing_entities':{'scenario':'play','action':'music'},
    }
    checks={}
    for name,value in cases.items():
        text=json.dumps(value,ensure_ascii=False)
        checks[name]={'jsonschema':validator.is_valid(value),'grammar':_is_grammar_accept_string(grammar,text)}
    legal=json.dumps(cases['legal_pair'])
    checks['double_object']={'jsonschema':False,'grammar':_is_grammar_accept_string(grammar,legal+'\n'+legal)}
    checks['reordered_legal']={'jsonschema':True,'grammar':_is_grammar_accept_string(grammar,json.dumps(dict(reversed(list(cases['legal_pair'].items())))))}
    actual=[]
    if responses:
        for row in json.loads(Path(responses).read_text()):
            text=row['text'];valid=False
            try:valid=validator.is_valid(json.loads(text))
            except json.JSONDecodeError:pass
            normalized = None
            if valid:
                value=json.loads(text)
                normalized=json.dumps({'scenario':value['scenario'],'action':value['action'],
                    'entities':[{'type':e['type'],'filler':e['filler']} for e in value['entities']]},ensure_ascii=False)
            actual.append({'identity':row['identity'],'response_sha256':hashlib.sha256(text.encode()).hexdigest(),
                           'jsonschema':valid,'grammar':_is_grammar_accept_string(grammar,text),
                           'grammar_stripped':_is_grammar_accept_string(grammar,text.strip()),
                           'grammar_canonical_order':False if normalized is None else _is_grammar_accept_string(grammar,normalized)})
    result={'schema':'saea-solver-protocol-cpu-probe-v1','format':rf,
            'format_sha256':hashlib.sha256(json.dumps(rf,sort_keys=True).encode()).hexdigest(),
            'inventory_sha256':hashlib.sha256(Path(inventory_path).read_bytes()).hexdigest(),
            'compiled_grammar_sha256':hashlib.sha256(str(grammar).encode()).hexdigest(),
            'checks':checks,'actual_responses':actual,'new_model_calls':0,'npu_cards':0,
            'limitation':'Grammar acceptance is CPU evidence, not live thinking/forced-close/throughput verification.'}
    out=Path(output)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',required=True);p.add_argument('--output',required=True);p.add_argument('--responses');a=p.parse_args()
    r=probe(a.inventory,a.output,a.responses);print(json.dumps({k:v for k,v in r.items() if k not in ('format','actual_responses')},ensure_ascii=False))
