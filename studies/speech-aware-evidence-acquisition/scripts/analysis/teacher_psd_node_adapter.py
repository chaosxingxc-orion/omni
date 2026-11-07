"""Project recorded ACE/Reflexion candidates into one PSD node, without model calls."""
import argparse
import copy
import hashlib
import json
import importlib.util
import io
import time
from pathlib import Path

from intent_slot_workload import NodeWorkload
from tfrl_hardset_a0_workload import confined


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def verify_wire_proposals(repo, wire, collected, prepared, profile, context, request):
    """Replay the pinned executor parser, then the original proposal validator."""
    from speech_aware_evidence_acquisition.rails.external_exploration import request_payload, validate_proposals
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
    executor = Path(repo) / 'scripts/environment/configured_external_learner.py'
    if digest(executor) != prepared['executor']['sha256']:
        raise ValueError('prepared executor differs')
    module_spec = importlib.util.spec_from_file_location('teacher_wire_executor', executor)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    parsed, _ = module.read_stream(io.BytesIO(Path(wire).read_bytes()), None,
        module.ResponseDiagnostics(time.monotonic()), persist_response=False)
    response = collected['response']['response']
    choice, = parsed['choices']
    if (parsed['id'] != response['provider_response_id'] or parsed['model'] != response['model_id']
            or parsed['usage'] != response['provider_usage'] or choice['finish_reason'] != 'stop'
            or choice['message']['content'] != response['text'] or sha(request) != collected['request_sha256']):
        raise ValueError('collected response differs from original wire')
    payload = request_payload(request)
    proposals = validate_proposals(choice['message']['content'], profile, context,
        evidence_ids=[group['query_id'] for group in payload['groups']])
    if proposals != collected['proposals']:
        raise ValueError('collected proposals differ from original wire')
    return proposals


def compile_candidate(repo, data, spec):
    repo, data = Path(repo), Path(data)
    def read_pin(pin):
        path = Path(pin['path'])
        if digest(path) != pin['sha256']:
            raise ValueError('source pin mismatch')
        return json.loads(path.read_text())
    parent = read_pin(spec['parent'])
    collected = read_pin(spec['collected'])
    prepared = read_pin(spec['prepared'])
    audit = read_pin(spec['exposure_audit'])
    source = read_pin(spec['source'])
    if spec['source']['sha256'] != prepared['source']['sha256']:
        raise ValueError('historical source pin differs')
    alg, node, index = spec['algorithm'], spec['node'], spec['candidate_index']
    response = collected['response']['response']
    if (alg not in ('ace', 'reflexion') or node not in ('slot', 'decision') or
        collected['proposals']['algorithm'] != alg or response['finish_reason'] != 'stop' or
        digest(Path(spec['prepared']['path'])) != collected['prepared_sha256'] or
        collected['request_sha256'] != response['request_sha256'] or
        (alg == 'ace' and node != 'slot')):
        raise ValueError('invalid completed teacher source or node')
    wire = Path(spec['wire']['path'])
    if digest(wire) != spec['wire']['sha256'] or digest(wire) != response['wire_response_sha256']:
        raise ValueError('wire receipt mismatch')
    if source['document']['model_id'] != audit['source_model_id'] or prepared['source']['sha256'] != audit['source']['sha256']:
        raise ValueError('historical source actor mismatch')
    fold = spec['fold']
    if fold not in ('A', 'B') or parent['population']['dataset_sha256'] != audit['folds'][fold]['sha256']:
        raise ValueError('fixed fold differs')
    if len(parent['sample_ids']) != 1080 or any(audit['folds'][fold]['intersection_counts'].values()):
        raise ValueError('fold exposure or size differs')
    pipeline = parent.get('pipeline', {})
    if parent['node_arm'] != 'enhanced-p-slot-decision' or pipeline.get('protocol_version') != 'v2' or not pipeline.get('coverage_recovery'):
        raise ValueError('complete v2 pipeline required')
    if 'slot_knowledge' in parent or 'tfgrpo_experience' in parent:
        raise ValueError('parent must retain locked knowledge')
    prepared_files = {}
    for name in ('profile.json', 'context.json', 'request.json'):
        pin = spec['prepared_files'][name]
        if pin['sha256'] != prepared['files'][name]['sha256']:
            raise ValueError('prepared file pin differs')
        prepared_files[name] = read_pin(pin)
    proposals = verify_wire_proposals(repo, wire, collected, prepared, prepared_files['profile.json'],
                                      prepared_files['context.json'], prepared_files['request.json'])
    candidate = proposals['candidates'][index]
    if canonical(candidate) != spec['candidate_sha256']:
        raise ValueError('teacher candidate mismatch')
    if alg == 'reflexion':
        if (index == 0) != (node == 'decision'):
            raise ValueError('candidate target mismatch')
        memory = candidate['memory']
        addition = '\n\nReflexion guidance (apply only within the existing node contract):\n' + '\n'.join(k.title()+': '+memory[k] for k in ('trigger','action','exception')) + '\n'
    else:
        ops = candidate['operations']
        if len(ops) != 1 or ops[0]['operation'] != 'ADD':
            raise ValueError('single ACE presentation proposal required')
        op = ops[0]
        addition = '\n\nACE conditional clarification of the supplied locked knowledge cards (do not modify or add knowledge; current audio and existing contracts govern):\n' + '\n'.join(k.title()+': '+op[k] for k in ('condition','content','exception')) + '\n'
    base, _, _, view, binding = NodeWorkload(parent).derive(repo, data, parent['material'])
    if view['runtime_view_sha256'] != audit['folds'][fold]['runtime_view_sha256']:
        raise ValueError('fold audio view differs from exposure audit')
    role = 'slot' if node == 'slot' else 'solver'
    if node == 'decision' and base['phase_material'][role]['response_format_id'] != 'slurp-solver-paired-label-json-v1':
        raise ValueError('decision response schema differs')
    original = confined(repo/'configs', base['phase_material'][role]['instruction_template']).read_text()
    dest = confined(repo, spec['template_relpath'])
    if not spec['template_relpath'].startswith('configs/probes/'):
        raise ValueError('probe template required')
    text = original.rstrip() + addition
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.read_text() != text:
        raise ValueError('existing template differs')
    if not dest.exists():
        dest.write_text(text)
    result = copy.deepcopy(parent)
    result['name'] = spec['name']
    result['run_id'] = spec['run_id']
    result[node+'_policy'] = {'thinking': copy.deepcopy(base['thinking_policy'][role]), 'instruction_template': {'relpath':spec['template_relpath'], 'sha256':digest(dest)}}
    derived = NodeWorkload(result).derive(repo, data, result['material'])[0]
    expected = copy.deepcopy(base)
    expected['phase_material'][role]['instruction_template'] = spec['template_relpath'][len('configs/'):]
    expected['native_contact_fit']['run_id'] = result['run_id']+'-FIT'
    if expected != derived:
        raise ValueError('derived full baseline changed beyond target and run identity')
    return result, {'algorithm':alg,'node':node,'candidate_id':candidate['id'], 'candidate_sha256':canonical(candidate),
        'historical_source_actor':source['document']['model_id'],'runtime_model_fragment':binding['model_fragment'],
        'teacher_model':response['model_id'],'historical_source_kind':'P-to-joint-Solver; cross-task adaptation',
        'source_sha256':prepared['source']['sha256'],'collected_sha256':spec['collected']['sha256'],
        'wire_sha256':digest(wire),'parent_sha256':spec['parent']['sha256'],
        'exposure_audit_sha256':spec['exposure_audit']['sha256'], 'fold':fold,
        'fold_dataset_sha256':parent['population']['dataset_sha256'], 'fold_runtime_view_sha256':view['runtime_view_sha256'],
        'locked_knowledge_sha256':canonical(base['knowledge_stores']), 'pipeline_sha256':canonical(parent['pipeline']),
        'template_sha256':digest(dest),'intervention_kind':('ace-context-intervention-not-knowledge-construction' if alg == 'ace' else 'reflexion-node-context-intervention'),'new_model_calls':0,'candidate_accepted':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','config','out','receipt'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    workload, receipt=compile_candidate(a.repo,a.data,json.loads(a.config.read_text()))
    for path,value in ((a.out,workload),(a.receipt,receipt)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')

if __name__=='__main__': main()
