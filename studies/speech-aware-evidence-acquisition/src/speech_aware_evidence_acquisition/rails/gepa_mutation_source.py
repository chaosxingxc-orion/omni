"""Rebuild reflection inputs from sealed native TRAIN evidence and source pins."""
import copy
import hashlib
import json
import re
from pathlib import Path
from . import solver_prompt_candidate as prompt
from .gepa_mutation_execution import require
from .tfgrpo_source import read_pin
from .tfgrpo_learning import sha, intent_slot_reward
from .tokenizer_runtime import load_text_token_counter

SCHEMA = 'saea-gepa-mutation-source-v1'
FOCUSES = {
    'intent-slot-reasoning': 'Improve intention and slot reasoning. Diagnose whether the predicted intent incorrectly constrains slots; preserve explicit semantic evidence and distinguish a missing slot from a later delivery filter. Do not change the ontology, P, or normalization.',
    'evidence-use': 'Improve knowledge consumption. Diagnose whether evidence was ignored, overgeneralized, contradicted by the utterance, or improperly filtered. State applicability conditions and evidence priority; do not invent new knowledge or change retrieval.'}


def historical_pin(pin,repo,data):
    require(isinstance(pin,dict) and set(pin)=={'root','relpath','sha256'}
        and pin['root'] in ('repo','data'),'exact historical seed pin required')
    return read_pin({**pin,'root':'repository' if pin['root']=='repo' else 'data'},
        repo_root=repo,data_root=data)


def source_repository(spec, *, repo, data):
    """Resolve an explicit immutable historical tree separately from live serving."""
    pin=spec.get('historical_repository')
    if pin is None:return Path(repo)
    require(isinstance(pin,dict) and set(pin)=={'root','relpath','commit'}
        and pin['root']=='data' and isinstance(pin['relpath'],str)
        and isinstance(pin['commit'],str) and re.fullmatch('[a-f0-9]{40}',pin['commit']),
        'exact pinned historical repository required')
    relative=Path(pin['relpath']);root=Path(data).resolve()
    require(not relative.is_absolute() and '..' not in relative.parts,
        'historical repository must remain within data root')
    target=root/relative
    require(target.is_dir() and target.resolve()==target, 'canonical historical repository required')
    from ..core.bundle.git_commit import require_clean_study_repo
    require(require_clean_study_repo(target)==pin['commit'],'historical repository commit changed')
    return target


def seed_contract(spec,*,repo,data):
    """Verify archived seed science without requiring its obsolete live topology."""
    doc=read_pin(spec,repo_root=repo,data_root=data)
    require(doc.get('schema')=='saea-gepa-seed-material-v1' and doc.get('seed_is_learned') is False,
        'exact historical seed material required')
    sources={k:historical_pin(p,repo,data) for k,p in doc['source_pins'].items()}
    require(set(sources)=={'source_baseline','model','runtime_binding','runtime_receipt','partition'},
        'complete archived seed sources required')
    material={k:historical_pin(doc[k],repo,data) for k in ('contract','candidate','tokenizer_identity')}
    contract,parent=material['contract'],material['candidate'];base=sources['source_baseline']
    prompt.validate_contract(contract,expected_contract_sha256=contract['contract_sha256'])
    partition=sources['partition'];part={k:v for k,v in partition.items() if k not in ('partition_sha256','input_files_sha256')}
    require(prompt.sha(part)==partition['partition_sha256']
        and contract['feedback_query_ids']==sorted(partition['feedback_ids'])
        and not set(contract['feedback_query_ids']) & set(partition['selection_ids'])
        and set(partition['selection_ids'])<=set(contract['excluded_query_ids']), 'fixed feedback partition changed')
    template_pin=doc['source_solver_template'];root=Path(repo if template_pin['root']=='repo' else data).resolve()
    path=(root/template_pin['relpath']).resolve()
    require(path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest()==template_pin['sha256'],
        'historical source Solver template changed')
    original=path.read_text();first,rest=original.split('\n\n',1)
    require(contract['base_template']==prompt.BLOCK_START+first+prompt.BLOCK_END+'\n\n'+rest,
        'seed contract does not preserve original Solver template')
    # Reconstruct the same frozen scientific identity from pinned historical bytes.
    templates={}
    for role,phase in base['phase_material'].items():
        target=(Path(repo)/'configs'/phase['instruction_template']).resolve()
        require(target.is_relative_to((Path(repo)/'configs').resolve()),'confined source template required')
        templates[role]={'definition':phase,'template_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    model=sources['model'];frozen={
        'knowledge_sha256':prompt.sha(base['knowledge_stores']),
        'perception_sha256':prompt.sha({'material':templates['perception'],'role':base['native_episode']['roles']['perception'],'thinking':base['thinking_policy']['perception']}),
        'supply_sha256':prompt.sha({'knowledge_stores':base['knowledge_stores'],'preparation_source_receipts':base.get('preparation_source_receipts',[])}),
        'model_sha256':prompt.sha(model),
        'decoding_sha256':prompt.sha({'model':{k:v for k,v in model.items() if k in ('decoding_params','reasoning_budget_tokens','answer_reserve_tokens','enable_thinking','reasoning_context_policy','forced_close_message_sha256','prompt_order','reasoning_content_guard')},'thinking_policy':base['thinking_policy']}),
        'roles_sha256':prompt.sha({'native_episode':base['native_episode'],'phase_material':templates}),
        'tools_sha256':prompt.sha(base.get('phase_tools',{})),
        'output_schema_sha256':prompt.sha({'solver_template_sha256':templates['solver']['template_sha256'],'response_format_id':base['phase_material']['solver'].get('response_format_id'),'fill_params':base['phase_material']['solver'].get('fill_params',{})}),
        'tokenizer_sha256':prompt.sha(material['tokenizer_identity'])}
    require(contract['frozen']==frozen,'archived frozen seed scientific identity changed')
    count,tokenizer=load_text_token_counter(Path(sources['runtime_receipt']['model']['model_dir']))
    try:
        require(tokenizer==material['tokenizer_identity'],'mutation tokenizer differs from seed')
        expected=prompt.seed_candidate(contract=contract,expected_contract_sha256=contract['contract_sha256'],count_tokens=count)
        require(parent==expected,'historical seed parent changed')
    finally:
        if hasattr(count,'close'):count.close()
    return {'document':doc,'contract':contract,'parent':parent,'tokenizer_identity':tokenizer,**sources}


def load_source(spec,*,repo,data):
    from ..core.bundle import verify_bundle
    from ..data.lock import load_lock, umbrella_lock_path
    from ..data.slurp import load_slurp_train_material, _governed_roots
    from .hashing import content_hash
    from .tfrl_bank_request_audit import checked_control
    fields={'schema','seed_material','source_manifest','feedback_query_ids','focus'}
    require(isinstance(spec,dict) and fields<=set(spec)<=fields|{'historical_repository','mutation_prompt_policy'} and spec['schema']==SCHEMA
        and spec['focus'] in FOCUSES,'exact configured mutation source required')
    policy=spec.get('mutation_prompt_policy')
    require(policy is None or policy=='explicit-feedback-query-ids-v1','unsupported mutation prompt policy')
    repo=source_repository(spec,repo=repo,data=data)
    seed=seed_contract(spec['seed_material'],repo=repo,data=data)
    ids=spec['feedback_query_ids'];require(isinstance(ids,list) and 0<len(ids)<=16
        and ids==sorted(set(ids)) and set(ids)<=set(seed['contract']['feedback_query_ids']),
        'bounded fixed feedback-only mutation input required')
    manifest=read_pin(spec['source_manifest'],repo_root=repo,data_root=data)
    root=Path(repo if spec['source_manifest']['root']=='repository' else data)
    bundle=verify_bundle(root/spec['source_manifest']['relpath'])
    require(bundle.attempt_state=='FINALIZED' and set(ids)<=set(bundle.split_ids)
        and bundle.manifest['plan']['split_role']=='discovery','finalized TRAIN source required')
    from ..core.config import compose
    fragments=manifest['config']['fragments']
    composed=compose(Path(repo),*(fragments[k] for k in ('model','dataset','baseline','experiment')))
    require(composed.config_hash==manifest['config']['config_hash'],'historical source config changed')
    for key in ('knowledge_stores','phase_material','thinking_policy','phase_tools','preparation_source_receipts'):
        require(composed.values.get(key)==seed['source_baseline'].get(key),'source changed frozen '+key)
    actual_model=Path(repo)/'configs/model'/(fragments['model']+'.json')
    require(json.loads(actual_model.read_bytes())==seed['model'],'source model differs from archived seed')
    score_meta=manifest['scores'];score_path=Path(score_meta['path'])
    require(hashlib.sha256(score_path.read_bytes()).hexdigest()==score_meta['sha256'],'sealed scores changed')
    scores=json.loads(score_path.read_bytes());require('slurp-slu-official-train' in scores,'official TRAIN source scores required')
    score_doc=scores['slurp-slu-official-train'];rows={r['sample_id']:r for r in score_doc['per_sample']}
    require(set(rows)==set(bundle.split_ids) and score_doc['aggregate']['identity_check']['passed'] is True,
        'complete official source scoring required')
    lock=load_lock(umbrella_lock_path(Path(repo)));_,_,metadata=_governed_roots(lock,Path(data))
    gold_path=metadata/'train.jsonl';gold_sha=hashlib.sha256(gold_path.read_bytes()).hexdigest();gold={}
    for item in load_slurp_train_material(lock,Path(data)):
        for rec in item.recordings:
            if rec.sample_id in ids:
                gold[rec.sample_id]={'scenario':item.scenario,'action':item.action,
                    'entities':[{'type':e.type,'filler':e.filler} for e in item.entities]}
    require(set(gold)==set(ids) and hashlib.sha256(gold_path.read_bytes()).hexdigest()==gold_sha,
        'complete immutable TRAIN gold required')
    observations=[];outputs={r['sample_id']:r for r in bundle.outputs_rows}
    original=seed['parent']['full_template'].replace(prompt.BLOCK_START,'').replace(prompt.BLOCK_END,'')
    for sid in ids:
        row=outputs[sid];visits=row['native_episode_receipt']['role_visits']
        solver=[v for v in visits if v['stage']=='solver'];perception=[v for v in visits if v['stage']=='perception']
        require(len(solver)==len(perception)==1,'one native P/S source required')
        contact=solver[0]['execution_receipt']['final_contact'];rid=contact['request_id']
        request=bundle.trace_records[rid+'/request'];response=bundle.trace_records[rid+'/response']
        payload=request['payload'];state=[h for h in payload['history'] if h.get('role')=='agent_state']
        require(payload['sample_id']==sid and content_hash(payload)==request['payload_sha256']==contact['request_sha256']
            and prompt.text_sha(response['text'])==contact['response_text_sha256']
            and response['response_sha256']==contact['response_sha256']
            and checked_control(payload)['messages'][1]['content'].startswith(original+'\n\nNative episode limits: ')
            and len(state)==1 and state[0]['source_response_sha256']==perception[0]['execution_receipt']['final_contact']['response_text_sha256'],
            'exact seed Solver instruction, source P and response custody required')
        projected=copy.deepcopy(payload)
        for key in ('raw_speech_query','speech_feature_parts'):
            if key in projected:projected[key]={'omitted_for_text_learning':True,'original_sha256':sha(projected[key])}
        observations.append({'query_id':sid,'request':projected,'raw_response':response['text'],
            'delivered_answer':row['text'],'official_score':rows[sid],'proxy_reward':intent_slot_reward(rows[sid]),
            'gold_train':gold[sid],'native_contact':contact})
    cost=manifest['cost'];historical={'calls':cost['calls_used'],'prompt_tokens':cost['prompt_tokens_total'],
        'completion_tokens':cost['completion_tokens_total']}
    require(all(type(v) is int and v>=0 for v in historical.values()),'actual historical costs required')
    return {**({'mutation_prompt_policy':policy} if policy is not None else {}),'schema':'saea-gepa-mutation-evidence-v1','contract':seed['contract'],'parent':seed['parent'],
        'feedback_query_ids':ids,'focus':spec['focus'],'observations':observations,
        'gold_train_sha256':gold_sha,'source_manifest':spec['source_manifest'],
        'historical_usage':historical,'tokenizer_identity':seed['tokenizer_identity'],
        'tokenizer_path':seed['runtime_receipt']['model']['model_dir'],
        'audio_available_to_reflector':False,'gold_runtime_use_forbidden':True,
        **({'historical_repository':copy.deepcopy(spec['historical_repository'])} if 'historical_repository' in spec else {})}


def messages(source):
    parent=source['parent'];contract=source['contract']
    system=('You improve one frozen multimodal agent Solver consumption instruction using observed TRAIN feedback. '
        'You cannot listen to audio: use only the recorded P and evidence; distinguish raw model mistakes from delivery transformations. '
        'Produce a generalizable replacement, not memorized answers. Preserve intent/slot ontology, output schema, knowledge, P, tools and decoding. '
        'Change only the consumption instruction. No braces, XML tags, template placeholders, query IDs, or per-example answers in solver_instruction. '
        'Return exactly one JSON object with fields schema,parent_sha256,module,evidence_query_ids,diagnosis,solver_instruction. '
        'No markdown. diagnosis may be detailed. evidence_query_ids must cite only supplied feedback. '
        'schema='+prompt.MUTATION_SCHEMA+'; parent_sha256='+parent['candidate_sha256']+'; module='+prompt.MODULE+'. '
        'Instruction limit '+str(contract['max_instruction_tokens'])+' tokenizer tokens. '+FOCUSES[source['focus']])
    policy=source.get('mutation_prompt_policy')
    require(policy is None or policy=='explicit-feedback-query-ids-v1','unsupported mutation prompt policy')
    if policy is not None:
        system += (' evidence_query_ids identifies feedback queries, not knowledge items. '
            'Choose a nonempty subset of allowed_feedback_query_ids, copying each selected ID exactly once. '
            'Never put evidence_id, knowledge keys, train-exemplars::value identifiers, or content hashes into that field. '
            'Use diagnosis for your analysis; do not enumerate or repeat retrieved knowledge IDs.')
    shared={};observations=[]
    def reference(value):
        digest=sha(value)
        previous=shared.setdefault(digest,value)
        require(previous==value,'reflection dedup collision')
        return {'content_sha256':digest}
    for original in source['observations']:
        row=copy.deepcopy(original);request=row.pop('request',None)
        if request is not None:
            # Lossless references avoid repeating the global ontology, framework
            # messages and identical knowledge for each feedback item. Full
            # projected requests remain bound in source custody.
            projected={}
            for key,value in request.items():
                if key in ('task_instruction','supplied_evidence','history'):
                    if key=='history':
                        history=[]
                        for item in value:
                            item=copy.deepcopy(item)
                            if item.get('role')=='control':
                                control=json.loads(item.pop('content'))
                                canonical=json.dumps(control,ensure_ascii=False,sort_keys=True,separators=(',',':'))
                                require(hashlib.sha256(canonical.encode()).hexdigest()==item['content_sha256'],
                                    'source control is not canonical for lossless dedup')
                                control['messages']=[{**msg,'content':reference(msg['content'])} for msg in control['messages']]
                                item['content_canonical_json']=control
                            elif 'content' in item:item['content']=reference(item['content'])
                            history.append(item)
                        projected[key]=history
                    elif isinstance(value,list):projected[key]=[reference(v) for v in value]
                    else:projected[key]=reference(value)
                else:projected[key]=value
            row['request']=projected
        observations.append(row)
    user={'current_instruction':parent['solver_instruction'],'fixed_surrounding_template':contract['base_template'],
        'shared_content_by_sha256':shared,'observations':observations,'audio_available_to_reflector':False,
        'content_reference_rule':'Replace each sole content_sha256 reference with the exact shared object. For content_canonical_json, serialize the expanded object as UTF-8 JSON with sorted keys, no whitespace, unescaped Unicode to recover original content. No evidence is omitted.',
        'goal':'Increase official intent and exact-slot performance while avoiding regressions. Return one changed instruction.'}
    if policy is not None:
        ids=source['feedback_query_ids']
        require(isinstance(ids,list) and ids and len(ids)==len(set(ids)),'unique allowed feedback IDs required')
        user['allowed_feedback_query_ids']=copy.deepcopy(ids)
        user['evidence_query_ids_contract']={'min_items':1,'max_items':len(ids),'unique_items':True,'source':'allowed_feedback_query_ids'}
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(user,ensure_ascii=False,sort_keys=True)}]
