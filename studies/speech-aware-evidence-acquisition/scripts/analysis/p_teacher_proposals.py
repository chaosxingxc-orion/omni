"""Offline candidate compiler. No model calls, registration, or auto-application.

Use with verified collected teacher JSON only after wire replay and source review.
Interface fixtures are never teacher results or accepted scientific candidates.
"""
from pathlib import Path
import argparse, copy, hashlib, json, math, re, sys

def sha(b):
    return hashlib.sha256(b).hexdigest()

def require(test, reason):
    if not test:
        raise ValueError(reason)

def no_sample_secrets(text, source):
    require(not re.search(r'slurp:(?:train|dev|test):|audio-\d+|(?:headset|headphone)\.flac|reference_feedback_private|reference_sentence',text,re.I),
            'sample identity or gold field in actor artifact')
    for group in source['document']['groups']:
        q = group['evidence']['query']
        require(q['sample_id'] not in text and q['media_relpath'] not in text, 'sample lookup in actor artifact')
        sentence = group['evidence']['reference_feedback_private'].get('reference_sentence','').strip()
        if len(sentence) >= 12:
            require(sentence.casefold() not in text.casefold(), 'reference transcript copied into actor artifact')

def section_spans(text):
    starts = [0]
    for heading in ('Activation boundary:', 'Confusable exclusions:', 'Uncertainty behavior:', 'Limitations:'):
        found = text.find('\n' + heading)
        require(found >= 0, 'required existing card section missing')
        starts.append(found + 1)
    require(starts == sorted(set(starts)), 'card section order malformed')
    return list(zip(starts, starts[1:] + [len(text)]))

def card_patch(operation, cards):
    required = {'operation','target_id','old','new','source_refs','semantic_equivalence'}
    require(set(operation) == required and operation['operation']=='REORDER_EXISTING_SECTIONS','exact card operation required')
    card = cards.get(operation['target_id'])
    require(card is not None and operation['old']==card['content'],'unknown card or changed old text')
    old,new,refs = operation['old'],operation['new'],operation['source_refs']
    require(isinstance(new,str) and old!=new and isinstance(refs,list) and refs,'nonempty non-noop patch required')
    require(isinstance(operation['semantic_equivalence'],str) and operation['semantic_equivalence'].strip(),'semantic rationale required')
    expected = section_spans(old)
    used = []; dest = []; proof = []
    for ref in refs:
        require(set(ref)=={'card_id','source_uri','content_sha256','old_start','old_end','new_start','new_end'},'exact source span fields required')
        require(ref['card_id']==card['id'] and ref['source_uri']==card['source_uri']
            and ref['content_sha256']==card['content_sha256']==sha(old.encode()),'card source/hash mismatch')
        a,b,c,d=(ref[k] for k in ('old_start','old_end','new_start','new_end'))
        require(all(type(v) is int for v in (a,b,c,d)) and 0<=a<b<=len(old) and 0<=c<d<=len(new),'invalid source span')
        matches = [(start, end) for start, end in expected
                   if a == start and (b == end or (end < len(old) and b == end - 1 and old[b:end] == '\n'))]
        require(len(matches)==1 and old[a:b]==new[c:d],'span must copy a complete unchanged existing section')
        start,end=matches[0]
        used.append((start,end)); dest.append((c,d,b != end))
        proof.append({'raw_old_span':[a,b], 'raw_new_span':[c,d],
                      'canonical_old_span':[start,end], 'excluded_separator':b != end})
    require(sorted(used)==expected,'all original sections must occur exactly once')
    ordered=sorted(dest)
    require(ordered[0][0]==0 and ordered[-1][1]==len(new) and all(new[x[1]:y[0]] in ('','\n') and x[1]<=y[0] for x,y in zip(ordered,ordered[1:])),
        'new content has additions, holes or overlaps')
    require(sum(1 for x,y in zip(ordered,ordered[1:]) if new[x[1]:y[0]]=='\n') == sum(item['excluded_separator'] for item in proof),
        'section separators must be preserved exactly')
    require(all(new[x[1]:y[0]] == ('' if new[x[1]-1:x[1]] == '\n' else '\n')
                for x,y in zip(ordered,ordered[1:])),
        'each section boundary must start at a line without an extra separator')
    header=next(r for r in refs if r['old_start']==0)
    require(header['new_start']==0 and new.startswith(old[:header['old_end']]),'Intent header moved or changed')
    return {'target_id':card['id'],'old_content_sha256':card['content_sha256'],'new_content_sha256':sha(new.encode()),
        'source_uri':card['source_uri'],'new':new,'source_refs':copy.deepcopy(refs),
        'derived_section_proof':{'coordinate_unit':'unicode_codepoints','sections':proof,
            'separator_convention':'single LF excluded from source section and retained between destination sections'},
        'semantic_equivalence_proof':'complete byte-identical section permutation; all original statements/conditions/exceptions retained',
        'production_status':'PROPOSAL_ONLY_SNAPSHOT_RELEASE_ADAPTER_REQUIRED'}

def retrieval_patch(candidate, parent):
    result=copy.deepcopy(parent)
    route=result['knowledge_stores']['intent-boundaries']['phase_supply']['perception']
    seen=set()
    for operation in candidate['operations']:
        require(set(operation)=={'operation','target','edits','mechanism','preserve_card_bytes'},'exact retrieval operation required')
        require(operation['operation']=='P_RETRIEVAL_CONFIG_PATCH'
            and operation['target']=='/knowledge_stores/intent-boundaries/phase_supply/perception'
            and operation['preserve_card_bytes'] is True,'only P retrieval patch allowed')
        for edit in operation['edits']:
            require(set(edit)=={'field','old','new'},'exact leaf edit required')
            field=edit['field']; require(field in {'k','tau','fuse_coverage/m','fuse_coverage/c','fuse_coverage/top_l'} and field not in seen,'forbidden or duplicate retrieval field')
            seen.add(field); pieces=field.split('/'); target=route
            for part in pieces[:-1]: target=target[part]
            leaf=pieces[-1]
            require(type(edit['old']) in (int,float) and target[leaf]==edit['old'],'old retrieval value differs')
            value=edit['new']
            require(type(value) in (int,float) and math.isfinite(value) and value!=edit['old'],'nonfinite or noop retrieval edit')
            require((0<=value<=1) if field=='tau' else (type(value) is int and 1<=value<=60),'retrieval knob outside v3 bounds')
            target[leaf]=value
    require(seen and route['fuse_coverage']['m']<=route['k'],'m must not exceed recall k')
    require(route['fuse_coverage']['top_l']<=route['fuse_coverage']['m']*route['fuse_coverage']['c'],'delivery cap exceeds possible selected values')
    expected=copy.deepcopy(result)
    expected['knowledge_stores']['intent-boundaries']['phase_supply']['perception']=copy.deepcopy(parent['knowledge_stores']['intent-boundaries']['phase_supply']['perception'])
    require(expected==parent,'non-P config changed')
    return result

def source_roles(counterexamples, groups):
    ids = {g['query_id'] for g in groups}
    successes = {g['query_id'] for g in groups if g['evidence']['reference_feedback_private']['gold_candidate_recall']}
    require(isinstance(counterexamples, list) and counterexamples and set(counterexamples) <= ids, 'counterexamples must belong to source')
    require(set(counterexamples) & successes, 'protected success reference required')
    return {'protected_success_query_ids': sorted(set(counterexamples) & successes),
            'failure_query_ids': sorted(set(counterexamples) - successes),
            'all_counterexample_query_ids': counterexamples}


def compile_one(candidate, *, method, axis, context, parent_context, source, count_tokens=None):
    from speech_aware_evidence_acquisition.rails import perception_prompt_candidate as pc
    refs=(candidate['evidence_query_ids'] if method=='reflexion' else candidate['reflection']['evidence_query_ids'])
    ids={g['query_id'] for g in source['document']['groups']}
    successes={g['query_id'] for g in source['document']['groups'] if g['evidence']['reference_feedback_private']['gold_candidate_recall']}
    require(set(refs)<=ids and refs,'non-source evidence references')
    counterexamples = candidate['validation']['counterexample_query_ids']
    roles = source_roles(counterexamples, source['document']['groups'])
    if method=='reflexion' and axis=='consumption':
        memory=candidate['memory']; mutation=memory['prompt_mutation']; text=json.dumps(mutation,ensure_ascii=False,separators=(',',':'))
        contract,parent=parent_context['contract'],parent_context['parent']
        pc.validate_mutation(text,contract=contract,expected_contract_sha256=contract['contract_sha256'],expected_parent_sha256=parent['candidate_sha256'])
        require(set(mutation['evidence_query_ids'])==set(refs),'memory mutation evidence mismatch')
        instruction=mutation['perception_instruction']
        require(all(isinstance(memory[k],str) and memory[k] and memory[k] in instruction for k in ('trigger','action','exception')),'memory must be fully expressed in the single prompt block')
        require(instruction!=parent['perception_instruction'],'noop prompt rejected')
        no_sample_secrets(instruction,source)
        before,_,after=pc._split(contract['base_template'])
        template=before.removesuffix(pc.BLOCK_START)+instruction+after.removeprefix(pc.BLOCK_END)
        output={'kind':'prompt_template','template':template,'mutation':mutation,
            'artifact_status':'OFFLINE_SYNTAX_VALIDATED_NO_TOKENIZER' if count_tokens is None else 'OFFLINE_COMPILED_WITH_ACTUAL_ACTOR_TOKENIZER',
            'production_status':'PROPOSAL_ONLY_WIRE_REVIEW_P_RUNTIME_REGISTRATION_ADMISSION_FIT_REQUIRED'}
        if count_tokens is not None:
            child=pc.derive_candidate(text,parent=parent,expected_parent_sha256=parent['candidate_sha256'],contract=contract,
                expected_contract_sha256=contract['contract_sha256'],count_tokens=count_tokens)
            compiled=pc.executable_template(child,contract=contract,expected_contract_sha256=contract['contract_sha256'],count_tokens=count_tokens,
                parent=parent,expected_parent_sha256=parent['candidate_sha256'],mutation_text=text)
            require(compiled==template,'native prompt compiler replay differs')
            output['candidate']=child
        return output
    if axis=='construction':
        operations=candidate['memory']['card_operations'] if method=='reflexion' else candidate['operations']
        require(len({o['target_id'] for o in operations})==len(operations),'duplicate card targets')
        patches=[card_patch(o,context['locked_intent_card_bodies']) for o in operations]
        return {'kind':'lossless_card_representation_patch','patches':patches,
            'production_status':'PROPOSAL_ONLY_SNAPSHOT_RELEASE_ADAPTER_REQUIRED'}
    require(method=='ace' and axis=='consumption','unsupported method/axis')
    baseline=retrieval_patch(candidate,context['parent_baseline_config'])
    return {'kind':'P_retrieval_baseline_copy','baseline':baseline,
        'source_roles': roles,
        'production_status':'PROPOSAL_ONLY_NEW_SUPPLY_PREPARE_ADMIT_FIT_P_RUNTIME_RECIPE_REQUIRED'}

def compile_document(doc, *, task, context, parent_context, source, count_tokens=None):
    import jsonschema
    from speech_aware_evidence_acquisition.rails.external_exploration import validate_proposals
    method,_,axis=task.split('-')
    jsonschema.Draft202012Validator(context['constraints']['teacher_output_schema']).validate(doc)
    validate_proposals(json.dumps(doc,ensure_ascii=False),{'algorithm':method,'min_candidates':2,'max_candidates':6},context,
        evidence_ids=[g['query_id'] for g in source['document']['groups']])
    return [compile_one(c,method=method,axis=axis,context=context,parent_context=parent_context,source=source,count_tokens=count_tokens) for c in doc['candidates']]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('source-root','context','parent-context','source','proposals','task','out'):
        parser.add_argument('--'+field,required=True)
    parser.add_argument('--actor-tokenizer')
    args=parser.parse_args(); sys.dont_write_bytecode=True
    sys.path.insert(0,str(Path(args.source_root)/'src'))
    read=lambda p:json.loads(Path(p).read_bytes())
    counter=None
    if args.actor_tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.actor_tokenizer,local_files_only=True,trust_remote_code=True)
        counter=lambda text:len(tokenizer.encode(text,add_special_tokens=False,truncation=False))
    result=compile_document(read(args.proposals),task=args.task,context=read(args.context),parent_context=read(args.parent_context),source=read(args.source),count_tokens=counter)
    out=Path(args.out);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    for n,item in enumerate(result):
        with (out/('candidate-'+str(n)+'.json')).open('x') as f:json.dump(item,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'offline_compiled_candidates':len(result),'native_admitted':False,'model_calls':0}))

if __name__=='__main__': main()
