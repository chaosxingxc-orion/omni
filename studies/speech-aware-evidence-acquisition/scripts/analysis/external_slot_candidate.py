"""Compile source-bound external slot proposals; never release or dispatch them."""
import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment'))
import configured_external_learner as learner
from speech_aware_evidence_acquisition.rails.hashing import content_hash
from speech_aware_evidence_acquisition.rails.knowledge import KnowledgeSnapshot
from speech_aware_evidence_acquisition.rails.semantic_knowledge_build import (
    AuthoredSemanticKnowledgeSpec, _slot, build_slot_contract_snapshot,
)
from speech_aware_evidence_acquisition.rails.semantic_knowledge_edits import apply_slot_value_edits


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def checked(pin):
    if set(pin) != {'path', 'sha256'}:
        raise ValueError('exact file pin required')
    path = Path(pin['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin['sha256']:
        raise ValueError('input pin changed')
    return path


def verify_response(config_pin, dynamic):
    cfg = read(checked(config_pin))
    root = Path(cfg['output_root'])
    intent, request, result = (read(root / n) for n in ('intent.json', 'request.json', 'result.json'))
    if intent != {'config': cfg, 'messages_sha256': cfg['messages']['sha256']}:
        raise ValueError('request intent differs')
    messages = read(checked(cfg['messages']))
    if messages != [{'role': 'user', 'content': messages[0]['content']}] or json.loads(messages[0]['content']) != dynamic:
        raise ValueError('dynamic construction evidence differs from sent messages')
    prompt = checked(cfg['research_prompt']).read_text()
    expected = {'model': cfg['model'], 'max_tokens': cfg['max_tokens'],
                'stream': cfg.get('stream', False),
                'messages': [{'role': 'system', 'content': prompt + learner.analysis_budget_prompt(cfg)}, *messages]}
    if expected['stream']:
        expected['stream_options'] = {'include_usage': True}
    if 'thinking' in cfg:
        expected['thinking'] = cfg['thinking']
    if request != expected or result.get('state') != 'response_received' or result.get('http_status') != 200:
        raise ValueError('request differs or no successful terminal response')
    if expected['stream']:
        wire = (root / 'response.sse').read_bytes()
        response, _ = learner.read_stream(io.BytesIO(wire), root,
            learner.ResponseDiagnostics(time.monotonic()), persist_response=False)
        if response != read(root / 'response.json'):
            raise ValueError('response differs from SSE replay')
    else:
        wire = (root / 'response.bin').read_bytes()
        response = json.loads(wire)
    wire_hash = hashlib.sha256(wire).hexdigest()
    choices = response['choices']
    if (wire_hash != result['response_sha256'] or len(choices) != 1
            or choices[0]['finish_reason'] != 'stop'
            or result['usage'] != response.get('usage')
            or result['model'] != response.get('model')):
        raise ValueError('terminal wire or cost binding differs')
    return json.loads(choices[0]['message']['content']), {
        'request_sha256': content_hash(request), 'response_sha256': wire_hash,
        'usage': result['usage'], 'model': result['model'],
        'cost_scope': 'whole teacher request; sibling candidates share this cost once',
        'elapsed_seconds': result.get('elapsed_seconds'),
    }


def compile_candidate(original, base, contract, dynamic, candidate, custody, forbidden):
    if dynamic['base_rows'] != base or dynamic['edit_contract'] != contract:
        raise ValueError('base or contract differs from teacher evidence')
    records = dynamic['records']
    groups = contract['evidence_groups']
    if set(groups) != {r['source_id'] for r in records} or any(not ids or any(not s.startswith(group + ':') for s in ids) for group, ids in groups.items()):
        raise ValueError('contract evidence differs from construction records')
    source = KnowledgeSnapshot.from_frozen_json(original)
    optimizer_sources = {s for ids in groups.values() for s in ids}
    # SLURP recordings/headsets from one utterance are not independent examples.
    identity = lambda s: ':'.join(s.split(':')[:3]) if s.startswith('slurp:') else s
    forbidden_identities = {identity(s) for s in forbidden}
    if {identity(s) for s in optimizer_sources} & forbidden_identities:
        raise ValueError('new optimizer evidence intersects forbidden samples')
    inherited_overlap = sorted(s for s in source.source_sample_ids if identity(s) in forbidden_identities)
    new_forbidden = frozenset(forbidden) - set(source.source_sample_ids)
    if source.source_partition_role != 'train':
        raise ValueError('TRAIN construction source required')
    for row in base:
        value = next(v for v in source.values if v.value_id == 'slot-contract:' + row['slot_id'])
        if value.content != _slot(row)[2] or set(value.source_sample_ids) != set(row['source_support']['source_sample_ids']) or set(value.provenance_refs) != set(row['source_support']['provenance_refs']):
            raise ValueError('typed base is not the original knowledge value')
    slots = {e['slot_id'] for e in candidate['proposal']['edits']}
    if slots != {candidate['slot_id']}:
        raise ValueError('candidate must edit exactly one declared slot')
    result = apply_slot_value_edits(base, candidate['proposal'], contract=contract,
        verified_generation_ref='artifact-run:' + custody['response_sha256'])
    rows = result['rows']
    sources = tuple(sorted({s for row in rows for s in row['source_support']['source_sample_ids']}))
    split = content_hash({'original_split': source.source_split_identity_hash,
                          'construction_pool': dynamic['pool'],
                          'construction_source_ids': sorted({s for ids in groups.values() for s in ids})})
    patch = build_slot_contract_snapshot(AuthoredSemanticKnowledgeSpec('external-slot-candidate', split, sources), rows=rows, forbidden_sample_ids=new_forbidden)
    changed_ids = {'slot-contract:' + s for s in slots}
    values = {v.value_id: v for v in patch.values}
    facets = {v.facet_id: v for v in patch.facets}
    merged_values = tuple(values[v.value_id] if v.value_id in changed_ids else v for v in source.values)
    merged_facets = tuple(facets[f.facet_id] if f.facet_id in changed_ids else f for f in source.facets)
    for before, after in zip(source.facets, merged_facets):
        if (before.facet_id, before.representation, before.parent_facet_ids) != (after.facet_id, after.representation, after.parent_facet_ids):
            raise ValueError('routing changed')
    all_sources = tuple(sorted({s for v in merged_values for s in v.source_sample_ids}))
    if {identity(s) for s in all_sources} & (forbidden_identities - {identity(s) for s in source.source_sample_ids}):
        raise ValueError('full snapshot intersects forbidden samples')
    snapshot = replace(source, snapshot_id='external-slot-candidate:' + content_hash({'source':source.content_hash(), 'proposal':candidate['proposal'], 'custody':custody})[:16],
        source_split_identity_hash=split, source_sample_ids=all_sources,
        values=merged_values, facets=merged_facets)
    return {'schema':'saea-external-slot-candidate-v1', 'admission':'candidate-only',
        'candidate':candidate, 'custody':custody, 'edit_result':result,
        'snapshot':snapshot.frozen_json(), 'original_snapshot_sha256':source.content_hash(),
        'changed_value_ids':sorted(changed_ids), 'promotion_allowed':False,
        'isolation':{'new_optimizer_forbidden_overlap':[],
                     'inherited_snapshot_forbidden_overlap':inherited_overlap,
                     'interpretation':'historical overlap retained; root must review evaluation eligibility'}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=read(a.config)
    if set(cfg) != {'schema','original','base','contract','dynamic','teacher','candidate_index','forbidden_sample_ids','output'} or cfg['schema']!='saea-external-slot-candidate-config-v1':
        raise ValueError('exact candidate compiler config required')
    original,base,contract,dynamic,forbidden=(read(checked(cfg[k])) for k in ('original','base','contract','dynamic','forbidden_sample_ids'))
    proposals,custody=verify_response(cfg['teacher'],dynamic)
    index=cfg['candidate_index']
    if type(index) is not int or index < 0 or index >= len(proposals['candidates']):
        raise ValueError('explicit candidate index required')
    result=compile_candidate(original,base,contract,dynamic,proposals['candidates'][index],custody,forbidden)
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
    for name,data in [('candidate.json',result),('snapshot.json',result['snapshot']),('rows.json',result['edit_result']['rows'])]:
        with (output/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'state':'candidate-only','changed_value_ids':result['changed_value_ids'],'snapshot_sha256':result['snapshot']['snapshot_sha256']}))


if __name__=='__main__':main()


def authored_release(candidate):
    """Round-trip every canonical slot card through the normal authored builder.

    Candidate-index routing requires the full typed source, not a proposal
    manifest. The rebuilt identity is distinct; reviewed card semantics stay
    byte-identical, and the original reviewed candidate is never overwritten.
    """
    snapshot=KnowledgeSnapshot.from_frozen_json(candidate['snapshot'])
    rows=[]
    for value in snapshot.values:
        text=value.content
        slot=text.splitlines()[0][5:-1]
        bindings=text.split('Intent bindings and semantic roles:\n- ',1)[1].split('\nExtraction cues:',1)[0].split('\n- ')
        row={'slot_id':slot,
             'intent_bindings':[{'intent_id':b.split(': ',1)[0],'semantic_role':b.split(': ',1)[1]} for b in bindings],
             'extraction_cues':text.split('Extraction cues:\n- ',1)[1].split('\nAdmissible formats:',1)[0].split('\n- '),
             'admissible_formats':text.split('Admissible formats:\n- ',1)[1].split('\nMissing behavior:',1)[0].split('\n- '),
             'missing_behavior':text.split('Missing behavior: ',1)[1].split('\nAmbiguity behavior:',1)[0],
             'ambiguity_behavior':text.split('Ambiguity behavior: ',1)[1].split('\nLimitations:',1)[0],
             'limitations':text.split('Limitations:\n- ',1)[1].split('\n- '),
             'source_support':{'source_sample_ids':list(value.source_sample_ids),'provenance_refs':list(value.provenance_refs)}}
        if _slot(row)[2]!=text:raise ValueError('canonical authored reconstruction differs')
        rows.append(row)
    rebuilt=build_slot_contract_snapshot(AuthoredSemanticKnowledgeSpec('slurp',snapshot.source_split_identity_hash,snapshot.source_sample_ids),rows=rows)
    if rebuilt.values!=snapshot.values or rebuilt.facets!=snapshot.facets:
        raise ValueError('authored rebuild changes reviewed cards or routing')
    return {'schema':'saea-authored-semantic-knowledge-source-v1','builder_policy_id':rebuilt.builder_policy_id,'source_split_identity_hash':rebuilt.source_split_identity_hash,'rows':rows},rebuilt.frozen_json()
