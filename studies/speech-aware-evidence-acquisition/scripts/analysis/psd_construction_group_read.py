#!/usr/bin/env python3
"""Read complete registered PSD construction episodes into TF-GRPO group evidence.

This reader does not call a model or a scorer. It verifies native bundle and
score custody through the existing system feedback reader before reading any
response or construction-only gold. Incomplete or resumed sources fail closed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/analysis')]
from psd_construction_campaign import check_compiled
from intent_slot_workload import NodeWorkload
from tfrl_hardset_a0_registration import pinned
from tfgrpo_seed_groups import gold_train, text_projection
from speech_aware_evidence_acquisition.analysis.native_episode_source import verify_native_episode_source
from speech_aware_evidence_acquisition.analysis.system_feedback import load_system_feedback
from speech_aware_evidence_acquisition.analysis.system_rollout_groups import initial_speech_input_hash
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.config import effective_request_decoding_params
from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
from speech_aware_evidence_acquisition.rails.finite_policy_artifacts import _pinned_json, _confined_file
from speech_aware_evidence_acquisition.rails.hashing import content_hash
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import canonical, sha, DENSE_REWARD, intent_slot_reward, validate_experience, empty_experience


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def verify_actor_parent(job, parent, *, repo, data, binding):
    """Bind claimed parent to the actor's reviewed, target-specific consumer.

    Unreviewed semantic mappings are not equivalent to the original parent.
    Empty parent is accepted only for the unmodified native Slot/Decision
    templates; arbitrary custom node templates have no auditable memory origin.
    """
    model = job['runtime_binding']['model_fragment']
    validate_experience(parent, model)
    descriptor = job.get('tfgrpo_experience')
    if descriptor is None:
        require(binding is None and parent == empty_experience(model), 'unconditioned actor requires empty unexposed parent')
        require('slot_policy' not in job and 'decision_policy' not in job,
                'custom node policy requires independently verified parent provenance')
        return []
    require(parent != empty_experience(model), 'experienced actor cannot use empty parent')
    require(isinstance(descriptor, dict) and descriptor.get('target') in ('slot', 'decision')
            and 'mapping' not in descriptor and 'deployment_binding' not in descriptor,
            'actor parent requires direct reviewed target and same actor')
    require(isinstance(binding, dict) and Path(binding.get('repo', '')).resolve() == Path(repo).resolve()
            and Path(binding.get('data', '')).resolve() == Path(data).resolve(), 'parent source roots differ')
    source, _ = _pinned_json(data, descriptor['source'], 'actor experience source')
    params_path = Path(binding['params']['path']).resolve()
    source_path = _confined_file(data, descriptor['source']['relpath']).resolve()
    require(params_path == source_path and binding['params']['sha256'] == descriptor['source']['sha256'],
            'actor and parent must bind the same reviewed source')
    from speech_aware_evidence_acquisition.rails.tfgrpo_psd_parent import verify
    return verify(parent, descriptor['target'], model, binding)


def knowledge_index(data, values):
    """Resolve pinned snapshot value sources, preserving ambiguous IDs."""
    index = {}
    for store_id, store in values.get('knowledge_stores', {}).items():
        snapshot = store.get('snapshot') if isinstance(store, dict) else None
        if not snapshot:
            continue
        doc, _ = _pinned_json(data, snapshot, 'knowledge snapshot')
        for value in doc.get('values', []):
            vid = value['value_id']
            samples = value.get('source_sample_ids')
            if not isinstance(samples, list) or any(not isinstance(x, str) for x in samples):
                continue
            item = {'store_id':store_id, 'snapshot':snapshot,
                    'source_utterances':sorted({':'.join(x.split(':')[:3]) for x in samples})}
            index.setdefault(vid, []).append(item)
            index.setdefault(store_id+'::'+vid, []).append(item)
    return index


def verify_fit_receipts(data, native, receipts, member):
    from speech_aware_evidence_acquisition.core.preparation import load_artifact
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity
    by_id = {}
    paid = 0
    for receipt in receipts:
        cid = receipt['agent_contact_id']
        require(cid not in by_id, 'duplicate FIT contact')
        path = Path(receipt['path']).resolve()
        require(path.is_relative_to(Path(data).resolve()), 'FIT artifact escapes data root')
        doc, _ = _pinned_json(data, {'relpath':str(path.relative_to(Path(data).resolve())), 'sha256':receipt['sha256']}, 'FIT artifact')
        output = load_artifact(path, expected_sha256=receipt['sha256'], kind='request-fit', inputs=doc['inputs'])
        require(doc['contact_class']=='tokenizer' and output['every_request_fits'] is True and len(output['rows'])==1
                and output['agent_contact_id']==cid and doc['inputs']['native_contact']['agent_contact_id']==cid,
                'invalid FIT artifact')
        plan = doc['inputs']['fit_plan']
        require(plan['run_id']==member['fit_run_id'] and plan['attempt_id']==member['fit_attempt_id'], 'FIT plan identity differs')
        require(type(output['contacts']) is int and output['contacts'] > 0, 'FIT paid tokenizer count required')
        paid += output['contacts']
        by_id[cid] = (doc, output)
    for sid in native.sample_ids:
        for visit in native.phase_results[sid]['native_episode_receipt']['role_visits']:
            for contact in visit['execution_receipt']['contacts']:
                cid = contact['agent_contact_id']
                require(cid in by_id, 'missing FIT artifact')
                doc, output = by_id[cid]
                payload = native.bundle.trace_records[contact['request_id']+'/request']['payload']
                fallback = native.bundle.trace_records.get(cid+'/reasoning-fallback')
                if fallback is not None:
                    payload = native.bundle.trace_records[fallback['first_request_id']+'/request']['payload']
                require(doc['inputs']['sources']['boundary_sha256']==boundary_identity((payload,))
                        and doc['inputs']['split_identity_hash']==native.plan.split_identity_hash
                        and output['sample_id']==sid, 'FIT request identity differs')
    require(paid <= member['fit_cap'], 'FIT tokenizer cost exceeds registered cap')
    return paid


def read_groups(compiled, bindings, *, repo, data, gold, gold_pin, parent, common, learning_config_sha256, bounds, parent_binding=None):
    check_compiled(compiled)
    members = compiled['members']
    require(len(bindings) == len(members), 'one source binding per registered member required')
    ids = members[0]['job_config']['sample_ids']
    model = members[0]['job_config']['runtime_binding']['model_fragment']
    validate_experience(parent, model)
    require(set(gold) == set(ids), 'complete unchanged construction gold required')
    require(common.get('reward_mode') == DENSE_REWARD, 'registered dense reward required')
    sources, seen_runs, seen_trajectories = [], set(), set()
    reference_science = None
    for member, pin in zip(members, bindings):
        binding, _ = _pinned_json(data, pin, 'system feedback binding')
        feedback = load_system_feedback(data_root=data, repo_root=repo, binding=binding)
        native = verify_native_episode_source(
            manifest_path=_confined_file(data, binding['manifest']['relpath']),
            expected_manifest_sha256=binding['manifest']['sha256'],
            phase_artifact_path=_confined_file(data, binding['phases']['relpath']),
            expected_phase_sha256=binding['phases']['sha256'], repo_root=repo, data_root=data)
        manifest, values = native.bundle.manifest, native.composed.values
        job = member['job_config']
        verify_actor_parent(job, parent, repo=repo, data=data, binding=parent_binding)
        require((native.plan.run_id, native.plan.attempt_id) == (job['run_id'], member['generation_attempt_id']), 'member run/attempt differs')
        require(native.plan.run_id not in seen_runs, 'duplicate member run')
        seen_runs.add(native.plan.run_id)
        require(list(native.sample_ids) == ids and feedback['run_id'] == native.plan.run_id, 'complete ordered construction population required')
        require(manifest['config']['fragments'] == {'model':model, 'dataset':job['population']['dataset'], 'baseline':job['name'], 'experiment':job['name']}, 'member fragments differ')
        candidate = json.loads(resolve_asset(repo, 'docs/readiness/'+job['name']+'/candidate.json').read_bytes())
        require(candidate.get('job_config') == job and candidate.get('composed_config_hash') == manifest['config']['config_hash'], 'registered member configuration differs')
        experiment = json.loads(resolve_asset(repo, 'configs/experiment/'+job['name']+'.json').read_bytes())
        view = pinned(repo, candidate['dataset'])
        NodeWorkload(job).validate(repo, data, job['name'], candidate, experiment, view)
        require(values.get('request_decoding_overrides') == job['collection_decoding'], 'registered collection decoding differs')
        require(values.get('native_episode', {}).get('intent_slot_pipeline', {}).get('node_rails') == 'sdk-handoff-v1', 'SDK PSD policy required')
        require(values.get('native_contact_fit', {}).get('run_id') == member['fit_run_id'] and values.get('native_contact_fit', {}).get('attempt_id') == member['fit_attempt_id'], 'separate registered FIT identity required')
        # Use all composed scientific fields, excluding per-run execution identity,
        # budgets and the seed alone. Source templates and knowledge remain bound.
        science = {key:copy.deepcopy(values.get(key)) for key in (
            'policy','arm','model_id','model_path','transport','decoding_params',
            'request_decoding_overrides','native_episode','knowledge_stores',
            'phase_material','thinking_policy','phase_tools','native_evidence_projection',
            'native_evidence_projection_params','deliver_normaliser','serving_slot_tokens')}
        overrides = science.get('request_decoding_overrides', {})
        overrides.pop('seed', None)
        science['decoding_params'] = dict(effective_request_decoding_params(values))
        science['decoding_params'].pop('seed', None)
        if reference_science is None: reference_science = science
        require(science == reference_science, 'mixed model, knowledge, template or sampling policy')
        scores, _ = _pinned_json(data, feedback['resolved_auxiliary_sources']['scores'], 'official scores')
        rows = scores[binding['scorer_id']]['per_sample']
        score_index = {r['sample_id']:r for r in rows}
        require(len(score_index) == len(ids) and set(score_index) == set(ids), 'score population differs')
        require(manifest.get('status') == 'COMPLETED' and native.bundle.attempt_state == 'FINALIZED', 'source is not final')
        cost = manifest.get('cost', {})
        require(type(cost.get('calls_used')) is int and cost['calls_used'] >= 0, 'recorded generation cost required')
        require(cost['calls_used'] <= member['generation_cap'], 'generation exceeds registered cap')
        receipts, _ = _pinned_json(data, binding['flight_receipts'], 'flight receipts')
        receipts = receipts.get('receipts', receipts)
        fit = receipts['stages']['run_phases']['dispatch']['native_contact_fit_receipts']
        require(isinstance(fit, list), 'native FIT receipts required')
        contacts = [c['agent_contact_id'] for sid in ids for visit in native.phase_results[sid]['native_episode_receipt']['role_visits'] for c in visit['execution_receipt']['contacts']]
        require(len(fit) == len(contacts) and len(set(contacts)) == len(contacts) and {x['agent_contact_id'] for x in fit} == set(contacts), 'FIT contact custody differs')
        require(len(fit) <= member['fit_cap'], 'FIT exceeds registered cap')
        fit_cost = verify_fit_receipts(data, native, fit, member)
        sources.append((member, native, feedback, score_index, cost, pin, fit_cost, knowledge_index(data, values)))
    groups = []
    knowledge_diagnostics = []
    for sid in ids:
        rollouts, input_hashes, query = [], [], None
        for member, native, feedback, score_index, cost, pin, fit_count, knowledge in sources:
            episode = native.phase_results[sid]['native_episode_receipt']
            visits = episode['role_visits']
            stages = [v['stage'] for v in visits]
            require(stages[:3] == ['perception','slot','solver'], 'complete P Slot Decision episode required')
            contacts = [c for visit in visits for c in visit['execution_receipt']['contacts']]
            require(all(visit['execution_receipt']['contacts'] for visit in visits[:3]), 'missing node contact')
            trace = native.bundle.trace_records
            first = trace[contacts[0]['request_id']+'/request']['payload']
            input_hashes.append(initial_speech_input_hash(first))
            if query is None:
                query = {'sample_id':sid, 'audio_available_to_reflector':False,
                         'task_instruction':first['task_instruction'],
                         'history':copy.deepcopy(first['history']),
                         'supplied_evidence':copy.deepcopy(first['supplied_evidence'])}
            projected = []
            for contact in contacts:
                request = trace[contact['request_id']+'/request']
                response = trace[contact['request_id']+'/response']
                require(isinstance(response.get('text'), str), 'missing response or UNKNOWN contact')
                projected.append({'request':text_projection(request['payload']), 'response_text':response['text'],
                                  'request_sha256':contact['request_sha256'], 'response_text_sha256':contact['response_text_sha256']})
                evidence = request['payload'].get('supplied_evidence')
                require(isinstance(evidence, list), 'actual supplied evidence inventory required')
                value_ids = [e.get('evidence_id') for e in evidence
                             if isinstance(e,dict) and e.get('knowledge_role')=='value']
                resolved = [{'value_id':vid, 'sources':knowledge.get(vid, [])} for vid in value_ids]
                complete = all(len(x['sources']) == 1 for x in resolved)
                own = ':'.join(sid.split(':')[:3])
                knowledge_diagnostics.append({
                    'run_id':member['job_config']['run_id'], 'sample_id':sid,
                    'request_id':contact['request_id'], 'request_sha256':contact['request_sha256'],
                    'supplied_values':resolved,
                    'same_utterance_supplied': (any(own in x['sources'][0]['source_utterances'] for x in resolved)
                                                 if complete else None),
                })
                fallback = trace.get(contact['agent_contact_id']+'/reasoning-fallback')
                if fallback is not None:
                    from speech_aware_evidence_acquisition.rails.solver_protocol_audit import canonical_request
                    from speech_aware_evidence_acquisition.core.reasoning_knobs import answer_only_decoding
                    require(fallback.get('schema')=='saea-thinking-answer-fallback-v1'
                            and fallback.get('agent_contact_id')==contact['agent_contact_id']
                            and fallback.get('reask_request_id')==contact['request_id'], 'fallback identity differs')
                    first_id = fallback['first_request_id']
                    request0, response0 = trace[first_id+'/request'], trace[first_id+'/response']
                    require(isinstance(response0.get('text'), str), 'missing original fallback response')
                    original, fmt = canonical_request(request0['payload'])
                    final, final_fmt = canonical_request(request['payload'])
                    original['decoding_params'] = answer_only_decoding(original['decoding_params'])
                    require(original == final and fmt == final_fmt, 'fallback changed request policy')
                    projected[-1]['answer_only_fallback'] = {
                        'receipt':copy.deepcopy(fallback), 'first_request':text_projection(request0['payload']),
                        'first_response_text':response0['text']}
            output = native.output_rows[sid]['text']
            score = score_index[sid]
            critique = {k:score[k] for k in ('scenario_action_exact','slot_f1_exact_tp','slot_f1_exact_fp','slot_f1_exact_fn')}
            reward = intent_slot_reward(critique)
            critique.update(reward=reward, reward_mode=DENSE_REWARD,
                            strict_task_success=int(critique['scenario_action_exact']==1 and critique['slot_f1_exact_fp']==0 and critique['slot_f1_exact_fn']==0),
                            scored_answer_sha256=hashlib.sha256(output.encode()).hexdigest(),
                            reward_scope='official score of delivered answer; audio unavailable to text reflector')
            trajectory = {'request':text_projection(first), 'response_text':projected[-1]['response_text'],
                          'contacts':projected, 'scored_answer_after_fixed_normalization':output}
            tid = sha({'run_id':member['job_config']['run_id'], 'sample_id':sid, 'contacts':[c['request_id'] for c in contacts]})
            require(tid not in seen_trajectories, 'trajectory identity reused')
            seen_trajectories.add(tid)
            rollouts.append({'trajectory_id':tid, 'trajectory':json.dumps(trajectory, ensure_ascii=False, sort_keys=True),
                             'critique':json.dumps(critique, sort_keys=True), 'reward':reward,
                             'model_id':model, 'policy_sha256':compiled['policy_sha256'], 'experience_sha256':parent['sha256']})
        require(len(set(input_hashes)) == 1, 'same original speech input required')
        query['speech_input_sha256'] = input_hashes[0]
        groups.append({'query_id':sid, 'query':json.dumps(query, sort_keys=True),
                       'answer':json.dumps(gold[sid], ensure_ascii=False, sort_keys=True), 'rollouts':rollouts})
    from speech_aware_evidence_acquisition.rails.tfgrpo_prompts import SOURCE_SHA256, UPSTREAM_COMMIT
    batch = {'schema':'saea-tfgrpo-batch-v1','groups':groups,'parent':parent,'common':common,
             'model_id':model,'policy_sha256':compiled['policy_sha256'],'rollout_kind':'full-same-policy',
             'learning_config_sha256':learning_config_sha256,'prompt_sha256':SOURCE_SHA256,
             'upstream_commit':UPSTREAM_COMMIT,'bounds':bounds}
    from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import project_group_evidence
    evidence = {'source_batch_sha256':sha(batch), 'model_id':model, 'parent':parent,
                'rollout_parent_sha256':parent['sha256'], 'rollout_policy_sha256':compiled['policy_sha256'],
                'rollout_kind':'full-same-policy', 'reward_mode':DENSE_REWARD, 'offline_stale_policy':False,
                'groups':[{'query_id':g['query_id'],'group_sha256':sha(g),
                           'evidence':project_group_evidence(g,evidence_view='full-query-v1')} for g in groups]}
    return {'schema':'saea-psd-construction-groups-v1', 'batch':batch,
            'tfgrpo_evidence':evidence,
            'knowledge_diagnostics':knowledge_diagnostics,
            'provenance':{'campaign_sha256':compiled['content_sha256'],'source_bindings':bindings,
                          'gold':gold_pin,'generation_calls_used':sum(s[4]['calls_used'] for s in sources),
                          'fit_tokenizer_calls_used':sum(s[6] for s in sources), 'new_model_calls':0},
            'claim':'same registered policy; identical outputs retained; seed is not diversity evidence'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','repo','data','recipe','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    compiled=json.loads(a.config.read_bytes()); recipe=json.loads(a.recipe.read_bytes())
    from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
    lock=umbrella_lock_path()
    ids=set(compiled['members'][0]['job_config']['sample_ids'])
    gold,gold_pin=gold_train(load_lock(lock),a.data,ids)
    parent,_=_pinned_json(a.data,recipe['parent'],'experience parent')
    result=read_groups(compiled,recipe['bindings'],repo=a.repo,data=a.data,gold=gold,gold_pin=gold_pin,
                       parent=parent,common=recipe['common'],learning_config_sha256=recipe['learning_config_sha256'],bounds=recipe['bounds'],parent_binding=recipe.get('parent_binding'))
    publish_new(a.output,canonical(result))
    publish_new(a.output.with_name(a.output.stem+'-batch.json'),canonical(result['batch']))
    publish_new(a.output.with_name(a.output.stem+'-evidence.json'),canonical({'document':result['tfgrpo_evidence']}))
    print(json.dumps({'groups':len(result['batch']['groups']),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}))


if __name__=='__main__': main()
