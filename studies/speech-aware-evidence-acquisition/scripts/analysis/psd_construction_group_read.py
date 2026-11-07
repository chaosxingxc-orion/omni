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
    Empty parent is accepted for explicitly pinned ordinary node templates;
    known generated TF-GRPO consumer templates require their descriptor.
    """
    model = job['runtime_binding']['model_fragment']
    validate_experience(parent, model)
    descriptor = job.get('tfgrpo_experience')
    if descriptor is None:
        require(binding is None and parent == empty_experience(model), 'unconditioned actor requires empty unexposed parent')
        for name in ('slot_policy', 'decision_policy'):
            if name not in job:
                continue
            pin = job[name]['instruction_template']
            require(isinstance(pin, dict) and set(pin) == {'relpath', 'sha256'}
                    and pin['relpath'].startswith('configs/probes/')
                    and '..' not in Path(pin['relpath']).parts, 'confined node template pin required')
            template = resolve_asset(repo, pin['relpath'])
            require(hashlib.sha256(template.read_bytes()).hexdigest() == pin['sha256'], 'node template pin differs')
            text = template.read_text()
            require('Use these conditional experiences only' not in text,
                    'TF-GRPO template requires reviewed parent provenance')
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
    from speech_aware_evidence_acquisition.rails.knowledge import KnowledgeSnapshot
    index = {}
    for store_id, store in values.get('knowledge_stores', {}).items():
        snapshot = store.get('snapshot') if isinstance(store, dict) else None
        if not snapshot:
            continue
        path = _confined_file(data, snapshot['relpath'])
        loaded = KnowledgeSnapshot.from_frozen_json(json.loads(path.read_bytes()))
        require(loaded.content_hash() == snapshot['sha256'], 'knowledge snapshot content digest differs')
        for value in loaded.values:
            vid = value.value_id
            samples = value.source_sample_ids
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
        require(not any(cost.get(key) for key in ('native_replay_source','replayed_predecessor_cost'))
                and not manifest.get('predecessor_attempt') and not manifest.get('predecessor_attempts'),
                'recovered source costs require complete ancestry reader')
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



def read_p_only_groups(compiled, bindings, *, repo, data, gold, gold_pin, legal_ids,
                       parent, common, learning_config_sha256, bounds):
    """Read finalized P node bundles; four members are four draws, repairs are not."""
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.analysis.perception_node_score import score
    from speech_aware_evidence_acquisition.rails.solver_protocol_audit import canonical_request
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import BINARY_REWARD
    from speech_aware_evidence_acquisition.rails.tfgrpo_prompts import SOURCE_SHA256, UPSTREAM_COMMIT
    from registered_workload import validate_workload

    check_compiled(compiled)
    members = compiled['members']
    require(len(members) == len(bindings) == 4, 'four independently registered P members required')
    ids = members[0]['job_config']['sample_ids']
    require(len(ids) == 80 and set(gold) == set(ids), 'complete construction population required')
    model = members[0]['job_config']['runtime_binding']['model_fragment']
    validate_experience(parent, model)
    require(parent == empty_experience(model), 'P actor has no experience input')
    require(set(common) == {'agent_objective','learning_objective','num_experiences'}, 'exact binary teacher objectives required')
    sources, all_request_ids = [], set()
    science_ref = None
    for member, pin in zip(members, bindings):
        binding, _ = _pinned_json(data, pin, 'P source binding')
        require(set(binding) == {'manifest', 'flight_receipts'}, 'manifest and FIT custody pins required')
        manifest_path = _confined_file(data, binding['manifest']['relpath'])
        _pinned_json(data, binding['manifest'], 'P manifest')
        bundle = verify_bundle(manifest_path)
        manifest = bundle.manifest
        job = member['job_config']
        require(manifest.get('status') == 'COMPLETED' and bundle.attempt_state == 'FINALIZED'
                and manifest.get('result_kind') == 'node', 'finalized P node bundle required')
        require(not manifest.get('predecessor_attempt') and not manifest.get('predecessor_attempts')
                and not any(manifest.get('cost', {}).get(k) for k in ('native_replay_source','replayed_predecessor_cost')),
                'recovered source requires complete ancestry accounting')
        require((manifest['plan']['run_id'], manifest['plan']['attempt_id']) ==
                (job['run_id'], member['generation_attempt_id']), 'run or attempt differs')
        require([r['sample_id'] for r in bundle.outputs_rows] == ids, 'ordered roster differs')
        fragments = manifest['config']['fragments']
        require(fragments == {'model':model, 'dataset':job['population']['dataset'],
                              'baseline':job['name'], 'experiment':job['name']}, 'fragments differ')
        candidate = json.loads(resolve_asset(repo, 'docs/readiness/'+job['name']+'/candidate.json').read_bytes())
        require(candidate['job_config'] == job and candidate['composed_config_hash'] == manifest['config']['config_hash'], 'registration differs')
        validate_workload(repo, data, job['name'], candidate)
        cfg = compose(repo, *(fragments[k] for k in ('model','dataset','baseline','experiment')))
        require(cfg.config_hash == manifest['config']['config_hash'], 'composed config differs')
        values = cfg.values
        require(values.get('request_decoding_overrides') == job['collection_decoding'], 'decoding differs')
        science = {k:copy.deepcopy(values.get(k)) for k in ('policy','arm','model_id','model_path','transport',
            'native_episode','knowledge_stores','phase_material','thinking_policy','phase_tools',
            'native_evidence_projection','native_evidence_projection_params','deliver_normaliser','serving_slot_tokens')}
        science['decoding'] = dict(effective_request_decoding_params(values))
        science['decoding'].pop('seed', None)
        if science_ref is None: science_ref = science
        require(science == science_ref, 'scientific policy differs')
        rows = {r['sample_id']:r for r in bundle.outputs_rows}
        roster = [{'sample_id':sid, 'gold_intent':gold[sid]} for sid in ids]
        scores = score(roster, bundle.outputs_rows, maximum=3, legal_ids=set(legal_ids))['rows']
        receipts, _ = _pinned_json(data, binding['flight_receipts'], 'flight receipts')
        fit = receipts.get('receipts', receipts)['stages']['run_phases']['dispatch']['native_contact_fit_receipts']
        contacts = [c for sid in ids for c in rows[sid]['node_outcome']['node_receipt']['contacts']]
        require(len(fit) == len(contacts) and len({c['agent_contact_id'] for c in contacts}) == len(contacts)
                and {f['agent_contact_id'] for f in fit} == {c['agent_contact_id'] for c in contacts}, 'FIT contact custody differs')
        fake = SimpleNamespace(sample_ids=ids, bundle=bundle, plan=SimpleNamespace(split_identity_hash=manifest['plan']['split_identity_hash']),
            phase_results={sid:{'native_episode_receipt':{'role_visits':[{'execution_receipt':{'contacts':rows[sid]['node_outcome']['node_receipt']['contacts']}}]}} for sid in ids})
        fit_cost = verify_fit_receipts(data, fake, fit, member)
        cost = manifest.get('cost', {})
        require(type(cost.get('calls_used')) is int and cost['calls_used'] <= member['generation_cap'], 'generation cost missing or exceeds cap')
        cost_rows = [v for k,v in bundle.trace_records.items() if k.endswith('/cost') and 'calls' in v]
        require(len(cost_rows) == len(contacts) and sum(v['calls'] for v in cost_rows) == cost['calls_used'],
                'unattributed generation cost requires ancestry accounting')
        sources.append((member, bundle, rows, dict(zip(ids, scores)), fit_cost, cost, pin))
    groups, detail = [], []
    for sid in ids:
        rollouts, reference = [], None
        for member, bundle, rows, scores, fit_cost, manifest_cost, pin in sources:
            node = rows[sid]['node_outcome']
            contacts = node['node_receipt']['contacts']
            require(1 <= len(contacts) <= 2, 'one primary and at most one repair required')
            if len(contacts) == 2:
                require(contacts[1]['agent_contact_id'] == contacts[0]['agent_contact_id'] + ':perception-length-repair', 'second contact is not P length repair')
            projected = []
            for contact in contacts:
                rid = contact['request_id']
                require(rid not in all_request_ids, 'request identity reused')
                all_request_ids.add(rid)
                req = bundle.trace_records[rid+'/request']
                resp = bundle.trace_records[rid+'/response']
                cost = bundle.trace_records.get(rid+'/cost')
                require(isinstance(resp.get('text'), str) and isinstance(cost, dict), 'response and cost required')
                require(all(type(cost.get(k)) in (int,float) and cost[k] >= 0 for k in ('calls','prompt_tokens','completion_tokens','latency_seconds')), 'complete contact cost required')
                projected.append({'request_id':rid, 'request_sha256':contact['request_sha256'],
                    'response_sha256':contact['response_sha256'], 'response_text_sha256':contact['response_text_sha256'],
                    'request':text_projection(req['payload']), 'response_text':resp['text'],
                    'finish_reason':cost.get('finish_reason'), 'usage':resp.get('usage'), 'cost':cost})
            if len(contacts) == 2:
                require(projected[0]['finish_reason'] == 'length' and bool(projected[0]['response_text']), 'repair requires original nonempty length failure')
            primary = bundle.trace_records[contacts[0]['request_id']+'/request']['payload']
            request, fmt = canonical_request(primary)
            require(request['decoding_params'].get('seed') == member['seed'], 'actual request seed differs')
            request['decoding_params'].pop('seed')
            request['decoding_params'].pop('id_slot', None)
            speech_fields = ('sample_id','carrier_lock_key','media_relpath','speech_ref','sample_rate_hz','audio_seconds')
            require(all(primary.get(k) is not None for k in speech_fields) and primary.get('raw_speech_query') is not None, 'complete original speech input required')
            speech_identity = content_hash({**{k:primary[k] for k in speech_fields},
                'raw_speech_query':primary['raw_speech_query'], 'speech_packet_sha256':primary.get('speech_packet_sha256')})
            comparable = (request, fmt, speech_identity)
            if reference is None: reference = comparable
            require(comparable == reference, 'actual prompt, audio, evidence, schema or decode differs')
            result = scores[sid]
            candidates = result['candidate_ids']
            reward = int(result['gold_recalled'])
            critique = {'status':result['status'], 'legal':result['status']=='valid', 'candidate_ids':candidates,
                        'top1':bool(candidates and candidates[0] == gold[sid]), 'recall3':bool(reward),
                        'rank':candidates.index(gold[sid])+1 if gold[sid] in candidates else None,
                        'native_status':node['status'], 'repair_used':len(contacts)==2}
            trajectory = {'request':text_projection(primary), 'response_text':node['response'], 'scored_answer_after_fixed_normalization':node['response'], 'contacts':projected, 'node_status':node['status'], 'delivered_response':node['response']}
            tid = sha({'run_id':member['job_config']['run_id'],'sample_id':sid,'primary_request':contacts[0]['request_id']})
            rollouts.append({'trajectory_id':tid,'trajectory':json.dumps(trajectory,ensure_ascii=False,sort_keys=True),
                'critique':json.dumps(critique,sort_keys=True),'reward':reward,'model_id':model,
                'policy_sha256':compiled['policy_sha256'],'experience_sha256':parent['sha256']})
        groups.append({'query_id':sid, 'query':json.dumps({'sample_id':sid,'request':text_projection(reference[0]),
            'response_format':reference[1], 'speech_input_sha256':reference[2], 'audio_available_to_reflector':False, 'task_instruction':reference[0]['task_instruction'], 'history':reference[0]['history'], 'supplied_evidence':reference[0]['supplied_evidence']},sort_keys=True),
            'answer':json.dumps({'gold_intent':gold[sid]},sort_keys=True),'rollouts':rollouts})
        values = [r['reward'] for r in rollouts]
        detail.append({'sample_id':sid,'mixed':len(set(values))>1,'all_correct':all(values),
            'all_wrong':not any(values),'ties':len(set(values))==1,
            'unique_responses':len({json.loads(r['trajectory'])['delivered_response'] for r in rollouts})})
    batch = {'schema':'saea-tfgrpo-batch-v1','groups':groups,'parent':parent,'common':common,
        'model_id':model,'policy_sha256':compiled['policy_sha256'],'rollout_kind':'full-same-policy',
        'learning_config_sha256':learning_config_sha256,'prompt_sha256':SOURCE_SHA256,
        'upstream_commit':UPSTREAM_COMMIT,'bounds':bounds}
    critiques = [json.loads(r['critique']) for g in groups for r in g['rollouts']]
    from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import project_group_evidence
    evidence = {'source_batch_sha256':sha(batch),'model_id':model,'parent':parent,
        'rollout_parent_sha256':parent['sha256'],'rollout_policy_sha256':compiled['policy_sha256'],
        'rollout_kind':'full-same-policy','reward_mode':BINARY_REWARD,'offline_stale_policy':False,
        'groups':[{'query_id':g['query_id'],'group_sha256':sha(g),
            'evidence':project_group_evidence(g,evidence_view='full-query-v1')} for g in groups]}
    return {'schema':'saea-p-only-construction-groups-v1','batch':batch,'tfgrpo_evidence':evidence,'groups':detail,
        'metrics':{'groups':len(groups),'draws':len(critiques),'mixed_groups':sum(d['mixed'] for d in detail),
            'ties':sum(d['ties'] for d in detail),'all_correct':sum(d['all_correct'] for d in detail),
            'all_wrong':sum(d['all_wrong'] for d in detail),'legal':sum(c['legal'] for c in critiques),
            'top1':sum(c['top1'] for c in critiques),'recall3':sum(c['recall3'] for c in critiques),
            'repaired_draws':sum(c['repair_used'] for c in critiques),
            'different_response_groups':sum(d['unique_responses']>1 for d in detail),
            'per_member':[{ 'run_id':sources[i][0]['job_config']['run_id'],
                'legal':sum(json.loads(g['rollouts'][i]['critique'])['legal'] for g in groups),
                'top1':sum(json.loads(g['rollouts'][i]['critique'])['top1'] for g in groups),
                'recall3':sum(g['rollouts'][i]['reward'] for g in groups)} for i in range(4)],
            'recall3_difference_vs_first':[sum(g['rollouts'][i]['reward']-g['rollouts'][0]['reward'] for g in groups) for i in range(4)]},
        'provenance':{'campaign_sha256':compiled['content_sha256'],'source_bindings':bindings,'gold':gold_pin,
            'generation_costs':[s[5] for s in sources], 'fit_tokenizer_calls_per_member':[s[4] for s in sources], 'fit_tokenizer_calls_used':sum(s[4] for s in sources),
            'new_model_calls':0},'claim':'feedback only; no semantic advantage without a teacher model call'}


def prepare_p_teacher(result, *, data, feedback_pin, roster_sha256, config):
    """Construct the existing dynamic TF-GRPO source and finite registration offline."""
    from dataclasses import asdict
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.rails.tfgrpo_runtime import learning_configuration, validate_registration
    from speech_aware_evidence_acquisition.rails.tfgrpo_learning import BINARY_REWARD, group_has_contrast
    require(result.get('schema') == 'saea-p-only-construction-groups-v1', 'verified P feedback required')
    saved, _ = _pinned_json(data, feedback_pin, 'verified P feedback')
    require(saved == result, 'feedback pin differs')
    batch = copy.deepcopy(result['batch'])
    contacts = []
    for group in batch['groups']:
        if group_has_contrast(group['rollouts'], BINARY_REWARD):
            contacts.extend({'stage':'summary','identity':r['trajectory_id'],'sample_id':group['query_id']} for r in group['rollouts'])
            contacts.extend({'stage':stage,'identity':group['query_id'],'sample_id':group['query_id']} for stage in ('advantage','group_update'))
    if not contacts:
        return {'status':'no-mixed-groups', 'source':None, 'registration':None, 'new_model_calls':0}
    contacts.append({'stage':'batch_update','identity':'batch','sample_id':batch['groups'][0]['query_id']})
    require(set(config) == {'plan','decoding_params','max_context_bytes','runtime_identity_sha256'}, 'exact teacher configuration required')
    plan_args = dict(config['plan']); plan_args['planned_model_calls'] = len(contacts); plan_args['protocol_hash'] = '0'*64
    plan = ExecutionPlan(**plan_args).validate()
    registration = {'schema':'saea-tfgrpo-runtime-binding-v1', 'plan':{k:v for k,v in asdict(plan).items() if k!='protocol_hash'},
        'model_id':batch['model_id'], 'construction_roster_sha256':roster_sha256, 'contacts':contacts,
        'decoding_params':config['decoding_params'], 'max_context_bytes':config['max_context_bytes'],
        'runtime_identity_sha256':config['runtime_identity_sha256']}
    registration['learning_config_sha256'] = sha(learning_configuration(registration))
    batch['learning_config_sha256'] = registration['learning_config_sha256']
    batch['bounds']['max_calls'] = len(contacts)
    registration['batch_sha256'] = sha(batch)
    validate_registration(registration, plan=plan, model_id=batch['model_id'])
    source = {'schema':'saea-tfgrpo-learning-source-v1','batch':batch,
              'provenance':[{'root':'data', **feedback_pin}]}
    return {'status':'prepared-not-admitted','source':source,'registration':registration,
            'plan':asdict(plan), 'new_model_calls':0}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','repo','data','recipe','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--teacher-config',type=Path)
    a=p.parse_args()
    compiled=json.loads(a.config.read_bytes()); recipe=json.loads(a.recipe.read_bytes())
    from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
    lock=umbrella_lock_path()
    ids=set(compiled['members'][0]['job_config']['sample_ids'])
    gold,gold_pin=gold_train(load_lock(lock),a.data,ids)
    parent,_=_pinned_json(a.data,recipe['parent'],'experience parent')
    if compiled['members'][0]['job_config']['schema'] == 'saea-perception-only-workload-v1':
        carrier,_ = _pinned_json(a.data, recipe['legal_ids'], 'legal intent carrier')
        require(isinstance(carrier, list) and len(carrier) == len(set(carrier)) and all(isinstance(x,str) and x for x in carrier), 'unique legal intent IDs required')
        intent_gold = {sid:row['scenario']+'_'+row['action'] for sid,row in gold.items()}
        result=read_p_only_groups(compiled,recipe['bindings'],repo=a.repo,data=a.data,gold=intent_gold,gold_pin=gold_pin,
            legal_ids=carrier,parent=parent,common=recipe['common'],learning_config_sha256=recipe['learning_config_sha256'],bounds=recipe['bounds'])
    else:
        result=read_groups(compiled,recipe['bindings'],repo=a.repo,data=a.data,gold=gold,gold_pin=gold_pin,
            parent=parent,common=recipe['common'],learning_config_sha256=recipe['learning_config_sha256'],bounds=recipe['bounds'],parent_binding=recipe.get('parent_binding'))
    publish_new(a.output,canonical(result))
    publish_new(a.output.with_name(a.output.stem+'-batch.json'),canonical(result['batch']))
    if 'tfgrpo_evidence' in result:
        publish_new(a.output.with_name(a.output.stem+'-evidence.json'),canonical({'document':result['tfgrpo_evidence']}))
    if a.teacher_config:
        teacher = json.loads(a.teacher_config.read_bytes())
        pin = {'relpath':str(a.output.resolve().relative_to(a.data.resolve())), 'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}
        prepared = prepare_p_teacher(result,data=a.data,feedback_pin=pin,roster_sha256=teacher['roster_sha256'],config=teacher['config'])
        publish_new(a.output.with_name(a.output.stem+'-teacher.json'),canonical(prepared))
        if prepared['source'] is not None:
            publish_new(a.output.with_name(a.output.stem+'-teacher-source.json'),canonical(prepared['source']))
            publish_new(a.output.with_name(a.output.stem+'-teacher-registration.json'),canonical(prepared['registration']))
    print(json.dumps({'groups':len(result['batch']['groups']),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}))


if __name__=='__main__': main()
