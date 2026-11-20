"""Successor of the sealed field adapter; sole learning implementation stays upstream.

This module extends retained-field admission/packing, never loss or training.
Publisher imports this file by exec, so sibling imports cannot be assumed.
"""
import copy
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sealed_consumer_adapter', HERE.parent/'training-consumer-adapter/adapter.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
require, digest, read_pin = legacy.require, legacy.digest, legacy.read_pin
unified, no_labels, numeric = legacy.unified, legacy.no_labels, legacy.numeric
to_dataproto = legacy.to_dataproto
CONSUMER_KIND = legacy.CONSUMER_KIND
CONSUMER_ID = 'unified-omni-actor-fields-successor-v1'
CONTRACT = 'unified-omni-retained-padded-v1'


class Session:
    """One immutable loading session. Each file identity is hashed once.

    Stat identity is checked on cache hits; conflicting pins fail closed. Raw
    waveform bytes are not needed for authenticated retained feature loading.
    """
    def __init__(self):
        self.cache = {}
        self.read_counts = Counter()

    def read(self, ref):
        path = str(Path(ref['path']).resolve())
        identity = (ref['sha256'], ref['bytes'])
        st = Path(path).stat()
        stat = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if path in self.cache:
            old_identity, old_stat, raw = self.cache[path]
            require(identity == old_identity and stat == old_stat, 'immutable session asset changed: '+path)
            return raw
        raw = read_pin(ref)
        after = Path(path).stat()
        require(stat == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'session asset changed during read')
        self.cache[path] = (identity, stat, raw)
        self.read_counts[path] += 1
        return raw

    def manifest(self, ref, schema):
        value = json.loads(self.read(ref))
        require(value['schema'] == schema and value['assets'], 'invalid pinned '+schema)
        for asset in value['assets']:
            self.read(asset)
        return value


def shape(value):
    if not isinstance(value, list):
        numeric([value], 'tensor scalar')
        return []
    if not value:
        return [0]
    sizes = [shape(v) for v in value]
    require(all(s == sizes[0] for s in sizes), 'ragged retained tensor')
    return [len(value)] + sizes[0]


def leaves(value):
    if isinstance(value, list):
        for child in value:
            yield from leaves(child)
    else:
        yield value


def recording_map(config, session):
    mapping = json.loads(session.read(config['recording_map_ref']))
    require(mapping['schema'] == 'retained-recording-map-v1', 'recording mapping schema')
    population = json.loads(session.read(mapping['population_ref']))['population']
    roster = {r['recording_ref_id']:r for r in population}
    require(len(roster) == len(population), 'duplicate population reference')
    result = {}
    for row in mapping['records']:
        ref = row['recording_ref_id']
        require(ref in roster and ref not in result, 'unknown/duplicate population mapping')
        for key in ('source_id', 'source', 'domain'):
            require(row[key] == roster[ref][key], 'population mapping mismatch: '+key)
        require(roster[ref]['split'] == 'TRAIN', 'non-TRAIN recording mapping')
        require(row['capture_recording_id'] and row['recording_filename'] == Path(row['audio_ref']['path']).name,
                'recording filename must bind the captured asset path')
        require(row['audio_ref']['bytes'] > 0 and len(row['audio_ref']['sha256']) == 64, 'unpinned audio identity')
        result[ref] = row
    return result


def validate_fields(call, config, session, origin):
    f = call['training_fields']
    p, a = f['prompt_token_ids'], f['action_token_ids']
    numeric(p, 'prompt tokens', True, True)
    require(isinstance(a, list), 'missing action tokens')
    if a:
        numeric(a, 'action tokens', True, True)
    else:
        require(call.get('zero_action_terminal_observed') is True and call.get('observed_terminal') is True
                and call['output']['terminal'] in ('model_invalid','model_length','protocol_error'),
                'zero action requires observed model failure')
    require(call['token_ids'] == a and f['action_mask'] == call['action_mask'] == [1]*len(a),
            'action/mask mismatch; uninterrupted generated suffix required')
    for key, value in [('behavior_logprobs', f['behavior_logprobs']), ('current_logprobs', call['current_logprobs']),
                       ('reference_logprobs', call['reference_logprobs'])]:
        require(isinstance(value, list) and len(value) == len(a), 'probability length mismatch: '+key)
        if value:
            numeric(value, key, logprob=True)
    require(f['behavior_logprobs'] == call['old_logprobs'], 'sampled probabilities differ')
    for key in ('behavior_policy_version','adapter_vector','probability_semantics'):
        require(f[key] == call[key], 'behavior identity mismatch: '+key)
    length = len(p)+len(a)
    require(f['attention_mask'] == [1]*length and all(type(v) is int for v in f['attention_mask']), 'retained full attention required')
    tokenizer = session.manifest(f['tokenizer_manifest'], 'retained-tokenizer-v1')
    processor = session.manifest(f['processor_manifest'], 'retained-audio-processor-v1')
    require(type(tokenizer['vocab_size']) is int and tokenizer['vocab_size'] > max(p+a), 'token outside pinned vocabulary')
    require(type(tokenizer['pad_token_id']) is int and 0 <= tokenizer['pad_token_id'] < tokenizer['vocab_size']
            and tokenizer['pad_token_id'] == config['pad_token_id'], 'configured tokenizer padding differs')
    schema = f['captured_tensor_schema']
    require(schema['processor_manifest_sha256'] == digest(f['processor_manifest'])
            and schema['consumer_contract_version'] == CONTRACT, 'captured tensor contract mismatch')
    specs = schema['tensors']
    for key in ('audio_features','feature_mask','position_ids','effective_audio_lengths'):
        actual = shape(f[key]); declared = specs[key]
        require(actual == declared['shape'] and len(actual) == len(declared['axes'])
                and len(set(declared['axes'])) == len(actual), 'captured tensor shape/axes mismatch: '+key)
    layout = processor['feature_layout']
    if layout == 'audio_feature_time':
        axes = dict(audio_features=['audio','feature','time'], feature_mask=['audio','time'])
        require(shape(f['audio_features'])[1] == processor['feature_bins'], 'feature bin count differs')
        masks = f['feature_mask']
    else:
        require(layout == 'frames_bins' and processor.get('consumer_conversion') == 'frames_bins_to_audio_feature_time',
                'feature conversion must be explicitly registered')
        axes = dict(audio_features=['frames','bins'], feature_mask=['frames'])
        require(shape(f['audio_features'])[1] == processor['feature_bins'], 'feature bin count differs')
        masks = [f['feature_mask']]
    for key, expected in axes.items():
        require(specs[key]['axes'] == expected, 'processor feature axes mismatch')
    for i, axis in enumerate(specs['feature_mask']['axes']):
        require(specs['feature_mask']['shape'][i] == specs['audio_features']['shape'][specs['audio_features']['axes'].index(axis)],
                'mask/feature dimensions mismatch')
    require(all(type(v) is int and v in (0,1) for v in leaves(masks)), 'invalid feature mask')
    require(all(m and m == sorted(m, reverse=True) and sum(m) > 0 for m in masks), 'audio mask coverage mismatch')
    require(specs['effective_audio_lengths']['axes'] == ['audio'] and f['effective_audio_lengths'] == list(map(sum,masks)),
            'effective audio length mismatch')
    pos = specs['position_ids']
    require(pos['axes'] == processor['position_axes'] and 'tokens' in pos['axes'], 'processor position axes mismatch')
    require(pos['shape'][pos['axes'].index('tokens')] == length, 'position token length mismatch')
    expected_non_token = processor['position_non_token_shape']
    require([s for s,axis in zip(pos['shape'],pos['axes']) if axis != 'tokens'] == expected_non_token, 'position non-token dimensions differ')
    require(all(type(v) is int and v >= 0 for v in leaves(f['position_ids'])), 'invalid retained positions')
    spans = []
    for index, token in enumerate(p):
        if token == processor['audio_pad_token_id']:
            if not spans or index != spans[-1][-1]+1:
                spans.append([])
            spans[-1].append(index)
    assets = f['modality_assets']
    require(len(spans) == len(assets) == len(masks) and f['audio_token_lengths'] == list(map(len,spans)),
            'ordered multi-audio placeholder coverage differs')
    require(f['recording_id'] == origin['capture_recording_id'], 'capture recording identity mismatch')
    for asset in assets:
        # Only retained immutable identities; no per-call raw waveform re-read.
        candidates = [origin['audio_ref']] + origin.get('context_audio_refs', [])
        require(asset['media_type'] == 'audio' and any(all(asset[k] == ref[k] for k in ('path','sha256','bytes')) for ref in candidates),
                'audio identity outside pinned recording mapping')
    require(any(all(asset[k] == origin['audio_ref'][k] for k in ('path','sha256','bytes')) for asset in assets), 'primary recording missing')
    require(f['sampling_distribution'] == config['sampling_distribution'], 'sampling distribution differs')
    no_labels(call['online_input'], 'call.online_input')
    task_keys = {'P':{'retrieved_candidates'},'S1':{'type_universe'},'S2':{'supplied_types'},'D':{'actual_candidates','actual_slot_values'}}
    require(set(call['online_input']) == {'messages','audio_assets'} | task_keys[call['node']], 'explicit online input envelope required')
    no_labels(f['prompt_messages'], 'prompt_messages')
    require(f['prompt_messages'] == call['online_input']['messages'] and assets == call['online_input']['audio_assets'], 'retained online input differs')
    require(isinstance(f['prompt_text'], str) and f['prompt_text'] and isinstance(f['action_text'],str), 'retained text required')
    require(f['capture_binding'] == dict(online_input_sha256=digest(call['online_input']),output_sha256=digest(call['output']),
                fields_sha256=digest({k:v for k,v in f.items() if k != 'capture_binding'})), 'capture binding differs')
    require(f['input_boundary'] == 'online_only_no_reference' and f['capture_origin'] in ('runtime_capture','synthetic_schema_fixture'), 'capture boundary missing')
    require(f['action_text'] == f['decoded_action_text'] == call['raw_output'], 'observed decode differs')
    return f


def restore(files, config, session=None):
    session = session or Session()
    require('refs' in files, 'publisher files.refs required; no output hash in loader config')
    rows = {}
    for name in ('online','offline','provenance'):
        ref = files['refs'][name]
        require(Path(files[name]).resolve() == Path(ref['path']).resolve(), 'publisher path/ref mismatch')
        rows[name] = json.loads(session.read(ref))
        require(isinstance(rows[name], list) and len(rows[name]) == files['count'], 'separated file count mismatch')
    require(type(files['count']) is int and files['count'] > 0, 'empty dataset')
    indexes = {}
    for name, items in rows.items():
        indexes[name] = {r['draw_index']:r for r in items}
        require(len(indexes[name]) == len(items) and set(indexes[name]) == set(range(files['count'])), 'draw join incomplete/duplicated')
    origins = recording_map(config, session)
    groups = []; synthetic = set(); provenance = []
    for draw in range(files['count']):
        online, offline, prov = (indexes[k][draw] for k in ('online','offline','provenance'))
        require(online['sample_id'] == offline['sample_id'] == prov['sample_id'], 'sample/provenance join mismatch')
        require(prov['duplicate_occurrence'] == 1, 'fresh group reuse forbidden')
        if prov.get('selection_design') in (
            'deterministic bounded TRAIN prefix; no sampling probability claimed',
            'deterministic bounded TRAIN window; no sampling probability claimed',
        ):
            plan = json.loads(session.read(config['selected_cohort_ref']))
            selected = plan['draws'][draw]
            require(plan.get('prepare_mode') == 'bounded_train_probe' and plan['synthetic_fixture'] is False
                    and plan['population_complete'] is False and plan['sampling']['purpose'] == 'bounded_train_probe'
                    and plan['sampling']['split'] == 'TRAIN' and plan['sampling']['replacement'] == 'forbid'
                    and selected['record']['recording_ref_id'] == prov['recording_ref_id']
                    and selected['record']['sample_id'] == prov['sample_id'] and selected['group_id'] == prov['group_id']
                    and selected['draw_index'] == prov['draw_index'] == draw
                    and selected['selection_design'] == prov['selection_design']
                    and prov['duplicate_occurrence'] == selected['duplicate_occurrence'] == 1,
                    'invalid deterministic TRAIN draw identity')
            for field in ('conditional_record_probability', 'inclusion_probability'):
                require(field in prov and prov[field] is None and selected[field] is None,
                        'deterministic draw cannot claim population probabilities')
            require(not any(k in prov or k in selected for k in ('conditional_source_probability',
                    'conditional_within_cell_probability', 'random_output_position_probability')),
                    'unexpected stochastic fields on deterministic draw')
        else:
            for key in ('conditional_source_probability','conditional_within_cell_probability','conditional_record_probability',
                        'random_output_position_probability','inclusion_probability'):
                v = prov[key]
                if v is None:
                    require(key in ('random_output_position_probability','inclusion_probability') and bool(prov.get('unavailable_probability_reason')), 'missing unavailable probability reason')
                else:
                    require(type(v) in (int,float) and 0 < v <= 1, 'invalid sampling probability')
            require(abs(prov['conditional_source_probability']*prov['conditional_within_cell_probability']-prov['conditional_record_probability']) < 1e-12,
                    'conditional sampling probability mismatch')
        origin = origins[prov['recording_ref_id']]
        for key in ('source_id','source','domain'):
            require(prov[key] == origin[key], 'source mapping mismatch: '+key)
        no_labels(online['online'], 'online.input')
        b = copy.deepcopy(online['behavior_bundle'])
        require(b['schema'] == 'unified-behavior-bundle-v1' and b['global_query_count'] == 1 and len(b['groups']) == 1
                and b['online_gold_injected'] is False and type(b['synthetic']) is bool, 'one complete fresh group per draw required')
        synthetic.add(b['synthetic'])
        g = b['groups'][0]
        require(g['recording_id'] == origin['capture_recording_id'], 'capture recording mapping differs')
        require(set(offline['group_supervision']) == {e['episode_id'] for e in g['episodes']}, 'offline episode roster mismatch')
        for e in g['episodes']:
            require('offline' not in e, 'offline labels leaked into online bundle')
            e['offline'] = offline['group_supervision'][e['episode_id']]
            for call in e['calls']:
                f = validate_fields(call, config, session, origin)
                require(b['synthetic'] == (f['capture_origin'] == 'synthetic_schema_fixture'), 'capture provenance mismatch')
        groups.append(g)
        provenance.append(dict(prov, capture_recording_id=origin['capture_recording_id'],recording_filename=origin['recording_filename']))
    require(len(synthetic) == 1, 'mixed real/synthetic evidence')
    return dict(schema='unified-behavior-bundle-v1',synthetic=synthetic.pop(),online_gold_injected=False,
                global_query_count=len(groups),groups=groups), provenance


def build(files, config, session=None):
    """Stable fields/mapping for the separate autograd owner; no loss here."""
    session = session or Session()
    require(Path(config['sealed_adapter_ref']['path']).resolve() == HERE.parent/'training-consumer-adapter/adapter.py', 'designated sealed adapter required')
    session.read(config['sealed_adapter_ref'])
    bundle, provenance = restore(files, config, session)
    learning_config = json.loads(session.read(config['learning_config_ref']))
    state = json.loads(session.read(config['clock_state_ref']))
    with unified(config) as (learning,schedule,_):
        result = learning.evaluate(bundle,schedule.Clock(learning_config,state))
    require(result['global_episode_denominator'] == len(bundle['groups'])*result['G'], 'fixed N*G denominator differs')
    advantages = {(g['group_id'],e['episode_id']):e['advantage'] for g in result['groups'] for e in g['episodes']}
    snap = result['snapshot']; selected = []; all_mapping = []
    for draw, g in enumerate(bundle['groups']):
        prov = provenance[draw]
        for e in g['episodes']:
            for c in e['calls']:
                trainable = snap['trainable_mask'][c['node']]
                count = len(c['training_fields']['action_token_ids'])
                m = dict(draw_index=draw,sample_id=prov['sample_id'],group_id=g['group_id'],query_draw_id=g['query_draw_id'],
                    query_id=g['query_id'],recording_id=g['recording_id'],capture_recording_id=prov['capture_recording_id'],
                    recording_ref_id=prov['recording_ref_id'],source_id=prov['source_id'],recording_filename=prov['recording_filename'],
                    sampling_round=g['sampling_round'],episode_id=e['episode_id'],replicate=e['replicate'],node=c['node'],
                    call_id=c['call_id'],request_identity=c['request_identity'],parents=c['parents'],trainable=trainable,
                    action_count=count,action_tokens=count,has_actions=bool(count),source_token_indices=list(range(count)),actor_row=None,
                    actor_row_exclusion='frozen_node' if not trainable else ('observed_zero_action' if not count else None),
                    captured_tensor_schema=c['training_fields']['captured_tensor_schema'],
                    modality_assets=c['training_fields']['modality_assets'],capture_binding=c['training_fields']['capture_binding'])
                all_mapping.append(m)
                if trainable and count:
                    m['actor_row'] = len(selected)
                    selected.append((c,m,advantages[g['group_id'],e['episode_id']]))
    import numpy as np
    import torch
    n = len(selected); pad = config['pad_token_id']
    pmax = max((len(c['training_fields']['prompt_token_ids']) for c,_,_ in selected),default=0)
    rmax = max((len(c['token_ids']) for c,_,_ in selected),default=0)
    require(pmax+rmax <= config['context_limit'], 'context limit exceeded; never truncate')
    tensors = dict(prompts=torch.full((n,pmax),pad,dtype=torch.long),responses=torch.full((n,rmax),pad,dtype=torch.long),
                   attention_mask=torch.zeros((n,pmax+rmax),dtype=torch.long),response_mask=torch.zeros((n,rmax),dtype=torch.long))
    for name in ('old_log_probs','ref_log_prob','advantages','unified_policy_node_weight','unified_kl_node_weight'):
        tensors[name] = torch.zeros((n,rmax),dtype=torch.float32)
    pos_spec = selected[0][0]['training_fields']['captured_tensor_schema']['tensors']['position_ids'] if selected else None
    nonshape = [s for s,a in zip(pos_spec['shape'],pos_spec['axes']) if a != 'tokens'] if pos_spec else []
    tensors['position_ids'] = torch.zeros((n,*nonshape,pmax+rmax),dtype=torch.long)
    non = {k:np.empty(n,dtype=object) for k in ('multi_modal_inputs','uid','record_identity','adapter_route')}
    token_mapping = []
    for row,(c,m,advantage) in enumerate(selected):
        f = c['training_fields']; plen = len(f['prompt_token_ids']); length = len(f['action_token_ids']); offset = pmax-plen
        tensors['prompts'][row,offset:] = torch.tensor(f['prompt_token_ids'])
        tensors['responses'][row,:length] = torch.tensor(f['action_token_ids'])
        tensors['attention_mask'][row,offset:pmax+length] = torch.tensor(f['attention_mask'])
        spec = f['captured_tensor_schema']['tensors']['position_ids']
        require(spec['axes'] == pos_spec['axes'] and [s for s,a in zip(spec['shape'],spec['axes']) if a != 'tokens'] == nonshape,
                'mixed position contracts require separate actor batches')
        positions = torch.tensor(f['position_ids'],dtype=torch.long).movedim(spec['axes'].index('tokens'),-1)
        tensors['position_ids'][row,...,offset:pmax+length] = positions
        tensors['response_mask'][row,:length] = 1
        tensors['old_log_probs'][row,:length] = torch.tensor(f['behavior_logprobs'])
        tensors['ref_log_prob'][row,:length] = torch.tensor(c['reference_logprobs'])
        tensors['advantages'][row,:length] = advantage
        weight = snap['loss_node_weights'][c['node']]
        tensors['unified_policy_node_weight'][row,:length] = weight
        tensors['unified_kl_node_weight'][row,:length] = weight if snap['loss_node_kl_weighting'] == 'policy_and_kl' else 1.
        features = torch.tensor(f['audio_features'],dtype=torch.float32)
        mask = torch.tensor(f['feature_mask'],dtype=torch.long)
        axes = f['captured_tensor_schema']['tensors']['audio_features']['axes']
        if axes == ['frames','bins']:
            features = features.T.unsqueeze(0); mask = mask.unsqueeze(0)
        # Target retained tensors stay exact, including each call's audio count
        # and time width. Existing actor owns any later concatenation/padding.
        non['multi_modal_inputs'][row] = dict(input_features=features,feature_attention_mask=mask)
        non['uid'][row] = m['query_draw_id']; non['record_identity'][row] = m
        non['adapter_route'][row] = snap['adapter_routing']['trainable_adapters'][c['node']]
        token_mapping.extend(dict(m,actor_row=row,response_token_index=t,source_token_index=t) for t in range(length))
    tensors['input_ids'] = torch.cat((tensors['prompts'],tensors['responses']),dim=-1)
    require(all(torch.isfinite(t).all().item() for t in tensors.values())
            and all(torch.isfinite(mm['input_features']).all().item() for mm in non['multi_modal_inputs']), 'nonfinite packed tensor')
    meta = dict(audio_contract=CONTRACT,requires_remove_padding=False,requires_sequence_parallel_size=1,
        group_size=result['G'],global_query_count=result['queries'],global_episode_denominator=result['global_episode_denominator'],
        reduction=result['reduction'],snapshot=snap,beta=result['beta'],kl_estimator=learning_config['kernel']['kl_estimator'],
        position_axes=['actor_row']+([a for a in pos_spec['axes'] if a != 'tokens'] if pos_spec else [])+['tokens'],
        advantages_source='unified-learning.learning.evaluate',recompute_advantages=False,
        gradient_status='actor_rows_available' if n else 'no_gradient',
        no_gradient_reason=None if n else 'complete_groups_have_no_trainable_action_tokens',
        trainer_integrated=False,training_ready=False,synthetic=bundle['synthetic'])
    report = dict(consumer=CONSUMER_ID,status='analysis_only',loaded_count=files['count'],required_fields_validated=True,
        actual_model_trainer_loader=False,trainer_integrated=False,training_ready=False,synthetic=bundle['synthetic'],
        actor_rows=n,action_tokens=len(token_mapping),all_call_mapping=all_mapping,token_mapping=token_mapping,
        tensor_shapes={k:list(v.shape) for k,v in tensors.items()},global_episode_denominator=result['global_episode_denominator'],
        gradient_status=meta['gradient_status'],no_gradient_reason=meta['no_gradient_reason'],snapshot=snap,provenance=provenance,
        group_results=result['groups'],component_scores=result['component_scores'],
        asset_hash_counts=dict(session.read_counts),raw_audio_reads=0,
        gaps=['real captured on-policy groups and authenticated model/processor parity',
              'existing actor forward, shared/per-node LoRA execution and parameter freezing need model acceptance',
              'unified-autograd-bridge owns differentiable loss wiring; no optimizer/model execution here'])
    return dict(tensors=tensors,non_tensors=non,meta_info=meta), report


def validate_dataset(*, files, config):
    try:
        return build(files,config)[1]
    except (ValueError,KeyError,TypeError,OSError,AssertionError,IndexError,RuntimeError) as error:
        return dict(consumer=CONSUMER_ID,loaded_count=0,required_fields_validated=False,actual_model_trainer_loader=False,
                    status='analysis_only',training_ready=False,trainer_integrated=False,error=str(error))
