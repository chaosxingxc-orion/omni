"""Thin separated-dataset -> padded actor fields. No reward/loss/optimizer loop."""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

CONSUMER_KIND = 'model_trainer'  # dataset.py plugin dispatch tag, NOT readiness evidence.
CONSUMER_ID = 'unified-omni-actor-fields-v1'
HERE = Path(__file__).resolve().parent
NODES = ('P', 'S1', 'S2', 'D')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_pin(ref):
    path = Path(ref['path']).resolve()
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'changed during read: '+str(path))
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'] and len(raw) == ref['bytes'], 'pin mismatch: '+str(path))
    return raw


@contextmanager
def unified(config):
    """Load the sole learning implementation; restore caller imports on return."""
    refs = config['source_pins']
    required = ('kernel.py', 'scoring.py', 'schedule.py', 'learning.py', 'dataset.py', 'objective.py', 'runtime.py')
    for name in required:
        read_pin(refs[name])
    root = HERE.parent/'unified-learning'
    # A sealed release can be mounted both at its physical archive location
    # and at the logical execution namespace. Authenticate BOTH copies and
    # always import the designated local implementation; paths are not identity.
    for name in required:
        designated = root/name if name in required[:5] else root.parent/'system-objective-cpu'/name
        read_pin(dict(refs[name], path=str(designated)))
    names = ('kernel', 'scoring', 'schedule', 'learning')
    saved = {name: sys.modules.get(name) for name in names}
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        for name in names:
            spec = importlib.util.spec_from_file_location(name, root/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        yield sys.modules['learning'], sys.modules['schedule'], sys.modules['kernel']
        for name in required:
            read_pin(refs[name])
    finally:
        sys.dont_write_bytecode = old_bytecode
        sys.path[:] = old_path
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


FORBIDDEN = {'gold', 'goldtext', 'goldintent', 'goldslots', 'labels', 'supervision',
             'offlinesupervision', 'offline', 'reference', 'referencetext', 'referencetranscript',
             'transcript', 'transcription', 'asrtext', 'asrtranscript', 'targettext', 'groundtruth'}


def no_labels(value, where='online'):
    if isinstance(value, dict):
        for key, child in value.items():
            norm = re.sub('[^a-z0-9]', '', str(key).lower())
            require(norm not in FORBIDDEN and not norm.startswith(('gold', 'reference', 'groundtruth')),
                    'offline/reference field forbidden in '+where+': '+str(key))
            no_labels(child, where+'.'+str(key))
    elif isinstance(value, list):
        for child in value:
            no_labels(child, where)


def numeric(values, name, integer=False, positive=False, logprob=False):
    require(isinstance(values, list) and values, 'missing '+name)
    for value in values:
        require((type(value) is int if integer else type(value) in (int, float)) and math.isfinite(value), 'invalid '+name)
        require(not positive or value >= 0, 'negative '+name)
        require(not logprob or value <= 0, 'positive '+name)


def manifest(ref, schema):
    value = json.loads(read_pin(ref))
    require(value['schema'] == schema, 'manifest schema mismatch: '+schema)
    for asset in value['assets']:
        read_pin(asset)  # hashes only: no processor/model/audio decoding.
    require(value['assets'], 'manifest must pin actual assets')
    return value


def validate_fields(call, config):
    """Validate retained evidence; missing data is never tokenized/reconstructed."""
    f = call['training_fields']
    p, a = f['prompt_token_ids'], f['action_token_ids']
    numeric(p, 'prompt tokens', True, True)
    numeric(a, 'action tokens', True, True)
    numeric(f['behavior_logprobs'], 'behavior probabilities', logprob=True)
    require(len(a) == len(f['behavior_logprobs']), 'action probability length mismatch')
    require(f['action_mask'] == call['action_mask'], 'action mask differs')
    # This padded actor path supports one uninterrupted generated suffix only.
    require(call['action_mask'] == [1]*len(a) and call['token_ids'] == a,
            'interleaved observation/tool tokens unsupported; do not compress action history')
    require(f['behavior_logprobs'] == call['old_logprobs'], 'sampled behavior probabilities differ')
    for name in ('current_logprobs','reference_logprobs'):
        numeric(call[name], name, logprob=True)
        require(len(call[name]) == len(a), 'aligned '+name+' required')
    require(f['behavior_policy_version'] == call['behavior_policy_version'] and f['adapter_vector'] == call['adapter_vector']
            and f['probability_semantics'] == call['probability_semantics'], 'behavior identity mismatch')
    require(len(f['attention_mask']) == len(p)+len(a) and
            all(type(x) is int and x == 1 for x in f['attention_mask']), 'retained unpadded full attention required')
    numeric(f['position_ids'], 'retained positions', True, True)
    require(len(f['position_ids']) == len(p)+len(a), 'position length mismatch')
    tokenizer = manifest(f['tokenizer_manifest'], 'retained-tokenizer-v1')
    processor = manifest(f['processor_manifest'], 'retained-audio-processor-v1')
    require(type(tokenizer['vocab_size']) is int and tokenizer['vocab_size'] > max(p+a), 'token outside pinned vocabulary')
    require(type(tokenizer['pad_token_id']) is int and 0 <= tokenizer['pad_token_id'] < tokenizer['vocab_size'], 'invalid pinned pad token')
    require(tokenizer['pad_token_id'] == config['pad_token_id'], 'configured pad differs from tokenizer')
    require(processor['feature_layout'] == 'frames_bins' and processor['position_layout'] == 'retained_1d', 'unsupported feature/position layout; no invented conversion')
    frames = f['audio_features']; mask = f['feature_mask']
    require(isinstance(frames, list) and frames, 'missing actual audio features')
    for frame in frames:
        numeric(frame, 'audio feature')
        require(len(frame) == processor['feature_bins'], 'feature bin count differs')
    require(len(mask) == len(frames) and all(type(x) is int and x in (0,1) for x in mask), 'feature mask mismatch')
    require(mask == sorted(mask, reverse=True) and sum(mask) > 0 and f['effective_audio_lengths'] == [sum(mask)], 'effective audio length mismatch')
    schema = f['captured_tensor_schema']
    require(schema['processor_manifest_sha256'] == digest(f['processor_manifest']) and schema['consumer_contract_version'] == 'unified-omni-retained-padded-v1', 'captured tensor contract mismatch')
    require(schema['tensors'] == dict(
        audio_features=dict(shape=[len(frames),processor['feature_bins']],axes=['frames','bins']),
        feature_mask=dict(shape=[len(mask)],axes=['frames']),
        position_ids=dict(shape=[len(p)+len(a)],axes=['tokens']),
        effective_audio_lengths=dict(shape=[1],axes=['audio'])), 'captured tensor shape/axes mismatch')
    spans = []; audio_pad = processor['audio_pad_token_id']
    for index, token in enumerate(p):
        if token == audio_pad:
            if not spans or index != spans[-1][-1]+1:
                spans.append([])
            spans[-1].append(index)
    require(len(spans) == 1 and f['audio_token_lengths'] == list(map(len, spans)), 'ordered audio placeholder coverage differs')
    require(len(f['modality_assets']) == 1, 'this path requires exactly one retained audio asset')
    asset = f['modality_assets'][0]
    read_pin(asset)
    require(asset['media_type'] == 'audio' and asset['recording_id'] == f['recording_id'], 'audio identity mismatch')
    require(f['sampling_distribution'] == config['sampling_distribution'], 'unregistered effective sampling distribution')
    no_labels(call['online_input'], 'call.online_input')
    task_keys = {'P':{'retrieved_candidates'},'S1':{'type_universe'},'S2':{'supplied_types'},
                 'D':{'actual_candidates','actual_slot_values'}}
    require(set(call['online_input']) == {'messages','audio_assets'} | task_keys[call['node']], 'explicit label-free input envelope required')
    no_labels(f['prompt_messages'], 'prompt_messages')
    require(f['prompt_messages'] == call['online_input']['messages'] and f['modality_assets'] == call['online_input']['audio_assets'], 'prompt/audio differs from actual online input')
    require(isinstance(f['prompt_text'], str) and f['prompt_text'] and isinstance(f['action_text'], str), 'retained prompt/action text required')
    # A pre-existing capture must bind exact online input, rendered prompt,
    # action text, tokenizer/processor, multimodal arrays and output together.
    binding = f['capture_binding']
    expected = dict(online_input_sha256=digest(call['online_input']), output_sha256=digest(call['output']),
                    fields_sha256=digest({k:v for k,v in f.items() if k != 'capture_binding'}))
    require(binding == expected, 'prompt/action/modality capture binding differs')
    require(f['input_boundary'] == 'online_only_no_reference' and f['capture_origin'] in ('runtime_capture', 'synthetic_schema_fixture'), 'explicit input boundary/capture provenance required')
    require(f['action_text'] == f['decoded_action_text'], 'captured decode differs from action text')
    require(f['action_text'] == call['raw_output'], 'action text differs from observed raw output')
    return f


def restore(files, config):
    rows = {}
    for name in ('online', 'offline', 'provenance'):
        if 'file_pins' in config:
            ref = config['file_pins'][name]
            require(Path(files[name]).resolve() == Path(ref['path']).resolve(), 'configured file path differs')
        else:
            # dataset.publish determines the output directory from its config.
            # Pin actual completed files here to avoid a config/hash cycle.
            path = Path(files[name]).resolve()
            raw = path.read_bytes()
            ref = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        rows[name] = json.loads(read_pin(ref))
        require(isinstance(rows[name], list) and len(rows[name]) == files['count'], 'separated file count mismatch')
    require(files['count'] > 0, 'empty dataset')
    indexes = {}
    for name, items in rows.items():
        indexes[name] = {row['draw_index']:row for row in items}
        require(len(indexes[name]) == len(items) and set(indexes[name]) == set(range(files['count'])), 'draw join incomplete or duplicated')
    groups = []; synthetic = set(); seen_draws = set()
    for draw in range(files['count']):
        online, offline, provenance = (indexes[k][draw] for k in ('online','offline','provenance'))
        require(online['sample_id'] == offline['sample_id'] == provenance['sample_id'], 'sample/provenance join mismatch')
        for key in ('recording_ref_id', 'source_id', 'source', 'domain', 'outcome', 'duplicate_occurrence',
                    'conditional_within_cell_probability', 'random_output_position_probability', 'inclusion_probability'):
            require(key in provenance, 'missing sampling provenance '+key)
        require(provenance['duplicate_occurrence'] == 1, 'fresh group replacement/reuse forbidden')
        for key in ('conditional_within_cell_probability','random_output_position_probability','inclusion_probability'):
            if provenance[key] is None and key != 'conditional_within_cell_probability':
                require(bool(provenance.get('unavailable_probability_reason')), 'missing unavailable sampling probability reason')
                continue
            require(type(provenance[key]) in (int,float) and math.isfinite(provenance[key]) and 0 < provenance[key] <= 1, 'invalid sampling probability')
        no_labels(online['online'], 'online.input')
        bundle = copy.deepcopy(online['behavior_bundle'])
        require(bundle is not None and bundle['schema'] == 'unified-behavior-bundle-v1', 'missing complete fresh behavior group')
        require(bundle['online_gold_injected'] is False and bundle['global_query_count'] == 1 and len(bundle['groups']) == 1, 'one whole group per draw required')
        require(type(bundle['synthetic']) is bool, 'explicit synthetic provenance required')
        synthetic.add(bundle['synthetic'])
        group = bundle['groups'][0]
        require(group['recording_id'] == provenance['recording_ref_id'], 'recording provenance mismatch')
        require(group['query_draw_id'] not in seen_draws, 'query_draw reused')
        seen_draws.add(group['query_draw_id'])
        supervision = offline['group_supervision']
        require(set(supervision) == {e['episode_id'] for e in group['episodes']}, 'offline episode roster mismatch')
        for episode in group['episodes']:
            require('offline' not in episode, 'offline labels leaked into online bundle')
            episode['offline'] = supervision[episode['episode_id']]
            require(episode.get('query_draw_id',group['query_draw_id']) == group['query_draw_id'], 'episode query_draw mismatch')
            for call in episode['calls']:
                fields = validate_fields(call, config)
                require(fields['recording_id'] == group['recording_id'], 'call/audio recording mismatch')
                require(bundle['synthetic'] == (fields['capture_origin'] == 'synthetic_schema_fixture'), 'synthetic capture masquerading as real')
        groups.append(group)
    require(len(synthetic) == 1, 'mixed synthetic/real groups')
    return dict(schema='unified-behavior-bundle-v1', synthetic=synthetic.pop(), online_gold_injected=False,
                global_query_count=len(groups), groups=groups), [indexes['provenance'][i] for i in range(files['count'])]


def build(files, config):
    """Return actor tensor/non-tensor dictionaries plus audit report, CPU only.

    Callers may wrap these dictionaries in DataProto in the existing trainer.
    No SFC compose/compute_advantages is called; full-group unified values win.
    """
    bundle, provenance = restore(files, config)
    learning_config = json.loads(read_pin(config['learning_config_ref']))
    state = json.loads(read_pin(config['clock_state_ref']))
    with unified(config) as (learning, schedule, kernel):
        clock = schedule.Clock(learning_config, state)
        result = learning.evaluate(bundle, clock)
    require(result['queries'] == len(bundle['groups']) and result['global_episode_denominator'] == result['queries']*result['G'], 'fixed whole-group denominator mismatch')
    advantages = {(g['group_id'],e['episode_id']):e['advantage'] for g in result['groups'] for e in g['episodes']}
    snapshot = result['snapshot']
    selected = []; all_mapping = []
    for draw, group in enumerate(bundle['groups']):
        for episode in group['episodes']:
            for call in episode['calls']:
                mapping = dict(draw_index=draw, sample_id=provenance[draw]['sample_id'], group_id=group['group_id'],
                    query_draw_id=group['query_draw_id'], query_id=group['query_id'], recording_id=group['recording_id'],
                    sampling_round=group['sampling_round'], episode_id=episode['episode_id'], replicate=episode['replicate'],
                    node=call['node'], call_id=call['call_id'], request_identity=call['request_identity'],
                    source_token_indices=list(range(len(call['token_ids']))), parents=call['parents'],
                    trainable=snapshot['trainable_mask'][call['node']])
                all_mapping.append(mapping)
                if mapping['trainable']:
                    selected.append((call, mapping, advantages[group['group_id'],episode['episode_id']]))
    require(selected, 'no trainable calls')
    import numpy as np
    import torch
    pmax = max(len(c['training_fields']['prompt_token_ids']) for c,_,_ in selected)
    rmax = max(len(c['token_ids']) for c,_,_ in selected)
    width = max(len(c['training_fields']['audio_features']) for c,_,_ in selected)
    require(pmax+rmax <= config['context_limit'], 'context limit exceeded; never truncate')
    n = len(selected); pad = config['pad_token_id']
    tensors = {k:torch.full((n,pmax), pad, dtype=torch.long) for k in ('prompts',)}
    tensors['responses'] = torch.full((n,rmax),pad,dtype=torch.long)
    for name in ('attention_mask','position_ids'):
        tensors[name] = torch.zeros((n,pmax+rmax),dtype=torch.long)
    tensors['response_mask'] = torch.zeros((n,rmax),dtype=torch.long)
    for name in ('old_log_probs','ref_log_prob','advantages','unified_policy_node_weight','unified_kl_node_weight'):
        tensors[name] = torch.zeros((n,rmax),dtype=torch.float32)
    non = {k:np.empty(n,dtype=object) for k in ('multi_modal_inputs','uid','record_identity','adapter_route')}
    token_mapping = []
    for row,(call,mapping,advantage) in enumerate(selected):
        f=call['training_fields']; plen=len(f['prompt_token_ids']); length=len(f['action_token_ids']); offset=pmax-plen
        tensors['prompts'][row,offset:] = torch.tensor(f['prompt_token_ids'])
        tensors['responses'][row,:length] = torch.tensor(f['action_token_ids'])
        tensors['attention_mask'][row,offset:pmax+length] = torch.tensor(f['attention_mask'])
        tensors['position_ids'][row,offset:pmax+length] = torch.tensor(f['position_ids'])
        tensors['response_mask'][row,:length] = 1  # only padding mask; real action masks were validated, not fabricated.
        tensors['old_log_probs'][row,:length] = torch.tensor(f['behavior_logprobs'])
        tensors['ref_log_prob'][row,:length] = torch.tensor(call['reference_logprobs'])
        tensors['advantages'][row,:length] = advantage
        weight = snapshot['loss_node_weights'][call['node']]
        tensors['unified_policy_node_weight'][row,:length] = weight
        tensors['unified_kl_node_weight'][row,:length] = weight if snapshot['loss_node_kl_weighting']=='policy_and_kl' else 1.
        features=torch.tensor(f['audio_features'],dtype=torch.float32).T.unsqueeze(0)
        mask=torch.tensor([f['feature_mask']],dtype=torch.long)
        non['multi_modal_inputs'][row] = dict(input_features=torch.nn.functional.pad(features,(0,width-features.shape[-1])),
                                            feature_attention_mask=torch.nn.functional.pad(mask,(0,width-mask.shape[-1])))
        non['uid'][row] = mapping['query_draw_id']  # never re-normalize by this UID downstream.
        non['record_identity'][row] = mapping
        routing=snapshot['adapter_routing']
        non['adapter_route'][row] = routing['trainable_adapters'][call['node']]
        token_mapping.extend(dict(actor_row=row,response_token_index=t,**mapping,source_token_index=t) for t in range(length))
    tensors['input_ids']=torch.cat((tensors['prompts'],tensors['responses']),dim=-1)
    require(all(torch.isfinite(t).all().item() for t in tensors.values()), 'nonfinite tensor after dtype conversion')
    meta=dict(audio_contract='unified-omni-retained-padded-v1',requires_remove_padding=False,requires_sequence_parallel_size=1,
              group_size=result['G'],global_query_count=result['queries'],global_episode_denominator=result['global_episode_denominator'],
              reduction=result['reduction'],snapshot=snapshot,beta=result['beta'],kl_estimator=learning_config['kernel']['kl_estimator'],
              advantages_source='unified-learning.learning.evaluate',recompute_advantages=False,
              trainer_integrated=False,training_ready=False,synthetic=bundle['synthetic'])
    report=dict(consumer=CONSUMER_ID,loaded_count=files['count'],required_fields_validated=True,actual_model_trainer_loader=False,
                status='analysis_only',trainer_integrated=False,training_ready=False,synthetic=bundle['synthetic'],
                actor_rows=n,action_tokens=len(token_mapping),all_call_mapping=all_mapping,token_mapping=token_mapping,
                tensor_shapes={k:list(v.shape) for k,v in tensors.items()},global_episode_denominator=meta['global_episode_denominator'],
                snapshot=snapshot,provenance=provenance,source='existing unified evaluate; no new reward/loss implementation',
                gaps=['actual actor forward/update not integrated or executed','query/G token-sum and KL native loss bridge missing',
                      'shared/per-node LoRA route execution and parameter freezing unverified','captured text/token/feature parity not recomputed'])
    return dict(tensors=tensors,non_tensors=non,meta_info=meta),report


def validate_dataset(*, files, config):
    """Exact dataset.py loader plugin API. Readiness remains false on success."""
    try:
        _, report = build(files, config)
        return report
    except (ValueError, KeyError, TypeError, OSError, AssertionError, IndexError) as error:
        return dict(consumer=CONSUMER_ID,loaded_count=0,required_fields_validated=False,actual_model_trainer_loader=False,
                    status='analysis_only',training_ready=False,trainer_integrated=False,error=str(error))


def to_dataproto(fields):
    """Optional actual verl transport only; not called by CPU validation."""
    from verl import DataProto
    return DataProto.from_dict(tensors=fields['tensors'],non_tensors=fields['non_tensors'],meta_info=fields['meta_info'])
