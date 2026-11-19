"""Replay retained audio-only inputs with vLLM's CPU processor, without weights."""
import argparse
import hashlib
import json
from pathlib import Path


def checked(root, ref):
    path = root / ref['file']
    if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != ref['sha256']:
        raise ValueError('retained input digest mismatch')
    return path


def compare(left, right):
    import torch
    if left.shape != right.shape:
        return {'left_shape': list(left.shape), 'right_shape': list(right.shape), 'equal': False}
    delta = (left.float() - right.float()).abs()
    return {'shape': list(left.shape), 'equal': torch.equal(left, right),
            'unequal': int(torch.count_nonzero(delta)), 'max_abs': float(delta.max())}


def run(config_path, model_path):
    import torch
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.multimodal.media.audio import AudioMediaIO
    from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeThinkerForConditionalGeneration
    spec = json.loads(config_path.read_text())
    root = config_path.parent
    cfg = ModelConfig(model=model_path, dtype='bfloat16', max_model_len=24576)
    processor = MULTIMODAL_REGISTRY.create_processor(cfg)
    # Position calculation uses only thinker config and the processed MM item.
    thinker = Qwen3OmniMoeThinkerForConditionalGeneration.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    thinker.config = processor.info.get_hf_config()
    rows = []
    if 'fresh_group' in spec:
        import sys
        sys.path.insert(0,str(root/'source'))
        from cohort_capture import prepare_packet
        hf_processor=processor.info.get_hf_processor()
        manifest=json.loads(checked(root,spec['fresh_group']).read_bytes())
        fresh=[]
        representatives=(r for r in manifest['rows'] if r['replicate']==0)
        behavior_rows={r['request_id']:r for r in json.loads(checked(root,spec['sampled']).read_bytes())['fresh_group']['records']}
        for row in representatives:
            packet=row['packet'];prepared=prepare_packet(hf_processor, packet)
            action=behavior_rows[packet['request_id']]
            fresh.append(dict(node='P',packet=packet,prepared=prepared,action=dict(action,prompt_token_ids=row['prompt_token_ids'])))
    else:
        fresh=[]
    for row in spec.get('rows',[]) + fresh:
        if 'action' in row:
            packet,prepared,action=row['packet'],row['prepared'],row['action']
        else:
            packet = json.loads(checked(root, row['packet']).read_bytes())
            behavior = json.loads(checked(root, row['behavior']).read_bytes())
            action = behavior['records'][row['action_index']]
            prepared = torch.load(checked(root, row['prepared']), map_location='cpu', weights_only=False)
            if packet['wire_sha256'] != action['wire_sha256'] or prepared['input_identity'] != action['input_identity']:
                raise ValueError('packet and retained action differ')
        audios = [part['input_audio'] for msg in packet['wire']['messages'] if isinstance(msg['content'], list)
                  for part in msg['content'] if part['type'] == 'input_audio']
        if len(audios) != 1:
            raise ValueError('single audio boundary required')
        audio = audios[0]
        wave, rate = AudioMediaIO().load_base64('audio/'+audio['format'], audio['data'])
        mm = processor.info.parse_mm_data({'audio': (wave, rate)})
        output = processor(prepared['prompt'], mm)
        ids = output['prompt_token_ids']
        item = output['mm_kwargs']['audio'][0]
        place = output['mm_placeholders']['audio'][0]
        feature = MultiModalFeatureSpec(data=item, modality='audio', identifier='cpu-replay', mm_position=place)
        pos, offset = thinker.get_mrope_input_positions(ids, [feature])
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration as HF
        from types import SimpleNamespace
        hf = SimpleNamespace(config=thinker.config, spatial_merge_size=thinker.config.vision_config.spatial_merge_size)
        full = torch.tensor([action['prompt_token_ids'] + action['token_ids']])
        # Both installed methods express destinations in prompt coordinates.
        # Compare those destinations independently of the audio tower values.
        prompt_ids = full[:, :len(ids)]
        _, _, hf_mask = HF.get_placeholder_mask(hf, prompt_ids, prompt_ids.unsqueeze(-1))
        hf_rows = torch.where(hf_mask[0, :, 0])[0].tolist()
        vl_ranges = place.extract_embeds_range()
        vl_rows = [row_id for start, end in vl_ranges for row_id in range(start, end + 1)]
        mapping = dict(equal=hf_rows == vl_rows, hf_count=len(hf_rows),
                       vllm_count=len(vl_rows), vllm_ranges=vl_ranges)
        if not hf_rows or len(vl_rows) != place.get_num_embeds():
            raise ValueError('missing audio placeholder rows or inconsistent range')
        hf_pos, _ = HF.get_rope_index(hf, input_ids=full, attention_mask=torch.ones_like(full),
                         audio_seqlens=prepared['encoded']['feature_attention_mask'].sum(-1), use_audio_in_video=False)
        enc = prepared['encoded']
        valid = enc['feature_attention_mask'][0].bool()
        left = enc['input_features'][0][:, valid]
        right = item['input_audio_features'].data
        rec = dict(node=row['node'], prompt_ids_equal=ids == action['prompt_token_ids'],
                   wire_sha256=action['wire_sha256'], encoded_identity=prepared.get('encoded_identity'),
                   audio_length=int(item['audio_feature_lengths'].data.item()), train_frames=int(valid.sum()),
                   waveform=compare(torch.as_tensor(prepared['waves'][0][:len(wave)]), torch.as_tensor(wave)),
                   features=compare(left, right), bf16_features=compare(left.bfloat16(), right.bfloat16()),
                   prompt_positions=compare(hf_pos[:, 0, :len(ids)].long(), pos.long()), position_shape=list(pos.shape), vllm_position_delta=offset,
                   vllm_positions_sha256=hashlib.sha256(pos.long().contiguous().numpy().tobytes()).hexdigest(),
                   audio_placeholder_mapping=mapping, selected_response_tokens=len(action['token_ids']))
        rows.append(rec)
    return dict(schema='retained-audio-cpu-processor-replay-v1', records=rows,
                used_model_weights=False, original_worker_intermediates_observed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.model)))
