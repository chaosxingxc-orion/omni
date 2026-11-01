"""CPU validation of frozen-audio preparation; no production weight training."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch', reason='audio boundary checks run in the pinned verl image')

SPEC = importlib.util.spec_from_file_location('omni_audio_boundary',
    Path(__file__).resolve().parents[2]/'scripts/environment/verl_omni/audio_feature_boundary.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ThinkerFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(audio_token_id=5, image_token_id=6, video_token_id=7)
        self.audio_tower = torch.nn.Linear(3, 4).eval()
        self.embeddings = torch.nn.Embedding(8, 4)
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embeddings

    def get_audio_features(self, input_features, feature_attention_mask, **kwargs):
        values = input_features.transpose(1, 2)[feature_attention_mask.bool()]
        return SimpleNamespace(last_hidden_state=self.audio_tower(values))

    def get_rope_index(self, input_ids, attention_mask, audio_seqlens, **kwargs):
        assert audio_seqlens.tolist() == [2]
        positions = (attention_mask.cumsum(-1)-1).clamp_min(0)
        return positions.unsqueeze(0).expand(3, -1, -1), torch.zeros((1, 1), dtype=torch.long)


def batch():
    return {'input_ids': torch.tensor([[1, 5, 5, 2]]), 'attention_mask': torch.ones(1, 4, dtype=torch.long),
            'response_mask': torch.tensor([[0, 0, 0, 1]]), 'input_features': torch.ones(1, 3, 2),
            'feature_attention_mask': torch.ones(1, 2, dtype=torch.long)}


def test_variable_audio_lengths_survive_generic_dim0_collation():
    examples = [{'input_features': torch.ones(1, 3, n),
                 'feature_attention_mask': torch.ones(1, n, dtype=torch.long)} for n in (2, 5)]
    padded = m.collate_audio_features(examples)
    assert torch.cat([r['input_features'] for r in padded]).shape == (2, 3, 5)
    assert torch.cat([r['feature_attention_mask'] for r in padded]).sum(-1).tolist() == [2, 5]
    assert examples[0]['input_features'].shape[-1] == 2


def test_frozen_boundary_preserves_text_and_replaces_exact_audio_positions():
    model = ThinkerFixture()
    source = batch()
    result = m.frozen_audio_inputs(model, **source)
    original = model.get_input_embeddings()(source['input_ids'])
    assert torch.equal(result['inputs_embeds'][:, [0, 3]], original[:, [0, 3]])
    assert not torch.equal(result['inputs_embeds'][:, 1:3], original[:, 1:3])
    assert not result['inputs_embeds'].requires_grad
    assert set(m.forward_inputs(result)) == {'input_ids', 'inputs_embeds', 'attention_mask', 'position_ids'}
    assert torch.equal(result['loss_mask'], source['response_mask'])


@pytest.mark.parametrize('fault', ['trainable_tower', 'trainable_embedding', 'tower_train_mode',
                                  'image', 'audio_loss', 'placeholder_length', 'mask_gap', 'nan'])
def test_rejects_changed_audio_or_alignment_contract(fault):
    model, source = ThinkerFixture(), batch()
    if fault == 'trainable_tower': model.audio_tower.requires_grad_(True)
    elif fault == 'trainable_embedding': model.embeddings.requires_grad_(True)
    elif fault == 'tower_train_mode': model.audio_tower.train()
    elif fault == 'image': source['input_ids'][0, 0] = 6
    elif fault == 'audio_loss': source['response_mask'][0, 1] = 1
    elif fault == 'placeholder_length': source['input_ids'][0, 1] = 1
    elif fault == 'mask_gap': source['feature_attention_mask'][0, :] = torch.tensor([0, 1])
    elif fault == 'nan': source['input_features'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        m.frozen_audio_inputs(model, **source)
