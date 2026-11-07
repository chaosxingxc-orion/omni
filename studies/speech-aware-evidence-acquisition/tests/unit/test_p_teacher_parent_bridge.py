"""Real parent patch, identity refusals, and native resolver integration."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY / 'scripts/analysis'))
sys.path.insert(0, str(STUDY / 'src'))
import p_teacher_bridge as bridge
from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
from speech_aware_evidence_acquisition.rails.phase_material import load_phase_material_specs, render_static_instruction

BASE = Path(os.environ.get('BRIDGE_REAL_INPUT', '/nonexistent'))
OUTPUT = Path(os.environ.get('BRIDGE_REAL_OUTPUT', '/nonexistent'))
pytestmark = pytest.mark.skipif(not BASE.is_file(), reason='requires pinned actual parent and compiled output')

@pytest.mark.parametrize('name', ['gepa3', 'reflexion-c1', 'reflexion-c2'])
def test_actual_patch_and_native_read(name):
    raw = BASE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '3933c579ec1610744a8202b267aae0f096df534d51284ebdb057afa274cff89a'
    parent = json.loads(raw)
    folder = OUTPUT / name
    patch = json.loads((folder / 'patch.json').read_bytes())
    result = json.loads((folder / 'result.json').read_bytes())
    applied = bridge.apply_parent_patch(parent, patch)
    old, new = (json.dumps(p['value']).encode() for p in patch)
    assert raw.count(old) == 1
    assert (folder / 'baseline.json').read_bytes() == raw.replace(old, new, 1)
    assert json.loads((folder / 'baseline.json').read_bytes()) == applied
    restored = copy.deepcopy(applied)
    restored['phase_material']['perception']['instruction_template'] = patch[0]['value']
    assert restored == parent and 'perception_policy' not in applied
    spec = load_phase_material_specs(applied, default_phase_program(applied['policy']))['perception']
    text = render_static_instruction(spec, folder, {}, folder)
    assert hashlib.sha256(text.encode()).hexdigest() == result['material']['instruction_template']['sha256']
    for index, field, value in [(0, 'path', '/perception_policy/instruction_template'), (0, 'value', 'probes/old.txt')]:
        bad = copy.deepcopy(patch); bad[index][field] = value
        with pytest.raises((ValueError, KeyError)):
            bridge.apply_parent_patch(parent, bad)


def test_input_drift_and_template_hash_refused(tmp_path):
    cfg_path = Path(os.environ['BRIDGE_CONFIG'])
    cfg = json.loads(cfg_path.read_text())
    raw = BASE.read_bytes()
    drift = tmp_path / 'parent.json'; drift.write_bytes(raw + b' ')
    bad = copy.deepcopy(cfg); bad['parent_baseline']['path'] = str(drift)
    with pytest.raises(ValueError): bridge.compile_config(bad)
    bad = copy.deepcopy(cfg); bad['parent_template']['sha256'] = '0' * 64
    with pytest.raises(ValueError): bridge.compile_config(bad)
