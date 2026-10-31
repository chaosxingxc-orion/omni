"""Fold A registration preserves drafts and admits only the exact unspent plan."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/flights/nx'))
import fold_a_encoding_admission as admission

DATA = Path('/mnt/cephfs/xingchao/omni')
LOCK = DATA / admission.draft.LIVE_LOCK_RELPATH
AT = '2026-10-30T22:00:00+00:00'


@pytest.fixture
def inputs(tmp_path):
    d = admission.draft
    if not (DATA / d.INVENTORY_RELPATH).exists():
        pytest.skip('Fold A source inventory is not mounted')
    repo, data = tmp_path / 'repo', tmp_path / 'data'
    for relative in (d.FOLD_B_FRAGMENT, d.FOLD_B_REGISTRATION, d.FOLD_A_FRAGMENT, d.FOLD_A_REGISTRATION,
                     d.FOLD_A_LEDGER_ROW, d.ROSTER_RECEIPT, d.DATASET_CONFIG, d.HISTORICAL_REGISTRATION, admission.AUTHORITY,
                     'configs/experiment/e0-model-free-closure.yaml'):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    (repo / admission.LEDGER).write_text('# Exposure ledger\n')
    encoder = json.loads((repo / d.FOLD_A_FRAGMENT).read_bytes())['encoder']['path'].removeprefix('${SAEA_DATA_ROOT}/')
    for relative in (d.INVENTORY_RELPATH, encoder):
        target = data / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / relative, target)
    return repo, data, LOCK


def build(inputs):
    return admission.build_registration(*inputs, head='a' * 40, at=AT)


def install(inputs):
    repo, _, _ = inputs
    for relative, blob in build(inputs).items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)


def test_exact_registration_preserves_drafts_and_passes_real_tool_gate(inputs):
    repo, data, _ = inputs
    before = (repo / admission.draft.FOLD_A_REGISTRATION).read_bytes()
    output = build(inputs)
    assert output == build(inputs)
    assert set(output) == {admission.REGISTRATION, admission.LEDGER}
    assert (repo / admission.draft.FOLD_A_REGISTRATION).read_bytes() == before
    assert (repo / admission.LEDGER).read_text() == '# Exposure ledger\n'
    install(inputs)
    cfg = admission.runtime_config(repo, data, head='b' * 40)
    assert cfg['study_head'] == 'b' * 40
    assert cfg['plan']['planned_model_calls'] == 1080
    assert cfg['plan']['planned_speech_audio_seconds'] == 3085
    assert cfg['plan']['run_id'] == admission.draft.FOLD_A_RUN
    assert not (data / 'attempts').exists()


@pytest.mark.parametrize('folder', ['attempts', 'preparation-runs', 'prepared', 'runs'])
def test_existing_encoder_identity_prevents_registration(inputs, folder):
    _, data, _ = inputs
    (data / folder / admission.draft.FOLD_A_RUN).mkdir(parents=True)
    with pytest.raises(ValueError, match='already has artifacts'):
        build(inputs)


def test_duplicate_registration_and_ledger_refused(inputs):
    repo, _, _ = inputs
    install(inputs)
    with pytest.raises(ValueError, match='already registered'):
        build(inputs)
    (repo / admission.REGISTRATION).unlink()
    with pytest.raises(ValueError, match='ledger row'):
        build(inputs)


def test_source_inventory_tampering_refused(inputs):
    _, data, _ = inputs
    path = data / admission.draft.INVENTORY_RELPATH
    doc = json.loads(path.read_bytes())
    doc['rows'][0]['source_frames'] += 1
    path.write_text(json.dumps(doc))
    with pytest.raises((ValueError, SystemExit)):
        build(inputs)


def test_runtime_config_refuses_modified_plan_and_consumed_ledger(inputs):
    repo, data, _ = inputs
    install(inputs)
    path = repo / admission.REGISTRATION
    original = json.loads(path.read_bytes())
    modified = copy.deepcopy(original)
    modified['plans'][0]['execution_plan']['planned_model_calls'] = 1081
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match='differs from verified frozen draft'):
        admission.runtime_config(repo, data, head='b' * 40)
    path.write_text(json.dumps(original))
    ledger = repo / admission.LEDGER
    ledger.write_text(ledger.read_text().replace(' | no | ', ' | yes | '))
    from speech_aware_evidence_acquisition.contracts.errors import GateClosed
    with pytest.raises(GateClosed, match='consumed'):
        admission.runtime_config(repo, data, head='b' * 40)


def test_encoder_inventory_drift_refused(inputs):
    repo, data, _ = inputs
    spec = json.loads((repo / admission.draft.FOLD_A_FRAGMENT).read_bytes())
    path = data / spec['encoder']['path'].removeprefix('${SAEA_DATA_ROOT}/')
    doc = json.loads(path.read_bytes())
    doc['first_core'] = 0
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='encoder inventory differs'):
        build(inputs)
