"""Local deployment profile: one entry, no local facts inside the base."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.experiments import runtime
from tools.experiments.config import ConfigError, load_config
from tools.experiments.local import SCHEMA


def profile(tmp_path, **changes):
    data = {
        'schema': SCHEMA,
        'hosts': {'training': 'train-host', 'rollout': 'rollout-host'},
        'storage': {'shared_root': str(tmp_path / 'shared'), 'hot_root': str(tmp_path / 'hot'),
                    'archive_root': str(tmp_path / 'archive')},
        'roots': {'model': str(tmp_path / 'models'), 'data': str(tmp_path / 'data')},
        'image': {'reference': 'fixture:1', 'digest': 'sha256:' + 'b' * 64},
        'identity': {'uid': 1000, 'gid': 1000},
    }
    data.update(changes)
    path = tmp_path / 'deployment.json'
    path.write_text(json.dumps(data))
    return path


def config(tmp_path, name='fixture'):
    shared = tmp_path / 'shared'
    data = tmp_path / 'data'
    (shared / 'manifests').mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)
    (data / 'manifest.json').write_text('{"samples": []}')
    cfg = {
        'schema_version': 1,
        'storage': {'shared_root': '${local.storage.shared_root}', 'run_root': '${shared_root}/runs'},
        'release': {'image_digest': '${local.image.digest}'},
        'experiment': {'id': name, 'model': 'fixture-no-model', 'thinking': False,
                       'dataset': {'manifest': '${local.roots.data}/manifest.json', 'version': 'fixture'},
                       'science': {'purpose': 'process regression only', 'host': '${local.hosts.training}'}},
        'execution': {'network': 'none', 'poll_seconds': 0.05,
                      'parallelism': {'cpu': 2, 'npu': 1},
                      'lanes': {'npu': [{'id': 'fixture', 'lock_path': '${run_root}/lanes/fixture.guard'}]}},
        'tasks': [{'id': 'one', 'resource': 'cpu', 'argv': ['/bin/true'], 'cwd': str(tmp_path)}],
    }
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps(cfg))
    return path, cfg


def test_local_values_expand_and_only_referenced_keys_are_frozen(tmp_path):
    doc = load_config(config(tmp_path)[0], profile(tmp_path))
    assert doc['local']['values'] == {
        'local.hosts.training': 'train-host',
        'local.roots.data': str(tmp_path / 'data'),
        'local.storage.shared_root': str(tmp_path / 'shared'),
        'local.image.digest': 'sha256:' + 'b' * 64,
    }
    cfg = doc['config']
    assert cfg['storage']['shared_root'] == str(tmp_path / 'shared')
    assert cfg['release']['image_digest'] == 'sha256:' + 'b' * 64
    assert cfg['experiment']['science']['host'] == 'train-host'
    assert doc['local']['local_sha256'] and doc['local']['file_sha256']


def test_unused_profile_edits_do_not_change_local_sha256(tmp_path):
    path, _ = config(tmp_path)
    first = load_config(path, profile(tmp_path))
    edited = profile(tmp_path)
    body = json.loads(edited.read_text())
    body['identity'] = {'uid': 4321, 'gid': 4321}
    edited.write_text(json.dumps(body))
    again = load_config(path, edited)
    assert again['local']['local_sha256'] == first['local']['local_sha256']
    assert again['local']['file_sha256'] != first['local']['file_sha256']
    body['roots']['data'] = str(tmp_path / 'other-data')
    edited.write_text(json.dumps(body))
    assert load_config(path, edited)['local']['local_sha256'] != first['local']['local_sha256']


def test_missing_local_profile_names_the_exact_key(tmp_path):
    with pytest.raises(ConfigError) as error:
        load_config(config(tmp_path)[0], None)
    assert 'local.storage.shared_root' in str(error.value)
    assert '--local-config' in str(error.value)


def test_profile_rejects_unknown_credential_macro_and_relative_fields(tmp_path):
    for change in ({'unknown': {}}, {'identity': {'uid': 1, 'gid': 1, 'token': 'x'}},
                   {'storage': {'shared_root': '${shared_root}', 'hot_root': '/hot', 'archive_root': '/arc'}},
                   {'roots': {'model': 'relative/models', 'data': '/data'}}):
        path = profile(tmp_path, **change)
        with pytest.raises(ConfigError):
            load_config(config(tmp_path)[0], path)


def test_prepare_freezes_local_snapshot_and_rejects_profile_drift(tmp_path):
    path, _ = config(tmp_path)
    doc = load_config(path, profile(tmp_path))
    receipt = runtime.prepare(doc, 'cpu')
    assert receipt['state'] == 'PREPARED'
    root = Path(doc['run_dir'])
    local = json.loads((root / 'LOCAL.json').read_text())
    lock = json.loads((root / 'CONFIG-LOCK.json').read_text())
    assert local['values'] == doc['local']['values']
    assert lock['local_sha256'] == doc['local']['local_sha256']
    assert 'path' not in local

    drifted = profile(tmp_path)
    body = json.loads(drifted.read_text())
    # A symlinked root resolves to the same bytes, so config_sha256 is unchanged
    # and only the frozen local identity can catch the drift.
    (tmp_path / 'data-link').symlink_to(tmp_path / 'data')
    body['roots']['data'] = str(tmp_path / 'data-link')
    drifted.write_text(json.dumps(body))
    changed = load_config(path, drifted)
    assert changed['sha256'] == doc['sha256']
    assert changed['local']['local_sha256'] != doc['local']['local_sha256']
    with pytest.raises(ConfigError) as error:
        runtime.assert_frozen(changed)
    assert 'local deployment profile changed' in str(error.value)


def test_observation_survives_profile_removal_through_snapshot(tmp_path):
    path, _ = config(tmp_path)
    deployment = profile(tmp_path)
    doc = load_config(path, deployment)
    runtime.prepare(doc, 'cpu')
    deployment.unlink()
    frozen = load_config(Path(doc['run_dir']) / 'CONFIG.json')
    observed = runtime.status(frozen, 'cpu')
    assert observed['prepared'] is True
    assert observed['config_sha256'] == doc['sha256']


def test_base_requests_only_declared_or_open_group_keys():
    """A host tool naming a key the schema rejects would fail only at deployment time.

    Extract every literal key the base asks the profile for -- both the shell
    bridge (``lp_get``/``lp_default``/``lp_opt``) and the Python bridge
    (``local_profile.get``) -- and hold it to the same declaration the loader
    enforces.  An open group (``paths.*``, ``ports.*``) is a deployment's own
    name by design; everything else must be a field the schema knows.
    """
    import re
    from tools.experiments.local import EXTRA_PREFIXES, FIELDS

    shell = re.compile(r"lp_(?:get|default|opt)\s+([a-z_]+(?:\.[a-z_.]+)+)")
    python = re.compile(r"local_profile\.(?:get|optional)\(\s*[\"']([a-z_.]+)[\"']")
    found = {}
    for path in sorted((REPO / 'scripts').rglob('*')):
        if not path.is_file() or path.suffix not in {'.py', '.sh', '.bash'}:
            continue
        for rule in (shell, python):
            for key in rule.findall(path.read_text(encoding='utf-8', errors='replace')):
                found.setdefault(key, set()).add(str(path.relative_to(REPO)))

    def declared(name):
        return name in FIELDS or any(name.startswith(p) and len(name) > len(p) for p in EXTRA_PREFIXES)

    unknown = {name: sorted(where) for name, where in found.items() if not declared(name)}
    assert unknown == {}, unknown
    assert 'storage.shared_root' in found and 'image.reference' in found


def test_committed_example_names_every_key_the_base_asks_for():
    """The template a newcomer copies must already carry every key the base reads.

    Otherwise "fill in this file" is a lie: the newcomer would hit a missing-key
    error on a key the example never mentioned.
    """
    import re

    example = json.loads((REPO / 'configs/local/deployment.example.json').read_text())

    def flat(value, prefix=''):
        for key, item in value.items():
            name = key if not prefix else prefix + '.' + key
            if isinstance(item, dict) and item:
                yield from flat(item, name)
            else:
                yield name

    present = {name for name in flat(example) if name != 'schema'}
    shell = re.compile(r"lp_(?:get|default|opt)\s+([a-z_]+(?:\.[a-z_.]+)+)")
    python = re.compile(r"local_profile\.(?:get|optional)\(\s*[\"']([a-z_.]+)[\"']")
    used = set()
    for path in sorted((REPO / 'scripts').rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.sh', '.bash'}:
            text = path.read_text(encoding='utf-8', errors='replace')
            used |= set(shell.findall(text)) | set(python.findall(text))
    assert used - present == set(), sorted(used - present)
