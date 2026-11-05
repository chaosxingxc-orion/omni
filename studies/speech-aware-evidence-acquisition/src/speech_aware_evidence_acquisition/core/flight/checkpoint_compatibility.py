"""Verify an explicitly reviewed, operational-only checkpoint transition.

Protocol hashes include attempt/readiness identities, so a fresh attempt moves
that hash even when the scientific method is unchanged. Recompose the pinned
predecessor, compare every configuration field, and bind the reviewed source
changes before allowing its stored rows into the successor. No model contacts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Mapping

SCHEMA = 'saea-checkpoint-compatibility-v1'
MULTI_SCHEMA = 'saea-checkpoint-compatibility-v2'
RELEASE_SCHEMA = 'saea-checkpoint-image-release-v1'
PREFIX = 'src/speech_aware_evidence_acquisition/'
IMPLEMENTATION = {'runtime': PREFIX + 'core/flight/runtime.py',
                  'runner': PREFIX + 'rails/runner.py',
                  'row_checkpoint': PREFIX + 'core/row_checkpoint.py'}
REVIEWABLE = {PREFIX + 'core/flight/' + name + '.py' for name in
              ('checkpoint_compatibility', 'runtime', 'resume', 'campaign', 'cli')} | {
                  PREFIX + 'reproduction/ojw/runner.py'}
FRAGMENTS = ('model', 'dataset', 'baseline', 'experiment')


class CheckpointCompatibilityError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CheckpointCompatibilityError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + str(repo), *args], cwd=repo)


def _declaration_map(document):
    from .resume import resume_declarations_from_document, ResumeRecoveryError
    try:
        declarations = resume_declarations_from_document(document)
    except ResumeRecoveryError as error:
        raise CheckpointCompatibilityError(str(error)) from error
    require(all(item.mode == 'checkpoint' for item in declarations),
            'compatibility requires checkpoint predecessors')
    return {item.attempt_id: {'attempt_id': item.attempt_id,
                             'checkpoint_sha256': item.checkpoint_sha256}
            for item in declarations}


def validate_bridge_declarations(compatibility, declarations):
    require(bool(declarations) and all(item.mode == 'checkpoint' for item in declarations),
            'compatibility requires checkpoint predecessors')
    ids = {item.attempt_id for item in declarations}
    require(compatibility.successor_attempt_id not in ids, 'successor cannot be its own predecessor')
    if isinstance(compatibility, MultiCheckpointCompatibility):
        require(set(compatibility.bridges) == ids,
                'multi compatibility bridges must exactly match resume predecessors')
    else:
        require(len(declarations) == 1, 'v1 compatibility requires exactly one checkpoint predecessor')


@dataclass(frozen=True)
class CheckpointCompatibility:
    predecessor_commit: str
    predecessor_fragments: Mapping[str, str]
    successor_attempt_id: str
    change: str
    reviewed_source_changes: Mapping[str, Mapping[str, str | None]]

    @classmethod
    def from_document(cls, document, *, attempt_id=None):
        if isinstance(document, Mapping) and document.get('schema') == RELEASE_SCHEMA:
            return ImageReleaseCompatibility.from_document(document, attempt_id=attempt_id)
        if isinstance(document, Mapping) and document.get('schema') == MULTI_SCHEMA:
            return MultiCheckpointCompatibility.from_document(document, attempt_id=attempt_id)
        fields = {'schema', 'predecessor_commit', 'predecessor_fragments',
                  'successor_attempt_id', 'change', 'reviewed_source_changes'}
        require(isinstance(document, Mapping) and set(document) == fields
                and document.get('schema') == SCHEMA, 'invalid checkpoint compatibility fields/schema')
        require(isinstance(document['predecessor_commit'], str)
                and re.fullmatch('[0-9a-f]{40}', document['predecessor_commit']), 'invalid predecessor commit')
        fragments = document['predecessor_fragments']
        require(isinstance(fragments, Mapping) and set(fragments) == set(FRAGMENTS)
                and all(isinstance(v, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', v)
                        for v in fragments.values()), 'invalid predecessor fragment names')
        successor = document['successor_attempt_id']
        require(isinstance(successor, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', successor),
                'invalid successor attempt')
        require(attempt_id is None or attempt_id == successor, 'successor attempt differs')
        require(document['change'] == 'contact-executor-width-v1', 'unreviewed transition kind')
        changes = document['reviewed_source_changes']
        require(isinstance(changes, Mapping) and set(changes) <= REVIEWABLE,
                'unreviewed source path')
        for value in changes.values():
            require(isinstance(value, Mapping) and set(value) == {'before', 'after'}, 'invalid source pin')
            require(value['before'] is None or isinstance(value['before'], str)
                    and re.fullmatch('[0-9a-f]{64}', value['before']), 'invalid before source hash')
            require(isinstance(value['after'], str) and re.fullmatch('[0-9a-f]{64}', value['after']),
                    'invalid after source hash')
        return cls(document['predecessor_commit'], dict(fragments), successor,
                   document['change'], copy.deepcopy(dict(changes)))

    def verify(self, *, repo_root, data_root, current_fragments, header, run_id,
               predecessor_attempt_id, current_protocol_hash, current_split_identity_hash,
               current_implementation, declaration):
        """Return predecessor identity only after validating actual files on both sides.

        Current protocol is recomputed, never embedded in the config itself (that
        would be a self-referential hash). Reviewed source pins are external to
        source bytes. All other source files and configuration values must match.
        """
        from ..config import compose, protocol_hash
        repo, data = Path(repo_root), Path(data_root).resolve()
        require(set(current_fragments) == set(FRAGMENTS), 'missing current fragments')
        require(header.get('run_id') == run_id and header.get('attempt_id') == predecessor_attempt_id,
                'predecessor checkpoint run/attempt differs')
        require(header.get('split_identity_hash') == current_split_identity_hash, 'split changed')
        record = json.loads((data / 'attempts' / run_id / (predecessor_attempt_id + '.json')).read_bytes())
        require(record.get('run_id') == run_id and record.get('attempt_id') == predecessor_attempt_id
                and record.get('state') == 'ABORTED', 'predecessor must be ABORTED')
        current = compose(repo, **current_fragments)
        cells = current.values.get('cells', [])
        require(len(cells) == 1, 'registered compatibility requires one runtime cell')
        registered = cells[0].get('checkpoint_compatibility')
        if isinstance(registered, Mapping) and registered.get('schema') == MULTI_SCHEMA:
            multi = MultiCheckpointCompatibility.from_document(registered,
                                                                attempt_id=self.successor_attempt_id)
            require(multi.bridges.get(predecessor_attempt_id) == self,
                    'registered compatibility bridge differs from runtime')
        else:
            require(self.from_document(registered,
                    attempt_id=self.successor_attempt_id) == self,
                    'registered compatibility differs from runtime')
        require(protocol_hash(current.config_hash, current.values['arm'], current_split_identity_hash)
                == current_protocol_hash, 'current protocol does not match actual config')
        names = git(repo, 'ls-tree', '-r', '--name-only', self.predecessor_commit, '--', 'configs', 'src').decode().splitlines()
        # One git archive avoids thousands of individual process launches. Its
        # member names come from the pinned commit, not user-supplied paths.
        import io
        import tarfile
        with tempfile.TemporaryDirectory(prefix='checkpoint-source-') as tmp:
            old_root = Path(tmp)
            archive = git(repo, 'archive', self.predecessor_commit, '--', 'configs', 'src')
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(old_root, filter='data')
            changing_configs = {'configs/' + kind + '/' + name + '.json'
                                for kind in ('baseline', 'experiment')
                                for name in (self.predecessor_fragments[kind], current_fragments[kind])}
            for path in names:
                if path.startswith('configs/') and path not in changing_configs:
                    require((repo / path).is_file() and (old_root / path).read_bytes() == (repo / path).read_bytes(),
                            'referenced configuration changed: ' + path)
            previous = compose(old_root, **self.predecessor_fragments)
            require(previous.config_hash == header.get('config_hash'), 'predecessor config hash differs')
            require(protocol_hash(previous.config_hash, previous.values['arm'], current_split_identity_hash)
                    == header.get('protocol_hash'), 'predecessor protocol differs')
            original_impl = {key: sha((old_root / path).read_bytes()) for key, path in IMPLEMENTATION.items()}
            require(original_impl == header.get('implementation'), 'checkpoint source hashes differ from predecessor commit')
            require({key: sha((repo / path).read_bytes()) for key, path in IMPLEMENTATION.items()}
                    == dict(current_implementation), 'current implementation hash differs')
            old_source = {path for path in names if path.startswith('src/')}
            current_source = {str(p.relative_to(repo)) for p in (repo / 'src').rglob('*')
                              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
            changes = {}
            for path in old_source | current_source:
                before = (old_root / path).read_bytes() if path in old_source else None
                after = (repo / path).read_bytes() if path in current_source else None
                if before != after:
                    changes[path] = {'before': sha(before) if before is not None else None,
                                     'after': sha(after) if after is not None else None}
            require(changes == self.reviewed_source_changes, 'actual source delta differs from reviewed pins')
            old_docs = {kind: json.loads((old_root / 'configs' / kind / (self.predecessor_fragments[kind] + '.json')).read_bytes()) for kind in FRAGMENTS}
            new_docs = {kind: json.loads((repo / 'configs' / kind / (current_fragments[kind] + '.json')).read_bytes()) for kind in FRAGMENTS}
            verify_config_delta(old_docs, new_docs, data=data, run_id=run_id,
                                predecessor_attempt_id=predecessor_attempt_id,
                                successor_attempt_id=self.successor_attempt_id, declaration=declaration)
        return {'protocol_hash': header['protocol_hash'], 'implementation': original_impl,
                'predecessor_commit': self.predecessor_commit, 'change': self.change,
                'reviewed_source_changes': self.reviewed_source_changes}



@dataclass(frozen=True)
class MultiCheckpointCompatibility:
    """A reviewed bridge for every checkpoint predecessor in a multi-hop resume.

    The v1 bridge remains unchanged.  v2 carries one v1-shaped bridge per
    predecessor attempt, so each pinned commit/config/source delta is verified
    independently before ``MergedPredecessors`` combines rows.
    """
    successor_attempt_id: str
    bridges: Mapping[str, CheckpointCompatibility]

    @classmethod
    def from_document(cls, document, *, attempt_id=None):
        require(isinstance(document, Mapping)
                and document.get('schema') == MULTI_SCHEMA
                and set(document) == {'schema', 'successor_attempt_id', 'bridges'},
                'invalid multi checkpoint compatibility fields/schema')
        successor = document['successor_attempt_id']
        require(isinstance(successor, str)
                and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', successor),
                'invalid successor attempt')
        require(attempt_id is None or attempt_id == successor,
                'successor attempt differs')
        raw = document['bridges']
        require(isinstance(raw, Mapping) and raw, 'multi compatibility requires bridges')
        bridges = {}
        for predecessor, value in raw.items():
            require(isinstance(predecessor, str)
                    and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', predecessor),
                    'invalid predecessor attempt key')
            # The nested bridge is deliberately v1-shaped and names the same
            # successor. This prevents a bridge from silently targeting a
            # different recovery attempt.
            require(isinstance(value, Mapping) and value.get('schema') in (SCHEMA, RELEASE_SCHEMA),
                    'nested compatibility bridge must use v1 schema or image-release schema')
            bridge = CheckpointCompatibility.from_document(value, attempt_id=successor)
            bridges[predecessor] = bridge
        require(successor not in bridges, 'multi compatibility cannot name itself as predecessor')
        return cls(successor, bridges)

    def validate_ancestry(self, repo, declarations):
        """Require a closed, acyclic ancestry graph from actual pinned configs."""
        validate_bridge_declarations(self, declarations)
        declared = {item.attempt_id: {'attempt_id': item.attempt_id,
                                     'checkpoint_sha256': item.checkpoint_sha256}
                    for item in declarations}
        graph = {}
        for predecessor, bridge in self.bridges.items():
            path = 'configs/experiment/' + bridge.predecessor_fragments['experiment'] + '.json'
            if isinstance(bridge, ImageReleaseCompatibility):
                old_files = bridge.old_source()
                document = json.loads(old_files[STUDY_PREFIX + path])
            else:
                document = json.loads(git(repo, 'show', bridge.predecessor_commit + ':' + path))
            cells = document.get('cells', [])
            require(len(cells) == 1 and cells[0].get('attempt_id') == predecessor,
                    'pinned predecessor attempt differs from bridge key')
            value = cells[0].get('resume_from')
            ancestors = _declaration_map(value) if value is not None else {}
            require(all(declared.get(key) == value for key, value in ancestors.items()),
                    'resume omits or changes a pinned ancestor')
            graph[predecessor] = set(ancestors)
        visiting, visited = set(), set()
        def visit(node):
            require(node not in visiting, 'cyclic checkpoint ancestry')
            if node in visited:
                return
            visiting.add(node)
            for ancestor in graph[node]:
                visit(ancestor)
            visiting.remove(node)
            visited.add(node)
        for predecessor in graph:
            visit(predecessor)

    def bridge_for(self, predecessor_attempt_id):
        try:
            return self.bridges[predecessor_attempt_id]
        except KeyError as error:
            raise CheckpointCompatibilityError(
                f'no compatibility bridge for predecessor {predecessor_attempt_id!r}'
            ) from error


def verify_config_delta(old, new, *, data, run_id, predecessor_attempt_id,
                        successor_attempt_id, declaration, allow_dispatch=False):
    """Reject any change beyond fresh bookkeeping and output-identical supply."""
    before, after = copy.deepcopy(old), copy.deepcopy(new)
    require(before['model'] == after['model'], 'model changed')
    require(before['dataset'] == after['dataset'], 'dataset changed')
    old_base, new_base = before['baseline'], after['baseline']
    if allow_dispatch:
        from ...rails.dispatch_slot_fill import parse_dispatch_slot_fill
        from ...rails.errors import RailsError
        for obj in (old_base, new_base):
            dispatch = obj.pop('dispatch', None)
            if dispatch is not None:
                try:
                    parse_dispatch_slot_fill({'dispatch': dispatch})
                except RailsError as exc:
                    raise CheckpointCompatibilityError('invalid dispatch configuration') from exc
    for key in ('run_id', 'attempt_id'):
        require(isinstance(new_base['native_contact_fit'][key], str)
                and new_base['native_contact_fit'][key]
                and new_base['native_contact_fit'][key] != old_base['native_contact_fit'][key], 'invalid new FIT identity')
        new_base['native_contact_fit'][key] = old_base['native_contact_fit'][key]
    old_supply, new_supply = old_base['prepared_phase_supply'], new_base['prepared_phase_supply']
    outputs = []
    for pin in (old_supply, new_supply):
        require(isinstance(pin, dict) and set(pin) == {'relpath', 'sha256'}, 'invalid supply pin')
        path = (data / pin['relpath']).resolve()
        require(path.is_relative_to(data.resolve()) and sha(path.read_bytes()) == pin['sha256'], 'supply pin differs')
        outputs.append(json.loads(path.read_bytes())['output'])
    require(outputs[0] == outputs[1], 'prepared supply output changed')
    new_base['prepared_phase_supply'] = old_supply
    require(old_base == new_base, 'baseline semantics changed')
    old_exp, new_exp = before['experiment'], after['experiment']
    require(len(old_exp['cells']) == len(new_exp['cells']) == 1, 'exact one-cell transition required')
    old_cell, new_cell = old_exp['cells'][0], new_exp['cells'][0]
    require(old_cell['run_id'] == new_cell['run_id'] == run_id, 'run changed')
    require(old_cell['attempt_id'] == predecessor_attempt_id and new_cell['attempt_id'] == successor_attempt_id
            and successor_attempt_id != predecessor_attempt_id, 'attempt transition differs')
    declared = _declaration_map(new_cell.get('resume_from'))
    require(declared.get(predecessor_attempt_id) == declaration, 'checkpoint declaration differs')
    require(successor_attempt_id not in declared, 'successor cannot be its own predecessor')
    ancestors = _declaration_map(old_cell.get('resume_from')) if old_cell.get('resume_from') is not None else {}
    require(predecessor_attempt_id not in ancestors, 'predecessor cannot be its own ancestor')
    require(all(declared.get(key) == value for key, value in ancestors.items()),
            'resume omits or changes a pinned ancestor')
    require('checkpoint_compatibility' in new_cell, 'missing registered compatibility')
    new_cell.pop('resume_from')
    new_cell.pop('checkpoint_compatibility')
    old_cell.pop('resume_from', None)
    old_cell.pop('checkpoint_compatibility', None)
    new_cell['attempt_id'] = old_cell['attempt_id']
    new_cell['baseline_fragment'] = old_cell['baseline_fragment']
    # Readiness remains independently checked by the ordinary campaign gates;
    # it cannot alter requests or the full roster and may pin fresh evidence.
    new_cell['readiness'] = old_cell['readiness']
    require(old_exp == new_exp, 'experiment semantics changed')

# Offline release migration primitives. These functions do not settle attempts,
# alter leases or authorize activation. Inputs are explicit immutable captures.
STUDY_PREFIX = 'studies/speech-aware-evidence-acquisition/'


def _git_object(kind, blob):
    return hashlib.sha1((kind + ' ' + str(len(blob)) + '\0').encode() + blob).hexdigest()


def authenticate_git_capture(commit, commit_object, listing_text, source_tar):
    """Authenticate a recursive Git tree and every supplied source blob."""
    import tarfile
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'invalid commit')
    raw = Path(commit_object).read_bytes()
    require(_git_object('commit', raw) == commit, 'commit object differs')
    lines = raw.splitlines()
    require(lines and lines[0].startswith(b'tree '), 'invalid commit tree')
    expected = lines[0][5:].decode('ascii')
    listing = {}; directories = {}
    for line in Path(listing_text).read_text().splitlines():
        attrs, path = line.split('\t', 1)
        mode, kind, oid = attrs.split()
        require(kind == 'blob' and mode in ('100644', '100755', '120000') and
                re.fullmatch('[0-9a-f]{40}', oid) and path not in listing, 'invalid tree entry')
        parts = path.split('/')
        require(all(x not in ('', '.', '..') for x in parts), 'invalid tree path')
        listing[path] = (mode, oid)
        parent = '/'.join(parts[:-1]); directories.setdefault(parent, {})[parts[-1]] = (mode, oid)
        for i in range(len(parts)-1): directories.setdefault('/'.join(parts[:i]), {}).setdefault(parts[i], ('40000', None))
    oids = {}
    for directory in sorted(directories, key=lambda s: len(s.split('/')) if s else 0, reverse=True):
        entries = directories[directory]; tree = b''
        for name, (mode, oid) in sorted(entries.items(), key=lambda item: item[0] + ('/' if item[1][0] == '40000' else '')):
            if mode == '40000': oid = oids[(directory + '/' if directory else '') + name]
            tree += mode.encode() + b' ' + name.encode() + b'\0' + bytes.fromhex(oid)
        oids[directory] = _git_object('tree', tree)
    require(oids.get('') == expected, 'Git root tree differs')
    files = {}
    with tarfile.open(source_tar) as archive:
        for member in archive:
            if not member.isfile():
                require(member.isdir(), 'source archive link or special file refused')
                continue
            name = member.name
            require(name in listing and name not in files and listing[name][0] != '120000', 'untracked or duplicate source blob')
            content = archive.extractfile(member).read()
            require(_git_object('blob', content) == listing[name][1], 'Git blob differs: ' + name)
            files[name] = content
    required_source = {name for name, (mode, _) in listing.items() if name.startswith(STUDY_PREFIX + 'src/') and mode != '120000'}
    require(required_source <= set(files), 'authenticated source archive incomplete')
    return files, listing


def authenticated_old_config(files, names, header, startup_data_root):
    """Recompose authenticated old fragments without reading external resolver env."""
    from ..config import _parse_fragment, _compose_fragments, protocol_hash, drop_null_arms
    from ..reasoning_knobs import fold_into_decoding
    from ..data_root_placeholder import collapse_data_root
    require(isinstance(startup_data_root, str) and startup_data_root.startswith('/') and startup_data_root != '/', 'startup data root required')
    docs = {}
    for kind in FRAGMENTS:
        name = names[kind]
        require(isinstance(name, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', name), 'invalid fragment name')
        key = STUDY_PREFIX + 'configs/' + kind + '/' + name + '.json'
        require(key in files, 'authenticated fragment missing: ' + kind)
        docs[kind] = _parse_fragment(files[key].decode('utf-8'), Path(key))
    # Collapse explicitly before composition, so neither successor data-root nor
    # external asset/workspace environment can change the predecessor hash.
    normalized = {key: collapse_data_root(value, startup_data_root) for key, value in docs.items()}
    composed = _compose_fragments(names, normalized, startup_data_root)
    merged = {}
    for kind in FRAGMENTS:
        require(not (set(merged) & set(normalized[kind])), 'fragment collision')
        merged.update(normalized[kind])
    merged = drop_null_arms(merged)
    merged = fold_into_decoding(merged) or merged
    config_hash = sha(json.dumps(merged, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
    require(config_hash == header.get('config_hash'), 'predecessor config hash differs')
    require(protocol_hash(config_hash, composed.values['arm'], header['split_identity_hash']) == header.get('protocol_hash'), 'predecessor protocol differs')
    composed = type(composed)(values=composed.values, config_hash=config_hash, fragments=composed.fragments)
    impl = {}
    for key, path in IMPLEMENTATION.items():
        source = STUDY_PREFIX + path
        require(source in files, 'authenticated implementation missing: ' + key)
        impl[key] = sha(files[source])
    require(impl == header.get('implementation'), 'predecessor implementation differs')
    return docs, composed


def verify_release_reference_closure(*, repo_root, old_files, references):
    """Compare every explicitly declared, pinned referenced asset using the official resolver.

    Callers must supply the complete reference closure; missing paths, mismatched
    hashes, or image fallback for an unpinned asset fail closed.
    """
    from ..experiment_assets import resolve_asset
    require(isinstance(references, Mapping) and references, 'reference closure required')
    result = {}
    for path, pin in references.items():
        rel = Path(path)
        require(not rel.is_absolute() and '..' not in rel.parts and path.startswith(('configs/', 'docs/')), 'invalid reference path')
        require(isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin), 'reference pin missing')
        old_key = STUDY_PREFIX + path
        require(old_key in old_files and sha(old_files[old_key]) == pin, 'old reference differs: ' + path)
        current = resolve_asset(repo_root, path)
        require(current.is_file() and sha(current.read_bytes()) == pin, 'current reference differs: ' + path)
        result[path] = pin
    return result


def verify_reviewed_source_delta(*, old_files, release_files, reviewed):
    """Require exact source delta with per-file immutable review and risk type."""
    prefix = STUDY_PREFIX + 'src/'
    old = {key[len(STUDY_PREFIX):]: sha(value) for key, value in old_files.items() if key.startswith(prefix)}
    new = {key[len(STUDY_PREFIX):]: value for key, value in release_files.items() if key.startswith(prefix)}
    changes = {key: {'before': old.get(key), 'after': new.get(key)} for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    require(isinstance(reviewed, Mapping) and set(reviewed) == set(changes), 'reviewed source delta incomplete')
    for path, actual in changes.items():
        entry = reviewed[path]
        require(isinstance(entry, Mapping) and set(entry) == {'before', 'after', 'type', 'review_sha256'}, 'invalid source review')
        require({k:entry[k] for k in ('before','after')} == actual, 'source review hashes differ')
        require(entry['type'] in ('operational-packaging', 'scheduling', 'trust-boundary'), 'unreviewed source change type')
        require(isinstance(entry['review_sha256'], str) and re.fullmatch('[0-9a-f]{64}', entry['review_sha256']), 'review evidence pin required')
    return changes


def referenced_repo_assets(value):
    """Discover direct repository template/receipt references in four fragments."""
    paths = set()
    def walk(item):
        if isinstance(item, Mapping):
            for key, child in item.items():
                if isinstance(child, str):
                    if key in ('instruction_template', 'task_instruction_template'):
                        paths.add('configs/' + child)
                    elif key == 'roster_receipt': paths.add(child)
                    elif key == 'runtime_receipt': paths.add('docs/' + child)
                walk(child)
        elif isinstance(item, list):
            for child in item: walk(child)
    walk(value)
    return paths


def verify_release_precheck(*, commit, commit_object, listing_text, source_tar, old_names,
                            header, startup_data_root, repo_root, current_names, reference_pins,
                            release_files, reviewed_source_changes):
    """Static precheck; NEVER an activation or terminal-state authorization."""
    from ..config import read_fragment
    old_files, _ = authenticate_git_capture(commit, commit_object, listing_text, source_tar)
    old_docs, old_composed = authenticated_old_config(old_files, old_names, header, startup_data_root)
    new_docs = {kind: read_fragment(repo_root, kind, current_names[kind]) for kind in FRAGMENTS}
    required = referenced_repo_assets(old_docs) | referenced_repo_assets(new_docs)
    require(required <= set(reference_pins), 'referenced asset pin missing')
    verify_release_reference_closure(repo_root=repo_root, old_files=old_files, references=reference_pins)
    changes = verify_reviewed_source_delta(old_files=old_files, release_files=release_files, reviewed=reviewed_source_changes)
    return {'predecessor_commit': commit, 'predecessor_config_hash': old_composed.config_hash,
            'reference_count':len(reference_pins), 'reviewed_source_count':len(changes),
            'activation_authorized':False, 'scientific_config_equivalence_verified':False,
            'reason':'scientific config delta and request-body equivalence require independent validation'}


def _pinned_bytes(pin):
    require(isinstance(pin, Mapping) and set(pin) == {'path', 'sha256'}, 'invalid evidence pin')
    require(isinstance(pin['path'], str) and Path(pin['path']).is_absolute(), 'evidence path must be absolute')
    require(isinstance(pin['sha256'], str) and re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'invalid evidence hash')
    path = Path(pin['path'])
    require(not path.is_symlink() and path.is_file(), 'pinned evidence missing')
    blob = path.read_bytes()
    require(sha(blob) == pin['sha256'], 'pinned evidence differs')
    return blob


def _release_inventory(repo_root, pin):
    """Use the release's own inventory verifier after authenticating its bytes."""
    import importlib.util
    root = Path(repo_root).resolve().parents[1]
    require(root / STUDY_PREFIX.rstrip('/') == Path(repo_root).resolve(), 'release study path differs')
    require(Path(pin['path']).resolve() == root / 'source-manifest.json', 'release manifest location differs')
    raw = _pinned_bytes(pin); doc = json.loads(raw)
    require(doc.get('schema') == 'saea-release-source-v1', 'invalid release manifest')
    verifier = root / 'verify-release.py'
    require(not verifier.is_symlink() and verifier.is_file() and
            sha(verifier.read_bytes()) == doc.get('files', {}).get('docker/saea-base/release/verify-release.py'), 'release verifier differs')
    spec = importlib.util.spec_from_file_location('_saea_release_verify', verifier)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    try:
        module.verify(pin['path'], root)
    except (ValueError, OSError, KeyError) as exc:
        raise CheckpointCompatibilityError('release source verification failed') from exc
    return doc['files']


def _verify_review(pin, old_files, release_files):
    doc = json.loads(_pinned_bytes(pin))
    require(isinstance(doc, Mapping) and set(doc) == {'schema', 'changes'} and
            doc['schema'] == 'saea-reviewed-source-delta-v1', 'invalid source review')
    changes = doc['changes']
    require(isinstance(changes, Mapping), 'invalid source review entries')
    old = {key[len(STUDY_PREFIX):]:sha(value) for key,value in old_files.items() if key.startswith(STUDY_PREFIX + 'src/')}
    new = {key[len(STUDY_PREFIX):]:value for key,value in release_files.items() if key.startswith(STUDY_PREFIX + 'src/')}
    actual = {key:{'before':old.get(key),'after':new.get(key)} for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    require(set(changes) == set(actual), 'reviewed source delta incomplete')
    for path, entry in changes.items():
        require(isinstance(entry, Mapping) and set(entry) == {'before','after','type','rationale'} and
                {k:entry[k] for k in ('before','after')} == actual[path] and
                entry['type'] in ('operational-packaging','scheduling','trust-boundary') and
                isinstance(entry['rationale'], str) and len(entry['rationale'].strip()) >= 12,
                'invalid reviewed source entry: ' + path)
    return len(actual)


def _verify_data_closure(data_root, documents, *, repo_root=None):
    """Validate declared asset pins and knowledge-store loader dependencies."""
    data = Path(data_root).resolve(); seen = {}; pending = []
    def scan(value):
        if isinstance(value, Mapping):
            rel = value.get('data_relpath', value.get('relpath'))
            if rel is not None and 'sha256' in value:
                require(isinstance(rel, str) and isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}',value['sha256']), 'invalid data reference pin')
                root_kind = 'data' if 'data_relpath' in value else value.get('root', 'data')
                require(root_kind in ('data', 'repository'), 'unknown reference root')
                pending.append((root_kind, rel, value['sha256']))
            for child in value.values(): scan(child)
        elif isinstance(value, list):
            for child in value: scan(child)
    from ...rails.knowledge_stores import load_knowledge_store_binding
    def scan_config(value):
        if isinstance(value, Mapping):
            if 'knowledge_stores' in value:
                load_knowledge_store_binding(value, data_root=data)
                stores = value['knowledge_stores']
                for spec in stores.values():
                    for key, child in spec.items():
                        if key not in ('snapshot', 'release', 'alternates'):
                            scan(child)
                for key, child in value.items():
                    if key != 'knowledge_stores': scan_config(child)
            else:
                for child in value.values(): scan_config(child)
                if ('relpath' in value or 'data_relpath' in value) and 'sha256' in value:
                    rel = value.get('data_relpath', value.get('relpath'))
                    kind = 'data' if 'data_relpath' in value else value.get('root', 'data')
                    require(kind in ('data','repository') and isinstance(rel,str) and isinstance(value['sha256'],str) and re.fullmatch('[0-9a-f]{64}',value['sha256']), 'invalid data reference pin')
                    pending.append((kind, rel, value['sha256']))
        elif isinstance(value, list):
            for child in value: scan_config(child)
    scan_config(documents)
    while pending:
        root_kind,rel,digest = pending.pop(); path = Path(rel)
        require(not path.is_absolute() and '..' not in path.parts, 'invalid data reference')
        key = root_kind + ':' + rel
        require(key not in seen or seen[key] == digest, 'conflicting data reference pin')
        if key in seen: continue
        if root_kind == 'repository':
            require(repo_root is not None, 'repository reference root required')
            from ..experiment_assets import resolve_asset
            target = resolve_asset(repo_root, rel)
        else:
            target = data / path
            require(target.resolve().is_relative_to(data), 'data reference escapes root')
        require(not target.is_symlink() and target.is_file(), 'data reference unavailable: ' + rel)
        h = hashlib.sha256()
        with target.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        require(h.hexdigest() == digest, 'data reference differs: ' + rel)
        seen[key] = digest
        # Official preparation validates explicitly declared source receipts.
        # Their internal lineage is provenance, not recursively loaded inputs;
        # generic recursion would promote archival references into live assets.
    return seen


@dataclass(frozen=True)
class ImageReleaseCompatibility(CheckpointCompatibility):
    evidence: Mapping[str, object]

    @classmethod
    def from_document(cls, document, *, attempt_id=None):
        keys = {'schema','predecessor_commit','predecessor_fragments','successor_attempt_id','change','evidence'}
        require(isinstance(document, Mapping) and set(document) == keys and document['schema'] == RELEASE_SCHEMA, 'invalid image release bridge')
        base = CheckpointCompatibility.from_document({'schema':SCHEMA,'predecessor_commit':document['predecessor_commit'],
            'predecessor_fragments':document['predecessor_fragments'],'successor_attempt_id':document['successor_attempt_id'],
            'change':'contact-executor-width-v1','reviewed_source_changes':{}}, attempt_id=attempt_id)
        require(document['change'] == 'image-release-operational-v1', 'unreviewed release transition')
        evidence = document['evidence']
        require(isinstance(evidence, Mapping) and set(evidence) == {'commit_object','tree_listing','source_archive','release_manifest','source_review','reference_pins','startup_data_root'}, 'invalid release bridge evidence')
        for key in ('commit_object','tree_listing','source_archive','release_manifest','source_review','reference_pins'):
            pin = evidence[key]
            require(isinstance(pin, Mapping) and set(pin) == {'path','sha256'} and isinstance(pin['path'],str) and Path(pin['path']).is_absolute() and isinstance(pin['sha256'],str) and re.fullmatch('[0-9a-f]{64}',pin['sha256']), 'invalid evidence pin')
        require(isinstance(evidence['startup_data_root'],str) and evidence['startup_data_root'].startswith('/') and evidence['startup_data_root'] != '/', 'invalid startup data root')
        return cls(base.predecessor_commit,base.predecessor_fragments,base.successor_attempt_id,document['change'],{},copy.deepcopy(dict(evidence)))

    def old_source(self):
        e = self.evidence
        for key in ('commit_object','tree_listing','source_archive'): _pinned_bytes(e[key])
        return authenticate_git_capture(self.predecessor_commit,e['commit_object']['path'],e['tree_listing']['path'],e['source_archive']['path'])[0]

    def precheck(self, *, repo_root, data_root, current_fragments, header, run_id,
                 predecessor_attempt_id, current_protocol_hash, current_split_identity_hash,
                 current_implementation, declaration):
        from ..config import compose, read_fragment, protocol_hash
        repo = Path(repo_root); data = Path(data_root).resolve()
        require(header.get('run_id') == run_id and header.get('attempt_id') == predecessor_attempt_id and header.get('split_identity_hash') == current_split_identity_hash, 'predecessor identity differs')
        old = self.old_source(); old_docs, previous = authenticated_old_config(old,self.predecessor_fragments,header,self.evidence['startup_data_root'])
        release = _release_inventory(repo,self.evidence['release_manifest'])
        reviews = _verify_review(self.evidence['source_review'], old, release)
        require({key:sha((repo/path).read_bytes()) for key,path in IMPLEMENTATION.items()} == dict(current_implementation), 'current implementation differs')
        new = {kind:read_fragment(repo,kind,current_fragments[kind]) for kind in FRAGMENTS}
        current = compose(repo,**current_fragments,data_root=data)
        require(protocol_hash(current.config_hash,current.values['arm'],current_split_identity_hash) == current_protocol_hash, 'current protocol differs')
        cells = current.values.get('cells', [])
        require(len(cells)==1, 'registered compatibility requires one runtime cell')
        registered = cells[0].get('checkpoint_compatibility')
        if isinstance(registered,Mapping) and registered.get('schema') == MULTI_SCHEMA:
            require(MultiCheckpointCompatibility.from_document(registered,attempt_id=self.successor_attempt_id).bridges.get(predecessor_attempt_id) == self, 'registered bridge differs')
        else:
            require(CheckpointCompatibility.from_document(registered,attempt_id=self.successor_attempt_id) == self, 'registered bridge differs')
        refs = json.loads(_pinned_bytes(self.evidence['reference_pins']))
        require(isinstance(refs,Mapping) and (referenced_repo_assets(old_docs) | referenced_repo_assets(new)) <= set(refs), 'referenced asset pin missing')
        verify_release_reference_closure(repo_root=repo,old_files=old,references=refs)
        verify_config_delta(old_docs,new,data=data,run_id=run_id,predecessor_attempt_id=predecessor_attempt_id,successor_attempt_id=self.successor_attempt_id,declaration=declaration,allow_dispatch=True)
        closure = _verify_data_closure(data, [old_docs,new], repo_root=repo)
        return {'protocol_hash':header['protocol_hash'],'implementation':header['implementation'],
                'predecessor_commit':self.predecessor_commit,'change':self.change,
                'release_manifest_sha256':self.evidence['release_manifest']['sha256'],
                'reviewed_source_count':reviews,'reference_count':len(refs),'data_pin_count':len(closure),
                'activation_authorized':False,'request_body_equivalence':'requires final integration evidence'}

    def verify(self, **kwargs):
        data = Path(kwargs['data_root']).resolve(); run = kwargs['run_id']; attempt = kwargs['predecessor_attempt_id']
        record = json.loads((data/'attempts'/run/(attempt+'.json')).read_bytes())
        require(record.get('run_id') == run and record.get('attempt_id') == attempt and record.get('state') == 'ABORTED', 'predecessor must be ABORTED')
        return self.precheck(**kwargs)
