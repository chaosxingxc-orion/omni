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
            require(isinstance(value, Mapping) and value.get('schema') == SCHEMA,
                    'nested compatibility bridge must use v1 schema')
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
                        successor_attempt_id, declaration):
    """Reject any change beyond fresh bookkeeping and output-identical supply."""
    before, after = copy.deepcopy(old), copy.deepcopy(new)
    require(before['model'] == after['model'], 'model changed')
    require(before['dataset'] == after['dataset'], 'dataset changed')
    old_base, new_base = before['baseline'], after['baseline']
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
