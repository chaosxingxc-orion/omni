"""Durable CPU controller for GEPA; native execution remains a Flight concern.

One immutable snapshot is published per transition under a local writer lock.
An unobserved stage reserves its entire allocation and blocks every later stage.
The local lock serializes metadata; it does not replace the shared NPU lease.
Scores/provenance must be verified by the caller's native adapter before observe;
this controller validates the state machine, not native custody or audio splits.
"""
from __future__ import annotations

import copy
import fcntl
import json
import random
import re
from contextlib import contextmanager
from pathlib import Path

from ..core.atomic_file import publish_new
from .gepa_search import (ScoredCandidate, best_candidate, compare_minibatch,
                          random_from_state, select_parent)
from .tfgrpo_learning import canonical, sha

SCHEMA = 'saea-gepa-search-state-v1'
USAGE = ('calls', 'prompt_tokens', 'completion_tokens')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value), 'SHA256 required')
    return value


def ids(values, name):
    require(isinstance(values, (list, tuple)) and values
            and all(isinstance(x, str) and x.startswith('slurp:train:') for x in values),
            name + ' requires nonempty TRAIN identities')
    require(len(set(values)) == len(values), name + ' contains duplicate identities')
    return list(values)


def usage(value):
    require(isinstance(value, dict) and set(value) == set(USAGE)
            and all(type(v) is int and v >= 0 for v in value.values()), 'exact nonnegative usage required')
    return copy.deepcopy(value)


def scored(candidate, sample_ids, scores):
    row = ScoredCandidate(candidate['candidate_id'], candidate['instruction_sha256'], sample_ids, scores)
    require(all(0 <= x <= 1 for x in row.scores), 'registered proxy scores must be in [0,1]')
    return row


def candidate(value):
    require(isinstance(value, dict) and set(value) == {'candidate_id', 'instruction_sha256'}
            and isinstance(value['candidate_id'], str) and value['candidate_id'].strip(),
            'exact candidate identity required')
    digest(value['instruction_sha256'])
    return copy.deepcopy(value)


class GepaSearchState:
    """Serial stage coordinator; stage allocations may cover parallel Solver calls.

    ``excluded_ids`` explicitly includes H2 identities, which use TRAIN prefixes.
    Corpus/audio duplicate isolation and native evidence validation belong to the
    registered partition builder/runtime; prefix checks alone cannot prove them.
    """

    @classmethod
    def create(cls, root, *, feedback_ids, selection_ids, excluded_ids,
               partition_sha256, frozen_system_sha256, initial_candidate,
               seed, batch_size, max_rounds, budget, inherited_usage):
        feedback = ids(feedback_ids, 'feedback'); selection = ids(selection_ids, 'selection')
        require(isinstance(excluded_ids, (list, tuple)) and all(isinstance(x, str) for x in excluded_ids),
                'explicit excluded evaluation identities required')
        require(not set(feedback) & set(selection)
                and not (set(feedback) | set(selection)) & set(excluded_ids), 'search/evaluation splits overlap')
        require(type(seed) is int and type(batch_size) is int and batch_size > 0
                and type(max_rounds) is int and max_rounds > 0, 'finite integer search bounds required')
        budget = usage(budget); require(all(v > 0 for v in budget.values()), 'positive search budget required')
        config = {'feedback_ids': feedback, 'selection_ids': selection,
            'excluded_ids': list(excluded_ids), 'partition_sha256': digest(partition_sha256),
            'frozen_system_sha256': digest(frozen_system_sha256),
            'initial_candidate': candidate(initial_candidate), 'seed': seed, 'batch_size': batch_size,
            'max_rounds': max_rounds, 'budget': budget, 'inherited_usage': usage(inherited_usage)}
        store = cls(root)
        with store._lock():
            require(not list(store.root.glob('*.json')), 'fresh GEPA state directory required')
            rng = random.Random(seed); order = list(feedback); rng.shuffle(order)
            state = {'phase': 'seed_eval', 'rng_state': rng.getstate(), 'feedback_order': order,
                'feedback_cursor': 0, 'rounds': [], 'pool': [], 'active': None, 'pending': None,
                'observed_stages': {}, 'known_usage': dict.fromkeys(USAGE, 0), 'failure': None}
            store._publish(None, 'create', config, state)
        return store

    def __init__(self, root):
        self.root = Path(root)

    @contextmanager
    def _lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root/'state.guard').open('a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _load(self):
        paths = sorted(self.root.glob('*.json'))
        require(paths, 'GEPA state absent')
        previous = None; config = None
        for index, path in enumerate(paths):
            require(path.name == f'{index:06d}.json', 'GEPA snapshot sequence has a gap or foreign file')
            doc = json.loads(path.read_bytes())
            require(doc.get('schema') == SCHEMA and doc.get('sequence') == index
                and doc.get('previous_sha256') == previous
                and doc.get('sha256') == sha({k:v for k,v in doc.items() if k != 'sha256'}),
                'GEPA snapshot integrity differs')
            if config is None: config = doc['config']
            require(doc['config'] == config, 'GEPA immutable configuration changed')
            previous = doc['sha256']
        random_from_state(doc['state']['rng_state'])
        return doc

    def _publish(self, previous, action, config, state):
        index = previous['sequence']+1 if previous else 0
        body = {'schema': SCHEMA, 'sequence': index, 'previous_sha256': previous['sha256'] if previous else None,
                'action': action, 'config': config, 'state': state}
        doc = {**body, 'sha256': sha(body)}
        publish_new(self.root/f'{index:06d}.json', canonical(doc))
        return json.loads(canonical(doc))

    def read(self):
        with self._lock(): return copy.deepcopy(self._load())

    def next_round(self):
        with self._lock():
            doc = self._load(); state = copy.deepcopy(doc['state']); config = doc['config']
            require(state['pending'] is None and state['failure'] is None, 'unknown or failed GEPA stage blocks continuation')
            if state['active'] is not None:
                # JSON publication normalizes RNG/selector tuples to lists. Return
                # the durable representation so an in-process retry equals a
                # fresh-process recovery byte-for-byte.
                return copy.deepcopy(state['active'])  # Do not redraw parent or batch after restart.
            require(state['phase'] == 'round_ready', 'GEPA is not ready for a round')
            require(len(state['rounds']) < config['max_rounds']
                    and state['feedback_cursor'] < len(state['feedback_order']), 'finite rounds/feedback exhausted')
            pool = [scored(r['candidate'], config['selection_ids'], r['scores']) for r in state['pool']]
            choice = select_parent(pool, config['selection_ids'], rng=random_from_state(state['rng_state']))
            start = state['feedback_cursor']; batch = state['feedback_order'][start:start+config['batch_size']]
            active = {'round_index': len(state['rounds']), 'parent': copy.deepcopy(state['pool'][choice.candidate_index]['candidate']),
                      'sample_ids': batch, 'selection': choice.to_dict()}
            state.update(active=active, phase='parent_eval', rng_state=choice.rng_after,
                         feedback_cursor=start+len(batch))
            published = self._publish(doc, 'next_round', config, state)
            return copy.deepcopy(published['state']['active'])

    def begin_stage(self, *, request_sha256, allocation):
        """Returns a new durable intent only once; dispatch belongs to its creator.

        Reopening with an existing intent raises even for identical input: until
        native reconciliation provides a response, a second dispatch is unsafe.
        """
        digest(request_sha256); allocation = usage(allocation)
        require(all(v > 0 for v in allocation.values()), 'positive stage reservation required')
        with self._lock():
            doc = self._load(); state = copy.deepcopy(doc['state']); config = doc['config']
            require(state['pending'] is None and state['failure'] is None, 'unknown or failed GEPA stage blocks continuation')
            phase = state['phase']
            require(phase in ('seed_eval', 'parent_eval', 'mutate', 'child_eval', 'selection_eval'),
                    'no executable GEPA stage')
            if phase in ('seed_eval', 'selection_eval'): samples = config['selection_ids']
            else: samples = state['active']['sample_ids']
            # One logical solver response per sample, one reflector response.
            require(allocation['calls'] >= (1 if phase == 'mutate' else len(samples)),
                    'stage allocation cannot cover complete sample roster')
            require(all(state['known_usage'][k]+allocation[k] <= config['budget'][k] for k in USAGE),
                    'GEPA budget cannot reserve whole next stage')
            body = {'phase': phase, 'round_index': len(state['rounds']), 'sample_ids': samples,
                    'request_sha256': request_sha256, 'allocation': allocation,
                    'frozen_system_sha256': config['frozen_system_sha256'], 'state_sha256': doc['sha256']}
            intent = {**body, 'stage_id': sha(body)}; state['pending'] = intent
            self._publish(doc, 'begin_stage', config, state)
            return copy.deepcopy(intent)

    def _finish_round(self, state, outcome):
        state['rounds'].append({**state['active'], 'outcome': outcome})
        state.update(active=None, phase='round_ready')

    def _apply_result(self, state, config, pending, result):
        phase = pending['phase']; active = state['active']
        if phase == 'mutate':
            require(set(result) == {'candidate'}, 'exact proposal result required')
            child = candidate(result['candidate'])
            if child['instruction_sha256'] in {x['candidate']['instruction_sha256'] for x in state['pool']}:
                self._finish_round(state, 'rejected-noop'); return
            require(child['candidate_id'] not in {x['candidate']['candidate_id'] for x in state['pool']},
                    'candidate ID collision')
            active['child'] = child; state['phase'] = 'child_eval'; return
        require(set(result) == {'candidate', 'sample_ids', 'scores'}, 'exact evaluation result required')
        expected = (config['initial_candidate'] if phase == 'seed_eval' else
                    active['parent'] if phase == 'parent_eval' else active['child'])
        require(candidate(result['candidate']) == expected and result['sample_ids'] == pending['sample_ids'],
                'evaluation candidate or complete ordered roster differs')
        evaluated = scored(expected, result['sample_ids'], result['scores'])
        if phase in ('seed_eval', 'selection_eval'):
            state['pool'].append({'candidate': expected, 'scores': list(evaluated.scores),
                                  'stage_id': pending['stage_id'], 'parent_id': active['parent']['candidate_id'] if active else None})
            if phase == 'seed_eval': state['phase'] = 'round_ready'
            else: self._finish_round(state, 'accepted')
        elif phase == 'parent_eval':
            active['parent_scores'] = list(evaluated.scores)
            if all(x == 1 for x in evaluated.scores): self._finish_round(state, 'skipped-perfect')
            else: state['phase'] = 'mutate'
        elif phase == 'child_eval':
            active['child_scores'] = list(evaluated.scores)
            decision = compare_minibatch(scored(active['parent'], active['sample_ids'], active['parent_scores']),
                                         evaluated, active['sample_ids'])
            if decision.accepted: state['phase'] = 'selection_eval'
            else: self._finish_round(state, decision.reason)

    def observe(self, stage_id, *, response_sha256, actual_usage, result):
        """Persist known response/cost even if malformed or over the allocation.

        ``response_sha256`` pins the external native/score evidence; the runtime
        must verify it. A failed observation is terminal for this search variant.
        """
        digest(response_sha256); actual_usage = usage(actual_usage)
        try:
            result = json.loads(canonical(result))
        except (ValueError, TypeError):
            # The raw native response remains pinned externally. Preserve a
            # diagnostic instead of letting a non-JSON score drop known spend.
            result = {'invalid_non_json_result_repr': repr(result)}
        observation = {'stage_id': stage_id, 'response_sha256': response_sha256,
                       'usage': actual_usage, 'result': copy.deepcopy(result)}
        with self._lock():
            doc = self._load(); state = copy.deepcopy(doc['state']); config = doc['config']
            old = state['observed_stages'].get(stage_id)
            if old is not None:
                require(old == observation, 'known GEPA response or cost changed')
                return copy.deepcopy(doc)
            pending = state['pending']
            require(pending is not None and pending['stage_id'] == stage_id, 'observation does not match pending stage')
            state['observed_stages'][stage_id] = observation
            for key in USAGE: state['known_usage'][key] += actual_usage[key]
            state['pending'] = None
            applied = copy.deepcopy(state)
            try:
                require(all(actual_usage[k] <= pending['allocation'][k] for k in USAGE), 'observed stage exceeds reservation')
                require(actual_usage['calls'] >= (1 if pending['phase'] == 'mutate' else len(pending['sample_ids'])),
                        'complete observed stage usage required')
                self._apply_result(applied, config, pending, result)
            except (ValueError, KeyError, TypeError) as error:
                state.update(failure={'stage_id': stage_id, 'reason': str(error)}, phase='failed')
            else: state = applied
            return self._publish(doc, 'observe', config, state)

    def finish(self):
        with self._lock():
            doc = self._load(); state = copy.deepcopy(doc['state']); config = doc['config']
            if state['phase'] == 'complete': return doc
            require(state['pending'] is None and state['failure'] is None and state['pool'],
                    'unknown, failed or uninitialized search cannot finish')
            pool = [scored(x['candidate'], config['selection_ids'], x['scores']) for x in state['pool']]
            best = best_candidate(pool, config['selection_ids'])
            state.update(phase='complete', winner_id=best.candidate_id,
                         cumulative_usage={k: state['known_usage'][k]+config['inherited_usage'][k] for k in USAGE})
            return self._publish(doc, 'finish', config, state)
