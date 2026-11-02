"""Pure CPU rules for GEPA Solver prompt search; no model/runtime integration.

Mechanism reference: arXiv:2507.19457v2, Algorithms 1--2, and gepa-ai/gepa
commit 8c7b697a71977980afc7537dac0881e784eef99f (gepa_utils.py). This is an
independent implementation of instance-winner cover pruning, not pairwise
numeric Pareto dominance. Equal-score pruning visits candidates in first
winner appearance order; within each instance, indices follow pool order.
This explicit ordering replaces the reference implementation's set iteration,
so the mechanism/weights match, but arbitrary tied set iteration is not a
promise of bit-for-bit upstream RNG equivalence. Pool order is creation order.

Only complete, finite, identically ordered evaluations enter selection. These
checks establish numerical contracts, not native provenance or gold isolation;
the eventual runtime must verify those before constructing these records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import re
from typing import Sequence

REFERENCE_COMMIT = '8c7b697a71977980afc7537dac0881e784eef99f'
SELECTOR_ID = 'gepa-8c7b697-cover-prune-ordered-v1'


def _ids(values, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'{name} must be a nonempty ordered list or tuple')
    result = tuple(values)
    if any(not isinstance(v, str) or not v.strip() for v in result):
        raise ValueError(f'{name} must contain nonempty string identities')
    if len(set(result)) != len(result):
        raise ValueError(f'{name} must be unique')
    return result


def _scores(values, count: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError('complete score vector required')
    result = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError('scores must be finite numbers, not bool or missing')
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError('scores must be finite') from exc
        if not math.isfinite(number):
            raise ValueError('scores must be finite')
        result.append(number)
    if not math.isfinite(sum(result)):
        raise ValueError('score aggregate must be finite')
    return tuple(result)


@dataclass(frozen=True)
class ScoredCandidate:
    """One instruction identity and a complete, defensively copied score vector."""

    candidate_id: str
    instruction_sha256: str
    sample_ids: tuple[str, ...]
    scores: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError('nonempty candidate identity required')
        if not isinstance(self.instruction_sha256, str) or not re.fullmatch(
                r'[0-9a-f]{64}', self.instruction_sha256):
            raise ValueError('lowercase instruction SHA256 required')
        ids = _ids(self.sample_ids, 'sample_ids')
        object.__setattr__(self, 'sample_ids', ids)
        object.__setattr__(self, 'scores', _scores(self.scores, len(ids)))

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores)


def _pool(candidates, selection_sample_ids) -> tuple[ScoredCandidate, ...]:
    expected = _ids(selection_sample_ids, 'selection_sample_ids')
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError('nonempty ordered candidate pool required')
    pool = tuple(candidates)
    if any(not isinstance(c, ScoredCandidate) for c in pool):
        raise ValueError('ScoredCandidate records required')
    if len({c.candidate_id for c in pool}) != len(pool):
        raise ValueError('duplicate candidate identity')
    if len({c.instruction_sha256 for c in pool}) != len(pool):
        raise ValueError('duplicate instruction/noop cannot enter pool')
    if any(c.sample_ids != expected for c in pool):
        raise ValueError('complete selection scores in EXACT registered sample order required')
    return pool


@dataclass(frozen=True)
class CoverPruneResult:
    """Indices refer to immutable creation order in the supplied pool."""

    candidate_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    mean_scores: tuple[float, ...]
    instance_winners: tuple[tuple[int, ...], ...]
    retained_winners: tuple[tuple[int, ...], ...]
    prune_order: tuple[int, ...]
    removed_indices: tuple[int, ...]
    sampling_indices: tuple[int, ...]
    sampling_weights: tuple[int, ...]

    def to_dict(self) -> dict:
        return {'selector': SELECTOR_ID, **asdict(self)}


def cover_prune(candidates: Sequence[ScoredCandidate], selection_sample_ids) -> CoverPruneResult:
    """Keep each sample covered, removing lower-average redundant winners first.

    Sampling weight is the number of surviving instance-winner memberships.
    It is not the number of all examples or the candidate's average reward.
    """
    pool = _pool(candidates, selection_sample_ids)
    fronts = []
    discovery = []
    for column in zip(*(c.scores for c in pool)):
        maximum = max(column)
        front = tuple(i for i, value in enumerate(column) if value == maximum)
        fronts.append(front)
        for index in front:
            if index not in discovery:
                discovery.append(index)
    means = tuple(c.mean_score for c in pool)
    order = sorted(discovery, key=lambda i: means[i])
    survivors = set(discovery)
    removed = []
    for index in order:
        covered = (front for front in fronts if index in front)
        if all(any(other != index and other in survivors for other in front)
               for front in covered):
            survivors.remove(index)
            removed.append(index)
    retained = tuple(tuple(i for i in front if i in survivors) for front in fronts)
    # First retained winner appearance, as in the reference frequency dictionary.
    weights = {}
    for front in retained:
        for index in front:
            weights[index] = weights.get(index, 0) + 1
    return CoverPruneResult(
        tuple(c.candidate_id for c in pool), pool[0].sample_ids, means,
        tuple(fronts), retained, tuple(order), tuple(removed),
        tuple(weights), tuple(weights.values()))


@dataclass(frozen=True)
class ParentSelection:
    candidate_id: str
    candidate_index: int
    coverage: CoverPruneResult
    rng_before: tuple
    rng_after: tuple

    def to_dict(self) -> dict:
        return {'selector': SELECTOR_ID, **asdict(self)}


def select_parent(candidates, selection_sample_ids, *, rng: random.Random) -> ParentSelection:
    """Consume one seeded RNG choice; record states for exact local replay.

    JSON can encode the state tuples directly. ``random_from_state`` restores
    either these tuples or the list representation produced by JSON decoding.
    All input validation finishes before the caller's RNG is advanced.
    """
    if type(rng) is not random.Random:
        raise ValueError('an explicit seeded random.Random instance is required')
    coverage = cover_prune(candidates, selection_sample_ids)
    tickets = tuple(i for i, weight in zip(coverage.sampling_indices,
                                          coverage.sampling_weights)
                    for _ in range(weight))
    before = rng.getstate()
    index = rng.choice(tickets)
    return ParentSelection(coverage.candidate_ids[index], index, coverage,
                           before, rng.getstate())


def random_from_state(state) -> random.Random:
    """Restore Python MT19937 v3 state without pickle or implicit reseeding."""
    if not isinstance(state, (list, tuple)) or len(state) != 3:
        raise ValueError('complete RNG state required')
    version, words, gaussian = state
    if type(version) is not int or version != 3:
        raise ValueError('RNG state version 3 required')
    if (not isinstance(words, (list, tuple)) or len(words) != 625
            or any(type(v) is not int or not 0 <= v < 2**32 for v in words[:-1])
            or type(words[-1]) is not int or not 0 <= words[-1] <= 624):
        raise ValueError('invalid MT19937 state')
    if gaussian is not None and (type(gaussian) not in (int, float)
                                 or not math.isfinite(gaussian)):
        raise ValueError('invalid Gaussian RNG cache')
    rng = random.Random()
    rng.setstate((version, tuple(words), gaussian))
    return rng


def best_candidate(candidates, selection_sample_ids) -> ScoredCandidate:
    """Highest full-selection mean; exact ties return earliest pool candidate."""
    pool = _pool(candidates, selection_sample_ids)
    return max(pool, key=lambda candidate: candidate.mean_score)


@dataclass(frozen=True)
class MinibatchDecision:
    accepted: bool
    reason: str
    parent_mean: float
    child_mean: float


def compare_minibatch(parent: ScoredCandidate, child: ScoredCandidate,
                      minibatch_sample_ids) -> MinibatchDecision:
    """Authorize full-selection evaluation only, not insertion into the pool.

    Parent/child must be evaluated on the SAME registered ordered minibatch;
    identical instruction bytes are a no-op regardless of stochastic scores.
    """
    expected = _ids(minibatch_sample_ids, 'minibatch_sample_ids')
    if not isinstance(parent, ScoredCandidate) or not isinstance(child, ScoredCandidate):
        raise ValueError('parent and child ScoredCandidate records required')
    if parent.sample_ids != expected or child.sample_ids != expected:
        raise ValueError('parent and child require EXACT same registered sample order')
    if parent.candidate_id == child.candidate_id:
        raise ValueError('parent and child candidate identities must differ')
    if parent.instruction_sha256 == child.instruction_sha256:
        reason = 'rejected-noop'
    elif sum(child.scores) > sum(parent.scores):
        reason = 'strict-improvement'
    else:
        reason = 'rejected-no-strict-improvement'
    return MinibatchDecision(reason == 'strict-improvement', reason,
                             parent.mean_score, child.mean_score)
