"""Deterministic TRAIN search partition; labels are used only for stratification.

Query/utterance/audio connected components cannot cross feedback and selection.
External H2/test identities and audio hashes are exclusion checks, not feedback.
Input content pins/actual media custody must be verified by the caller.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import re

SCHEMA = 'saea-gepa-partition-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(row, *, train):
    require(isinstance(row, dict) and set(row) == ({'sample_id', 'utterance_id', 'audio_sha256', 'intent'}
            if train else {'sample_id', 'utterance_id', 'audio_sha256'}), 'exact identity fields required')
    for name in ('sample_id', 'utterance_id'):
        require(isinstance(row[name], str) and row[name].strip() == row[name] and row[name], 'nonempty identity required')
        if train:
            require(row[name].startswith('slurp:train:'), 'search population must be TRAIN')
    require(isinstance(row['audio_sha256'], str) and re.fullmatch('[a-f0-9]{64}', row['audio_sha256']), 'audio SHA256 required')
    if train:
        require(isinstance(row['intent'], str) and row['intent'].strip() == row['intent'] and row['intent'], 'original intent required')


def build_partition(rows, excluded, *, target_selection, seed):
    require(isinstance(rows, list) and len(rows) > 1 and isinstance(excluded, list), 'explicit population and exclusions required')
    require(type(target_selection) is int and 0 < target_selection < len(rows) and type(seed) is int,
            'finite nonempty feedback and selection required')
    for row in rows: identity(row, train=True)
    for row in excluded: identity(row, train=False)
    require(len({r['sample_id'] for r in rows}) == len(rows), 'duplicate search query')
    require(len({r['sample_id'] for r in excluded}) == len(excluded), 'duplicate exclusion query')
    for field in ('sample_id', 'utterance_id', 'audio_sha256'):
        require(not {r[field] for r in rows} & {r[field] for r in excluded}, 'search/exclusion overlap: ' + field)
    rows = sorted(rows, key=lambda r:r['sample_id'])
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for field in ('utterance_id', 'audio_sha256'):
            key = (field, row[field])
            if key in seen: parent[find(i)] = find(seen[key])
            else: seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[find(i)].append(row)
    strata = defaultdict(list)
    for group in groups.values():
        # Mixed-intent duplicate components stay together in a disclosed joint
        # stratum; no synthetic intent/slot combinations are introduced.
        stratum = tuple(sorted({r['intent'] for r in group}))
        strata[stratum].append(group)
    # Each stratum offers exact reachable counts from whole components. Across
    # strata minimize squared proportional-allocation error, then maximize the
    # reachable count <= target. Seed affects ties, never labels or scores.
    options = []
    for stratum, components in sorted(strata.items()):
        components.sort(key=lambda g:digest({'seed':seed, 'ids':[r['sample_id'] for r in g]}))
        choices = {0: []}
        for component in components:
            size = len(component)
            for count, selected in list(sorted(choices.items(), reverse=True)):
                if count + size <= target_selection:
                    choices.setdefault(count + size, selected + [component])
        population = sum(map(len, components))
        options.append((stratum, population, choices))
    dp = {0: (Fraction(0), [])}
    for stratum, population, choices in options:
        expected = Fraction(target_selection * population, len(rows))
        next_dp = {}
        for total, (cost, selected) in sorted(dp.items()):
            for count, components in sorted(choices.items()):
                if total + count > target_selection: continue
                proposal = (cost + (count - expected)**2, selected + components)
                old = next_dp.get(total + count)
                if old is None or proposal[0] < old[0]: next_dp[total + count] = proposal
        dp = next_dp
    actual = max(dp)
    require(actual > 0, 'no complete component fits selection target')
    chosen = {r['sample_id'] for group in dp[actual][1] for r in group}
    selection = [r for r in rows if r['sample_id'] in chosen]
    feedback = [r for r in rows if r['sample_id'] not in chosen]
    counts = lambda rs: dict(sorted(Counter(r['intent'] for r in rs).items()))
    doc = {'schema':SCHEMA, 'algorithm':'component-stratified-count-dp-v1', 'seed':seed,
           'population_sha256':digest(rows), 'exclusions_sha256':digest(sorted(excluded,key=lambda r:r['sample_id'])),
           'target_selection':target_selection, 'actual_selection':actual,
           'feedback_ids':[r['sample_id'] for r in feedback], 'selection_ids':[r['sample_id'] for r in selection],
           'components':sorted([{'sample_ids':sorted(r['sample_id'] for r in g),
                               'side':'selection' if g[0]['sample_id'] in chosen else 'feedback',
                               'intents':sorted({r['intent'] for r in g})} for g in groups.values()],
                               key=lambda c:c['sample_ids']),
           'coverage':{'population':counts(rows), 'feedback':counts(feedback), 'selection':counts(selection)},
           'selection_is_unseen_validation':False, 'new_model_calls':0,
           'limitation':'TRAIN internal search selection; historical knowledge may have seen both sides. Audio hashes are caller-verified identities.'}
    return {**doc, 'partition_sha256':digest(doc)}
