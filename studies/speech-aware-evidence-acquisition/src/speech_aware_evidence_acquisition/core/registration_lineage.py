"""Bounded registration traversal, shared by compilation and business validation.

predecessor_registration pins the root registration. New successors additionally
pin their immediate predecessor; legacy one-hop documents use the root pin for
both roles. No attempt names are interpreted here.
"""
import re
from pathlib import PurePosixPath

MAX_REGISTRATION_ANCESTORS = 128
LINEAGE_FIELDS = ('predecessor_registration', 'immediate_predecessor_registration',
                  'immediate_predecessor_plan')


def registration_pin(pin):
    if (not isinstance(pin, dict) or set(pin) != {'relpath', 'sha256'}
            or not isinstance(pin['relpath'], str)
            or not pin['relpath'].startswith('docs/')
            or '..' in PurePosixPath(pin['relpath']).parts or '\\' in pin['relpath']
            or not isinstance(pin['sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', pin['sha256'])):
        raise ValueError('exact confined registration lineage pin required')
    return pin


def registration_lineage(spec, load):
    """Return (root pin, root, immediate-to-root documents), checking all root pins.

    ``load`` must authenticate bytes against each pin, without fallback to an
    unlisted asset when used by a compiler. Traversal is iterative and bounded.
    """
    current, seen, documents, roots = spec, set(), [], []
    root_pin = None
    while True:
        if 'predecessor_registration' in current:
            roots.append(registration_pin(current['predecessor_registration']))
        link = current.get('immediate_predecessor_registration',
                           current.get('predecessor_registration'))
        if link is None:
            break
        registration_pin(link)
        key = (link['relpath'], link['sha256'])
        if key in seen or len(seen) >= MAX_REGISTRATION_ANCESTORS:
            raise ValueError('cyclic or excessive registration lineage')
        seen.add(key)
        current = load(link)
        if not isinstance(current, dict) or current.get('run_id') != spec.get('run_id'):
            raise ValueError('registration lineage same-run identity differs')
        documents.append(current)
        root_pin = link
    if any(pin != root_pin for pin in roots):
        raise ValueError('registration lineage root pin differs')
    return root_pin, current, documents


def scientific_registration(document):
    """Remove only explicitly permitted operational successor fields."""
    import copy
    value = copy.deepcopy(document)
    for key in LINEAGE_FIELDS:
        value.pop(key, None)
    if (value.get('execution') != value.get('job_config', {}).get('execution')
            or set(value.get('execution', {})) != {'admitted_level'}):
        raise ValueError('registration width identity differs')
    value['execution'].pop('admitted_level')
    value['job_config']['execution'].pop('admitted_level')
    for key in ('attempt_id', 'fit_run_id', 'fit_attempt_id', 'composed_config_hash', 'protocol_hash'):
        value.pop(key, None)
    return value
