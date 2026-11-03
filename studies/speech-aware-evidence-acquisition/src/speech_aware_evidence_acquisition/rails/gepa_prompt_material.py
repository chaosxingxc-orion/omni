"""Publish verified GEPA text for the existing native template-variant path.

This CPU operation is not Flight registration, native FIT, search completion or
candidate adoption. Use an independent writable configuration tree, never a live
pinned tree. The caller supplies independently verified tokenizer/lineage custody.
"""
from pathlib import Path
import re

from ..core.atomic_file import publish_new
from .solver_prompt_candidate import executable_template, sha, text_sha
from .supply_variants import SupplyVariantTemplate


def publish_template(repo, *, candidate, contract, expected_contract_sha256,
                     count_tokens, tokenizer_identity, expected_tokenizer_sha256,
                     parent=None, expected_parent_sha256=None, mutation_text=None):
    """Verify first, then exclusively publish content-addressed UTF-8 bytes.

    Returns the exact ``instruction_template`` pin shape plus an independent
    text count for the arm's ``template_tokens``. Native FIT must still measure
    the complete speech/P/knowledge/framework request at execution time.
    """
    if (not isinstance(expected_tokenizer_sha256, str)
            or re.fullmatch('[a-f0-9]{64}', expected_tokenizer_sha256) is None
            or sha(tokenizer_identity) != expected_tokenizer_sha256
            or contract['frozen']['tokenizer_sha256'] != expected_tokenizer_sha256):
        raise ValueError('externally pinned tokenizer identity differs')
    if tokenizer_identity.get('rendering') != {
            'mode': 'text', 'add_generation_prompt': False,
            'add_special_tokens': False, 'truncation': False}:
        raise ValueError('untruncated plain-text tokenizer required')
    text = executable_template(candidate, contract=contract,
        expected_contract_sha256=expected_contract_sha256, count_tokens=count_tokens,
        parent=parent, expected_parent_sha256=expected_parent_sha256,
        mutation_text=mutation_text)
    count = count_tokens(text)
    if type(count) is not int or not 0 < count <= contract['max_template_tokens']:
        raise ValueError('executable template token bound exceeded')
    repo = Path(repo).resolve(strict=True)
    digest = text_sha(text)
    relative = 'probes/gepa/' + digest + '.txt'
    target = repo / 'configs' / relative
    # Refuse symlink redirection, including an existing same-content target.
    if target.resolve() != target:
        raise ValueError('template destination redirected')
    blob = text.encode('utf-8')
    try:
        publish_new(target, blob)
    except FileExistsError:
        if not target.is_file() or target.is_symlink() or target.read_bytes() != blob:
            raise ValueError('existing template conflicts with verified candidate') from None
    return {'instruction_template': SupplyVariantTemplate(relative, digest).as_json(),
        'template_tokens': count, 'candidate_sha256': candidate['candidate_sha256'],
        'contract_sha256': expected_contract_sha256,
        'tokenizer_sha256': expected_tokenizer_sha256,
        'new_model_calls': 0, 'native_fit_completed': False}
