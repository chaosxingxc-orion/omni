"""Authenticate an independently sealed scoring tool against a generation release."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

STUDY = Path('studies/speech-aware-evidence-acquisition')


def verify_separate_scoring_tool(generation_repo: Path, tool_repo: Path) -> dict:
    """Verify both complete inventories; never substitute either commit for the other.

    The generation release keeps its standard /opt/saea-release identity and
    resolves its original workspace. The separate tool is an explicit, pinned
    complete release, not a Python module overlay. Execute the verifier shipped
    with the authenticated generation release against both inventories.
    """
    generation_root = Path('/opt/saea-release')
    if Path(generation_repo).resolve() != generation_root / STUDY:
        raise ValueError('generation release location differs')
    tool_root_text = os.environ.get('SAEA_SCORE_TOOL_RELEASE')
    expected = os.environ.get('SAEA_SCORE_TOOL_RELEASE_SHA256', '')
    if not tool_root_text or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('separate scoring tool requires release and SHA256 pin')
    tool_root = Path(tool_root_text)
    if not tool_root.is_absolute() or any(p.is_symlink() for p in (tool_root, *tool_root.parents)):
        raise ValueError('scoring tool release path must be absolute and nonsymlink')
    if Path(tool_repo).resolve() != tool_root.resolve() / STUDY:
        raise ValueError('imported scoring tool differs from pinned release')
    from .bundle.git_commit import require_clean_load_bearing_surface
    generation_commit, _ = require_clean_load_bearing_surface(generation_repo)
    manifest = tool_root / 'source-manifest.json'
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('scoring tool release manifest digest differs')
    verifier = generation_root / 'verify-release.py'
    verified = subprocess.run([sys.executable, str(verifier), str(manifest), str(tool_root)],
                              capture_output=True, text=True, check=True)
    tool_commit = json.loads(verified.stdout)['commit']
    if tool_commit != json.loads(raw)['commit']:
        raise ValueError('scoring tool release commit differs')
    return {'generation_commit': generation_commit,
            'generation_manifest_sha256': os.environ['SAEA_RELEASE_MANIFEST_SHA256'],
            'tool_commit': tool_commit, 'tool_manifest_sha256': expected}
