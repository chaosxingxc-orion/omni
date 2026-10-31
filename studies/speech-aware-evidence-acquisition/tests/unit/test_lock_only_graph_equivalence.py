"""Real episode graph / scripted transport proof of removing diagnostic branches.

This synthetic structural proof is necessary but not an actual-data FIT or a
proof of equivalent output under changed serving/batching.
"""
import json
from pathlib import Path
from dataclasses import replace

import pytest

from speech_aware_evidence_acquisition.rails.native_episode_config import NativeEpisodeConfig, NativeEpisodeRole
from speech_aware_evidence_acquisition.rails.supply_variants import parse_perception_sampling_variants
from speech_aware_evidence_acquisition.reproduction.ojw import openjiuwen_available
from test_llama_server_transport import _spawn_live_matching_process  # noqa: F401
from test_native_episode_runner import ACTIONS, PHASES, _ActualEpisodeHarness

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not openjiuwen_available(), reason='OpenJiuwen unavailable')
def test_two_contact_wire_matches_five_branch_delivering_lock(tmp_path):
    source = json.loads((ROOT / 'configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json').read_bytes())
    group = parse_perception_sampling_variants(source['native_episode']['roles']['perception']['sampling_variants'], phase_id='perception')
    runs = []
    for branched in (True, False):
        harness = _ActualEpisodeHarness(tmp_path / ('branched' if branched else 'plain'),
            sample_count=1, max_revisions=0, max_contacts=17 if branched else 2,
            planner_actions=(), arm='raw-speech-agent-native-tools', query_source='speech_audio')
        harness.program = replace(harness.program, program_id='phase-native-module-v1-c17' if branched else 'phase-native-module-v1-c2')
        harness.config = NativeEpisodeConfig(episode_id='lock-wire-equivalence-v1', max_revisions=0,
            actions=ACTIONS, roles=tuple(NativeEpisodeRole(phase_id=phase, role_kind=phase,
                max_iterations=0 if phase in {'reviewer', 'planner'} else 1,
                tool_names=(), perception_branches=group if branched and phase == 'perception' else None)
                for phase in PHASES), max_contacts=17 if branched else 2)
        if not branched:
            harness.config = replace(harness.config, prompt_contact_limits='legacy-c17-lock-v1')
        harness._output = lambda key: (json.dumps({'candidates': [{'intent_id': 'i:a'}]}) if key[1] == 'perception'
                                      else json.dumps({'scenario': 's', 'action': 'a', 'entities': []}))
        try:
            result = harness.run(decoding_params={'temperature': 0, 'max_tokens': 64, 'seed': 20260913})
        finally:
            harness.close()
        assert len(result.results) == 1
        assert len(result.results[0].contacts) == (17 if branched else 2)
        runs.append((harness, result.results[0]))
    branched, plain = runs
    assert branched[1].final_response_text == plain[1].final_response_text
    for phase in ('perception', 'solver'):
        old = [(key, value['rendered']) for key, value in branched[0].prepared.items() if key[1] == phase]
        new = [(key, value['rendered']) for key, value in plain[0].prepared.items() if key[1] == phase]
        assert len(new) == 1
        # Delivering lock is last in the registered source's branch inventory.
        left = json.dumps(max(old, key=lambda pair: pair[0][-1])[1], sort_keys=True, indent=2)
        right = json.dumps(new[0][1], sort_keys=True, indent=2)
        import difflib
        assert left == right, "\n".join(difflib.unified_diff(left.splitlines(), right.splitlines()))
