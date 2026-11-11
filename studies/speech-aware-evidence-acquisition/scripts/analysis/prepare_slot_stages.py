"""Prepare one S stage from external knowledge/audio; no model/network calls."""
import argparse
import json
from pathlib import Path

from speech_aware_evidence_acquisition.rails.slot_stages import Query, StageRails


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--knowledge', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--sample-id', required=True)
    p.add_argument('--audio-part', required=True, type=Path)
    p.add_argument('--source-component')
    p.add_argument('--s1-result', type=Path, help='Observed S1 handoff containing selected_types; absent => prepare S1')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    rails = StageRails(json.loads(a.knowledge.read_text()), json.loads(a.config.read_text()))
    query = Query(a.sample_id, json.loads(a.audio_part.read_text()), a.source_component)
    if a.s1_result:
        result = json.loads(a.s1_result.read_text())
        if result['sample_id'] != a.sample_id or result['knowledge_id'] != rails.knowledge['knowledge_id']:
            raise ValueError('S1 source/knowledge identity mismatch')
        stage, branches = 'S2', sorted(set(result['selected_types']))
    else:
        stage, branches = 'S1', [f['fold_id'] for f in rails.knowledge['folds']]
    packets = [rails.before_model_call(stage, query, branch) for branch in branches]
    # A single batch artifact, not one duplicated audio file per request.
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'stage': stage, 'packets': packets, 'model_calls': 0,
                                   'production_dispatch_authorized': False}, ensure_ascii=False) + '\n')
    print(json.dumps({'stage': stage, 'requests_prepared': len(packets), 'model_calls': 0}))


if __name__ == '__main__':
    main()
