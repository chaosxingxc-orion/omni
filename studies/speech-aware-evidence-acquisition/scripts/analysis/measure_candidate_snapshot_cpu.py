"""Model-free measurement of pinned three-store candidate source custody.

Run in the fixed image with baseline config and data root mounted read-only;
output root must be an isolated writable directory. It verifies actual source
pins and all three released snapshots, not retrieval relevance or PSD throughput.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from speech_aware_evidence_acquisition.rails.knowledge_stores import load_knowledge_store_binding
from speech_aware_evidence_acquisition.rails.candidate_handoff import persist_shared_snapshot, load_shared_snapshot, project_snapshot_messages
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    t = perf_counter(); binding = load_knowledge_store_binding(config, data_root=a.data_root)
    loaded_s = perf_counter() - t
    snapshot = binding.composite_snapshot; snapshot_sha = binding.binding_sha256
    t = perf_counter(); ref = persist_shared_snapshot(snapshot, a.output_root)
    write_s = perf_counter() - t
    t = perf_counter(); verified = load_shared_snapshot(ref)
    verify_s = perf_counter() - t
    values = {v.value_id:v for v in verified.values}; facets = {v.facet_id:v for v in verified.facets}
    counts = {sid: {'facets':len(binding.store(sid).snapshot.facets), 'values':len(binding.store(sid).snapshot.values)} for sid in binding.store_ids}
    # Measurements exercise source/body/scope projection against actual records,
    # not candidate retrieval or a model request. Use bounded per-node fixtures.
    chosen = {sid:[v for v in verified.values if v.value_id.startswith(sid+'::')][:limit]
              for sid,limit in [('intent-boundaries',8),('slot-contracts',28),('train-exemplars',20)]}
    timings = {}; sizes = {}
    for node, ids in [('P',['intent-boundaries']), ('Slot',['slot-contracts','train-exemplars']),
                      ('Decision',['slot-contracts','train-exemplars'])]:
        parts=[{'type':'input_audio'}]
        for sid in ids:
            for value in chosen[sid]:
                parts.append({'type':'text','text':value.content,'knowledge_role':'value','evidence_id':value.value_id,
                              'source_uri':'knowledge-snapshot://'+snapshot_sha+'/'+value.value_id})
        messages=[{'role':'system','content':'test'}, {'role':'user','content':parts}]
        start=perf_counter()
        checked, groups = project_snapshot_messages(messages, snapshot_sha256=snapshot_sha, values=values, facets=facets)
        timings[node]=perf_counter()-start
        sizes[node]={'supplied':len(parts)-1,'groups':len(groups),
                     'request_evidence_bytes':len(json.dumps(checked,ensure_ascii=False,separators=(',',':')).encode()),
                     'handoff_knowledge_bytes':len(json.dumps(groups,ensure_ascii=False,separators=(',',':')).encode())}
    refs={'org_knowledge_snapshot_ref':ref, 'org_knowledge_snapshot_sha256':snapshot_sha,
          'org_knowledge_store_pins':{sid:binding.store_receipt(sid) for sid in binding.store_ids}}
    print(json.dumps({'schema':'saea-candidate-source-cpu-measurement-v1','counts':counts,'pins':refs['org_knowledge_store_pins'],
       'snapshot_sha256':snapshot_sha, 'shared_file_bytes':Path(ref['path']).stat().st_size,
       'per_row_reference_and_pins_bytes':len(json.dumps(refs,ensure_ascii=False,separators=(',',':')).encode()),
       'shared_source_cold_load_verify_s':verify_s,'three_store_load_and_compose_s':loaded_s,
       'shared_source_write_and_verify_s':write_s,'node_projection_s':timings,'node_projection':sizes,
       'composite_id':verified.snapshot_id,'pin_identity_ok':verified.snapshot_id.endswith(digest({sid:{'snapshot_sha256':v['store_snapshot_sha256'],
            'release_sha256':v['store_release_sha256']} for sid,v in refs['org_knowledge_store_pins'].items()}))},indent=2))

if __name__ == '__main__': main()
