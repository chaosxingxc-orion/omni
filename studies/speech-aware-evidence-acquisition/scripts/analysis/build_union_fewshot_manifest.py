"""Build a gold-free Q-source/D union from the frozen ranking assets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--frozen', required=True)
p.add_argument('--candidate-metadata', required=True)
p.add_argument('--output', required=True)
p.add_argument('--question-depth', type=int, default=50)
p.add_argument('--description-depth', type=int, default=5)
a = p.parse_args()
z = np.load(a.frozen, allow_pickle=False)
meta = {r['sample_id']:r for r in json.loads(Path(a.candidate_metadata).read_text())['rows']}
labels = z['labels'].tolist()
Q = z['question_first_source_rank']; D = z['description_rank']; legal_all = z['legal_cards']; fewshots = z['fewshot_member_indices']
rows = []
for i, sid in enumerate(z['sample_ids'].tolist()):
    q, d, legal = Q[i], D[i], legal_all[i]
    qi = sorted((j for j in range(len(labels)) if legal[j] and q[j] <= a.question_depth), key=lambda j:int(q[j]))
    di = sorted((j for j in range(len(labels)) if legal[j] and d[j] <= a.description_depth), key=lambda j:int(d[j]))
    order = qi + [j for j in di if j not in qi]
    candidates = []
    for j in order:
        candidates.append({'label':labels[j], 'knowledge_id':'intent-boundaries::intent-boundary:'+labels[j],
            'routes':(['question'] if j in qi else []) + (['description'] if j in di else []),
            'question_source_rank':int(q[j]) if q[j] < 100000 else None,
            'description_rank':int(d[j]) if d[j] < 100000 else None,
            'fewshot_member_indices_top5':[int(v) for v in fewshots[i,j] if v >= 0]})
    rows.append({'sample_id':sid, 'excluded_source_groups':meta[sid]['excluded_source_groups'],
                 'candidate_labels':[labels[j] for j in order], 'candidates':candidates})
out = {'gold_read_for_candidates':False, 'config':{'question_depth':a.question_depth, 'description_depth':a.description_depth},
       'source_sha256':{'frozen':hashlib.sha256(Path(a.frozen).read_bytes()).hexdigest(),
                        'candidate_metadata':hashlib.sha256(Path(a.candidate_metadata).read_bytes()).hexdigest()}, 'rows':rows}
Path(a.output).write_text(json.dumps(out, ensure_ascii=False, separators=(',',':'))+'\n')
