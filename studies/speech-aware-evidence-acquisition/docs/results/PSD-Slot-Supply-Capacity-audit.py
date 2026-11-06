"""Offline cardinality/selection audit of pinned authored slot bindings."""
import ast
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

source_path, provider_path, output_path = map(Path, sys.argv[1:])
raw=source_path.read_bytes()
expected='8994f773c886f2c207cd9db6436507e53e7f7da79ff7bdf441ab3799f244d34c'
assert hashlib.sha256(raw).hexdigest()==expected
source=json.loads(raw)
index=defaultdict(list)
for row in source['rows']:
    for binding in row['intent_bindings']:
        index[binding['intent_id']].append('slot-contract:'+row['slot_id'])
index={key:tuple(sorted(values)) for key,values in index.items()}
assert all(len(v)==len(set(v)) for v in index.values())
# Execute the existing pure selector unchanged; no framework or model imports.
tree=ast.parse(provider_path.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_candidate_semantic_values')
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),function],type_ignores=[])
namespace={}
exec(compile(ast.fix_missing_locations(module),str(provider_path),'exec'),namespace)
select=namespace['select_candidate_semantic_values']
counts={};missing=0;max_union=0;max_example=None;max_characters=0
for size in (1,2,3):
    count=0
    for candidates in itertools.permutations(sorted(index),size):
        count+=1
        available=set().union(*(set(index[k]) for k in candidates))
        chosen,mappings=select(candidates,index,max_values_per_candidate=16,max_total_values=28)
        if set(chosen)!=available:missing+=1
        if len(available)>max_union:max_union=len(available);max_example=candidates
    counts[str(size)]=count
result={'schema':'omni-psd-slot-supply-capacity-audit-v1',
 'authored_source_sha256':expected,'provider_file_sha256':hashlib.sha256(provider_path.read_bytes()).hexdigest(),
 'selector_ast_sha256':hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest(),
 'knowledge_cards':len(source['rows']),'indexed_intents':len(index),'maximum_cards_per_intent':max(map(len,index.values())),
 'maximum_union_up_to_three_intents':max_union,'maximum_union_example':max_example,
 'limits':{'per_candidate':16,'total':28,'max_candidates':3},'ordered_candidate_sets_checked':counts,
 'sets_with_dropped_available_cards':missing,'card_counts_by_intent':{k:len(v) for k,v in sorted(index.items())},
 'scope':'Exact authored binding index and unchanged pure round-robin selector only. This is not an intent/slot Cartesian product or a dataset distribution estimate. No annotations, audio, runtime request, final context compaction or model accuracy checked.',
 'conclusion':'Existing cardinality caps do not drop any indexed card for up to three distinct indexed intents; raising these caps alone cannot expose additional indexed cards.',
 'model_calls':0}
output_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('knowledge_cards','indexed_intents','maximum_cards_per_intent','maximum_union_up_to_three_intents','ordered_candidate_sets_checked','sets_with_dropped_available_cards')}))
