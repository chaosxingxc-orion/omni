"""Score a sealed paired window against original wire hashes (only aggregate output)."""
import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path
from io_utils import doc, pin, require, canonical
from summarize_engine_window import summarize


def evaluate(config_path, gold_path, parser_path):
    config = doc(pin(config_path))
    root = Path(config['output'])
    gold = json.loads(Path(gold_path).read_bytes())
    labels = {item['wire']: item for item in gold}
    require(len(labels) == len(gold), 'duplicate gold wire')
    spec = importlib.util.spec_from_file_location('ranked_candidate_indices', parser_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = []
    for ref in doc(pin(root/'RESULT.json'))['rounds']:
        r = doc(ref)
        n, = config['grouped_concurrency']
        salt = hashlib.sha256(canonical([config['measurement_id'], r['block']])).hexdigest()
        arms = {s['name']: s for s in config['services']}
        indexed = {}
        for row in r['requests']:
            i = n * row['group'] + row['replicate']
            src = r['input_refs'][i]
            g = labels[src['sha256']]
            arm = row['service']
            wire = dict(doc(src), cache_salt=salt)
            wire['model'] = arms[arm].get('stock_adapter_alias', wire['model'])
            resp = doc(row['receipt_ref'])
            intent = doc(pin(Path(row['receipt_ref']['path']).with_name('INTENT.json')))
            require(intent['wire_sha256'] == hashlib.sha256(canonical(wire)).hexdigest()
                    and resp['status'] == 200, 'unmatched request or response')
            msg = json.loads(resp['raw'])['choices'][0]['message']['content']
            try:
                ids = [x['label'] for x in mod.parse(msg, g['ranking'], 3)]
            except ValueError:
                ids = []
            reward = (1., .75, .5)[ids.index(g['gold'])] if g['gold'] in ids else 0.
            require((i, arm) not in indexed, 'duplicate arm/sample')
            indexed[i, arm] = ids, reward, row['elapsed_seconds']
        x, y = arms
        indices = range(len(r['input_refs']))
        require(len(indexed) == 2 * len(indices), 'unpaired panel')
        out.append(dict(n=len(indices), paired_valid_equal=sum(bool(indexed[i,x][0]) and
                    indexed[i,x][0] == indexed[i,y][0] for i in indices),
                    per_arm={a:dict(valid=sum(bool(indexed[i,a][0]) for i in indices),
                             p_at_3=sum(indexed[i,a][1]>0 for i in indices)/len(indices),
                             mean_reward=statistics.mean(indexed[i,a][1] for i in indices))
                             for a in arms},
                    paired_median_latency_delta_seconds=statistics.median(
                        indexed[i,y][2]-indexed[i,x][2] for i in indices),
                    delta_order=[y,x]))
    return dict(performance=summarize(root), quality=out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ('config', 'gold', 'parser'):
        p.add_argument(arg, type=Path)
    print(json.dumps(evaluate(**{k+'_path':v for k,v in vars(p.parse_args()).items()}),indent=2))
