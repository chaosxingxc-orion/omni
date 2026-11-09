"""Local archive admission only. Default preview; --apply is for reviewed deployment."""
import argparse
import collections
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

EPS = 1e-12


class Decision(Exception):
    def __init__(self, status, reason):
        self.status, self.reason = status, reason
        super().__init__(reason)


def need(ok, reason, status='REJECT'):
    if not ok:
        raise Decision(status, reason)


def read(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def index(rows, n):
    d = {r['sample_id']: r for r in rows}
    need(len(d) == len(rows) == n, 'duplicate IDs or denominator mismatch')
    return d


def near(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= EPS


def metrics(rows, key, all_three=False):
    """Aggregate saved counts only; never parse/model-score or change normalization."""
    counts = dict(tp=0, fp=0, fn=0, all_slots_exact=0, invalid=0)
    rewards = []
    for r in rows:
        v = r[key]
        valid = r['valid'] and (not all_three or r['invalid_branches'] == 0)
        if not valid:
            v = dict(tp=0, fp=0, fn=len(r['gold']), all_slots_exact=0, invalid=1)
        for k in counts:
            counts[k] += v[k]
        den = 2*v['tp'] + v['fp'] + v['fn']
        rewards.append((2*v['tp']/den if den else 1.) if valid else 0.)
    den = 2*counts['tp'] + counts['fp'] + counts['fn']
    return dict(counts, denominator=len(rows), F1=2*counts['tp']/den if den else 1.,
                mean_sample_reward=sum(rewards)/len(rows) if rows else None)


class Gate:
    def __init__(self, config, state, publisher=None, now=time.time):
        self.c = read(config)
        self.root = Path(self.c['root']).resolve()
        self.state = Path(state).resolve()
        self.publisher, self.now = publisher, now
        self.config_sha = sha(config)
        self.cache = {}

    def p(self, path):
        p = Path(path)
        return p if p.is_absolute() else self.root/p

    def jsonlines(self, path):
        return [json.loads(x) for x in self.p(path).read_bytes().splitlines() if x.strip()]

    def pins(self, pins):
        need(bool(pins), 'missing pins', 'WAIT')
        for path, pin in pins.items():
            p = self.p(path)
            need(p.is_file(), 'missing pinned file: '+path, 'WAIT')
            need(sha(p) == pin, 'SHA mismatch: '+path)

    def terminal(self, t, n):
        need(t.get('unknown') == 0, 'unknown contacts; preserved; no resend', 'WAIT')
        need(t.get('status') not in ('FAILED', 'ERROR', 'ABORTED', 'failed'), 'known terminal failure')
        need(t.get('status') == 'COMPLETE' and t.get('completed') == t.get('total') == n,
             'not full COMPLETE terminal', 'WAIT')
        need(t.get('absent', 0) == 0 and not t.get('error'), 'failed/absent terminal')

    def task_state(self, task_id, n):
        p = self.p(self.c['execution_tasks'])/(task_id+'.json')
        if not p.exists():
            return
        s = read(p)
        if s.get('terminal'):
            self.terminal(s['terminal'], n)
        need(s.get('status') not in ('failed', 'blocked', 'rejected'),
             task_id+': failed/blocked execution record; no resend')

    def archive(self, spec):
        """Use owner SHA-receipted local tar; keep selected bytes in memory, no extraction."""
        if spec['task_id'] in self.cache:
            return self.cache[spec['task_id']]
        self.task_state(spec['task_id'], spec['branches'])
        source = read(self.p(spec['config']))
        need(source['run_id'] == spec['run_id'], 'source run identity changed')
        required = ['runtime-config.json', 'LAUNCH.json', 'run/TERMINAL.json',
                    'run/predictions.private.jsonl', 'score/samples.private.jsonl',
                    'score/typed-normalized.json', 'score/typed-samples.private.jsonl']
        found = []
        for receipt in sorted(self.p(self.c['archives']).glob('*.json')):
            tar = receipt.with_suffix('.tar')
            if not tar.is_file():
                continue
            with tarfile.open(tar) as tf:
                members = {str(Path(m.name)): m for m in tf.getmembers()}
                if 'runtime-config.json' not in members:
                    continue
                cfg = json.load(tf.extractfile(members['runtime-config.json']))
                if cfg.get('run_id') != spec['run_id']:
                    continue
                need(cfg.get('attempt_id') in spec['attempt_ids'], 'unexpected attempt; no automatic retry adoption')
                need(len(members) == len(tf.getmembers()), 'duplicate archive members')
                r = read(receipt)
                need(tar.stat().st_size == r['bytes'] and sha(tar) == r['sha256'], 'archive SHA/size mismatch')
                need(all(x in members and members[x].isfile() for x in required),
                     spec['task_id']+': archive lacks full score/terminal', 'WAIT')
                raw = {x: tf.extractfile(members[x]).read() for x in required}
            doc = {k: ([json.loads(x) for x in v.splitlines() if x.strip()] if k.endswith('.jsonl') else json.loads(v))
                   for k, v in raw.items()}
            launch = doc['LAUNCH.json']
            need((launch['run_id'], launch['attempt_id']) == (cfg['run_id'], cfg['attempt_id']), 'launch/run identity')
            if spec.get('require_launch_task_id', True):
                need(launch.get('task_id') == spec['task_id'], 'launch task identity')
            need(cfg['manifest_sha256'] == source['manifest_sha256'], 'runtime/source manifest identity')
            need(cfg['expected_rows'] == spec['branches'] and cfg['full_denominator'] == spec['samples'], 'runtime denominator')
            terminal = doc['run/TERMINAL.json']; self.terminal(terminal, spec['branches'])
            manifest = index([json.loads(x) for x in self.p(source['manifest']).read_bytes().splitlines() if x.strip()], spec['branches'])
            need(sha(self.p(source['manifest'])) == cfg['manifest_sha256'], 'source manifest SHA')
            roster = index([json.loads(x) for x in self.p(source['samples']).read_bytes().splitlines() if x.strip()], spec['samples'])
            typed = index(doc['score/typed-samples.private.jsonl'], spec['samples'])
            summary = index(doc['score/samples.private.jsonl'], spec['samples'])
            preds = index(doc['run/predictions.private.jsonl'], spec['branches'])
            side = doc['score/typed-normalized.json']
            need(side['status'] == 'COMPLETE_TYPED_STRICT_OFFICIAL_SIDECAR' and side['denominator'] == spec['samples'], 'score status/denominator')
            need(side['usage'] == terminal['usage'], 'terminal/score historical usage')
            for field, path in self.c['score_rules'].items():
                need(side[field] == sha(self.p(path)), 'accepted score rule SHA')
            need(set(typed) == set(summary) == set(roster), 'score/roster IDs mismatch')
            if source.get('reference_CPU_only'):
                refs = index([json.loads(x) for x in self.p(source['reference_CPU_only']).read_bytes().splitlines() if x.strip()], spec['samples'])
                gold = {sid:r.get('entities', r.get('gold')) for sid,r in refs.items()}
                intents = {sid:r.get('intent', r.get('gold_intent')) for sid,r in refs.items()}
            else:
                gold = {r['sample_id']:r['gold'] for r in self.jsonlines(self.c['gold545'])}
                intents = {r['sample_id']:r['gold_OFFLINE_ONLY'] for r in self.jsonlines(self.c['intent_ledger']) if r['cohort'] == 'fit'}
            need(set(typed) == set(gold) and set(typed) <= set(intents), 'gold full sample identity')
            need(set(preds) == set(manifest), 'branch manifest IDs mismatch')
            for sid, row in summary.items():
                tr = typed[sid]
                gold_branch = next((b for b in row['branches'] if b['candidate_id'] == intents[sid]), None)
                need(gold_branch is not None, 'offline gold branch missing')
                legal = gold_branch['status'] == 'valid'
                need(tr['gold'] == gold[sid] and tr['valid'] == legal and tr['prediction'] ==
                     (gold_branch['result']['entities'] if legal else []), 'saved score gold branch projection')
                if source.get('dataset_partition'):
                    ref = refs[sid]
                    need(all(tr[k] == ref[k] for k in ('reviewed_hold', 'teacher_update_allowed')), 'TRAIN hold mask changed')
                    need(not tr['reviewed_hold'] and not tr.get('evaluation_only', False) and tr['teacher_update_allowed'],
                         'only eligible remainder; no hold/S_valid promotion')
                need(row['candidate_ids'] == roster[sid]['candidate_ids'], 'ordered P candidates')
                need(len(row['branches']) == 3, 'P3 branch denominator')
                need([(b['branch_key'], b['candidate_id'], b['candidate_index']) for b in row['branches']] ==
                     [(b['branch_key'], b['candidate_id'], b['candidate_index']) for b in roster[sid]['branches']], 'roster branch identity')
                need(tr['invalid_branches'] == sum(b['status'] == 'invalid' for b in row['branches']), 'typed branch invalid counts')
                for b in row['branches']:
                    key = b['branch_key']; pr = preds[key]; mr = manifest[key]
                    need(b['status'] in ('valid', 'invalid'), 'unknown branch score', 'WAIT')
                    need(mr['source_sample_id'] == sid and bool(pr['valid']) == (b['status'] == 'valid'), 'branch source/validity')
                    need(pr['packet_sha256'] == mr['packet_sha256'] and pr['wire_sha256'] == mr['wire_sha256'], 'response packet/wire identity')
                    need(pr['raw_response_sha256'] == b['raw_response_sha256'] and bool(pr['raw_response_sha256']), 'response score SHA')
                for key in ('strict', 'normalized'):
                    v = tr[key]
                    need(all(type(v[k]) is int and v[k] >= 0 for k in ('tp', 'fp', 'fn', 'all_slots_exact', 'invalid')), 'noninteger count')
                    need(tr['valid'] or (v['tp'] == v['fp'] == 0 and v['fn'] == len(tr['gold'])), 'invalid must preserve gold FN without exact/reward credit')
                    need(v['all_slots_exact'] in (0, 1) and v['invalid'] == int(not tr['valid']), 'exact/invalid count')
                    need(v['tp']+v['fn'] == len(tr['gold']) and v['tp']+v['fp'] == len(tr['prediction']), 'sample count denominator')
                    den = 2*v['tp']+v['fp']+v['fn']
                    reward = (2*v['tp']/den if den else 1.) if tr['valid'] else 0.
                    need(near(v['F1'], reward), 'saved sample F1 mismatch')
                    need(v['all_slots_exact'] == int(tr['valid'] and v['fp'] == v['fn'] == 0), 'saved exact mismatch')
            invalid = sum(not p['valid'] for p in preds.values())
            need(terminal['known_model_invalid'] == invalid and terminal['valid'] == spec['branches']-invalid, 'terminal invalid denominator')
            result = dict(task_id=spec['task_id'], typed=typed, summary=summary, terminal=terminal,
                          invalid=invalid, length=sum(p.get('finish_reason') == 'length' for p in preds.values()),
                          metrics={name: metrics(list(typed.values()), key) for name, key in [('strict', 'strict'), ('typed', 'normalized')]},
                          archive=str(tar), archive_sha256=r['sha256'], receipt_sha256=sha(receipt),
                          file_pins={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
                          identity=dict(run_id=cfg['run_id'], attempt_id=cfg['attempt_id']))
            for name, key in [('strict', 'strict'), ('typed', 'normalized')]:
                aggregate = side['groups']['all'][key]; actual = result['metrics'][name]
                need(all(aggregate[k] == actual[k] for k in ('tp', 'fp', 'fn', 'all_slots_exact', 'invalid', 'denominator')) and near(aggregate['F1'], actual['F1']), 'saved aggregate mismatch')
            found.append(result)
        need(len(found) <= 1, spec['task_id']+': duplicate whole task/run archives; manual reconciliation')
        need(bool(found), spec['task_id']+': missing local full scored terminal archive', 'WAIT')
        self.cache[spec['task_id']] = found[0]
        return found[0]

    def scientific(self, candidate, parent, gate):
        failures = []
        for name in ('strict', 'typed'):
            m = candidate['metrics'][name]
            if m['all_slots_exact'] < gate[name+'_exact_min']:
                failures.append(name+' exact')
            for field, threshold in [('F1', 'micro_F1_min'), ('mean_sample_reward', 'mean_sample_reward_min')]:
                if m[field] + EPS < gate[name+'_'+threshold]:
                    failures.append(name+' '+field)
        if candidate['invalid'] >= gate['invalid_branches_strictly_below']:
            failures.append('branch invalid')
        parent_length = gate['length_branches_parent']
        if not (candidate['length'] < parent_length if parent_length > 0 else candidate['length'] == 0):
            failures.append('length')
        for name in ('strict', 'typed'):
            need(parent['metrics'][name]['all_slots_exact'] == gate[name+'_exact_min'], 'parent exact does not match frozen gate')
            need(near(parent['metrics'][name]['F1'], gate[name+'_micro_F1_min']) and
                 near(parent['metrics'][name]['mean_sample_reward'], gate[name+'_mean_sample_reward_min']), 'parent metric does not match frozen gate')
        need(parent['invalid'] == gate['invalid_branches_strictly_below'] and parent['length'] == parent_length, 'parent branch gate mismatch')
        paired = {}
        need(set(parent['typed']) == set(candidate['typed']), 'paired full sample identity')
        for sid, r in candidate['typed'].items():
            need(r['gold'] == parent['typed'][sid]['gold'], 'paired gold changed')
            need(candidate['summary'][sid]['candidate_ids'] == parent['summary'][sid]['candidate_ids'], 'paired P3 changed')
        for name, key in [('strict', 'strict'), ('typed', 'normalized')]:
            rows = []
            for sid, r in candidate['typed'].items():
                p = parent['typed'][sid]
                rows.append(dict(sample_id=sid, repair=bool(r[key]['all_slots_exact'] and not p[key]['all_slots_exact']),
                                 regression=bool(p[key]['all_slots_exact'] and not r[key]['all_slots_exact']),
                                 reward_delta=r[key]['F1']-p[key]['F1'], gold_empty=not r['gold'], gold_nonempty=bool(r['gold']), both_legal=r['valid'] and p['valid']))
            paired[name] = dict(repairs=sum(r['repair'] for r in rows), regressions=sum(r['regression'] for r in rows), samples=rows,
                                all_three={tag: metrics(list(a['typed'].values()), key, True) for tag, a in [('parent', parent), ('candidate', candidate)]},
                                strata={stratum: {tag: metrics([a['typed'][r['sample_id']] for r in rows if r[stratum]], key)
                                                  for tag, a in [('parent', parent), ('candidate', candidate)]}
                                        for stratum in ('gold_empty', 'gold_nonempty', 'both_legal')})
        save(self.state/'SCIENCE.json', dict(status='REJECT' if failures else 'ACCEPT', failed_gates=failures,
             promotion_gate=gate, paired=paired, costs={tag:a['terminal'] for tag,a in [('parent',parent),('candidate',candidate)]},
             archives={tag:{k:a[k] for k in ('archive','archive_sha256','receipt_sha256','file_pins','identity','metrics','invalid','length')}
                       for tag,a in [('parent',parent),('candidate',candidate)]}))
        need(not failures, 'scientific failure: '+', '.join(failures))

    def cpu_tasks(self):
        ready = read(self.p(self.c['cpu_ready'])); doc = read(self.p(self.c['cpu_tasks']))
        need(ready['status'] == 'CONDITIONAL_CPU_VERIFIED_NOT_PUBLISHED', 'formal CPU READY not available', 'WAIT')
        need(ready['CPU_verified_samples'] == 3734 and ready['CPU_verified_branches'] == 11202, 'CPU READY full denominator')
        need(self.c['cpu_tasks'] in ready['pins'], 'CPU READY must pin TASKS')
        self.pins(ready['pins'])
        tasks = doc['tasks']
        need([t['task_id'] for t in tasks] == [p['candidate_task_id'] for p in self.c['parts']], 'only four fixed remainder candidates allowed')
        all_ids = set(); begin = 0
        for task, part in zip(tasks, self.c['parts']):
            self.pins(task['pins'])
            need(task['parent_task_id'] == part['control']['task_id'], 'corresponding TRAIN control')
            need(task['samples'] == part['control']['samples'] and task['branches'] == task['samples']*3, 'TRAIN denominator')
            need(task['status'] == 'CONDITIONAL_CPU_VERIFIED_NOT_PUBLISHED', 'candidate CPU status')
            cfg = read(self.p(task['config'])); custody = cfg['partition_custody']
            need(sha(self.p(task['parent_prepared']+'/single_intent_fewshot/native-UNBOUND.json')) == task['parent_config_sha256'] and
                 task['parent_manifest_sha256'] == read(self.p(part['control']['config']))['manifest_sha256'], 'frozen TRAIN parent pins')
            need(cfg['dataset_partition'] == 'eligible_remainder' and custody['parent_denominator'] == 3734 and
                 custody['range_begin'] == begin and custody['range_end_exclusive'] == begin+task['samples'], 'fixed contiguous TRAIN partition')
            need(cfg['runtime']['status'] == 'UNBOUND', 'candidate already bound; no duplicate execution')
            begin += task['samples']
            ids = set(index([json.loads(x) for x in self.p(cfg['samples']).read_bytes().splitlines() if x.strip()], task['samples']))
            need(not all_ids.intersection(ids), 'whole TRAIN/overlapping sample duplication'); all_ids.update(ids)
            need(cfg['expected_rows'] == task['branches'] and cfg['full_denominator'] == task['samples'], 'native denominator')
            for path in (task['config'], cfg['manifest'], task['CPU_verification'], task['budget'], task['runner'], task['bind_script']):
                need(path in task['pins'], 'required CPU pin missing: '+path)
            cpu = read(self.p(task['CPU_verification'])); budget = read(self.p(task['budget']))
            need(cpu['status'] == 'PASS_FIXED_IMAGE_CPU' and cpu['image'] == self.c['fixed_image'] and
                 cpu['samples'] == task['samples'] and cpu['rows'] == task['branches'] and
                 cpu['actual_tokenizer'] and cpu['validate_fit_replayed'] and cpu['model_calls'] == 0, 'formal fixed image CPU evidence')
            need(cpu['manifest_sha256'] == budget['manifest_sha256'] == cfg['manifest_sha256'] and cpu['max_total_upper'] < 65536, 'CPU budget manifest pins')
            manifest = index(self.jsonlines(cfg['manifest']), task['branches'])
            roster = index(self.jsonlines(cfg['samples']), task['samples'])
            need(set(manifest) == {b['branch_key'] for r in roster.values() for b in r['branches']}, 'CPU sample/manifest branch IDs')
            need(all(manifest[b['branch_key']]['source_sample_id'] == sid for sid,r in roster.items() for b in r['branches']), 'CPU manifest source identity')
            # producer.publish requires these existing canonical evidence paths; never run verify/copy.
            for field, filename in [('CPU_verification','CPU-VERIFIED.json'), ('budget','budget.json')]:
                canonical = self.p(self.c['producer_output'])/'evidence'/task['task_id']/filename
                need(canonical.is_file() and sha(canonical) == sha(self.p(task[field])), 'producer evidence not ready', 'WAIT')
        need(begin == len(all_ids) == 3734, 'TRAIN union denominator')
        return tasks

    def inventory(self, task):
        doc = read(self.p(self.c['published_tasks']))
        for other in doc['tasks']:
            if other['task_id'] == task['task_id']:
                need(other['config'] == task['config'], 'published task ID collision')
                return 'ALREADY_SUBMITTED'
            need(other.get('config') != task['config'] and other.get('prepared') != task['prepared'], 'whole task duplicate identity')
        if (self.p(self.c['execution_tasks'])/(task['task_id']+'.json')).exists():
            return 'ALREADY_SUBMITTED'
        if (self.p(self.c['producer_output'])/'evidence'/task['task_id']/'PUBLISHED.json').exists():
            return 'ALREADY_SUBMITTED'
        return None

    def publish(self, task):
        if self.publisher:
            return self.publisher(task)
        self.pins(self.c['publisher_pins'])
        path = self.p(self.c['publisher'])
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location('successor_producer', path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        need(mod.ROOT == self.root and mod.OUT == self.p(self.c['producer_output']), 'publisher roots differ')
        need(self.now() < read(self.p(self.c['window']))['end_epoch'], 'WINDOW deadline before publisher', 'STOP')
        mod.publish(task)  # retains existing ready-writer.guard; no verify, SSH or dispatch

    def once(self, apply=False):
        self.cache = {}
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state/'admission.guard').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = dict(at=self.now(), mode='apply' if apply else 'preview', parts={})
            try:
                window = read(self.p(self.c['window']))
                frozen = self.state/'FROZEN.json'
                current = dict(config_sha256=self.config_sha, window=window)
                if frozen.exists():
                    need(read(frozen) == current, 'window/config changed; do not reset admission window')
                else:
                    save(frozen, current)
                need(self.c['cadence_seconds'] == 600, 'cadence must be 600 seconds')
                need(self.now() < window['end_epoch'], 'WINDOW deadline reached; stop admission', 'STOP')
                self.pins(self.c['input_pins'])
                promotion = read(self.p(self.c['promotion_task']))
                gate = promotion['promotion_gate']
                need(gate == self.c['promotion_gate'], 'full promotion_gate changed/omitted')
                need(self.c['promotion']['samples'] == self.c['parent']['samples'] == gate['full_samples'] and
                     self.c['promotion']['branches'] == self.c['parent']['branches'] == gate['full_branches'], 'full promotion/parent denominators')
                need(promotion['task_id'] == self.c['promotion']['task_id'] and promotion['samples'] == gate['full_samples'] and
                     promotion['branches'] == gate['full_branches'] and gate['unknown'] == 0, 'promotion dimensions')
                candidate = self.archive(self.c['promotion']); parent = self.archive(self.c['parent'])
                self.scientific(candidate, parent, gate)
                tasks = self.cpu_tasks()
                for task, part in zip(tasks, self.c['parts']):
                    name = task['task_id']; journal = self.state/(name+'.json')
                    try:
                        existing = self.inventory(task)
                        if existing:
                            result['parts'][name] = existing
                            continue
                        if journal.exists():
                            old = read(journal)
                            result['parts'][name] = old['status']
                            continue  # intent/unknown/failed are never replayed
                        control = self.archive(part['control'])
                        cfg = read(self.p(task['config']))
                        candidate_ids = set(index([json.loads(x) for x in self.p(cfg['samples']).read_bytes().splitlines() if x.strip()], task['samples']))
                        need(candidate_ids == set(control['typed']), 'TRAIN paired sample IDs')
                        candidate_roster = index(self.jsonlines(cfg['samples']), task['samples'])
                        for sid, row in candidate_roster.items():
                            need(row['candidate_ids'] == control['summary'][sid]['candidate_ids'] and
                                 [(b['branch_key'], b['candidate_id'], b['candidate_index']) for b in row['branches']] ==
                                 [(b['branch_key'], b['candidate_id'], b['candidate_index']) for b in control['summary'][sid]['branches']],
                                 'TRAIN paired P3 identity')
                        # Normalize only the new publication copy; keep CPU inputs/pins intact.
                        task = task | dict(parent_version=part['control']['task_id'],
                             method='compact_serialization_transfer_on_TRAIN_fewshot',
                             purpose='Transfer the existing compact JSON instruction to the corresponding frozen TRAIN fewshot parent; no weights update',
                             preferred_lane=part['preferred_lane'])
                        if not apply:
                            result['parts'][name] = 'WOULD_PUBLISH'
                            continue
                        need(self.now() < window['end_epoch'], 'WINDOW deadline reached', 'STOP')
                        self.pins(task['pins'])
                        existing = self.inventory(task)
                        if existing:
                            result['parts'][name] = existing
                            continue
                        need(self.now() < window['end_epoch'], 'WINDOW deadline reached', 'STOP')
                        save(journal, dict(status='PUBLISH_INTENT_HOLD_NO_RETRY', at=self.now(), config_sha256=self.config_sha,
                             task_id=name, control_archive_sha256=control['archive_sha256']))
                        try:
                            self.publish(task)
                        except Exception as exc:
                            save(journal, dict(status='PUBLISH_UNKNOWN_HOLD_NO_RETRY', at=self.now(), task_id=name, error=str(exc)))
                            result['parts'][name] = 'PUBLISH_UNKNOWN_HOLD_NO_RETRY'
                            continue
                        save(journal, dict(status='PUBLISHED_ONCE', at=self.now(), task_id=name))
                        result['parts'][name] = 'PUBLISHED_ONCE'
                    except FileNotFoundError as exc:
                        result['parts'][name] = dict(status='WAIT', reason='missing local evidence: '+str(exc))
                    except Decision as exc:
                        result['parts'][name] = dict(status=exc.status, reason=exc.reason)
                    except (ValueError, KeyError, TypeError, tarfile.TarError) as exc:
                        result['parts'][name] = dict(status='REJECT', reason='invalid control evidence: '+str(exc))
                result['status'] = 'ACCEPT_SCIENCE_CONDITIONAL_PART_ADMISSION'
            except FileNotFoundError as exc:
                result.update(status='WAIT', reason='missing local evidence: '+str(exc))
            except Decision as exc:
                result.update(status=exc.status, reason=exc.reason)
            except (ValueError, KeyError, TypeError, tarfile.TarError) as exc:
                result.update(status='REJECT', reason='invalid evidence: '+str(exc))
            save(self.state/('EVENT-'+str(time.time_ns())+'.json'), result)
            save(self.state/'STATUS.json', result)
            return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--state', required=True, help='persistent local directory; preserve across restarts')
    p.add_argument('--apply', action='store_true', help='reviewed deployment only; invokes producer.publish')
    p.add_argument('--watch', action='store_true')
    a = p.parse_args(); gate = Gate(a.config, a.state)
    while True:
        result = gate.once(a.apply)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not a.watch or result['status'] in ('STOP', 'REJECT'):
            return
        # No dependency on NPU scheduler, service or CPU preparation process.
        deadline = time.monotonic()+gate.c['cadence_seconds']
        while time.monotonic() < deadline:
            time.sleep(min(30, deadline-time.monotonic()))


if __name__ == '__main__':
    main()
