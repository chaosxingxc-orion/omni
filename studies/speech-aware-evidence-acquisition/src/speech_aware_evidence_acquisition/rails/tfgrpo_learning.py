"""Model-driven TF-GRPO experience updates through an injected metered caller.

This learning-side adapter owns no network client or inference authorization.
The caller must bind registered same-core contacts to the existing runtime.
Published versions are candidates, not accepted deployment policies.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from ..core.atomic_file import publish_new
from .tfgrpo_prompts import PROMPTS, SOURCE_SHA256, UPSTREAM_COMMIT


class LearningError(ValueError):
    pass


class UnknownLearningContact(LearningError):
    """A durable intent has no observed response; reconcile, never auto resend."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise LearningError(f'{name} must be nonempty text')
    return value


def render(key, **values):
    template = PROMPTS[key]
    names = set(re.findall(r'{{\s*(\w+)\s*}}', template))
    if not names <= values.keys():
        raise LearningError('missing upstream template variables')
    # One pass: trajectory text containing template-like tokens stays data.
    return re.sub(r'{{\s*(\w+)\s*}}', lambda m: str(values[m[1]]), template)


def empty_experience(model_id):
    require_text(model_id, 'model identity')
    body = {'schema': 'saea-tfgrpo-experience-v1', 'model_id': model_id,
            'items': {}, 'learned_query_ids': [], 'parent_sha256': None,
            'status': 'candidate', 'deployment_accepted': False}
    return {**body, 'sha256': sha(body)}


def validate_experience(version, model_id):
    if not isinstance(version, dict) or version.get('schema') != 'saea-tfgrpo-experience-v1':
        raise LearningError('experience schema differs')
    body = {k:v for k,v in version.items() if k != 'sha256'}
    if version.get('sha256') != sha(body) or version.get('model_id') != model_id:
        raise LearningError('experience bytes or frozen model differ')
    if not isinstance(version.get('items'), dict):
        raise LearningError('experience items must be an object')
    for key, value in version['items'].items():
        require_text(key, 'experience id'); require_text(value, 'experience content')
    ids = version.get('learned_query_ids')
    if not isinstance(ids, list) or any(not isinstance(x,str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise LearningError('unique learning query identities required')
    return version


def experience_context(version, query_id, *, purpose="held-out-evaluation"):
    """Render candidate experience with explicit learning/evaluation semantics.

    Production acceptance and source/label leakage checks remain the runtime's
    responsibility; this is not an acceptance or exposure authorization API.
    """
    validate_experience(version, version['model_id'])
    require_text(query_id, 'query id')
    if purpose not in ('learning','held-out-evaluation'):
        raise LearningError('explicit context purpose required')
    if purpose == 'held-out-evaluation' and query_id in version['learned_query_ids']:
        raise LearningError('query cannot be held-out after contributing to this experience')
    return '\n'.join(f'[{key}]. {value}' for key,value in sorted(version['items'].items())) or 'None'


def parse_operations(text, items, *, max_operations):
    require_text(text, 'model operation response')
    stripped = text.strip()
    if stripped.startswith('```json') and stripped.endswith('```'):
        stripped = stripped[7:-3].strip()
    try:
        operations = json.loads(stripped)
    except (ValueError, TypeError) as error:
        raise LearningError('model operations must be JSON') from error
    if not isinstance(operations, list) or len(operations) > max_operations:
        raise LearningError('bounded operation array required')
    touched = set()
    for operation in operations:
        if not isinstance(operation, dict) or set(operation) != {'operation', 'id', 'content'}:
            raise LearningError('exact operation schema required')
        action, target, content = (operation[k] for k in ('operation','id','content'))
        if action not in ('ADD','UPDATE','DELETE','NONE') or not isinstance(content, str):
            raise LearningError('unsupported model operation')
        if action in ('UPDATE','DELETE'):
            if not isinstance(target,str) or target not in items or target in touched:
                raise LearningError('unknown or multiply modified experience id')
            touched.add(target)
        elif target is not None:
            raise LearningError('ADD/NONE must not target an existing id')
        if action in ('ADD','UPDATE'):
            require_text(content, 'updated content')
    return operations


def apply_operations(items, operations, *, max_items, max_characters):
    result = copy.deepcopy(items)
    for operation in operations:
        action, target, content = (operation[k] for k in ('operation','id','content'))
        if action == 'ADD':
            key = 'E' + hashlib.sha256(content.encode()).hexdigest()
            if key in result:
                raise LearningError('duplicate ADD; model must choose NONE or UPDATE')
            result[key] = content
        elif action == 'UPDATE':
            result[target] = content
        elif action == 'DELETE':
            del result[target]
    if len(result) > max_items or sum(len(v) for v in result.values()) > max_characters:
        raise LearningError('experience size exceeds declared limit; no silent truncation')
    return result


class ExperienceLearner:
    """One immutable learning batch with bounded, durable model-driven stages.

    ``metered_call(request)`` returns JSON-compatible ``text``, exact ``model_id``,
    nonempty ``receipt_id`` and ``usage`` with calls=1 and integer input/output
    token counts. The production adapter must obtain these from actual metering,
    not synthesize them. ``learning_config_sha256`` pins the registered learner
    decoding, rendering and runtime configuration independently of rollout policy.
    Optional ``preflight(request)`` validates/prepares without generation before
    intent creation. Optional ``reconcile(request)`` is read-only and returns None
    or an exact request/response hash envelope from a verified runtime receipt;
    it must never dispatch. Caller exceptions retain UNKNOWN custody until such
    a receipt is found, never automatically re-send.
    """
    def __init__(self, root, *, model_id, metered_call, learning_config_sha256, max_calls, concurrency=4,
                 preflight=None, reconcile=None,
                 max_operations=32, max_items=128, max_characters=32000):
        self.root = Path(root)
        self.model_id = require_text(model_id, 'model identity')
        if not callable(metered_call):
            raise LearningError('externally metered callable required')
        for value in (max_calls, concurrency, max_operations, max_items, max_characters):
            if type(value) is not int or value < 1:
                raise LearningError('positive integer bounds required')
        if not isinstance(learning_config_sha256,str) or not re.fullmatch('[a-f0-9]{64}',learning_config_sha256):
            raise LearningError('pinned learning generation configuration required')
        for callback in (preflight,reconcile):
            if callback is not None and not callable(callback):
                raise LearningError('preflight and reconciliation must be callable')
        self.learning_config_sha256 = learning_config_sha256
        self.preflight, self.reconcile = preflight, reconcile
        self.caller = metered_call
        self.max_calls, self.concurrency = max_calls, concurrency
        self.max_operations, self.max_items, self.max_characters = max_operations, max_items, max_characters
        self._reservation_lock = threading.Lock()
        self._batch = None
        self.last_invocation = {'new_calls': 0, 'reused_calls': 0}

    @contextmanager
    def _exclusive_batch(self):
        missing=[]; current=self.root/'contacts'
        while not current.exists():
            missing.append(current); current=current.parent
        for path in reversed(missing):
            path.mkdir(exist_ok=True)
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        with (self.root/'runner.lock').open('a+b') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise LearningError('batch is already being executed') from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _observed_text(self, observed):
        if not isinstance(observed,dict) or observed.get('model_id') != self.model_id:
            raise LearningError('response came from another model')
        require_text(observed.get('receipt_id'), 'metering receipt')
        usage = observed.get('usage')
        if not isinstance(usage,dict) or usage.get('calls') != 1 or type(usage.get('calls')) is not int:
            raise LearningError('one metered call receipt required')
        for field in ('prompt_tokens','completion_tokens'):
            if type(usage.get(field)) is not int or usage[field] < 0:
                raise LearningError('nonnegative metered token counts required')
        return require_text(observed.get('text'), 'model text')

    def _verified_response(self, record, request):
        if (not isinstance(record,dict)
                or set(record) != {'request_sha256','response','response_sha256'}
                or record['request_sha256'] != sha(request)
                or record['response_sha256'] != sha(record['response'])):
            raise LearningError('observed response custody differs')
        self._observed_text(record['response'])
        return record['response']

    def _call(self, stage, identity, messages):
        request = {'schema': 'saea-tfgrpo-learning-request-v1', 'batch_sha256': self._batch,
                   'model_id': self.model_id, 'learning_config_sha256':self.learning_config_sha256,
                   'stage': stage, 'identity': identity, 'messages': messages}
        key = sha({'stage':stage, 'identity':identity})
        intent_path = self.root/'contacts'/(key+'.intent.json')
        response_path = self.root/'contacts'/(key+'.response.json')
        with self._reservation_lock:
            known = intent_path.exists()
            if known and intent_path.read_bytes() != canonical(request):
                raise LearningError('known contact request changed')
            if not known and response_path.exists():
                raise LearningError('orphan response')
        if known:
            if response_path.exists():
                observed = self._verified_response(json.loads(response_path.read_bytes()),request)
            else:
                # Read only: the runtime may have durably observed the response
                # before the learner process exited. Reconcile that exact receipt,
                # never invoke the generation callback for an existing intent.
                record = None if self.reconcile is None else self.reconcile(copy.deepcopy(request))
                if record is None:
                    raise UnknownLearningContact('contact outcome unknown; no automatic retry')
                observed = self._verified_response(record,request)
                publish_new(response_path,canonical(record))
            with self._reservation_lock:
                self.last_invocation['reused_calls'] += 1
        else:
            # Deterministic preparation failures must not become unknown sends.
            # Keep preparation outside the budget lock so independent FITs overlap.
            if self.preflight is not None:
                self.preflight(copy.deepcopy(request))
            with self._reservation_lock:
                if intent_path.exists() or response_path.exists():
                    raise LearningError('contact concurrently claimed')
                if len(list((self.root/'contacts').glob('*.intent.json'))) >= self.max_calls:
                    raise LearningError('learning contact budget exhausted')
                publish_new(intent_path, canonical(request))
                self.last_invocation['new_calls'] += 1
            observed = self.caller(copy.deepcopy(request))
            # Preserve invalid but observed outputs too; they must not be resampled.
            response = {'request_sha256': sha(request), 'response': observed, 'response_sha256':sha(observed)}
            publish_new(response_path, canonical(response))
        return self._observed_text(observed)

    def accounting(self):
        """Report cumulative attempts and known usage, including failed batches.

        Unknown contacts retain their reservation. This never estimates missing
        token spend or treats a failed batch as free.
        """
        totals={key:0 for key in ('calls','prompt_tokens','completion_tokens')}
        intents=list((self.root/'contacts').glob('*.intent.json'))
        observed=0; unknown=0; ids=set(); response_names=set()
        for path in intents:
            request=json.loads(path.read_bytes())
            response_path=path.with_name(path.name.replace('.intent.json','.response.json'))
            if not response_path.exists():
                unknown+=1
                continue
            response_names.add(response_path.name)
            record=json.loads(response_path.read_bytes())
            if record.get('request_sha256')!=sha(request) or record.get('response_sha256')!=sha(record.get('response')):
                raise LearningError('accounting contact custody differs')
            response=record['response']
            if not isinstance(response,dict) or response.get('model_id')!=self.model_id:
                raise LearningError('accounting response model differs')
            receipt=require_text(response.get('receipt_id'),'metering receipt')
            if receipt in ids:raise LearningError('duplicate metering receipt identity')
            ids.add(receipt)
            usage=response.get('usage')
            if not isinstance(usage,dict) or type(usage.get('calls')) is not int or usage['calls']!=1:
                raise LearningError('accounting requires one-call metering receipt')
            for key in totals:
                value=usage.get(key)
                if type(value) is not int or value<0:raise LearningError('invalid metered usage')
                totals[key]+=value
            observed+=1
        if response_names!={p.name for p in (self.root/'contacts').glob('*.response.json')}:
            raise LearningError('orphan accounting response')
        return {'attempted_contacts':len(intents),'observed_contacts':observed,
                'unknown_contacts':unknown,'known_usage':totals,
                'unknown_spend_excluded_from_known_usage':bool(unknown)}

    def _messages(self, prefix, common, **values):
        return [{'role':'system','content':render(prefix+'_SP', **common)},
                {'role':'user','content':render(prefix+'_UP', **values)}]

    def learn(self, groups, parent, *, agent_objective, learning_objective,
              rollout_kind='full-same-policy', policy_sha256, num_experiences=2):
        """Produce a candidate version; no gold enters a task inference request."""
        groups, parent = copy.deepcopy(groups), copy.deepcopy(parent)
        validate_experience(parent, self.model_id)
        if rollout_kind not in ('full-same-policy','shared-perception-adaptation'):
            raise LearningError('explicit rollout semantics required')
        if not isinstance(policy_sha256,str) or not re.fullmatch('[a-f0-9]{64}',policy_sha256):
            raise LearningError('pinned rollout policy required')
        if type(num_experiences) is not int or not 1 <= num_experiences <= self.max_operations:
            raise LearningError('invalid per-group experience bound')
        for value in (agent_objective,learning_objective):require_text(value,'objective')
        seen_queries, seen_trajectories = set(), set()
        for group in groups:
            if set(group) != {'query_id','query','answer','rollouts'}:
                raise LearningError('exact learning group schema required')
            qid=require_text(group['query_id'],'query id')
            if qid in seen_queries:raise LearningError('duplicate learning query')
            seen_queries.add(qid)
            for field in ('query','answer'):require_text(group[field],field)
            if not isinstance(group['rollouts'],list) or len(group['rollouts']) < 2:
                raise LearningError('at least two rollouts per group required')
            for rollout in group['rollouts']:
                if set(rollout) != {'trajectory_id','trajectory','critique','reward','model_id','policy_sha256','experience_sha256'}:
                    raise LearningError('exact rollout provenance required')
                rid=require_text(rollout['trajectory_id'],'trajectory id')
                if rid in seen_trajectories:raise LearningError('duplicate trajectory identity')
                seen_trajectories.add(rid)
                for field in ('trajectory','critique'):require_text(rollout[field],field)
                if type(rollout['reward']) is not int or rollout['reward'] not in (0,1):
                    raise LearningError('baseline requires binary reward, not continuous SLU-F1')
                if (rollout['model_id'] != self.model_id or rollout['policy_sha256'] != policy_sha256
                        or rollout['experience_sha256'] != parent['sha256']):
                    raise LearningError('all rollouts must share model, policy and experience')
        common={'agent_objective':agent_objective,'learning_objective':learning_objective,'num_experiences':num_experiences}
        binding={'schema':'saea-tfgrpo-batch-v1','groups':groups,'parent':parent,'common':common,
                 'model_id':self.model_id,'policy_sha256':policy_sha256,'rollout_kind':rollout_kind,
                 'learning_config_sha256':self.learning_config_sha256,
                 'prompt_sha256':SOURCE_SHA256,'upstream_commit':UPSTREAM_COMMIT,
                 'bounds':{'max_calls':self.max_calls,'concurrency':self.concurrency,
                           'max_operations':self.max_operations,'max_items':self.max_items,
                           'max_characters':self.max_characters}}
        with self._exclusive_batch():
            self.last_invocation = {'new_calls': 0, 'reused_calls': 0}
            binding_path=self.root/'batch.json'
            if binding_path.exists():
                if binding_path.read_bytes()!=canonical(binding):raise LearningError('batch binding changed')
            else:publish_new(binding_path,canonical(binding))
            self._batch=sha(binding)
            mixed=[g for g in groups if {r['reward'] for r in g['rollouts']}=={0,1}]
            planned=sum(len(g['rollouts'])+2 for g in mixed)+(1 if mixed else 0)
            if planned>self.max_calls:raise LearningError('complete update exceeds contact budget before first call')
            result_path=self.root/'candidate.json'
            if result_path.exists():
                candidate=json.loads(result_path.read_bytes());validate_experience(candidate,self.model_id)
                if candidate.get('batch_sha256')!=self._batch:raise LearningError('candidate belongs to another batch')
                actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((self.root/'contacts').glob('*.response.json'))}
                if candidate.get('contact_response_sha256')!=actual:raise LearningError('candidate contact evidence changed')
                if candidate.get('accounting')!=self.accounting():raise LearningError('candidate accounting changed')
                return candidate
            current='\n'.join(f'[{k}]. {v}' for k,v in sorted(parent['items'].items())) or 'None'
            def summarize(pair):
                group,rollout=pair
                from .tfgrpo_projection import summary_trajectory
                return self._call('summary',rollout['trajectory_id'],self._messages(
                    'SINGLE_ROLLOUT_SUMMARY_TEMPLATE',common,question=group['query'],answer=group['answer'],
                    critique=rollout['critique'],trajectory=summary_trajectory(group['query'],rollout['trajectory'])))
            summary_by_id={}
            def advantage(group):
                formatted='\n\n'.join(f"Attempt {i+1} (Reward {r['reward']}):\n{summary_by_id[r['trajectory_id']]}" for i,r in enumerate(group['rollouts']))
                result=self._call('advantage',group['query_id'],self._messages(
                    'SINGLE_QUERY_GROUP_ADVANTAGE',common,question=group['query'],answer=group['answer'],trajectories=formatted))
                matches=re.findall(r'<Experiences>\s*(.*?)\s*</Experiences>',result,re.S|re.I)
                if len(matches)!=1 or not matches[0].strip():raise LearningError('one nonempty semantic advantage block required')
                return matches[0].strip()
            def update_group(group,experience):
                proposed=self._call('group_update',group['query_id'],self._messages(
                    'GROUP_EXPERIENCE_UPDATE_TEMPLATE',common,existing_experiences=current,new_experiences=experience))
                return parse_operations(proposed,parent['items'],max_operations=self.max_operations)
            # Only the coordinator submits work; workers never wait on another
            # future. Ready downstream contacts take priority over unsent summaries.
            pending=deque(('summary',i,r) for i,g in enumerate(mixed) for r in g['rollouts'])
            ready=deque(); active={}; counts=[0]*len(mixed)
            grouped_operations=[None]*len(mixed)
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                try:
                    while pending or ready or active:
                        while len(active)<self.concurrency and (ready or pending):
                            stage,index,value=(ready if ready else pending).popleft()
                            group=mixed[index]
                            if stage=='summary':future=pool.submit(summarize,(group,value))
                            elif stage=='advantage':future=pool.submit(advantage,group)
                            else:future=pool.submit(update_group,group,value)
                            active[future]=(stage,index,value)
                        completed,_=wait(active,return_when=FIRST_COMPLETED)
                        for future in completed:
                            stage,index,value=active.pop(future)
                            result=future.result()
                            if stage=='summary':
                                summary_by_id[value['trajectory_id']]=result
                                counts[index]+=1
                                if counts[index]==len(mixed[index]['rollouts']):
                                    ready.append(('advantage',index,None))
                            elif stage=='advantage':ready.append(('group_update',index,result))
                            else:grouped_operations[index]=result
                except BaseException:
                    for future in active:future.cancel()
                    raise
            operations=[operation for group_ops in grouped_operations for operation in group_ops]
            final_ops=[]
            if mixed:
                material=json.dumps({'existing_experiences':parent['items'],'proposed_operations':operations},ensure_ascii=False,sort_keys=True)
                merged=self._call('batch_update','batch',self._messages(
                    'BATCH_EXPERIENCE_UPDATE_TEMPLATE',common,experiences_and_operations=material))
                final_ops=parse_operations(merged,parent['items'],max_operations=self.max_operations)
            items=apply_operations(parent['items'],final_ops,max_items=self.max_items,max_characters=self.max_characters)
            receipts=[json.loads(p.read_bytes())['response'] for p in sorted((self.root/'contacts').glob('*.response.json'))]
            if len({r['receipt_id'] for r in receipts})!=len(receipts):raise LearningError('duplicate metering receipt identity')
            accounting=self.accounting()
            usage=accounting['known_usage']
            body={'schema':'saea-tfgrpo-experience-v1','model_id':self.model_id,'items':items,
                  'learned_query_ids':sorted(set(parent['learned_query_ids'])|{g['query_id'] for g in mixed}),
                  'parent_sha256':parent['sha256'],'batch_sha256':self._batch,
                  'status':'candidate','deployment_accepted':False,
                  'rollout_kind':rollout_kind,'learning_query_reuse':sorted(seen_queries & set(parent['learned_query_ids'])),'mixed_groups':len(mixed),'skipped_tied_groups':len(groups)-len(mixed),
                  'planned_learning_calls':planned,'operations':final_ops,'learning_usage':usage,'accounting':accounting,
                  'contact_response_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((self.root/'contacts').glob('*.response.json'))}}
            candidate={**body,'sha256':sha(body)}
            publish_new(result_path,canonical(candidate))
            return candidate
