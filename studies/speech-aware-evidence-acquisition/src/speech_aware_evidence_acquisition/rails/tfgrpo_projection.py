"""Lossless summary-only deduplication against the prompt's full question.

The sealed source remains unchanged. References replace only equal JSON
values already present in the same model request's Question section.
"""
import copy
import hashlib
import json

VERSION = 'saea-tfgrpo-summary-question-references-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def restore(question, projection):
    if projection.get('schema') != VERSION:
        raise ValueError('unknown summary projection')
    if digest(question) != projection['question_sha256']:
        raise ValueError('summary question changed')
    original=copy.deepcopy(projection['trajectory'])
    request=original['request']
    for field in projection['request_fields_from_question']:
        if field not in ('task_instruction','supplied_evidence','sample_id') or field in request:
            raise ValueError('invalid summary reference')
        request[field]=copy.deepcopy(question[field])
    if projection['history_prefix_from_question']:
        request['history']=copy.deepcopy(question['history'])+request['history']
    if digest(original) != projection['original_trajectory_sha256']:
        raise ValueError('summary reconstruction changed source')
    return original


def summary_trajectory(question_text, trajectory_text):
    """Generic trajectories stay byte-identical; exported request JSON dedups."""
    try:
        question=json.loads(question_text);original=json.loads(trajectory_text)
    except (TypeError, ValueError):
        return trajectory_text
    if (not isinstance(question,dict) or not isinstance(original,dict)
            or not isinstance(original.get('request'),dict)):
        return trajectory_text
    projected=copy.deepcopy(original);request=projected['request'];fields=[]
    for field in ('task_instruction','supplied_evidence','sample_id'):
        if field in request and field in question and request[field]==question[field]:
            del request[field];fields.append(field)
    history=question.get('history');actual=request.get('history')
    prefix=(isinstance(history,list) and bool(history) and isinstance(actual,list)
            and actual[:len(history)]==history)
    if prefix:request['history']=actual[len(history):]
    if not fields and not prefix:return trajectory_text
    document={'schema':VERSION,'question_sha256':digest(question),
        'original_trajectory_sha256':digest(original),'trajectory':projected,
        'request_fields_from_question':fields,'history_prefix_from_question':prefix,
        'reading_instruction':'Reconstruct trajectory.request by copying the named fields from the full Question above. If history_prefix_from_question is true, prepend Question.history to trajectory.request.history. These exact repeated values are omitted only to avoid duplication; all request conditions and the response remain available.'}
    if restore(question,document)!=original:
        raise ValueError('lossless summary projection failed')
    return canonical(document)
