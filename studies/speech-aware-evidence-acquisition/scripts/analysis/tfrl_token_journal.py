"""Durable token-only requests: reuse completed responses, refuse ambiguous spend."""
import fcntl
import hashlib
import json
from pathlib import Path
from speech_aware_evidence_acquisition.core.atomic_file import publish_new


def raw(value):return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()


class JournalTokenizer:
    def __init__(self,client,root,*,identity,cap,guard):
        if type(cap) is not int or cap<1:raise ValueError('positive token contact cap required')
        self.client,self.root,self.identity,self.cap,self.guard=client,Path(root),identity,cap,guard
        self.root.mkdir(parents=True,exist_ok=True)
        path=self.root/'identity.json';binding={'identity':identity,'contact_cap':cap}
        if path.exists():
            if path.read_bytes()!=raw(binding):raise ValueError('token journal identity differs')
        else:publish_new(path,raw(binding))

    def request_json(self,method,path,payload=None):
        if (method,path)!=('POST','/tokenize'):raise ValueError('only token measurement allowed')
        request={'method':method,'path':path,'payload':payload,'identity':self.identity}
        digest=hashlib.sha256(raw(request)).hexdigest()
        start=self.root/(digest+'.started.json');done=self.root/(digest+'.complete.json')
        with (self.root/'journal.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if done.exists():
                doc=json.loads(done.read_bytes())
                if not start.exists() or start.read_bytes()!=raw(request) or doc['request_sha256']!=digest:
                    raise ValueError('token response custody differs')
                if doc['response_sha256']!=hashlib.sha256(raw(doc['response'])).hexdigest():
                    raise ValueError('token response hash differs')
                return doc['response']
            if start.exists():raise ValueError('unfinished token request; inspect ambiguity before continuation')
            if sum(1 for _ in self.root.glob('*.started.json'))>=self.cap:raise ValueError('token contact cap exhausted')
            self.guard()
            publish_new(start,raw(request))
            response=self.client.request_json(method,path,payload)
            # Validate before making this response reusable; failed contacts stay counted.
            if (not isinstance(response,dict) or type(response.get('count')) is not int or response['count']<0
                    or not isinstance(response.get('tokens'),list) or len(response['tokens'])!=response['count']
                    or any(type(t) is not int for t in response['tokens'])):
                raise ValueError('invalid tokenizer count or tokens')
            publish_new(done,raw({'request_sha256':digest,'response':response,
                'response_sha256':hashlib.sha256(raw(response)).hexdigest()}))
            return response

    def accounting(self):
        started=sum(1 for _ in self.root.glob('*.started.json'));complete=sum(1 for _ in self.root.glob('*.complete.json'))
        return {'tokenizer_attempts':started,'completed_tokenizer_requests':complete,'unresolved_tokenizer_requests':started-complete,
            'cap':self.cap,'generation_calls':0}
