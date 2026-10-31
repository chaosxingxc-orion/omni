import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/analysis')]
from tfrl_token_journal import JournalTokenizer


class Client:
    calls=0
    def request_json(self,*a):self.calls+=1;return {'count':2,'tokens':[1,2]}


def test_completed_measurements_reused_across_restart(tmp_path):
    client=Client();j=JournalTokenizer(client,tmp_path,identity={'model':'pinned'},cap=2,guard=lambda:None)
    response=j.request_json('POST','/tokenize',{'text':'hello'})
    again=JournalTokenizer(client,tmp_path,identity={'model':'pinned'},cap=2,guard=lambda:None)
    assert again.request_json('POST','/tokenize',{'text':'hello'})==response and client.calls==1
    assert again.accounting()['tokenizer_attempts']==1
    with pytest.raises(ValueError):JournalTokenizer(client,tmp_path,identity={'model':'changed'},cap=2,guard=lambda:None)


def test_unresolved_request_never_replayed(tmp_path):
    class Fails(Client):
        def request_json(self,*a):self.calls+=1;raise TimeoutError('ambiguous')
    client=Fails();j=JournalTokenizer(client,tmp_path,identity={},cap=2,guard=lambda:None)
    with pytest.raises(TimeoutError):j.request_json('POST','/tokenize',{})
    with pytest.raises(ValueError,match='unfinished'):j.request_json('POST','/tokenize',{})
    assert client.calls==1 and j.accounting()['unresolved_tokenizer_requests']==1


def test_cap_routes_guard_and_corruption(tmp_path):
    client=Client();j=JournalTokenizer(client,tmp_path,identity={},cap=1,guard=lambda:None)
    with pytest.raises(ValueError):j.request_json('POST','/v1/chat/completions',{})
    j.request_json('POST','/tokenize',{})
    with pytest.raises(ValueError,match='cap'):j.request_json('POST','/tokenize',{'new':1})
    p=next(tmp_path.glob('*.complete.json'));r=json.loads(p.read_text());r['response']['count']=99;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='hash'):j.request_json('POST','/tokenize',{})
    assert client.calls==1


def test_guard_prevents_any_contact(tmp_path):
    def refuse():raise ValueError('resident changed')
    client=Client();j=JournalTokenizer(client,tmp_path,identity={},cap=1,guard=refuse)
    with pytest.raises(ValueError):j.request_json('POST','/tokenize',{})
    assert client.calls==0 and j.accounting()['tokenizer_attempts']==0
