"""Capture existing loopback serving metrics; never sends inference requests."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def capture(ports):
    if not ports or len(set(ports))!=len(ports) or any(p<1024 or p>65535 for p in ports):raise ValueError('distinct unprivileged ports required')
    values=[]
    for port in ports:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=10) as response:
            body=response.read().decode()
        values.append({'port':port,'observed_at':time.time(),'prometheus_text':body})
    return {'schema':'saea-real-engine-metrics-v1','instances':values,'model_calls':0,
            'scope':'Operational counters and gauges; generation/FIT costs require independent scientific accounting.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=capture(a.port)
    with a.output.open('x') as f:json.dump(result,f);f.write('\n')
