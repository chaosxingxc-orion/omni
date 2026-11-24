"""CPU integration checks using the actual training package and service preparation receipt."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from build import save
from deploy_resident_graph_lanes import prepare
from paired_experiment import HERE, compile_experiment, digest, graph, load, rank_profile


def check(root, start, deployment, model):
    def read(p):
        return json.loads(p.read_bytes())
    os.environ['OMNI_DEPLOY_MODEL_PATH'] = model
    package = read(root/'PACKAGE.json')
    exp = load(root/'EXPERIMENT.json', package['experiment_sha256'])
    actor = read(root/'preview/ACTOR.json')
    service = read(root/'service-prepared.json')
    assert actor['experiment_sha256'] == service['experiment_sha256'] == digest(exp)
    assert service['graph'] == graph(exp) and actor['actor_batch'] == exp['actor_batch']
    assert actor['initial_actor'] == read(start)
    with tempfile.TemporaryDirectory(dir=root) as t:
        work = Path(t)
        template = read(HERE/'P-EXPERIMENT.template.json')
        template['actor_batch']['gradient_accumulation_steps'] = 1
        template['graph_widths'] = template['graph_widths'][:-1]
        template['episode_concurrency'] = template['graph_widths'][-1]
        tmp = work/'variant.json'
        save(tmp, template)
        variant = work/'compiled'
        compile_experiment(tmp, HERE/'EXPERIMENT-DEPLOYMENT.example.json', variant)
        variant_exp = load(variant/'EXPERIMENT.json')
        policy = read(variant/'TRAINING-POLICY.json')
        assert read(variant/'rank-profile.json')['actor_batch']['gradient_accumulation_steps'] == 1
        assert policy['acceptance']['recordings_per_step'] == 16 and policy['rollout_dispatch']['episode_concurrency'] == 64
        assert read(variant/'graph.json')['max_num_seqs'] == 64
        os.environ['OMNI_DEPLOY_MODEL_PATH'] = '/changed-after-first-resolution'
        compile_experiment(tmp, HERE/'EXPERIMENT-DEPLOYMENT.example.json', variant)
        assert load(variant/'EXPERIMENT.json') == variant_exp
        cfg = read(deployment)['config']
        inputs = work/'inputs'
        inputs.mkdir()
        save(inputs/'EXPERIMENT.json', exp)
        cfg['overlays']['policy_identity.py'] = '0'*64
        try:
            prepare(work, cfg)
        except ValueError as exc:
            assert 'different paired adapter reader: policy_identity.py' == str(exc)
        else:
            raise AssertionError('incompatible service did not stop')
        assert not (work/'bundle').exists()
    return save(root/'CHECK.json',dict(experiment_sha256=digest(exp),training_service_match=True,
                variant=dict(recordings=16,episodes=64,max_sequences=64),resume_environment_frozen=True,
                rejected_old_reader_before_preparation=True,npu_dispatch=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root','start','deployment'):
        parser.add_argument(name,type=Path)
    parser.add_argument('model')
    print(json.dumps(check(**vars(parser.parse_args()))))
