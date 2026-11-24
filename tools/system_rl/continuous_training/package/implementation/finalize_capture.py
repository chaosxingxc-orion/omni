"""Same original finalize_on_worker on the resident DP8 worker, local write clone."""
from pathlib import Path
import inspect
from dependencies import ROOT,require,load
from run_package import read,pin,write


def run(worker,adapted,config,hot,rank):
    from budget import check
    check(config,additional=config['update_token_upper'])
    return restored_worker(worker,adapted,config,hot,rank)


def restored_worker(worker,adapted,config,hot,rank,*,sources=None,accepted_sink=None):
    """Finalize on the caller's already authenticated and restored worker."""
    from fresh_adapter import scope,clone_generation
    import torch.distributed as dist
    directory=Path(hot)/(sources['cohort_dir'] if sources else 'finalized-cohort')
    if rank==0:
        clone_generation(config['pending_capture_ref'],config['generation_inventory_ref'],directory)
        if sources:
            captured=read(directory/'INPUT.json')
            official=read(captured['capture']['official_source_config_ref'])
            official['index_cache_root']=sources['index_root']
            write(directory/'LOCAL-OFFICIAL-SOURCE.json',official)
            consumer=read(directory/'CONSUMER.json')
            local=pin(ROOT/'training-consumer-adapter/adapter.py')
            require(pin(consumer['sealed_adapter_ref'])['sha256']==local['sha256'],
                    'sealed consumer adapter changed')
            consumer['sealed_adapter_ref']=local
            write(directory/'LOCAL-CONSUMER.json',consumer)
    dist.barrier()
    native=read(config['pending_capture_ref']);native=dict(native,enabled=True,parity=config['parity'],model_path=config['model']['path'])
    if sources:
        native['local_official_source_ref']=pin(directory/'LOCAL-OFFICIAL-SOURCE.json')
        native['local_consumer_ref']=pin(directory/'LOCAL-CONSUMER.json')
        native['prepared_cache_bytes']=sources['prepared_cache_bytes']
        if 'row_encoding' in sources:native['row_encoding']=sources['row_encoding']
        if sources.get('reference_batch'):
            native['reference_batch']=sources['reference_batch']
        if sources.get('reuse_group_current_scores'):
            native['reuse_group_current_scores']=True
    with scope():
        source=Path(sources['capture_source']) if sources else ROOT/'fresh-cohort-successor/cohort_capture.py'
        if sources:
            load(sources['cohort_source'],'cohort')
            if sources.get('reference_batch'):
                load(sources['reference_batch_source'],'reference_batch')
        module=load(source,'dp8_finalize_original')
        # Original current/reference scorer imports these exact pinned functions.
        import forward,validate_npu
        origins={n:pin(inspect.getfile(obj)) for n,obj in dict(forward=forward.forward,inputs_for=validate_npu.inputs_for).items()}
        require(Path(origins['forward']['path']).resolve()==(ROOT/'native-model-integration/forward.py').resolve(),'wrong actual forward namespace')
        require(Path(origins['inputs_for']['path']).resolve()==(ROOT/'native-training-successor/validate_npu.py').resolve(),'wrong actual input namespace')
        module.finalize_on_worker(worker,adapted,native,hot,rank,directory=directory,resident=True,
            **({'accepted_sink':accepted_sink} if accepted_sink is not None else {}),
            cost_receipt=dict(dp8_reserved_logical_upper=config['update_token_upper']))
    dist.barrier()
    write(Path(hot)/f'rank-{rank}/FINALIZE-RESULT.json',dict(trajectory_ref=pin(directory/'COMPLETE.json'),
        optimizer_steps=0,module_origins=origins,source_generation_inventory_ref=config['generation_inventory_ref']))
