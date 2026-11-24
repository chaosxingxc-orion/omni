"""DP8 configuration adaptation of pinned native-training-successor/worker.py.

CPU/meta load and original-buffer materialization come from Jev. All optimizer
scheduling/backward/clipping stays inside DataParallelPPOActor.update_policy.
"""
import json,os,inspect
from pathlib import Path
from runtime import commit,require,sources

def construct(config,root):
 import torch,torch_npu,torch.distributed as dist
 from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision,ShardingStrategy
 from torch.distributed.fsdp.wrap import CustomPolicy
 from multimodal_jev.omni_fsdp_loading import load_cpu_thinker,build_meta_thinker,MetaParameterMaterializer,WRAP_CLASSES,buffer_manifest
 from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3
 initialize_verl_preserving_qwen3()
 from verl.workers.fsdp_workers import ActorRolloutRefWorker,FSDPCheckpointManager
 from verl.workers.actor.dp_actor import DataParallelPPOActor
 from omegaconf import OmegaConf
 from peft import LoraConfig,get_peft_model
 rank=dist.get_rank();world=dist.get_world_size();require(world==config['world_size'],'configured real NPU ranks required')
 device=torch.device('npu',rank);torch.npu.set_device(device);torch.manual_seed(config['seed'])
 out=Path(root)/f'rank-{rank}'
 commit(out/'LOAD-INTENT.json',{'rank':rank,'device':str(device),'loader':'Jev load_cpu_thinker rank0/build_meta_thinker peers','world':world})
 mode=config.get('checkpoint_storage_mode','full')
 require(mode in ('full','lora_training_state'),'unknown checkpoint storage_mode')
 if mode=='lora_training_state':
  from lora_training_state import verify_base_files
  from resident import collectively
  collectively(lambda:verify_base_files(config['model_path'],config['base_assets_ref']) if rank==0 else None)
 if rank==0:model,loading=load_cpu_thinker(config['model_path']);commit(out/'LOADING.json',{k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()})
 else:model=build_meta_thinker(json.loads((Path(config['model_path'])/'config.json').read_text()))
 buffers=buffer_manifest(model)
 model.config.use_cache=False
 model.gradient_checkpointing_disable()
 lora=LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],bias='none',target_modules=[n for n,m in model.named_modules() if __import__('re').fullmatch(r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)',n)],exclude_modules=r'.*(audio_tower|visual|projector|merger).*',task_type=None)
 adapted=get_peft_model(model,lora)
 if config.get('expanded_publication') and rank==0:
  from rank_profile import load_initial_publication
  load_initial_publication(adapted,config['expanded_publication'],config)
 names=[n for n,p in adapted.named_parameters() if p.requires_grad]
 require(names and all('.model.layers.' in n and ('.lora_A.' in n or '.lora_B.' in n) for n in names),'only text attention LoRA can train')
 # PEFT creates meta adapters beside meta base on peer ranks. Materialize ONLY
 # these tiny FP32 masters on CPU; sync_module_states copies rank0 values.
 for module_name,module in adapted.named_modules():
  if '.lora_A.' in module_name or '.lora_B.' in module_name:
   for name,p in list(module._parameters.items()):
    if p is not None:
     module._parameters[name]=torch.nn.Parameter(torch.empty(p.shape,dtype=torch.float32,device='cpu') if p.is_meta else p.detach().float(),requires_grad=True)
 # Separate FSDP units keep mixed BF16 frozen weights / FP32 LoRA masters out
 # of one FlatParameter. Parent decoder units still shard actual 30B experts.
 units={m for n,m in adapted.named_modules() if type(m).__name__ in WRAP_CLASSES or isinstance(m,torch.nn.Linear) and ('.lora_A.' in n or '.lora_B.' in n)}
 base_materializer=MetaParameterMaterializer(device)
 def materialize(module):
  params=list(module.parameters(recurse=False))
  if any(p.requires_grad for p in params):
   require(all(p.requires_grad and p.dtype==torch.float32 and not p.is_meta for p in params),'isolated real FP32 LoRA unit required')
   for name,p in list(module._parameters.items()):
    if p is not None:module._parameters[name]=torch.nn.Parameter(p.detach().to(device),requires_grad=True)
  else:base_materializer(module)
 wrapped=FSDP(adapted,auto_wrap_policy=CustomPolicy(lambda m:m in units),param_init_fn=materialize,
   device_id=device,sharding_strategy=ShardingStrategy.FULL_SHARD,use_orig_params=True,sync_module_states=True,
   mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32,keep_low_precision_grads=False),
   forward_prefetch=False,backward_prefetch=None,limit_all_gathers=True)
 from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper,apply_activation_checkpointing,CheckpointImpl
 from functools import partial
 apply_activation_checkpointing(wrapped,checkpoint_wrapper_fn=partial(checkpoint_wrapper,checkpoint_impl=CheckpointImpl.NO_REENTRANT),check_fn=lambda m:isinstance(m,FSDP) and type(m.module).__name__=='Qwen3OmniMoeThinkerTextDecoderLayer')
 require(sum(type(m).__name__=='CheckpointWrapper' for m in wrapped.modules())==48,'48 decoder FSDP units must be inside checkpoint wrappers')
 local_count=torch.tensor(sum(p.numel() for p in wrapped.parameters()),device=device,dtype=torch.int64);dist.all_reduce(local_count)
 require(int(local_count)==31719205488+(835584*config['lora_rank']),'global sharded parameter coverage differs')
 commit(out/'FSDP-BUILT.json',{'rank':rank,'world_size':world,'global_parameters':int(local_count),'local_parameters':sum(p.numel() for p in wrapped.parameters()),'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved(),'peak_allocated':torch.npu.max_memory_allocated(),'physical_device_nodes':sorted(str(f) for f in Path('/dev').glob('davinci[0-9]*')),'visible_count':torch.npu.device_count()})
 optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=config['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
 cfg=OmegaConf.create(dict(use_remove_padding=False,use_fused_kernels=False,ulysses_sequence_parallel_size=1,use_dynamic_bsz=False,use_prefix_grouper=False,entropy_from_logits_with_chunking=False,use_torch_compile=False,fsdp_config={'dtype':'bfloat16'},entropy_checkpointing=False,calculate_sum_pi_squared=False,ppo_mini_batch_size=1,ppo_micro_batch_size_per_gpu=1,ppo_epochs=1,use_kl_loss=False,calculate_entropy=False,entropy_coeff=0.,loss_agg_mode='unified-token-sum',use_rollout_log_probs=True,policy_loss={'loss_mode':'native_successor'},clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.2,clip_ratio_c=3.,global_batch_info={},grad_clip=1.))
 actor=DataParallelPPOActor(cfg,wrapped,optimizer)
 from transformers import Qwen3OmniMoeProcessor
 processor=Qwen3OmniMoeProcessor.from_pretrained(config['model_path'],local_files_only=True)
 class NativeOmniWorker(ActorRolloutRefWorker):
  """Builds concrete resident state; inherits original save/load implementation.

  Initialization is specialized for unsupported Omni CPU/meta loading. No Ray
  controller or hybrid vLLM copy is allocated inside this bounded worker.
  """
  def __init__(self):
   self._rank=rank;self._world_size=world;self.actor=actor;self.actor_module_fsdp=wrapped;self.actor_module=adapted
   self.actor_optimizer=optimizer;self.actor_lr_scheduler=scheduler;self.processor=processor;self.tokenizer=processor.tokenizer
   self._is_actor=True;self._is_rollout=False;self._is_offload_param=False;self._is_offload_optimizer=False
   self._is_lora=True
   self.checkpoint_manager=FSDPCheckpointManager(model=wrapped,optimizer=optimizer,lr_scheduler=scheduler,processing_class=processor,
    checkpoint_config=OmegaConf.create({'save_contents':['model','optimizer','extra'],'load_contents':['model','optimizer','extra'],'async_save':False}))
  def save_checkpoint(self,local_path,hdfs_path=None,global_step=0,max_ckpt_to_keep=None):
   require(getattr(self,'checkpoint_storage_mode','full')=='full','LoRA exact save must use original checkpoint.save API')
   # Keep the original checkpoint manager, replacing ONLY broken PEFT export.
   from verl.utils.fsdp_utils import layered_summon_lora_params
   from safetensors.torch import save_file
   import os
   self.checkpoint_manager.save_checkpoint(local_path=local_path,hdfs_path=hdfs_path,global_step=global_step,max_ckpt_to_keep=max_ckpt_to_keep)
   self.export_adapter(local_path)
  def export_adapter(self,local_path):
   from verl.utils.fsdp_utils import layered_summon_lora_params
   from safetensors.torch import save_file
   Path(local_path).mkdir(parents=True,exist_ok=True)
   dist.barrier()
   # Original exporter recognizes FSDP decoder children directly; checkpoint
   # wrappers are transparent for state_dict but not isinstance(FSDP). At this
   # post-backward export boundary temporarily expose the same resident units.
   from policy_identity import resident_lora_weights
   params=resident_lora_weights(self)
   if rank==0:
    dest=Path(local_path)/'lora_adapter';dest.mkdir(exist_ok=True)
    require(params and all(v.numel()>0 for v in params.values()),'empty adapter export')
    require(len(params)==384 and sum(v.numel() for v in params.values())==(835584*config['lora_rank']),'48decoder complete configured qkvo LoRA required')
    published={k:v.detach().cpu().to(torch.bfloat16).contiguous() for k,v in params.items()}
    save_file(published,str(dest/'adapter_model.safetensors'))
    commit(dest/'PUBLICATION-DTYPE.json',{'master_dtype':'float32','publication_dtype':'bfloat16','conversion':'master.to(bfloat16), same as FSDP param_dtype','keys':384,'elements':(835584*config['lora_rank'])})
    adapted.peft_config['default'].save_pretrained(str(dest))
    from safetensors import safe_open
    with safe_open(str(dest/'adapter_model.safetensors'),framework='pt') as stream:
     require(set(stream.keys())==set(params),'adapter readback keys changed')
    require(json.loads((dest/'adapter_config.json').read_text())['task_type'] is None,'keep generic PEFT wrapper')
   dist.barrier()
 worker=NativeOmniWorker()
 from policy_identity import base_assets
 worker.base_identity=base_assets(config['model_path'],config['base_assets_ref'])
 commit(out/'WORKER-BUILT.json',{'runtime':sources({'actor':DataParallelPPOActor,'original_worker':ActorRolloutRefWorker,'loader':load_cpu_thinker,'checkpoint':FSDPCheckpointManager}),'world_size':world,'rank':rank,'sharding':'FULL_SHARD','use_orig_params':True,'trainable_names':names,'local_trainable_parameters':sum(p.numel() for p in wrapped.parameters() if p.requires_grad),'buffers_before':buffers,'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved(),'original_update_policy':inspect.getfile(DataParallelPPOActor),'no_full_model_to_device':True})
 from multimodal_jev.context_optimizer_audit import audit_adam_shards
 worker.audit_master_optimizer = lambda step: audit_adam_shards([(n,p) for n,p in wrapped.named_parameters() if p.requires_grad], optimizer, step)
 if mode=='lora_training_state':
  from lora_training_state import initialize
  from dependencies import source_pins
  initialize(worker,source_pins(),integrity_policy=config.get('checkpoint_integrity'))
 if 'actor_arithmetic_profile' in config:
  from actor_arithmetic import install
  worker.close_actor_arithmetic = install(adapted,config['actor_arithmetic_profile'],world)
 return worker,adapted,names
