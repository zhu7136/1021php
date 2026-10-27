"""Simulator child. Receives no account credential or signed transfer URL."""
import argparse,json,os,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
 plan=json.loads(a.plan.read_text())
 from flux.five_distill_contract import validate_scope,TEACHERS,digest
 validate_scope(plan['scope'])
 import torch
 base=None
 for path,source in zip(plan['teachers'],TEACHERS):
  assert digest(path)==source[4]
  ck=torch.load(path,map_location='cpu',weights_only=True)
  assert ck['iter']==source[2]
  c=ck['experiment_config']
  assert c['robot']['actions_dim']==29 and c['algo']['config']['empirical_normalization'] is False
  if base is not None:assert c['robot']==base['robot'] and c['observation']==base['observation']
  else:base=c
 from flux.preflight import runtime_checks
 runtime_checks(Path(plan['datasets'][0]))
 from wbt_training.config_values.experiment import g1_wbt_terrain_warp_distill
 from wbt_training.utils.isaacsim_clock import use_php_isaacsim
 from holosoma.utils.sim_utils import setup_simulator_imports
 from holosoma.train_agent import train
 from wbt_training.utils.multi_motion_helpers import pull_paired_from_wandb_registry
 # Prove combined route table before creating a simulator.
 counts,motion,terrain,obj=pull_paired_from_wandb_registry([Path(x).resolve().as_uri() for x in plan['datasets']])
 import numpy as np
 with np.load(motion) as d:assert set(np.unique(d['motion_idxs']))==set(range(5))
 assert counts==[100,56,60,40,40]
 cfg=g1_wbt_terrain_warp_distill
 output=Path(plan['output']);output.mkdir(parents=True,exist_ok=True)
 name='g1-five-skill-depth-student-30m'
 cfg=replace(cfg,training=replace(cfg.training,headless=True,num_envs=80,checkpoint=None,
   teacher_checkpoint=','.join(plan['teachers']),wandb_path=None,
   registry_name=','.join(Path(x).resolve().as_uri() for x in plan['datasets']),
   preprocess_hook_kwargs=json.dumps({'add_onpath_obstacle':True,'num_variants':3,'obstacle_seed':42}),
   project='HuD04_WBT',name=name),
  logger=replace(cfg.logger,mode='offline',base_dir=str(output),name=name,
   entity='zhul136-university-of-science-and-technology-of-china',project='HuD04_WBT',video=replace(cfg.logger.video,enabled=False)),
  algo=replace(cfg.algo,_target_='flux.five_distill_algo.FiveSkillDistillation',config=replace(cfg.algo.config,
   num_learning_iterations=100000,save_interval=100,num_steps_per_env=24,num_learning_epochs=2,num_mini_batches=8,
   distillation_warmup_steps=5)))
 os.environ['PHP_FLUX_OUTPUT']=str(output)
 import wandb
 original=wandb.init
 def init(*a,**kw):
  config=dict(kw.get('config') or {});config.update(_wb_agent=plan['provenance'],record_role='native_training',
   flux_task_id=plan['task_id'],skills=plan['scope']['skills'],mode='masked_pure_dagger',teachers_frozen=True,student_count=1)
  kw['config']=config;return original(*a,**kw)
 wandb.init=init
 cfg=use_php_isaacsim(cfg);sys.argv=[sys.argv[0]]
 print(json.dumps({'event':'five_distill_simulator_initializing','num_envs':80,'motion_counts':counts}),flush=True)
 setup_simulator_imports(cfg);train(cfg)

if __name__=='__main__':main()