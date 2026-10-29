"""Resume the matched full teacher with genuine simulator cleanup."""
import argparse,json,os,sys
from pathlib import Path
from dataclasses import replace
def main():
 p=argparse.ArgumentParser()
 for n in ['payload','output','site','thread-id','turn-id']:p.add_argument('--'+n,required=True)
 a=p.parse_args();payload=Path(a.payload);out=Path(a.output);root=payload/'bundle'
 for path in [payload/'native-source',Path(a.site),root/'runtime/php',root/'runtime/holosoma/src/holosoma',root/'overlay',root,payload/'teacher-code']:sys.path.insert(0,str(path))
 from runtime_versions import check_runtime_versions
 (out/'runtime-versions.json').write_text(json.dumps(check_runtime_versions(),indent=2))
 from portable_config import resolve
 from holosoma.config_types.experiment import ExperimentConfig
 from holosoma.config_types.logger import DisabledLoggerConfig
 raw=json.loads((payload/'teacher-code/config.json').read_text());cfg=ExperimentConfig(**resolve(raw,root,out))
 assert cfg.training.num_envs==4096 and len(cfg.robot.dof_names)==31 and cfg.robot.actions_dim==29
 assert not cfg.training.checkpoint and not cfg.training.teacher_checkpoint and not cfg.training.registry_name
 checkpoint=payload/'teacher-code/continuation-source.pt';assert checkpoint.is_file()
 cfg=replace(cfg,training=replace(cfg.training,checkpoint=str(checkpoint)),simulator=replace(cfg.simulator,_target_='wbt_training.simulators.isaacsim.IsaacSim'),algo=replace(cfg.algo,_target_='teacher_algo.FullTeacherPPO',config=replace(cfg.algo.config,load_optimizer=True)),logger=DisabledLoggerConfig(base_dir=str(out)))
 os.environ['PHP_FLUX_OUTPUT']=str(out);os.environ['PHP_FLUX_STOP_FILE']=str(out/'STOP_AFTER_UPDATE');sys.argv=[sys.argv[0]]
 (out/'provenance.json').write_text(json.dumps({'_wb_agent':{'thread_id':a.thread_id,'turn_id':a.turn_id},'robot':'Hu-D04','fresh_training':False,'teacher_converged':False,'task_id':os.environ['PHP_FLUX_TASK_ID']}))
 from holosoma.utils.sim_utils import setup_simulator_imports
 from holosoma.utils.eval_utils import init_sim_imports
 from holosoma import train_agent
 import close_audit
 report={'close_started':False,'close_returned':False,'train_returned':False,'task_id':os.environ['PHP_FLUX_TASK_ID']}
 setup_simulator_imports(cfg);app=None
 try:
  app=init_sim_imports(cfg);close_audit.configure(app,out,report)
  context=type('ExistingAppContext',(),{'simulation_app':app})()
  train_agent.train(cfg,training_context=context)
  report['train_returned']=True;close_audit.write(out,report)
 finally:
  if app is not None:close_audit.close(app,out,report)
 report['child_returned']=True;close_audit.write(out,report)
if __name__=='__main__':main()