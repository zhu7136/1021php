"""Native resumed PPO child, genuine close observed; never mask a nonzero exit."""
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
 cfg=ExperimentConfig(**resolve(json.loads((payload/'teacher-code/config.json').read_text()),root,out))
 assert cfg.training.num_envs==4096 and len(cfg.robot.dof_names)==31 and cfg.robot.actions_dim==29
 assert cfg.training.checkpoint=='/workspace/resume-input/model_source.pt' and cfg.algo.config.load_optimizer
 assert not cfg.training.teacher_checkpoint and not cfg.training.registry_name and cfg.algo.config.num_learning_iterations==100000
 cp=Path(os.environ['PHP_RESUME_CHECKPOINT']);assert cp.is_file() and not cp.is_symlink()
 cfg=replace(cfg,training=replace(cfg.training,checkpoint=str(cp)),simulator=replace(cfg.simulator,_target_='wbt_training.simulators.isaacsim.IsaacSim'),algo=replace(cfg.algo,_target_='teacher_algo.FullTeacherPPO'),logger=DisabledLoggerConfig(base_dir=str(out)))
 os.environ['PHP_FLUX_OUTPUT']=str(out);os.environ['PHP_FLUX_STOP_FILE']=str(out/'STOP_AFTER_UPDATE');sys.argv=[sys.argv[0]]
 (out/'provenance.json').write_text(json.dumps(dict(_wb_agent=dict(thread_id=a.thread_id,turn_id=a.turn_id),robot='Hu-D04',fresh_training=False,source_task='TASK_20261123_410',source_completed_updates=200,teacher_converged=False,task_id=os.environ['PHP_FLUX_TASK_ID'])))
 from holosoma.utils.sim_utils import setup_simulator_imports
 from holosoma.utils.eval_utils import init_sim_imports
 from holosoma import train_agent
 import close_audit
 report=dict(close_started=False,close_returned=False,train_returned=False,task_id=os.environ['PHP_FLUX_TASK_ID'])
 setup_simulator_imports(cfg);app=None
 try:
  app=init_sim_imports(cfg);close_audit.configure(app,out,report)
  train_agent.train(cfg,training_context=type('ExistingAppContext',(),{'simulation_app':app})())
  report['train_returned']=True;close_audit.write(out,report)
 finally:
  if app is not None:close_audit.close(app,out,report)
 report['child_returned']=True;close_audit.write(out,report)
if __name__=='__main__':main()