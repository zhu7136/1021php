"""Hu-D04 4096-environment native continuation with real lifecycle markers."""
import argparse
from dataclasses import replace
import json,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for name in ['payload','output','site']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--thread-id',required=True);p.add_argument('--turn-id',required=True)
    a=p.parse_args();root=a.payload/'bundle'
    for path in [a.payload/'native-source',a.site,root/'runtime/php',root/'runtime/holosoma/src/holosoma',root/'overlay',root,a.payload/'trial-code']:
        sys.path.insert(0,str(path))
    from runtime_versions import check_runtime_versions
    (a.output/'runtime-versions.json').write_text(json.dumps(check_runtime_versions(),indent=2))
    from portable_config import resolve
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.config_types.logger import DisabledLoggerConfig
    raw=json.loads((a.payload/'trial-code/config.json').read_text())
    assert raw['training']['num_envs']==128 and raw['algo']['config']['num_learning_iterations']==2
    assert raw['randomization']['setup_terms']['setup_dof_pos_bias']['params']['enabled'] is False
    raw['training'].update(num_envs=4096,name='hud04-formal-resume-4096env-30min')
    raw['algo']['config'].update(num_learning_iterations=100000,save_interval=100)
    cfg=ExperimentConfig(**resolve(raw,root,a.output))
    assert len(cfg.robot.dof_names)==31 and cfg.robot.actions_dim==29
    assert cfg.training.num_envs==4096 and cfg.algo.config.num_learning_iterations==100000
    assert not cfg.training.checkpoint and not cfg.training.teacher_checkpoint and not cfg.training.registry_name
    cfg=replace(cfg,simulator=replace(cfg.simulator,_target_='wbt_training.simulators.isaacsim.IsaacSim'),
        algo=replace(cfg.algo,_target_='formal_resume.ResumePPO'),logger=DisabledLoggerConfig(base_dir=str(a.output)))
    cfg=replace(cfg,training=replace(cfg.training,checkpoint=str(a.payload/'formal-resume/source-checkpoint.pt')))
    os.environ['PHP_FLUX_OUTPUT']=str(a.output);os.environ['PHP_FLUX_STOP_FILE']=str(a.output/'STOP_AFTER_UPDATE')
    sys.argv=[sys.argv[0]]
    (a.output/'provenance.json').write_text(json.dumps({'_wb_agent':{'thread_id':a.thread_id,'turn_id':a.turn_id},
        'robot':'Hu-D04','source_run':'ot7i239c','teacher_converged':False,'fresh_training':False,'num_envs':4096,'worker_seconds_max':1800}))
    from holosoma.utils.sim_utils import setup_simulator_imports
    from holosoma import train_agent
    original_close=train_agent.close_simulation_app
    def observed_close(app):
        (a.output/'simulation-close-started.json').write_text(json.dumps({'simulation_close_started':True}))
        result=original_close(app)
        (a.output/'simulation-close-returned.json').write_text(json.dumps({'simulation_close_returned':True}))
        return result
    train_agent.close_simulation_app=observed_close
    setup_simulator_imports(cfg)
    train_agent.train(cfg)
    (a.output/'train-returned.json').write_text(json.dumps({'train_returned':True}))

if __name__=='__main__':main()