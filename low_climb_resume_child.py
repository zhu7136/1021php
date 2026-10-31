"""Original-input low-climb-76 continuation child. Bounded parent required."""
import argparse,faulthandler,hashlib,json,os,sys
from pathlib import Path
from dataclasses import replace
SOURCE_SHA='9a0b61cdb689886c1171f77eb725ff85f4910573d7fa6376fdf4594415e45711'

def main():
    # Capture native shutdown crashes; never force exit or suppress their status.
    faulthandler.enable(all_threads=True)
    p=argparse.ArgumentParser()
    for name in ('payload','output','site','thread-id','turn-id','source-checkpoint'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();payload=Path(a.payload);out=Path(a.output);root=payload/'bundle';source=Path(a.source_checkpoint).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:raise ValueError('Checkpoint integrity failure')
    if not os.environ.get('PHP_FLUX_TASK_ID') or not os.environ.get('PHP_FLUX_STOP_FILE'):
        raise ValueError('Bounded task parent required')
    out.mkdir(parents=True,exist_ok=True)
    for path in (payload/'native-source',Path(a.site),root/'runtime/php',root/'runtime/holosoma/src/holosoma',root/'overlay',root,payload/'teacher-code',Path(__file__).resolve().parent):
        sys.path.insert(0,str(path))
    from runtime_versions import check_runtime_versions
    (out/'runtime-versions.json').write_text(json.dumps(check_runtime_versions(),indent=2))
    from portable_config import resolve
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.config_types.logger import DisabledLoggerConfig
    raw=json.loads((payload/'teacher-code/config.json').read_text())
    cfg=ExperimentConfig(**resolve(raw,root,out))
    if cfg.training.checkpoint or cfg.training.teacher_checkpoint or cfg.training.registry_name:
        raise ValueError('Unexpected inherited checkpoint configuration')
    if cfg.training.num_envs!=4096 or len(cfg.robot.dof_names)!=31 or cfg.robot.actions_dim!=29:
        raise ValueError('Original training scope changed')
    if not cfg.algo.config.load_optimizer or cfg.algo.config.num_learning_iterations!=100000:
        raise ValueError('Optimizer/update policy changed')
    cfg=replace(cfg,training=replace(cfg.training,checkpoint=str(source)),simulator=replace(cfg.simulator,_target_='wbt_training.simulators.isaacsim.IsaacSim'),algo=replace(cfg.algo,_target_='low_climb_resume_guard.LowClimbResumePPO'),logger=DisabledLoggerConfig(base_dir=str(out)))
    os.environ['PHP_FLUX_OUTPUT']=str(out)
    os.environ['PHP_LOW_STEP_SOURCE_CHECKPOINT']=str(source)
    sys.argv=[sys.argv[0]]
    (out/'provenance.json').write_text(json.dumps({'_wb_agent':{'thread_id':a.thread_id,'turn_id':a.turn_id},'robot':'Hu-D04','skill':'low-climb','dataset':'motions-low-climb-76','fresh_training':False,'source_run':'puljx4ho','source_checkpoint_sha256':SOURCE_SHA,'source_iteration':660,'first_new_iteration':661,'task_id':os.environ['PHP_FLUX_TASK_ID'],'teacher_converged':False}))
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