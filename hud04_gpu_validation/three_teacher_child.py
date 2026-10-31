"""One native Hu-D04 DAgger+PPO validation child, invoked by bounded parent."""
import argparse,faulthandler,json,os,sys,time
from pathlib import Path

def main():
    faulthandler.enable(all_threads=True)
    p=argparse.ArgumentParser()
    for name in ('payload','data-root','output','site','thread-id','turn-id','train-deadline'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();payload=Path(a.payload).resolve();data=Path(a.data_root).resolve();out=Path(a.output).resolve();bundle=payload/'bundle'
    if not os.environ.get('HUD04_FLUX_TASK_ID') or float(a.train_deadline)<=time.monotonic():
        raise ValueError('Live bounded task parent required')
    # The overlay and pinned runtime share a nested regular package. Repair
    # its package path before either parent package can enter sys.modules.
    if any(n == 'wbt_training' or n.startswith('wbt_training.') for n in sys.modules):
        raise ValueError('PHP package imported before overlay repair')
    from overlay_package_fix import repair_overlay
    (out/'overlay-package-repair.json').write_text(json.dumps(repair_overlay(bundle),indent=2))
    checkout=Path(__file__).resolve().parents[1]
    for path in (payload/'native-source',Path(a.site),bundle/'runtime/php',bundle/'runtime/holosoma/src/holosoma',bundle/'overlay',bundle,checkout/'hud04_three_teacher',Path(__file__).resolve().parent):
        sys.path.insert(0,str(path))
    from runtime_versions import check_runtime_versions
    (out/'runtime-versions.json').write_text(json.dumps(check_runtime_versions(),indent=2))
    from three_adapter.entry import verify_data
    from three_adapter.ppo_config import build_ppo_smoke_dict
    manifest=verify_data(data)
    import numpy as np
    with np.load(data/'terrain_mesh.npz',allow_pickle=False) as mesh:
        raw=build_ppo_smoke_dict(json.loads((data/'locomotion-teacher-config.json').read_text()),
            bundle=bundle,smoke=data,teachers=[data/'teachers'/f'{i}.pt' for i in range(3)],
            output=out,mesh_vertices=mesh['vertices'].tolist(),mesh_faces=mesh['faces'].tolist())
    raw['command']['setup_terms']['motion_command']['params']['motion_config']['motion_file']=str(data/'train-motion.npz')
    from holosoma.config_types.algo import DistillationPPOAlgoConfig,DistillationPPOConfig
    from holosoma.config_types.experiment import ExperimentConfig
    raw['algo']=DistillationPPOAlgoConfig(_target_='three_adapter.native_ppo.HuD04ThreeTeacherPPO',
        _recursive_=False,config=DistillationPPOConfig(**raw['algo']['config']))
    cfg=ExperimentConfig(**raw)
    if cfg.training.num_envs!=6 or cfg.algo.config.num_learning_iterations!=5 or cfg.training.checkpoint or cfg.training.multigpu:
        raise ValueError('Authorized six-env/five-update/fresh/single-GPU scope changed')
    os.environ.update(HUD04_THREE_GPU_EXECUTION_APPROVED='1',HUD04_TRAIN_DEADLINE=a.train_deadline,
        HUD04_SMOKE_OUTPUT=str(out),HUD04_ASSET_ROOT=str(bundle/'asset'),
        HUD04_ARIA_THREAD_ID=a.thread_id,HUD04_ARIA_TURN_ID=a.turn_id)
    sys.argv=[sys.argv[0]]
    (out/'provenance.json').write_text(json.dumps(dict(task_id=os.environ['HUD04_FLUX_TASK_ID'],
        _wb_agent=dict(thread_id=a.thread_id,turn_id=a.turn_id),algorithm='dagger_ppo',
        skills=manifest['skills'],num_envs=6,max_updates=5,training_seconds_max=600,
        teacher_iterations=[1361,1272,1373],fresh_student=True,skill_success_verified=False),indent=2))
    from holosoma.utils.sim_utils import setup_simulator_imports
    from holosoma.utils.eval_utils import init_sim_imports
    from holosoma import train_agent
    import close_audit
    report=dict(task_id=os.environ['HUD04_FLUX_TASK_ID'],close_started=False,close_returned=False,train_returned=False)
    setup_simulator_imports(cfg);app=None
    try:
        app=init_sim_imports(cfg);close_audit.configure(app,out,report)
        context=type('ExistingAppContext',(),{'simulation_app':app})()
        print(json.dumps(dict(event='hud04_native_validation_started',num_envs=6,max_updates=5)),flush=True)
        train_agent.train(cfg,training_context=context)
        report['train_returned']=True;close_audit.write(out,report)
    except BaseException as exc:
        # Audit SystemExit as well as ordinary exceptions; never convert a
        # failed learner/teardown into a successful native process exit.
        report['train_exception_type']=type(exc).__name__
        if isinstance(exc,SystemExit):
            report['train_system_exit_code']=exc.code if isinstance(exc.code,(int,str,type(None))) else 'non_scalar'
        close_audit.write(out,report)
        raise
    finally:
        if app is not None:close_audit.close(app,out,report)
    report['child_returned']=True;close_audit.write(out,report)

if __name__=='__main__':main()