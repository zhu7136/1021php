"""One approved fresh PPO pilot;1GPU/32env/200updates OR300s;1800sworker/no retry."""
from __future__ import annotations
import argparse,atexit,dataclasses,json,os,signal,subprocess,sys,time
from pathlib import Path
import train_low_noise as base

ROOT=base.ROOT
OUT=ROOT/'outputs/fresh_pilot'
WORKER_SECONDS=1800
MAX_UPDATES=200
TRAIN_SECONDS=300
def emit(event,**fields):base.emit(event.replace('low_noise_','fresh_'),**fields)

class TrackingCapture:
    """Observe the native processed targets and tracking errors before automatic resets."""
    def __init__(self,env,motion,path):
        self.phase='initialization';self.step=0
        self.sink=Path(path).open('w',buffering=1)
        self.original=env._check_termination
        names=list(env.robot_config.dof_names);hip=names.index('left_hip_roll_joint')
        bnames=list(motion.motion_cfg.body_names_to_track);ankle=bnames.index('left_ankle_roll_link')
        import torch
        def check():
            result=self.original()
            if self.phase=='initialization':return result
            terms=[t for t in env.action_manager._term_instances.values() if hasattr(t,'_applied_position_targets')]
            if len(terms)!=1:raise ValueError('Native position-target term missing')
            target=terms[0]._applied_position_targets
            native=env.simulator._robot.data.joint_pos_target[:,env.simulator.dof_ids]
            if not torch.equal(target,native):raise ValueError('Native target readback mismatch')
            limits=env.simulator.hard_dof_pos_limits
            outside=(target<limits[:,0])|(target>limits[:,1])
            error=torch.linalg.vector_norm(motion.body_pos_relative_w[:,ankle]-motion.robot_body_pos_w[:,ankle],dim=-1)
            row=dict(phase=self.phase,control_step=self.step,
                left_hip_target_mean_rad=float(target[:,hip].mean()),left_hip_target_min_rad=float(target[:,hip].min()),
                left_hip_target_max_rad=float(target[:,hip].max()),left_hip_target_outside_fraction=float(outside[:,hip].float().mean()),
                all_target_outside_fraction=float(outside.float().mean()),left_ankle_error_median_m=float(error.median()),
                left_ankle_error_max_m=float(error.max()),left_ankle_exceeded_count=int((error>.5).sum()),
                target_reference_abs_error_mean_rad=float((target-motion.joint_pos).abs().mean()),
                done_count=int(env.reset_buf.bool().sum()))
            self.sink.write(json.dumps(row,allow_nan=False)+'\n');self.step+=1
            return result
        env._check_termination=check
    def set_phase(self,phase):self.phase=phase;self.step=0
    def close(self):self.sink.close()

def worker(run_path):
    out=OUT/'fresh';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT/'.flux-deps'))
    from train_hud04 import prepare_runtime
    runtime=prepare_runtime();sys.path[:0]=[str(ROOT/'.flux-deps'),str(ROOT),str(runtime)];os.chdir(ROOT)
    import torch,numpy as np,random
    from holosoma.config_types.logger import DisabledLoggerConfig
    from wbt_training.config_values.hu_d04.experiment import hu_d04_wbt_terrain_ref_noscale as preset
    from wbt_training.utils.checkpoint import apply_preprocess_hook
    from holosoma.utils.helpers import get_class
    import holosoma.utils.sim_utils as sim_utils
    from startup_diagnostics import bounded_launcher
    from low_noise_tools import EnvironmentStats,evaluate,train
    from checkpoint_contract import attach_metadata,roundtrip
    cfg=dataclasses.replace(preset,
        training=dataclasses.replace(preset.training,num_envs=32,headless=True,seed=1401,
            checkpoint=None,teacher_checkpoint=None,wandb_path=None,export_onnx=False,
            registry_name=(OUT/'candidates').resolve().as_uri(),
            preprocess_hook_kwargs=json.dumps(dict(add_onpath_obstacle=True,num_variants=32))),
        algo=dataclasses.replace(preset.algo,config=dataclasses.replace(preset.algo.config,
            load_optimizer=True,num_learning_iterations=MAX_UPDATES,save_interval=10000000)),
        logger=DisabledLoggerConfig(base_dir=str(out),headless_recording=False))
    torch.manual_seed(1401);torch.cuda.manual_seed_all(1401);np.random.seed(1401);random.seed(1401)
    cfg=apply_preprocess_hook(cfg);cfg.save_config(str(out/'holosoma_config.yaml'))
    if cfg.training.checkpoint is not None or cfg.training.teacher_checkpoint is not None:raise ValueError('Fresh-only trial')
    sys.argv=[sys.argv[0]]
    sim_utils.setup_isaaclab_launcher=bounded_launcher(sim_utils.setup_isaaclab_launcher,limit_seconds=300)
    env,device,app=sim_utils.setup_simulation_environment(cfg);atexit.register(sim_utils.close_simulation_app,app)
    if device!='cuda:0' or env.num_envs!=32:raise ValueError('Allocation mismatch')
    ids=list(env.simulator.dof_ids);native=list(env.simulator._robot.joint_names)
    if len(set(ids))!=29 or [native[i] for i in ids]!=list(cfg.robot.dof_names):raise ValueError('Joint mapping mismatch')
    algo=get_class(cfg.algo._target_)(env=env,config=cfg.algo.config,log_dir=str(out),device=device)
    algo.setup();attach_metadata(algo,cfg,run_path)
    if algo.current_learning_iteration!=0 or algo.actor_optimizer.state or algo.critic_optimizer.state:
        raise ValueError('Fresh initialization unexpectedly contains training state')
    preflight=roundtrip(algo,out/'model_preflight.pt')
    base.save(out/'save_reload_preflight.json',preflight);emit('fresh_save_reload_preflight_pass',**preflight)
    motion=base.resolve_motion_command(env);stats=EnvironmentStats(env)
    capture=TrackingCapture(env,motion,out/'tracking.jsonl');atexit.register(capture.close)
    capture.set_phase('evaluation_before');before=evaluate(algo,motion,stats)
    base.save(out/'evaluation_before.json',before);emit('fresh_evaluation_before',**before)
    # Restore only THIS trial's random initialization, never any previous task checkpoint.
    algo.load(str(out/'model_preflight.pt'))
    # Iteration zero denotes zero completed updates in our fresh preflight file.
    # Start native logging at one; this also avoids PPO's unconditional export at it==0.
    algo.current_learning_iteration=1
    torch.manual_seed(1403);torch.cuda.manual_seed_all(1403);np.random.seed(1403);random.seed(1403)
    capture.set_phase('training')
    emit('fresh_training_begin',from_scratch=True,source_checkpoint=None,initial_iteration=1,completed_updates_before=0,
        actor_lr=algo.actor_learning_rate,critic_lr=algo.critic_learning_rate,
        initial_std_mean=float(algo.actor.std.mean()),max_updates=MAX_UPDATES,training_seconds_cap=TRAIN_SECONDS)
    def training_deadline(signum,frame):raise TimeoutError('Strict 300-second training deadline; no partial-update final save')
    signal.signal(signal.SIGALRM,training_deadline);signal.setitimer(signal.ITIMER_REAL,TRAIN_SECONDS)
    try:trained=train(algo,stats,out,emit)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    capture.set_phase('initialization')
    final=roundtrip(algo,out/'model_final.pt')
    base.save(out/'final_save_reload.json',final);emit('fresh_checkpoint_saved',**final)
    capture.set_phase('evaluation_after');after=evaluate(algo,motion,stats)
    base.save(out/'evaluation_after.json',after);emit('fresh_evaluation_after',**after)
    capture.close()
    result=dict(from_scratch=True,source_checkpoint=None,preflight=preflight,final_checkpoint=final,
        before=before,after=after,**trained,automatic_retries=0,
        limitation='One short plumbing/early-behavior check, not a convergence or success-rate estimate')
    if not 1<=trained['completed_updates']<=MAX_UPDATES:raise ValueError('Update bounds violated')
    base.save(out/'summary.json',result);base.save(OUT/'summary.json',result);emit('fresh_pilot_complete',**result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upload-url');p.add_argument('--upload-headers',default='{}')
    p.add_argument('--wandb-run-path',required=True);p.add_argument('--runtime-child',action='store_true');args=p.parse_args()
    if args.runtime_child:worker(args.wandb_run_path);return
    if OUT.exists() or not args.upload_url:raise ValueError('Fresh output and scoped upload required; no retry')
    OUT.mkdir(parents=True);started=time.monotonic();child=None
    base.OUT=OUT;base.ARMS=(('fresh',1.),);base.BUNDLE_NAME='fresh_pilot_bundle.zip'
    def deadline(signum,frame):
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        os._exit(124)
    signal.signal(signal.SIGALRM,deadline);signal.alarm(WORKER_SECONDS)
    emit('fresh_entrypoint_begin',from_scratch=True,num_envs=32,max_updates=MAX_UPDATES,
         training_seconds_cap=TRAIN_SECONDS,worker_seconds_cap=WORKER_SECONDS,automatic_retries=0)
    success=False
    try:
        base.bootstrap()
        from runtime_preflight import resolve_isaac_launcher
        from startup_diagnostics import wait_for_child
        remaining=int(WORKER_SECONDS-(time.monotonic()-started)-75)
        if remaining<60:raise TimeoutError('Insufficient worker budget')
        env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
        for n in ['key','github','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:env.pop(n,None)
        child=subprocess.Popen([str(resolve_isaac_launcher()),str(Path(__file__).resolve()),
            '--runtime-child','--wandb-run-path',args.wandb_run_path],env=env,start_new_session=True)
        try:code=wait_for_child(child,ROOT,remaining)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5);code=124
        if code!=0 or not (OUT/'summary.json').is_file():raise RuntimeError('Pilot failed; no replacement or retry')
        success=True
    except BaseException as exc:
        base.save(OUT/'failure.json',dict(exception_type=type(exc).__name__,worker_seconds=time.monotonic()-started))
        emit('fresh_failure',exception_type=type(exc).__name__)
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        try:base.seal(args.upload_url,json.loads(args.upload_headers))
        except Exception as exc:success=False;emit('fresh_upload_failure',exception_type=type(exc).__name__)
        signal.alarm(0)
    if not success:raise RuntimeError('Fresh pilot incomplete; no retry')
    emit('fresh_entrypoint_complete',worker_seconds=time.monotonic()-started)

if __name__=='__main__':main()