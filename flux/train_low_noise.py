"""One approved sequential two-arm PPO pilot:1GPU,32env,1800s hard worker budget, no retry."""
from __future__ import annotations
import argparse,atexit,dataclasses,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/low_noise_trial'
ENTRYPOINT=Path(__file__).resolve()
SHA='634f343946005e72303b9c100aa90f6a9ff76e8224316a59af1f0842e3edaf85'
ARMS=(('resume_control',1.),('resume_low_noise',.25))
WORKER_SECONDS=1800
ARM_SECONDS=825

def emit(event,**fields):
    print(json.dumps(dict(event=event,**fields),allow_nan=False),flush=True)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    tmp=Path(str(path)+'.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def bootstrap():
    from source_preflight import verify_sources
    from runtime_preflight import verify_runtime
    emit('low_noise_source_verified',**verify_sources(ROOT));emit('low_noise_runtime_verified',**verify_runtime(ROOT))
    from git_smoke_v8 import ARCHIVE_BYTES,ARCHIVE_SHA256,extract_candidates
    from git_parts_v8 import assemble_parts
    archive=OUT/'v8.zip'
    assemble_parts(ROOT/'flux/data/hud04-v8-parts',archive,ARCHIVE_BYTES,ARCHIVE_SHA256)
    dataset=OUT/'candidates';extract_candidates(archive,dataset)
    from offline_dependencies import install
    install(ROOT)
    return dataset

def worker(arm):
    name,scale=next(a for a in ARMS if a[0]==arm)
    out=OUT/name;out.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT/'.flux-deps'))
    from train_hud04 import prepare_runtime
    runtime=prepare_runtime();sys.path[:0]=[str(ROOT/'.flux-deps'),str(ROOT),str(runtime)]
    os.chdir(ROOT)
    import torch
    from holosoma.utils.eval_utils import CheckpointConfig
    from holosoma.config_types.logger import DisabledLoggerConfig
    from wbt_training.utils.checkpoint import load_saved_experiment_config,apply_preprocess_hook
    from holosoma.utils.helpers import get_class
    import holosoma.utils.sim_utils as sim_utils
    from startup_diagnostics import bounded_launcher
    from low_noise_tools import EnvironmentStats,evaluate,set_initial_noise,train
    checkpoint=ROOT/'flux/replay-input/model_00790.pt'
    if digest(checkpoint)!=SHA:raise ValueError('Input checkpoint mismatch')
    saved,_=load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    if (saved.algo.config.num_steps_per_env,saved.algo.config.num_learning_epochs,
        saved.algo.config.num_mini_batches,saved.algo.config.entropy_coef)!=(24,5,4,.005):
        raise ValueError('Original PPO configuration differs')
    if saved.algo.config.use_symmetry or saved.reward.only_positive_rewards:
        raise ValueError('Telemetry assumes native unaugmented reward path')
    dataset=OUT/'candidates'
    cfg=dataclasses.replace(saved,
        training=dataclasses.replace(saved.training,num_envs=32,headless=True,seed=1401,
            export_onnx=False,registry_name=dataset.resolve().as_uri()),
        algo=dataclasses.replace(saved.algo,config=dataclasses.replace(saved.algo.config,
            load_optimizer=True,num_learning_iterations=200,save_interval=10000000)),
        logger=DisabledLoggerConfig(base_dir=str(out),headless_recording=False))
    cfg=apply_preprocess_hook(cfg);cfg.save_config(str(out/'holosoma_config.yaml'))
    sys.argv=[sys.argv[0]]
    sim_utils.setup_isaaclab_launcher=bounded_launcher(sim_utils.setup_isaaclab_launcher,limit_seconds=300)
    env,device,app=sim_utils.setup_simulation_environment(cfg)
    atexit.register(sim_utils.close_simulation_app,app)
    if device!='cuda:0' or env.num_envs!=32:raise ValueError('Compute allocation mismatch')
    native=list(env.simulator._robot.joint_names);ids=list(env.simulator.dof_ids)
    if len(ids)!=29 or len(set(ids))!=29 or [native[i] for i in ids]!=list(cfg.robot.dof_names):
        raise ValueError('Runtime joint mapping mismatch')
    algo=get_class(cfg.algo._target_)(env=env,config=cfg.algo.config,log_dir=str(out),device=device)
    algo.setup();algo.load(str(checkpoint))
    if algo.current_learning_iteration!=790:raise ValueError('Source iteration mismatch')
    if algo.actor_learning_rate!=1e-5 or algo.critic_learning_rate!=1e-5:raise ValueError('Optimizer LR mismatch')
    if algo.empirical_normalization:raise ValueError('Unexpected observation normalization')
    motion=env.command_manager.get_term('motion_command')
    stats=EnvironmentStats(env)
    baseline=evaluate(algo,motion,stats)
    save(out/'evaluation_before.json',baseline)
    emit('low_noise_evaluation_before',arm=name,**baseline)
    # Native restore AGAIN: evaluation must not contaminate checkpoint environment/optimizer state.
    algo.load(str(checkpoint));algo.current_learning_iteration+=1
    intervention=set_initial_noise(algo.actor,scale,completed_rollouts=0)
    save(out/'intervention.json',intervention)
    # Both arms share the training RNG seed; original adaptive frame sampler is restored.
    import random,numpy as np
    torch.manual_seed(1403);np.random.seed(1403);random.seed(1403)
    emit('low_noise_training_begin',arm=name,source_iteration=790,next_iteration=791,
         optimizer_restored=True,initial_actor_lr=algo.actor_learning_rate,
         initial_critic_lr=algo.critic_learning_rate,**intervention)
    trained=train(algo,stats,out,lambda event,**kw:emit(event,arm=name,**kw))
    # Save TRAINING state before evaluation changes the environment.
    final=out/'model_final.pt';algo.save(str(final))
    if not final.is_file() or final.stat().st_size<100000:raise ValueError('No output checkpoint')
    emit('low_noise_checkpoint_saved',arm=name,sha256=digest(final),**trained)
    after=evaluate(algo,motion,stats)
    save(out/'evaluation_after.json',after)
    emit('low_noise_evaluation_after',arm=name,**after)
    before_rm=baseline['restricted_mean_survival_seconds'];after_rm=after['restricted_mean_survival_seconds']
    before_track=baseline['tracking_positive_mean_reward'];after_track=after['tracking_positive_mean_reward']
    result=dict(arm=name,initial_std_multiplier=scale,checkpoint_input_sha256=SHA,
                checkpoint_input_unchanged=digest(checkpoint)==SHA,final_checkpoint_sha256=digest(final),
                final_checkpoint_bytes=final.stat().st_size,before=baseline,after=after,
                initial_noise=intervention,**trained,
                survival_ratio=after_rm/before_rm if before_rm else None,
                pilot_continuation_gate=bool(before_rm>0 and after_rm>=2*before_rm and after_track>=before_track),
                limitation='One-seed short pilot; no convergence or statistical significance claim')
    if not result['checkpoint_input_unchanged']:raise ValueError('Source checkpoint modified')
    save(out/'summary.json',result);emit('low_noise_arm_complete',**result)

def seal(url,headers):
    import urllib.parse,urllib.request,zipfile
    parsed=urllib.parse.urlsplit(url)
    if (parsed.scheme!='https' or parsed.username or parsed.password or parsed.fragment or
        parsed.port not in (None,443) or not (parsed.hostname=='storage.googleapis.com' or
        str(parsed.hostname).endswith('.storage.googleapis.com'))):raise ValueError('Only scoped signed GCS PUT allowed')
    if any(k.lower()!='content-type' and not k.lower().startswith('x-goog-') for k in headers):
        raise ValueError('API credentials forbidden in output upload')
    files=[p for p in OUT.glob('*.json') if p.is_file()]
    for name,_ in ARMS:
        directory=OUT/name
        files.extend(p for p in directory.glob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.pt','.yaml'))
    if any(p.is_symlink() or not p.resolve().is_relative_to(OUT.resolve()) for p in files):raise ValueError('Unsafe output')
    if sum(p.stat().st_size for p in files)>128*1024*1024:raise ValueError('Raw output bound exceeded')
    bundle=ROOT/'outputs/low_noise_trial_bundle.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(files):target.write(path,arcname=str(path.relative_to(OUT)))
    if bundle.stat().st_size>32*1024*1024:raise ValueError('Compressed output bound exceeded')
    with zipfile.ZipFile(bundle) as check:
        if check.testzip() is not None:raise ValueError('Output ZIP integrity failure')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    req=urllib.request.Request(url,data=bundle.read_bytes(),headers=headers,method='PUT')
    with urllib.request.build_opener(NoRedirect()).open(req,timeout=30) as response:
        if response.status not in (200,201):raise RuntimeError('Output upload rejected')
    emit('low_noise_bundle_uploaded',bytes=bundle.stat().st_size,sha256=digest(bundle))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upload-url');p.add_argument('--upload-headers',default='{}')
    p.add_argument('--runtime-child',choices=[a[0] for a in ARMS]);args=p.parse_args()
    if args.runtime_child:
        try:worker(args.runtime_child)
        except BaseException as exc:
            emit('low_noise_arm_failure',arm=args.runtime_child,exception_type=type(exc).__name__)
            raise
        return
    if OUT.exists() or not args.upload_url:raise ValueError('Fresh output and scoped upload required; no restart')
    # Credential-free remote worker: signed output capability only, never platform/GitHub/W&B API keys.
    OUT.mkdir(parents=True);started=time.monotonic();child=None
    def deadline(signum,frame):
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        emit('low_noise_hard_deadline',worker_limit_seconds=WORKER_SECONDS)
        os._exit(124)  # Hard cap: never enter packaging/upload beyond the approved deadline.
    signal.signal(signal.SIGALRM,deadline);signal.alarm(WORKER_SECONDS)
    emit('low_noise_entrypoint_begin',worker_limit_seconds=WORKER_SECONDS,arms=ARMS,
         num_envs=32,max_iterations_per_arm=200,training_seconds_per_arm=300,automatic_retries=0)
    success=False
    try:
        bootstrap()
        from runtime_preflight import resolve_isaac_launcher
        launcher=resolve_isaac_launcher()
        from startup_diagnostics import wait_for_child
        for name,_ in ARMS:
            remaining=WORKER_SECONDS-(time.monotonic()-started)-75
            if remaining<60:raise TimeoutError('Insufficient worker budget for next arm')
            limit=min(ARM_SECONDS,int(remaining))
            child_env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
            for n in ['key','github','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:child_env.pop(n,None)
            child=subprocess.Popen([str(launcher),str(ENTRYPOINT),'--runtime-child',name],
                env=child_env,start_new_session=True)
            try:code=wait_for_child(child,ROOT,limit)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5);code=124
            if code!=0:raise RuntimeError('Arm failed; no retry or replacement task')
            if not (OUT/name/'summary.json').is_file():raise RuntimeError('Arm summary missing')
        records=[json.loads((OUT/n/'summary.json').read_text()) for n,_ in ARMS]
        if not all(r['checkpoint_input_unchanged'] and 1<=r['completed_updates']<=200 for r in records):raise ValueError('Arm bounds')
        # Require baseline pairing evidence; seed equality alone is insufficient.
        if records[0]['before']['initial_motion_frames']!=records[1]['before']['initial_motion_frames']:
            raise ValueError('Evaluation baseline frame mismatch')
        q0,q1=records[0]['before']['initial_joint_positions'],records[1]['before']['initial_joint_positions']
        if max(abs(x-y) for a,b in zip(q0,q1) for x,y in zip(a,b))>1e-5:raise ValueError('Evaluation baseline pose mismatch')
        save(OUT/'summary.json',dict(arms=records,worker_seconds=time.monotonic()-started,
             worker_limit_seconds=WORKER_SECONDS,automatic_retries=0,baseline_pose_and_frames_matched=True))
        success=True
    except BaseException as exc:
        save(OUT/'failure.json',dict(exception_type=type(exc).__name__,worker_seconds=time.monotonic()-started))
        emit('low_noise_failure',exception_type=type(exc).__name__)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        try:seal(args.upload_url,json.loads(args.upload_headers))
        except Exception as exc:
            success=False;emit('low_noise_upload_failure',exception_type=type(exc).__name__)
        signal.alarm(0)
    if not success:raise RuntimeError('Trial incomplete; no auto retry')
    emit('low_noise_entrypoint_complete',worker_limit_seconds=WORKER_SECONDS)

if __name__=='__main__':main()