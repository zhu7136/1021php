"""One approved GPU task: 3 paired seeds x 2 frozen models; zero PPO updates."""
from __future__ import annotations
import argparse,atexit,dataclasses,hashlib,json,os,signal,subprocess,sys,time,urllib.parse,urllib.request,zipfile
from pathlib import Path
import train_low_noise as base

ROOT=base.ROOT
OUT=ROOT/'outputs/protected_multiseed_eval'
SEEDS=(1402,2402,3402)
ARMS=tuple((f'{label}-seed{seed}',0.) for seed in SEEDS for label in ('initial','trained'))
WORKER_SECONDS=1800
INPUT_HASHES={'model_preflight.pt':'86e249d10da6a6c48ce0bb867660a3bc1faca5b4c9fd25fdeff305914631237b',
              'model_final.pt':'3dd4286a4b5e825f8e224f476cc5cc510a375c77a9ffabd168f518d946c680f1'}
BUNDLE_SHA='c30ecf9b2d7d8ab50e7864b79a97c56b356beea63f61106d746ee8047d93de69'
BUNDLE_BYTES=13513018

def emit(event,**fields):base.emit(event,**fields)

def extract_inputs(archive,directory):
    if archive.stat().st_size!=BUNDLE_BYTES or base.digest(archive)!=BUNDLE_SHA:
        raise ValueError('Source TASK097 bundle hash/size mismatch')
    directory.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Input ZIP corruption')
        if sum(i.file_size for i in z.infolist())>128*1024*1024:raise ValueError('Input expansion bound')
        for name,sha in INPUT_HASHES.items():
            info=z.getinfo('fresh/'+name)
            if info.file_size>16*1024*1024:raise ValueError('Checkpoint size bound')
            data=z.read(info)
            if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Input model hash mismatch')
            with (directory/name).open('xb') as f:f.write(data)
    return dict(input_bundle_sha256=BUNDLE_SHA,checkpoint_sha256=INPUT_HASHES)

def fetch_inputs(url):
    parsed=urllib.parse.urlsplit(url)
    if (parsed.scheme!='https' or parsed.username or parsed.password or parsed.fragment or
        parsed.port not in (None,443) or not (parsed.hostname=='storage.googleapis.com' or
        str(parsed.hostname).endswith('.storage.googleapis.com'))):raise ValueError('Only scoped GCS GET allowed')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    archive=OUT/'source_bundle.zip'
    with urllib.request.build_opener(NoRedirect()).open(url,timeout=45) as response,archive.open('xb') as sink:
        count=0
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            count+=len(chunk)
            if count>BUNDLE_BYTES:raise ValueError('Input download bound exceeded')
            sink.write(chunk)
    result=extract_inputs(archive,OUT/'inputs');base.save(OUT/'input_verification.json',result)
    emit('paired_eval_inputs_verified',**result)

def worker(arm,run_path,synchronized_diagnostics=False):
    if arm not in dict(ARMS):raise ValueError('Undeclared evaluation arm')
    label,seed_text=arm.split('-seed');seed=int(seed_text)
    out=OUT/arm;out.mkdir(exist_ok=False)
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
    from low_noise_tools import EnvironmentStats,evaluate
    from environment_tensor_boundary import install_environment_tensor_boundary
    from paired_evaluation_tools import FirstEpisodeCapture,INPUTS
    from wbt_training.utils.target_guard import protected_config
    if INPUTS!=INPUT_HASHES:raise ValueError('Input contract drift')
    checkpoint=OUT/'inputs'/('model_preflight.pt' if label=='initial' else 'model_final.pt')
    if base.digest(checkpoint)!=INPUT_HASHES[checkpoint.name]:raise ValueError('Input checkpoint changed')
    # Same physical/environment configuration as TASK097; training is never called.
    cfg=dataclasses.replace(preset,
        training=dataclasses.replace(preset.training,num_envs=32,headless=True,seed=1401,
            checkpoint=None,teacher_checkpoint=None,wandb_path=None,export_onnx=False,
            registry_name=(OUT/'candidates').resolve().as_uri(),
            preprocess_hook_kwargs=json.dumps(dict(add_onpath_obstacle=True,num_variants=32))),
        algo=dataclasses.replace(preset.algo,config=dataclasses.replace(preset.algo.config,
            load_optimizer=False,num_learning_iterations=0,save_interval=10000000)),
        logger=DisabledLoggerConfig(base_dir=str(out),headless_recording=False))
    cfg=protected_config(cfg)
    torch.manual_seed(1401);torch.cuda.manual_seed_all(1401);np.random.seed(1401);random.seed(1401)
    cfg=apply_preprocess_hook(cfg);cfg.save_config(str(out/'holosoma_config.yaml'))
    sys.argv=[sys.argv[0]]
    sim_utils.setup_isaaclab_launcher=bounded_launcher(sim_utils.setup_isaaclab_launcher,limit_seconds=300)
    env,device,app=sim_utils.setup_simulation_environment(cfg);atexit.register(sim_utils.close_simulation_app,app)
    if device!='cuda:0' or env.num_envs!=32:raise ValueError('Allocation mismatch')
    ids=list(env.simulator.dof_ids);native=list(env.simulator._robot.joint_names)
    if len(set(ids))!=29 or [native[i] for i in ids]!=list(cfg.robot.dof_names):raise ValueError('Joint mapping mismatch')
    emit('paired_eval_tensor_boundary',arm=arm,**install_environment_tensor_boundary(env))
    algo=get_class(cfg.algo._target_)(env=env,config=cfg.algo.config,log_dir=str(out),device=device)
    algo.setup();algo.attach_checkpoint_metadata(cfg,run_path)
    algo.load(str(checkpoint))
    expected_iteration=0 if label=='initial' else 200
    if algo.current_learning_iteration!=expected_iteration or algo.empirical_normalization:
        raise ValueError('Source model iteration or normalization differs')
    if algo.actor_optimizer.state or algo.critic_optimizer.state:raise ValueError('Optimizers unexpectedly loaded')
    def forbid(*args,**kwargs):raise RuntimeError('Evaluation-only task cannot train, step optimizer, or save a checkpoint')
    algo.learn=forbid;algo.save=forbid
    algo.actor_optimizer.step=forbid;algo.critic_optimizer.step=forbid
    weights_before={k:v.detach().clone() for k,v in algo.actor.state_dict().items()}
    motion=base.resolve_motion_command(env);stats=EnvironmentStats(env)
    capture=FirstEpisodeCapture(env,motion,stats,out/'first_episode_tracking.jsonl')
    synchronized=None
    if synchronized_diagnostics:
        from synchronized_diagnostics import SynchronizedDiagnostics
        synchronized=SynchronizedDiagnostics(env,motion,stats,out)
    emit('paired_eval_arm_begin',arm=arm,seed=seed,source_iteration=expected_iteration,
         checkpoint_sha256=base.digest(checkpoint),optimizer_loaded=False,training_updates=0)
    try:
        result=evaluate(algo,motion,stats,seed=seed)
        diagnostics=capture.finish(result)
        if synchronized is not None:
            result['synchronized_diagnostics']=synchronized.finish()
    finally:
        if synchronized is not None:synchronized.close()
    for key,value in algo.actor.state_dict().items():
        if not torch.equal(value,weights_before[key]):raise ValueError('Evaluation changed model weights')
    if algo.actor_optimizer.state or algo.critic_optimizer.state:raise ValueError('Unexpected optimizer state')
    if base.digest(checkpoint)!=INPUT_HASHES[checkpoint.name]:raise ValueError('Source model changed')
    result.update(arm=arm,source_iteration=expected_iteration,checkpoint_sha256=INPUT_HASHES[checkpoint.name],
        model_weights_unchanged=True,optimizer_loaded=False,training_updates=0,
        independent_simulator_process=True)
    base.save(out/'initial_state.json',capture.initial)
    base.save(out/'first_failures.json',diagnostics)
    base.save(out/'evaluation.json',result)
    emit('paired_eval_arm_complete',arm=arm,seed=seed,training_updates=0,
        restricted_mean_survival_seconds=result['restricted_mean_survival_seconds'],
        survivors_at_horizon=result['survivors_at_horizon'],first_failures=len(diagnostics['first_failures']))

def summarize():
    # Parent is plain Python; import torch-dependent comparison under the simulator launcher instead.
    pairs=[]
    for seed in SEEDS:
        paths={label:OUT/f'{label}-seed{seed}' for label in ['initial','trained']}
        ev={k:json.loads((p/'evaluation.json').read_text()) for k,p in paths.items()}
        states={k:json.loads((p/'initial_state.json').read_text()) for k,p in paths.items()}
        if states['initial']!=states['trained']:raise ValueError('Initial simulator state is not exactly paired')
        for key in ['seed','horizon_steps','dt','initial_motion_frames','initial_joint_positions']:
            if ev['initial'][key]!=ev['trained'][key]:raise ValueError('Pair contract differs: '+key)
        b=ev['initial']['first_episode_steps'];a=ev['trained']['first_episode_steps']
        if len(a)!=32 or len(b)!=32:raise ValueError('Pair size mismatch')
        pairs.append(dict(seed=seed,initial_state_exact=True,
            before_seconds=ev['initial']['restricted_mean_survival_seconds'],
            after_seconds=ev['trained']['restricted_mean_survival_seconds'],
            improved=sum(x>y for x,y in zip(a,b)),unchanged=sum(x==y for x,y in zip(a,b)),
            worse=sum(x<y for x,y in zip(a,b)),before_survivors=ev['initial']['survivors_at_horizon'],
            after_survivors=ev['trained']['survivors_at_horizon']))
    result=dict(evaluation_only=True,training_updates=0,automatic_retries=0,
        seeds=list(SEEDS),num_envs=32,horizon_steps=128,paired_evaluations=3,model_evaluations=6,
        independent_simulator_per_model_seed=True,initial_state_exact=True,pairs=pairs,
        input_checkpoint_sha256=INPUT_HASHES,source_task='TASK_20261024_097',
        before_mean_seconds=sum(p['before_seconds'] for p in pairs)/3,
        after_mean_seconds=sum(p['after_seconds'] for p in pairs)/3,
        limitation='Short 2.56s horizon; 3 seeds are not convergence or deployment success proof')
    for key in ['improved','unchanged','worse','before_survivors','after_survivors']:
        result[key]=sum(p[key] for p in pairs)
    base.save(OUT/'summary.json',result);emit('paired_eval_complete',**result)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-url');parser.add_argument('--upload-url');parser.add_argument('--upload-headers',default='{}')
    parser.add_argument('--wandb-run-path',required=True);parser.add_argument('--runtime-child',choices=list(dict(ARMS)))
    parser.add_argument('--synchronized-diagnostics',action='store_true',
        help='Opt-in pre-reset control/substep ankle/contact/push telemetry; no training changes')
    args=parser.parse_args()
    if args.runtime_child:worker(args.runtime_child,args.wandb_run_path,args.synchronized_diagnostics);return
    if OUT.exists() or not args.input_url or not args.upload_url:raise ValueError('Fresh output and scoped capabilities required')
    OUT.mkdir(parents=True);started=time.monotonic();child=None
    base.OUT=OUT;base.ARMS=ARMS;base.BUNDLE_NAME='protected_multiseed_eval_bundle.zip'
    def deadline(signum,frame):
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        os._exit(124)
    signal.signal(signal.SIGALRM,deadline);signal.alarm(WORKER_SECONDS)
    emit('paired_eval_entrypoint_begin',seeds=list(SEEDS),num_envs=32,horizon_steps=128,
        model_evaluations=6,training_updates=0,worker_seconds_cap=WORKER_SECONDS,automatic_retries=0)
    success=False
    try:
        base.bootstrap();fetch_inputs(args.input_url)
        from runtime_preflight import resolve_isaac_launcher
        from startup_diagnostics import wait_for_child
        for arm,_ in ARMS:
            remaining=int(WORKER_SECONDS-(time.monotonic()-started)-75)
            if remaining<60:raise TimeoutError('Insufficient approved worker time')
            env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
            for name in ['key','github','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:env.pop(name,None)
            command=[str(resolve_isaac_launcher()),str(Path(__file__).resolve()),
                '--runtime-child',arm,'--wandb-run-path',args.wandb_run_path]
            if args.synchronized_diagnostics:command.append('--synchronized-diagnostics')
            child=subprocess.Popen(command,env=env,start_new_session=True)
            try:code=wait_for_child(child,ROOT,remaining)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5);code=124
            if code!=0 or not (OUT/arm/'evaluation.json').is_file():raise RuntimeError('Evaluation arm failed; no retry')
        summarize();success=True
    except BaseException as exc:
        base.save(OUT/'failure.json',dict(exception_type=type(exc).__name__,worker_seconds=time.monotonic()-started))
        emit('paired_eval_failure',exception_type=type(exc).__name__)
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        try:base.seal(args.upload_url,json.loads(args.upload_headers))
        except Exception as exc:success=False;emit('paired_eval_upload_failure',exception_type=type(exc).__name__)
        signal.alarm(0)
    if not success:raise RuntimeError('Evaluation incomplete; no retry')
    emit('paired_eval_entrypoint_complete',worker_seconds=time.monotonic()-started,training_updates=0)

if __name__=='__main__':main()