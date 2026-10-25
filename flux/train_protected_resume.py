"""One approved continuation: 1GPU/32env, <=200 new updates OR300s,1800s worker,no retry."""
from __future__ import annotations
import argparse,atexit,dataclasses,json,os,signal,subprocess,sys,time
from pathlib import Path
import train_low_noise as base
import evaluate_protected_multiseed as frozen

ROOT=base.ROOT
OUT=ROOT/'outputs/protected_resume'
WORKER_SECONDS=1800
MAX_UPDATES=200
TRAIN_SECONDS=300
SOURCE_ITERATION=200
SOURCE_SHA=frozen.INPUT_HASHES['model_final.pt']
SEEDS=(1402,2402,3402)
ARMS=(('training',0.),)+tuple((f'{label}-seed{seed}',0.) for seed in SEEDS for label in ('before','after'))

def emit(event,**fields):base.emit(event,**fields)

def equal_state(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,type(a)) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b

def verify_source_restore(algo,checkpoint):
    """Native full load must preserve weights, populated moments, iteration and current LRs."""
    import torch
    if base.digest(checkpoint)!=SOURCE_SHA:raise ValueError('Source checkpoint hash mismatch')
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if data['iter']!=SOURCE_ITERATION or algo.current_learning_iteration!=SOURCE_ITERATION:
        raise ValueError('Source iteration mismatch')
    for module,key in [(algo.actor,'actor_model_state_dict'),(algo.critic,'critic_model_state_dict'),
                       (algo.actor_optimizer,'actor_optimizer_state_dict'),(algo.critic_optimizer,'critic_optimizer_state_dict')]:
        if not equal_state(module.state_dict(),data[key]):raise ValueError('Native load differs: '+key)
    for opt,lr in [(algo.actor_optimizer,algo.actor_learning_rate),(algo.critic_optimizer,algo.critic_learning_rate)]:
        if not opt.state or any(g['lr']!=lr for g in opt.param_groups):raise ValueError('Optimizer moments/LR missing')
    # TASK097 has normalization disabled and no simulator/RNG snapshot; do not imply exact trajectory resume.
    if algo.empirical_normalization or data.get('actor_obs_normalizer_state_dict') is not None or data.get('critic_obs_normalizer_state_dict') is not None:
        raise ValueError('Unexpected normalization contract')
    return dict(source_iteration=SOURCE_ITERATION,source_sha256=SOURCE_SHA,
        actor_critic_optimizer_exact=True,actor_lr=algo.actor_learning_rate,critic_lr=algo.critic_learning_rate,
        observation_normalization_enabled=False,simulator_state_resumed=False,rng_state_resumed=False)

def setup(arm,run_path):
    sys.path.insert(0,str(ROOT/'.flux-deps'))
    from train_hud04 import prepare_runtime
    runtime=prepare_runtime();sys.path[:0]=[str(ROOT/'.flux-deps'),str(ROOT),str(runtime)];os.chdir(ROOT)
    import torch,numpy as np,random
    from holosoma.utils.eval_utils import CheckpointConfig
    from holosoma.config_types.logger import DisabledLoggerConfig
    from wbt_training.utils.checkpoint import load_saved_experiment_config,apply_preprocess_hook
    from holosoma.utils.helpers import get_class
    import holosoma.utils.sim_utils as sim_utils
    from startup_diagnostics import bounded_launcher
    from environment_tensor_boundary import install_environment_tensor_boundary
    source=OUT/'inputs/model_final.pt'
    if base.digest(source)!=SOURCE_SHA:raise ValueError('Source checkpoint changed')
    saved,_=load_saved_experiment_config(CheckpointConfig(checkpoint=str(source)))
    cfg=dataclasses.replace(saved,
        training=dataclasses.replace(saved.training,num_envs=32,headless=True,seed=1401,
            checkpoint=None,teacher_checkpoint=None,wandb_path=None,export_onnx=False,
            registry_name=(OUT/'candidates').resolve().as_uri(),
            preprocess_hook_kwargs=json.dumps(dict(add_onpath_obstacle=True,num_variants=32))),
        algo=dataclasses.replace(saved.algo,config=dataclasses.replace(saved.algo.config,
            load_optimizer=(arm=='training'),num_learning_iterations=MAX_UPDATES if arm=='training' else 0,
            save_interval=10000000)),
        logger=DisabledLoggerConfig(base_dir=str(OUT/arm),headless_recording=False))
    # No protected_config reapplication: keep the saved reward, action, termination, PPO and physical recipe.
    for name in ['reward','action','termination','robot','randomization','command']:
        if getattr(cfg,name)!=getattr(saved,name):raise ValueError('Recipe changed: '+name)
    torch.manual_seed(1401);torch.cuda.manual_seed_all(1401);np.random.seed(1401);random.seed(1401)
    cfg=apply_preprocess_hook(cfg);cfg.save_config(str(OUT/arm/'holosoma_config.yaml'))
    sys.argv=[sys.argv[0]]
    sim_utils.setup_isaaclab_launcher=bounded_launcher(sim_utils.setup_isaaclab_launcher,limit_seconds=300)
    env,device,app=sim_utils.setup_simulation_environment(cfg);atexit.register(sim_utils.close_simulation_app,app)
    if device!='cuda:0' or env.num_envs!=32:raise ValueError('Allocation mismatch')
    ids=list(env.simulator.dof_ids);native=list(env.simulator._robot.joint_names)
    if len(set(ids))!=29 or [native[i] for i in ids]!=list(cfg.robot.dof_names):raise ValueError('Joint mapping mismatch')
    emit('resume_tensor_boundary',arm=arm,**install_environment_tensor_boundary(env))
    algo=get_class(cfg.algo._target_)(env=env,config=cfg.algo.config,log_dir=str(OUT/arm),device=device)
    algo.setup();algo.attach_checkpoint_metadata(cfg,run_path)
    return algo,base.resolve_motion_command(env)

def train_worker(run_path):
    from runtime_import_boundary import activate
    emit('resume_import_boundary_verified',arm='training',**activate(ROOT))
    import torch,numpy as np,random
    from low_noise_tools import EnvironmentStats,train
    from train_fresh_pilot import TrackingCapture
    from checkpoint_contract import roundtrip
    out=OUT/'training';out.mkdir(exist_ok=False)
    algo,motion=setup('training',run_path)
    source=OUT/'inputs/model_final.pt';algo.load(str(source))
    restored=verify_source_restore(algo,source)
    base.save(out/'source_restore.json',restored);emit('resume_source_restore_verified',**restored)
    preflight=roundtrip(algo,OUT/'inputs/resume_preflight.pt')
    base.save(out/'save_reload_preflight.json',preflight)
    # Native learn starts inclusively at current_learning_iteration; source iter200 means 200 COMPLETED updates.
    algo.current_learning_iteration=SOURCE_ITERATION+1
    torch.manual_seed(1403);torch.cuda.manual_seed_all(1403);np.random.seed(1403);random.seed(1403)
    stats=EnvironmentStats(algo.env);capture=TrackingCapture(algo.env,motion,out/'tracking.jsonl',actor=algo.actor)
    capture.set_phase('training');atexit.register(capture.close)
    emit('resume_training_begin',initial_iteration=201,completed_updates_before=200,max_new_updates=MAX_UPDATES,
        training_seconds_cap=TRAIN_SECONDS,actor_lr=algo.actor_learning_rate,critic_lr=algo.critic_learning_rate)
    def deadline(signum,frame):raise TimeoutError('Strict 300s training cap; no partial-update final checkpoint')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,TRAIN_SECONDS)
    try:trained=train(algo,stats,out,emit)
    finally:signal.setitimer(signal.ITIMER_REAL,0);capture.close()
    if not 1<=trained['completed_updates']<=MAX_UPDATES:raise ValueError('Update bound violated')
    if algo.current_learning_iteration!=SOURCE_ITERATION+trained['completed_updates']:raise ValueError('Iteration accounting mismatch')
    final=roundtrip(algo,out/'model_final.pt')
    if base.digest(source)!=SOURCE_SHA:raise ValueError('Source modified')
    result=dict(from_scratch=False,source_iteration=200,source_sha256=SOURCE_SHA,restore=restored,
        preflight=preflight,final_checkpoint=final,automatic_retries=0,**trained)
    base.save(out/'summary.json',result);emit('resume_training_complete',**result)

def eval_worker(arm,run_path):
    from runtime_import_boundary import activate
    emit('resume_import_boundary_verified',arm=arm,**activate(ROOT))
    import torch
    from low_noise_tools import EnvironmentStats,evaluate
    from paired_evaluation_tools import FirstEpisodeCapture
    from synchronized_diagnostics import SynchronizedDiagnostics
    label,seed_text=arm.split('-seed');seed=int(seed_text)
    if seed not in SEEDS or label not in ('before','after'):raise ValueError('Undeclared eval arm')
    out=OUT/arm;out.mkdir(exist_ok=False)
    algo,motion=setup(arm,run_path)
    training=json.loads((OUT/'training/summary.json').read_text())
    checkpoint=OUT/('inputs/model_final.pt' if label=='before' else 'training/model_final.pt')
    expected_sha=SOURCE_SHA if label=='before' else training['final_checkpoint']['sha256']
    iteration=200 if label=='before' else training['final_checkpoint']['iteration']
    if base.digest(checkpoint)!=expected_sha:raise ValueError('Eval checkpoint hash differs')
    algo.load(str(checkpoint))
    if algo.current_learning_iteration!=iteration or algo.empirical_normalization:raise ValueError('Eval load mismatch')
    if algo.actor_optimizer.state or algo.critic_optimizer.state:raise ValueError('Eval optimizer loaded')
    def forbid(*a,**kw):raise RuntimeError('Frozen eval cannot update or save')
    algo.learn=forbid;algo.save=forbid;algo.actor_optimizer.step=forbid;algo.critic_optimizer.step=forbid
    weights={k:v.detach().clone() for k,v in algo.actor.state_dict().items()}
    stats=EnvironmentStats(algo.env)
    capture=FirstEpisodeCapture(algo.env,motion,stats,out/'first_episode_tracking.jsonl')
    sync=SynchronizedDiagnostics(algo.env,motion,stats,out)
    try:
        result=evaluate(algo,motion,stats,seed=seed,continuous_segments=True)
        failures=capture.finish(result);result['synchronized_diagnostics']=sync.finish()
    finally:sync.close()
    if not equal_state(weights,algo.actor.state_dict()) or base.digest(checkpoint)!=expected_sha:
        raise ValueError('Frozen evaluation modified weights')
    result.update(source_iteration=iteration,checkpoint_sha256=expected_sha,model_weights_unchanged=True,
        training_updates=0,optimizer_loaded=False,independent_simulator_process=True)
    for name,data in [('initial_state',capture.initial),('first_failures',failures),
                      ('continuous_segments',result['continuous_segments']),('evaluation',result)]:base.save(out/(name+'.json'),data)
    emit('resume_eval_arm_complete',arm=arm,source_iteration=iteration,training_updates=0)

def summarize():
    pairs=[]
    for seed in SEEDS:
        paths={label:OUT/f'{label}-seed{seed}' for label in ('before','after')}
        states={k:json.loads((p/'initial_state.json').read_text()) for k,p in paths.items()}
        if states['before']!=states['after']:raise ValueError('Evaluation initial states not exactly paired')
        ev={k:json.loads((p/'evaluation.json').read_text()) for k,p in paths.items()}
        for key in ['seed','horizon_steps','dt','initial_motion_frames','initial_joint_positions']:
            if ev['before'][key]!=ev['after'][key]:raise ValueError('Pair mismatch: '+key)
        pairs.append(dict(seed=seed,initial_state_exact=True,
            evaluations={k:{field:ev[k][field] for field in ['source_iteration','restricted_mean_survival_seconds',
                'survivors_at_horizon','continuous_segments']} for k in paths}))
    result=json.loads((OUT/'training/summary.json').read_text())
    result.update(pairs=pairs,seeds=list(SEEDS),model_evaluations=6,num_envs=32,
        continuous_segment_accounting=True,initial_state_exact=True,
        limitation='Short continuation and 2.56s evaluations; no convergence/deployment success claim. Internal resets are censored, not successful survival.')
    base.save(OUT/'summary.json',result);emit('resume_pilot_complete',completed_updates=result['completed_updates'],initial_state_exact=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-url');p.add_argument('--upload-url');p.add_argument('--upload-headers',default='{}')
    p.add_argument('--wandb-run-path',required=True);p.add_argument('--runtime-child',choices=list(dict(ARMS)))
    args=p.parse_args()
    if args.runtime_child:
        if args.runtime_child=='training':train_worker(args.wandb_run_path)
        else:eval_worker(args.runtime_child,args.wandb_run_path)
        return
    if OUT.exists() or not args.input_url or not args.upload_url:raise ValueError('Fresh output and scoped capabilities required')
    OUT.mkdir(parents=True);started=time.monotonic();child=None
    base.OUT=OUT;base.ARMS=ARMS;base.BUNDLE_NAME='protected_resume_bundle.zip';frozen.OUT=OUT
    def deadline(signum,frame):
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        os._exit(124)
    signal.signal(signal.SIGALRM,deadline);signal.alarm(WORKER_SECONDS)
    emit('resume_entrypoint_begin',source_iteration=200,num_envs=32,max_new_updates=200,
        training_seconds_cap=300,worker_seconds_cap=1800,automatic_retries=0)
    success=False
    try:
        base.bootstrap();frozen.fetch_inputs(args.input_url)
        from runtime_preflight import resolve_isaac_launcher
        from startup_diagnostics import wait_for_child
        for arm,_ in ARMS:
            remaining=int(WORKER_SECONDS-(time.monotonic()-started)-75)
            if remaining<60:raise TimeoutError('Insufficient worker budget')
            env=dict(os.environ,PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
            for name in ['key','github','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:env.pop(name,None)
            child=subprocess.Popen([str(resolve_isaac_launcher()),str(Path(__file__).resolve()),
                '--runtime-child',arm,'--wandb-run-path',args.wandb_run_path],env=env,start_new_session=True)
            try:code=wait_for_child(child,ROOT,remaining)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5);code=124
            required=OUT/arm/('summary.json' if arm=='training' else 'evaluation.json')
            if code!=0 or not required.is_file():raise RuntimeError('Continuation stage failed; no retry: '+arm)
        summarize();success=True
    except BaseException as exc:
        base.save(OUT/'failure.json',dict(exception_type=type(exc).__name__,worker_seconds=time.monotonic()-started))
        emit('resume_failure',exception_type=type(exc).__name__)
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        try:base.seal(args.upload_url,json.loads(args.upload_headers))
        except Exception as exc:success=False;emit('resume_upload_failure',exception_type=type(exc).__name__)
        signal.alarm(0)
    if not success:raise RuntimeError('Continuation incomplete; no retry')
    emit('resume_entrypoint_complete',worker_seconds=time.monotonic()-started)

if __name__=='__main__':main()