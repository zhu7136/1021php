"""Native deterministic full-clip evaluation of task283; no learn/update calls."""
import argparse,copy,json,os,random,time,types
from dataclasses import replace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flux.g1_eval283_protocol import *

def main():
    p=argparse.ArgumentParser()
    for name in ('data','checkpoint','output','approval-file'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--soft-deadline',type=float,required=True)
    a=p.parse_args();scope=validate_scope(json.loads(a.approval_file.read_text()))
    out=a.output;out.mkdir(exist_ok=False)
    if digest(a.checkpoint)!=CHECKPOINT_SHA256:raise ValueError('Wrong evaluation checkpoint')
    from flux.g1_lowstep_input import runtime_checks
    runtime_checks(a.data)
    from wbt_training.utils.checkpoint import load_saved_experiment_config,apply_preprocess_hook
    from holosoma.utils.eval_utils import CheckpointConfig
    from wbt_training.utils.isaacsim_clock import use_php_isaacsim
    from holosoma.utils.sim_utils import setup_simulation_environment
    from wbt_training.utils.registry import bundle_pairs
    import numpy as np
    import torch
    pairs=bundle_pairs(a.data)
    if len(pairs)!=56 or any(t is None for _,t,_ in pairs):raise ValueError('Wrong lowstep pair set')
    lengths=[];names=[]
    for motion,terrain,_ in pairs:
        with np.load(motion,allow_pickle=False) as d:
            if float(d['fps'])!=50:raise ValueError('Unexpected reference frame rate')
            if 'motion_ends' in d and np.count_nonzero(d['motion_ends'])!=1:raise ValueError('Nested original clips unsupported')
            lengths.append(len(d['joint_pos']));names.append(motion.relative_to(a.data).as_posix())
    cfg,_=load_saved_experiment_config(CheckpointConfig(checkpoint=str(a.checkpoint)))
    if cfg.training.num_envs!=4096 or cfg.algo.config.empirical_normalization:raise ValueError('Unexpected source recipe')
    saved_kwargs=json.loads(cfg.training.preprocess_hook_kwargs)
    if saved_kwargs!={'add_onpath_obstacle':True,'num_variants':64}:raise ValueError('Unexpected terrain recipe')
    # Explicit, reproducible regenerated terrain seed, not a claim of the historical training layout.
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    cfg=replace(cfg,training=replace(cfg.training,headless=True,num_envs=56,seed=42,
                checkpoint=str(a.checkpoint),registry_name=a.data.resolve().as_uri(),
                export_onnx=False,preprocess_hook_kwargs=json.dumps({**saved_kwargs,'obstacle_seed':42})),
                logger=replace(cfg.logger,mode='offline',base_dir=str(out),name='g1-task283-frozen-eval',
                video=replace(cfg.logger.video,enabled=True,interval=1000000,save_dir=str(out/'videos'),
                              upload_to_wandb=False,show_command_overlay=False)),
                algo=replace(cfg.algo,_target_='holosoma.agents.ppo.ppo.PPO'))
    cfg=use_php_isaacsim(cfg);cfg=apply_preprocess_hook(cfg)
    # Evaluation-only scheduling: per-clip [start,end), no early native clip-end resampling.
    term=cfg.command.setup_terms['motion_command'];mc=term.params['motion_config']
    overrides=dict(use_adaptive_timesteps_sampler=False,resample_on_motion_end=False,
                   start_at_timestep_zero_prob=0.0,freeze_at_timestep_zero_prob=0.0)
    mc={**mc,**overrides} if isinstance(mc,dict) else replace(mc,**overrides)
    terms=dict(cfg.command.setup_terms);terms['motion_command']=replace(term,params={**term.params,'motion_config':mc})
    cfg=replace(cfg,command=replace(cfg.command,setup_terms=terms),
                terrain=replace(cfg.terrain,terrain_term=replace(cfg.terrain.terrain_term,
                                spawn=replace(cfg.terrain.terrain_term.spawn,randomize_tiles=True))),
                simulator=replace(cfg.simulator,config=replace(cfg.simulator.config,
                    sim=replace(cfg.simulator.config.sim,max_episode_length_s=12.0))))
    cfg.save_config(str(out/'eval_config.yaml'))
    print(json.dumps({'event':'eval283_simulator_initializing','planned_trials':168,'training_updates':0}),flush=True)
    env,device,app=setup_simulation_environment(cfg)
    if str(device)!='cuda:0' or env.num_envs!=56:raise ValueError('Wrong evaluation device/count')
    if env._update_tasks_before_termination:raise ValueError('Require pre-reset termination ordering')
    from holosoma.agents.ppo.ppo import PPO
    from holosoma.managers.command.terms import wbt
    from flux.resume_algo import snapshot,require_loaded_state
    from flux.teacher_algo import _require_same_state
    from holosoma.utils.rotations import quat_error_magnitude
    algo=PPO(device=device,env=env,config=cfg.algo.config,log_dir=str(out),multi_gpu_cfg=None)
    algo.setup();algo.load(str(a.checkpoint))
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    before=snapshot(algo);require_loaded_state(algo,saved,before)
    if algo.current_learning_iteration!=SOURCE_ITERATION:raise ValueError('Wrong source iteration')
    algo._eval_mode();env.set_is_evaluating()
    for module in (algo.actor,algo.critic,algo.actor_obs_normalizer,algo.critic_obs_normalizer):
        module.requires_grad_(False)
    def forbidden(*args,**kwargs):raise RuntimeError('Training is forbidden during evaluation')
    algo.learn=forbidden;algo.actor_optimizer.step=forbidden;algo.critic_optimizer.step=forbidden
    policy=algo.get_inference_policy()
    cmd=env.command_manager.get_state('motion_command')
    base=cmd.motion;ends=base.motion_ends.nonzero(as_tuple=False).flatten()+1
    starts=torch.cat([torch.zeros(1,device=device,dtype=torch.long),ends[:-1]])
    expected=np.tile(lengths,64)
    if len(ends)!=3584 or not np.array_equal((ends-starts).cpu().numpy(),expected):
        raise ValueError('Combined motion boundaries do not match56x64 ordered pairs')
    class SegmentedMotion:
        num_motions=3584
        motion_start_idx=starts
        motion_end_idx=ends
        def __getattr__(self,key):return getattr(base,key)
    cmd.motion=SegmentedMotion()
    namespace=dict(vars(wbt))
    source=Path(wbt.__file__).read_text();exec(compile(scheduled_reset_source(source),'scheduled_native_reset','exec'),namespace)
    cmd.reset=types.MethodType(namespace['reset'],cmd)
    cmd.eval_motion_ids=torch.arange(56,device=device,dtype=torch.long)
    observed={};original=env._check_termination
    def capture():
        original()
        observed['failed']=env.termination_manager.terminated.clone()
        observed['timeout']=env.termination_manager.time_outs.clone()
        observed['frame']=cmd.time_steps.clone()
        observed['ref_pos']=torch.linalg.vector_norm(cmd.ref_pos_w-cmd.robot_ref_pos_w,dim=-1).cpu().numpy()
        observed['body_pos']=torch.linalg.vector_norm(cmd.body_pos_relative_w-cmd.robot_body_pos_w,dim=-1).mean(-1).cpu().numpy()
        observed['ref_ori']=quat_error_magnitude(cmd.ref_quat_w,cmd.robot_ref_quat_w,w_last=True).cpu().numpy()
        # Native failure term is BadTracking; expose its component reasons without changing flags.
        reasons={}
        for name,obj in env.termination_manager._term_instances.items():
            if hasattr(obj,'bad_ref_pos'):
                for label in ('bad_ref_pos','bad_ref_ori','bad_motion_body_pos'):
                    reasons[name+'/'+label]=getattr(obj,label)(cmd).clone().cpu().tolist()
        observed['reasons']=reasons
    env._check_termination=capture
    rows=[{'clip_index':i,'motion':names[i],'seed':seed,'variant_index':j,'status':'not_tested',
           'steps':0,'progress':0.0,'failure_reasons':[]} for j,seed in enumerate(SEEDS) for i in range(56)]
    protocol={'source_run':SOURCE_RUN,'source_sha256':CHECKPOINT_SHA256,'pairs':56,'repeats':3,
              'seeds':list(SEEDS),'terrain_seed':42,'terrain_variants':64,'selected_variants':[0,1,2],
              'seed_scope':'global Python/NumPy/Torch rollout seeds; native stateless DR episode counters retained',
              'new_regenerated_layout_not_historical_training_layout':True,
              'clip_mapping_validated_against_all_original_frame_lengths':True,
              'evaluation_overrides':['56env','explicit clip assignment','disable adaptive sampling',
                  'disable clip-end resampling until exclusive end','fixed terrain seed42','video','base PPO inference only',
                  'episode timeout10s->12s to cover longest10.76s clip; failure thresholds unchanged'],
              'initialization':'native reset_all performs one zero-action simulation step at frame0; included in trial statistics',
              'failure_thresholds_preserved':True,'episode_timeout_seconds':12.0,
              'training_updates':0,'video_scope':'env0, original clip0, each of3repeats'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def save_progress():
        (out/'trials.json').write_text(json.dumps(rows,indent=2))
        summary=summarize(rows)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({'event':'eval283_progress',**summary}),flush=True)
    save_progress()
    for j,seed in enumerate(SEEDS):
        if time.monotonic()>=a.soft_deadline:break
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        cmd.eval_motion_ids=torch.arange(56,device=device,dtype=torch.long)+56*j
        active=np.ones(56,dtype=bool);samples=[{'ref_pos':[],'body_pos':[],'ref_ori':[]} for _ in range(56)]
        block=rows[j*56:(j+1)*56]
        for r in block:r['status']='truncated';r['truncation_reason']='worker_budget'
        recorder=env.simulator.video_recorder
        if recorder is not None:recorder.stop_recording()
        with torch.inference_mode():
            obs=env.reset_all()
            if recorder is not None:recorder.start_recording(j+1)
            for step in range(max(lengths)+2):
                # Consume captured pre-reset outcome of native reset warmup or previous policy step.
                fail=observed['failed'].cpu().numpy();timeout=observed['timeout'].cpu().numpy()
                wrapped=cmd.motion_end_reset.cpu().numpy()
                frames=observed['frame'].cpu().numpy()
                for i in np.flatnonzero(active):
                    r=block[i];r['steps']+=1
                    r['progress']=min(1.0,max(0.0,(int(frames[i])-int(starts[56*j+i])+1)/lengths[i]))
                    for key in samples[i]:samples[i][key].append(float(observed[key][i]))
                    if int(frames[i])<int(starts[56*j+i]) or int(frames[i])>=int(ends[56*j+i]):
                        raise ValueError('Active clip escaped its scheduled boundary')
                    status=classify(bool(fail[i]),bool(timeout[i]),bool(wrapped[i]))
                    if status:
                        r['status']=status;r.pop('truncation_reason',None)
                        if status=='truncated':r['truncation_reason']='native_episode_timeout'
                        r['failure_reasons']=[key for key,flags in observed['reasons'].items() if flags[i]] if fail[i] else []
                        active[i]=False
                if not active.any() or time.monotonic()>=a.soft_deadline:break
                actor_obs=torch.cat([obs[k] for k in algo.actor_obs_keys],dim=1)
                actions=policy({'actor_obs':actor_obs})
                if not torch.isfinite(actions).all():raise ValueError('Nonfinite evaluation actions')
                obs,_,_,_=env.step({'actions':actions})
                if step%100==0:save_progress()
        if recorder is not None:recorder.stop_recording()
        for i,r in enumerate(block):
            for key,values in samples[i].items():
                if values:r[key+'_mean']=float(np.mean(values));r[key+'_p95']=float(np.percentile(values,95))
        save_progress()
    after=snapshot(algo);_require_same_state(before,after,'frozen_evaluation_state')
    unchanged=digest(a.checkpoint)==CHECKPOINT_SHA256
    if not unchanged:raise ValueError('Source checkpoint mutated')
    result={**summarize(rows),'pipeline_passed':True,'frozen_model_and_optimizer_verified':True,
            'source_checkpoint_unchanged':True,'iteration_unchanged':SOURCE_ITERATION,
            'source_sha256':CHECKPOINT_SHA256,'video_files':[str(p.relative_to(out)) for p in out.rglob('*.mp4')]}
    (out/'evaluation_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'event':'eval283_frozen_evaluation_passed',**result}),flush=True)
    # The supervisor cleans up the simulation group after evidence has been flushed.
    sys.stdout.flush();os._exit(0)

if __name__=='__main__':main()