"""One bounded read-only GPU diagnosis; no policy updates or checkpoint writes."""
from __future__ import annotations
import argparse
import atexit
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/short_episode_diagnostic'
SHA='634f343946005e72303b9c100aa90f6a9ff76e8224316a59af1f0842e3edaf85'
MODES=('policy_mean','policy_sample','reference_pd')
NUM_ENVS=32
STEPS_PER_MODE=256
WORKER_SECONDS=900

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def emit(event,**fields):
    print(json.dumps(dict(event=event,**fields),allow_nan=False),flush=True)

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))

def heights_at_xy(mesh,points):
    """Highest mesh surface at XY. Pure numpy vertical triangle intersections."""
    import numpy as np
    tri=np.asarray(mesh.triangles,dtype=np.float64)
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    valid=abs(den)>1e-12; a,b,c,den=a[valid],b[valid],c[valid],den[valid]
    result=[]
    for p in points:
        u=((b[:,1]-c[:,1])*(p[0]-c[:,0])+(c[:,0]-b[:,0])*(p[1]-c[:,1]))/den
        v=((c[:,1]-a[:,1])*(p[0]-c[:,0])+(a[:,0]-c[:,0])*(p[1]-c[:,1]))/den
        inside=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
        z=u*a[:,2]+v*b[:,2]+(1-u-v)*c[:,2]
        result.append(float(z[inside].max()) if inside.any() else None)
    return result

def diagnose(dataset):
    import numpy as np
    import torch
    from holosoma.utils.eval_utils import CheckpointConfig
    from holosoma.config_types.logger import DisabledLoggerConfig
    from wbt_training.utils.checkpoint import load_saved_experiment_config,apply_preprocess_hook
    from holosoma.utils.helpers import get_class
    import holosoma.utils.sim_utils as sim_utils
    from startup_diagnostics import bounded_launcher
    import trimesh
    checkpoint=ROOT/'flux/replay-input/model_00790.pt'
    assert digest(checkpoint)==SHA,'Checkpoint mismatch'
    saved,_=load_saved_experiment_config(CheckpointConfig(checkpoint=str(checkpoint)))
    cfg=dataclasses.replace(saved,
        training=dataclasses.replace(saved.training,num_envs=NUM_ENVS,headless=True,
            export_onnx=False,registry_name=dataset.resolve().as_uri()),
        algo=dataclasses.replace(saved.algo,config=dataclasses.replace(saved.algo.config,load_optimizer=False)),
        logger=DisabledLoggerConfig(base_dir=str(OUT),headless_recording=False))
    cfg=apply_preprocess_hook(cfg)
    cfg.save_config(str(OUT/'diagnostic_config.yaml'))
    sys.argv=[sys.argv[0]]
    sim_utils.setup_isaaclab_launcher=bounded_launcher(sim_utils.setup_isaaclab_launcher)
    env,device,app=sim_utils.setup_simulation_environment(cfg)
    atexit.register(sim_utils.close_simulation_app,app)
    assert device=='cuda:0' and env.num_envs==NUM_ENVS
    torch.set_grad_enabled(False)
    sim=env.simulator
    native=list(sim._robot.joint_names); ids=list(sim.dof_ids)
    assert len(ids)==len(set(ids))==29
    assert [native[i] for i in ids]==list(cfg.robot.dof_names)
    assert torch.equal(sim.dof_pos,sim._robot.data.joint_pos[:,ids])
    save('joint_mapping.json',[dict(policy_index=i,joint=n,isaac_index=ids[i]) for i,n in enumerate(cfg.robot.dof_names)])
    reset_all=env.reset_all
    env.reset_all=lambda:None
    try:
        algo=get_class(cfg.algo._target_)(env=env,config=cfg.algo.config,log_dir=str(OUT/'tensorboard'),device=device)
    finally:
        env.reset_all=reset_all
    def forbidden(*args,**kwargs):raise RuntimeError('Read-only diagnostic forbids training/save')
    algo.learn=forbidden;algo.save=forbidden
    algo.setup();algo._restore_env_state=lambda state:None;algo.load(str(checkpoint));algo._eval_mode()
    assert algo.current_learning_iteration==790
    def weights_hash():
        h=hashlib.sha256()
        for name,t in sorted(algo.actor.state_dict().items()):
            h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=weights_hash()
    motion=env.command_manager.get_state('motion_command')
    term=env.termination_manager._term_instances['bad_tracking']
    all_ids=torch.arange(NUM_ENVS,device=device)
    mesh=trimesh.load(cfg.terrain.terrain_term.obj_file_path,force='mesh')
    # Probe exact reference reset with no noise, before a single physics control step.
    # No target alignment is advanced here; all comparisons are absolute world FK.
    old_noise=motion.init_pose_cfg
    motion.init_pose_cfg=dataclasses.replace(old_noise,overall_noise_scale=0.0)
    torch.manual_seed(1401);torch.cuda.manual_seed_all(1401)
    env.reset_envs_idx(all_ids)
    sim.set_actor_root_state_tensor_robots(all_ids,sim.robot_root_states)
    sim.set_dof_state_tensor_robots(all_ids,sim.dof_state)
    sim.write_state_updates();sim.refresh_sim_tensors()
    refpos=motion.body_pos_w; realpos=motion.robot_body_pos_w
    cached_error=float(torch.linalg.vector_norm(refpos-realpos,dim=-1).max())
    # Flush kinematics explicitly, without a physics step, and invalidate lazy body
    # buffers. Keep the pre-forward error to distinguish stale readback from FK.
    sim.sim.forward();sim.scene.update(dt=1e-6);sim.refresh_sim_tensors()
    refpos=motion.body_pos_w; realpos=motion.robot_body_pos_w
    errors=torch.linalg.vector_norm(refpos-realpos,dim=-1)
    footnames=['contact_foot_center_L','contact_foot_center_R']
    mbody=list(cfg.robot.body_names)
    feet=motion.motion._body_pos_w[motion.time_steps][:,[mbody.index(n) for n in footnames]]+sim.scene.env_origins[:,None,:]
    f=feet.detach().cpu().numpy().reshape(-1,3)
    z=heights_at_xy(mesh,f)
    clear=[float(p[2]-h) if h is not None else None for p,h in zip(f,z)]
    init=dict(probe='no_noise_no_physics',environments=NUM_ENVS,
        tracked_body_cached_pre_forward_error_max_m=cached_error,
        root_pos_error_max_m=float(torch.linalg.vector_norm(sim.robot_root_states[:,:3]-motion.root_pos_w,dim=-1).max()),
        joint_pos_error_max_rad=float(abs(sim.dof_pos-motion.joint_pos).max()),
        tracked_body_absolute_error_max_m=float(errors.max()),
        tracked_body_error_by_name_m={n:float(errors[:,i].max()) for i,n in enumerate(motion.motion_cfg.body_names_to_track)},
        motion_frame_indices=motion.time_steps.tolist(),
        scene_env_origin_abs_max_m=float(abs(sim.scene.env_origins).max()),
        terrain_bounds=mesh.bounds.tolist(),reference_feet=f.tolist(),terrain_height_at_feet=z,
        foot_center_clearance_m=clear,feet_with_no_surface=sum(v is None for v in z),
        foot_center_below_surface_2cm=sum(v is not None and v<-.02 for v in clear),
        note='Foot-center height is not collision-mesh penetration; no-noise joint clipping and simulator FK can affect readback.')
    save('initialization_probe.json',init);emit('diagnostic_initialization_probe',**init)
    motion.init_pose_cfg=old_noise
    # Uniform training-frame resets avoid eval's frame-zero-only branch and adaptive
    # distribution drift across the three controls. Dynamics/pushes/noise stay original.
    motion.motion_cfg=dataclasses.replace(motion.motion_cfg,use_adaptive_timesteps_sampler=False)
    env.is_evaluating=False
    phase={'name':None,'step':-1}; counters={}; results=[]
    original_check=env._check_termination
    event_file=(OUT/'events.jsonl').open('w',buffering=1)
    def checked():
        original_check()
        if phase['name'] is None:return
        flags={n:getattr(term,n)(motion) for n in ['bad_ref_pos','bad_ref_ori','bad_motion_body_pos']}
        combined=flags['bad_ref_pos']|flags['bad_ref_ori']|flags['bad_motion_body_pos']
        if motion.motion.has_object:
            for n in ['bad_object_pos','bad_object_ori']:
                flags[n]=getattr(term,n)(motion);combined|=flags[n]
        done=env.reset_buf.bool();timeout=env.time_out_buf.bool()
        expected=combined|timeout
        assert torch.equal(done,expected),'Termination instrumentation disagrees with original'
        counters['checks']+=NUM_ENVS
        for n,v in flags.items():counters[n]=counters.get(n,0)+int(v.sum())
        counters['timeouts']+=int(timeout.sum())
        counters['resets']+=int(done.sum())
        if done.any():
            lengths=env.episode_length_buf[done].tolist();counters['lengths']+=lengths
            bi=term.bad_motion_body_pos_body_indexes
            err=torch.linalg.vector_norm(motion.body_pos_relative_w[:,bi]-motion.robot_body_pos_w[:,bi],dim=-1)
            for j,n in enumerate(term.bad_motion_body_pos_body_names):
                counters['body_causes'][n]+=int((err[:,j]>term.bad_motion_body_pos_threshold).sum())
            event_file.write(json.dumps(dict(mode=phase['name'],step=phase['step'],env_ids=done.nonzero().flatten().tolist(),
                episode_lengths=lengths,causes={n:v[done].tolist() for n,v in flags.items()},
                tracked_errors_m=err[done].tolist(),ref_pos_error_m=torch.linalg.vector_norm(motion.ref_pos_w-motion.robot_ref_pos_w,dim=-1)[done].tolist()))+'\n')
    env._check_termination=checked
    try:
        for mode in MODES:
            torch.manual_seed(1402);torch.cuda.manual_seed_all(1402)
            counters=dict(checks=0,timeouts=0,resets=0,lengths=[],body_causes={n:0 for n in term.bad_motion_body_pos_body_names})
            phase.update(name=None,step=-1)
            obs=env.reset_all() # One initialization zero-action step excluded from phase lengths.
            env.episode_length_buf.zero_()
            phase['name']=mode
            for step in range(STEPS_PER_MODE):
                phase['step']=step
                if mode=='reference_pd':actions=(motion.joint_pos-env.default_dof_pos)/env.action_scales
                else:
                    actor_obs=torch.cat([obs[k] for k in algo.actor_obs_keys],dim=1)
                    normalized=algo._normalize_actor_obs(actor_obs,update=False)
                    actions=algo.actor.act_inference({'actor_obs':normalized}) if mode=='policy_mean' else algo.actor.act({'actor_obs':normalized})
                assert torch.isfinite(actions).all() and torch.isfinite(sim.dof_pos).all()
                obs,_,_,_=env.step({'actions':actions})
                if (step+1)%64==0:emit('diagnostic_progress',mode=mode,steps=step+1,resets=counters['resets'])
            phase['name']=None
            lengths=np.asarray(counters.pop('lengths'))
            row=dict(mode=mode,control_steps=STEPS_PER_MODE,num_envs=NUM_ENVS,seed=1402,
                reset_sampler='uniform_training_frames',init_noise='original',pushes='original',
                completed_episode_mean_steps=float(lengths.mean()) if len(lengths) else None,
                completed_episode_median_steps=float(np.median(lengths)) if len(lengths) else None,
                completed_episode_p95_steps=float(np.percentile(lengths,95)) if len(lengths) else None,
                right_censored_environment_ages=env.episode_length_buf.tolist(),**counters)
            results.append(row);save('partial_results.json',results);emit('diagnostic_mode_result',**row)
        summary=dict(checkpoint_sha256=SHA,checkpoint_iteration=790,training_updates=0,
            checkpoint_unchanged=digest(checkpoint)==SHA,actor_weights_unchanged=weights_hash()==before,
            joint_mapping_runtime_verified=True,gpu='1x4090D24GB',num_envs=NUM_ENVS,
            control_steps_per_mode=STEPS_PER_MODE,total_control_steps=3*(STEPS_PER_MODE+1),
            action_noise_std=algo.actor.std.detach().cpu().tolist(),results=results,
            limitations=['One short diagnostic, no success-rate/convergence estimate.',
              'Terrain regenerated, not original task034 mesh; original seed was unspecified.',
              'Uniform training-frame reset sampler; adaptive failure distribution intentionally disabled.',
              'Mean episode length excludes right-censored survivors; see environment ages.',
              'Reference PD is a controller probe, not a trained policy or dynamics certificate.'])
        assert summary['checkpoint_unchanged'] and summary['actor_weights_unchanged']
        save('summary.json',summary);emit('diagnostic_complete',**summary)
    finally:event_file.close()

def worker():
    from source_preflight import verify_sources
    from runtime_preflight import verify_runtime
    emit('diagnostic_source_verified',**verify_sources(ROOT));emit('diagnostic_runtime_verified',**verify_runtime(ROOT))
    from git_smoke_v8 import ARCHIVE_BYTES,ARCHIVE_SHA256,extract_candidates
    from git_parts_v8 import assemble_parts
    archive=OUT/'v8.zip';OUT.mkdir(parents=True,exist_ok=False)
    assemble_parts(ROOT/'flux/data/hud04-v8-parts',archive,ARCHIVE_BYTES,ARCHIVE_SHA256)
    dataset=OUT/'candidates';extract_candidates(archive,dataset)
    from offline_dependencies import install
    install(ROOT)
    from train_hud04 import prepare_runtime
    runtime=prepare_runtime();sys.path[:0]=[str(ROOT/'.flux-deps'),str(ROOT),str(runtime)]
    os.chdir(ROOT)
    from startup_diagnostics import install as startup
    startup(ROOT)
    diagnose(dataset)

def upload(url,headers):
    # Do not include extracted input datasets/archive in the output bundle.
    import zipfile,urllib.request,urllib.parse
    p=urllib.parse.urlsplit(url)
    assert p.scheme=='https' and (p.hostname=='storage.googleapis.com' or str(p.hostname).endswith('.storage.googleapis.com'))
    assert all(k.lower()=='content-type' or k.lower().startswith('x-goog-') for k in headers)
    files=[f for f in OUT.glob('*') if f.is_file() and f.suffix in ['.json','.jsonl','.yaml']]
    assert all(not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files)<40*1024*1024
    bundle=ROOT/'outputs/replay_bundle.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,arcname=f.name)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*a,**k):return None
    r=urllib.request.Request(url,data=bundle.read_bytes(),headers=headers,method='PUT')
    with urllib.request.build_opener(NoRedirect()).open(r,timeout=40) as response:assert response.status in [200,201,204]
    receipt=dict(sha256=digest(bundle),bytes=bundle.stat().st_size)
    (ROOT/'outputs/diagnostic_upload_receipt.json').write_text(json.dumps(receipt))
    emit('diagnostic_bundle_uploaded',**receipt)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime-child',action='store_true')
    p.add_argument('--upload-url',required=True);p.add_argument('--upload-headers',default='{}');args=p.parse_args()
    if args.runtime_child:
        try:worker()
        except BaseException as e:
            OUT.mkdir(parents=True,exist_ok=True);save('failure.json',dict(exception_type=type(e).__name__))
            emit('diagnostic_failure',exception_type=type(e).__name__);raise
        finally:
            if OUT.exists():
                try:upload(args.upload_url,json.loads(args.upload_headers))
                except BaseException as e:emit('diagnostic_upload_failure',exception_type=type(e).__name__)
        return
    assert not OUT.exists(),'Fresh diagnostic task required; never restart'
    from runtime_preflight import resolve_isaac_launcher
    launcher=resolve_isaac_launcher()
    e=os.environ.copy();e.update(PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',WANDB_MODE='disabled')
    for n in ['key','github','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY']:e.pop(n,None)
    emit('diagnostic_worker_start',num_envs=NUM_ENVS,modes=MODES,steps_per_mode=STEPS_PER_MODE,worker_limit_seconds=WORKER_SECONDS,training_updates=0,automatic_retries=0)
    child=subprocess.Popen([str(launcher),str(Path(__file__).resolve()),'--runtime-child','--upload-url',args.upload_url,'--upload-headers',args.upload_headers],env=e,start_new_session=True)
    from startup_diagnostics import wait_for_child
    try:code=wait_for_child(child,ROOT,WORKER_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10);raise SystemExit(124)
    assert code==0 and (OUT/'summary.json').is_file(),'No sealed diagnostic summary'
    result=json.loads((OUT/'summary.json').read_text())
    assert result['checkpoint_unchanged'] and result['actor_weights_unchanged'] and len(result['results'])==3
    assert (ROOT/'outputs/diagnostic_upload_receipt.json').is_file(),'Output upload unverified'
    emit('diagnostic_entrypoint_complete',checkpoint_sha256=SHA,training_updates=0)

if __name__=='__main__':main()