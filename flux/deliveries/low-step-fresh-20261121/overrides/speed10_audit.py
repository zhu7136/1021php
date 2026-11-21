"""Read-only native velocity-limit and measured-speed evidence; no state clipping."""
import json
from pathlib import Path
import torch

def is_target(name):
    return name.removesuffix('_joint').removeprefix('left_').removeprefix('right_') in ('hip_pitch','hip_roll','hip_yaw','knee')

def install_native_audit(env,out):
    out=Path(out);sim=env.simulator;robot=sim._robot
    names=list(env.robot_config.dof_names);physical=list(robot.joint_names)
    assert len(names)==len(physical)==31 and set(names)==set(physical)
    expected=torch.tensor(env.robot_config.dof_vel_limit_list,dtype=torch.float32)
    actual=robot.root_physx_view.get_dof_max_velocities().detach().cpu()[:,[physical.index(n) for n in names]]
    assert actual.shape==(env.num_envs,31) and torch.isfinite(actual).all()
    assert torch.allclose(actual,expected[None,:].expand_as(actual),atol=1e-4,rtol=1e-5), 'native_velocity_limits_mismatch'
    ids=[i for i,n in enumerate(names) if is_target(n)]
    assert len(ids)==8
    for i in ids:
        suffix=names[i].removesuffix('_joint').removeprefix('left_').removeprefix('right_')
        assert expected[i].item()=={'hip_pitch':32.,'hip_roll':20.,'hip_yaw':32.,'knee':20.}[suffix], 'g1_target_limit_mismatch'
    report={'event':'native_velocity_limits_verified','source':'Articulation.root_physx_view.get_dof_max_velocities','num_envs':env.num_envs,'joint_names':names,'expected_rad_s':expected.tolist(),'native_min_rad_s':actual.amin(0).tolist(),'native_max_rad_s':actual.amax(0).tolist(),'target_joints':[names[i] for i in ids],'all_envs_matched':True}
    (out/'native-velocity-limits.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    state={'physics_steps':0,'peak':torch.zeros(31,device=sim.dof_vel.device),'over_count':torch.zeros(31,device=sim.dof_vel.device,dtype=torch.int64)}
    limits=expected.to(sim.dof_vel.device);original_step=sim.simulate_at_each_physics_step
    def observed_step(*args,**kwargs):
        result=original_step(*args,**kwargs)
        velocity=sim.dof_vel.detach().abs()
        state['peak']=torch.maximum(state['peak'],velocity.amax(0))
        state['over_count']+=(velocity>limits[None,:]+1e-3).sum(0)
        state['physics_steps']+=1
        return result
    sim.simulate_at_each_physics_step=observed_step
    def flush(update):
        result={'event':'measured_joint_speeds','update':int(update),'physics_steps':state['physics_steps'],'joint_names':names,'limits_rad_s':expected.tolist(),'peak_abs_rad_s':state['peak'].cpu().tolist(),'over_limit_env_physics_samples':state['over_count'].cpu().tolist(),'measurement':'after each physics step, all environments; cumulative since learn entry; no clipping'}
        (out/'measured-joint-speeds.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    return flush