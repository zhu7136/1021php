"""Independent candidate read-back gates. No simulation/dynamics certification."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from kinematics import ROOT,Model,robot_config
from retarget_hud04 import SOURCE,G1_URDF


def validate(dataset):
    model=Model();g1=Model(G1_URDF);config=robot_config();bodies=config['_HU_D04_BODY_NAMES']
    zero=np.zeros((1,29));origin=np.zeros((1,3));identity=np.array([[1.,0,0,0]])
    hp,_,_,_=model.forward(zero,origin,identity);gp,_,_,_=g1.forward(zero,origin,identity)
    reports=[];manifest={}
    if not list(dataset.glob('*_motion.npz')):
        raise ValueError(f'No candidate motions found in {dataset}')
    for path in sorted(dataset.glob('*_motion.npz')):
        with np.load(path,allow_pickle=False) as loaded:
            data={k:loaded[k] for k in loaded.files}
        with np.load(SOURCE/path.name,allow_pickle=False) as loaded:
            src={k:loaded[k] for k in loaded.files}
        audit=json.loads(path.with_name(path.stem+'_audit.json').read_text())
        if audit['source_sha256'] != hashlib.sha256((SOURCE/path.name).read_bytes()).hexdigest():
            raise ValueError('Source digest mismatch')
        assert data['joint_names'].tolist()==model.names
        assert data['body_names'].tolist()==bodies
        assert all(np.isfinite(v).all() for v in data.values() if v.dtype.kind in 'fiu')
        q=data['joint_pos'][:,7:];root=data['joint_pos'][:,:3];quat=data['joint_pos'][:,3:7]
        frames=len(q);fps=float(data['fps'][0])
        if fps!=50 or data['joint_pos'].shape!=(frames,36) or data['joint_vel'].shape!=(frames,35):
            raise ValueError(f'Invalid FPS or root/joint array shape: {path.name}')
        for key,width in [('body_pos_w',3),('body_quat_w',4),('body_lin_vel_w',3),('body_ang_vel_w',3)]:
            if data[key].shape!=(frames,32,width):
                raise ValueError(f'Invalid {key} shape: {path.name}')
        quaternion_norm_error=float(max(np.max(abs(np.linalg.norm(quat,axis=-1)-1)),np.max(abs(np.linalg.norm(data['body_quat_w'],axis=-1)-1))))
        p,r,_,_=model.forward(q,root,quat)
        positions=np.stack([p[b] for b in bodies],axis=1)
        fkerror=float(np.linalg.norm(positions-data['body_pos_w'],axis=-1).max())
        rt=Rotation.from_quat(data['body_quat_w'][:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(len(q),32,3,3)
        rotationerror=float(Rotation.from_matrix((np.swapaxes(np.stack([r[b] for b in bodies],axis=1),-1,-2)@rt).reshape(-1,3,3)).magnitude().max())
        old=np.linspace(0,len(src['joint_pos'])-1,len(q))
        foot_errors=[];wrist_errors=[]
        for foot in ['contact_foot_center_L','contact_foot_center_R']:
            target=PchipInterpolator(np.arange(len(src['joint_pos'])),src['body_pos_w'][:,bodies.index(foot)],axis=0)(old)
            foot_errors.append(np.linalg.norm(p[foot]-target,axis=-1))
        for side in ['left','right']:
            shoulder=f'{side}_shoulder_pitch_link';wrist=f'{side}_wrist_roll_link';source_wrist=f'{side}_wrist_yaw_link'
            scale=model.chain_length(shoulder,wrist)/g1.chain_length(shoulder,source_wrist)
            delta=src['body_pos_w'][:,bodies.index(source_wrist)]-src['body_pos_w'][:,bodies.index(shoulder)]
            target=p[shoulder]+scale*PchipInterpolator(np.arange(len(delta)),delta,axis=0)(old)
            wrist_errors.append(np.linalg.norm(p[wrist]-target,axis=-1))
        feet=np.stack(foot_errors,axis=1);wrists=np.stack(wrist_errors,axis=1)
        angles_ok=bool(((q>=model.lower-1e-7)&(q<=model.upper+1e-7)).all())
        speed_ratio=float((abs(data['joint_vel'][:,6:])/model.velocity).max())
        velocity_error=float(abs(data['joint_vel'][:,6:]-np.gradient(q,1/float(data['fps'][0]),axis=0,edge_order=2)).max())
        body_linear_error=float(np.max(abs(data['body_lin_vel_w']-np.gradient(data['body_pos_w'],1/fps,axis=0,edge_order=2))))
        lo=np.maximum(np.arange(frames)-1,0);hi=np.minimum(np.arange(frames)+1,frames-1)
        body_rot=Rotation.from_quat(data['body_quat_w'][:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(frames,32,3,3)
        relative=body_rot[hi]@np.swapaxes(body_rot[lo],-1,-2)
        angular=Rotation.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(frames,32,3)*fps/(hi-lo)[:,None,None]
        body_angular_error=float(np.max(abs(data['body_ang_vel_w']-angular)))
        root_velocity_error=float(max(np.max(abs(data['joint_vel'][:,:3]-data['body_lin_vel_w'][:,0])),np.max(abs(data['joint_vel'][:,3:6]-data['body_ang_vel_w'][:,0]))))
        terrain=path.with_name(path.name.replace('_motion.npz','_terrain.npy'))
        terrain_same=hashlib.sha256(terrain.read_bytes()).digest()==hashlib.sha256((SOURCE/terrain.name).read_bytes()).digest()
        # Verify foot orientation after resampling rather than trusting audit.
        from scipy.spatial.transform import Slerp
        foot_angles=[]
        _,hr,_,_=model.forward(zero,origin,identity);_,gr,_,_=g1.forward(zero,origin,identity)
        for foot,gfoot in [('contact_foot_center_L','left_foot_contact_point'),('contact_foot_center_R','right_foot_contact_point')]:
            source_rot=Rotation.from_quat(src['body_quat_w'][:,bodies.index(foot),:][:,[1,2,3,0]])
            target_rot=Slerp(np.arange(len(src['joint_pos'])),source_rot)(old).as_matrix()@(gr[gfoot][0].T@hr[foot][0])
            foot_angles.append(np.rad2deg(Rotation.from_matrix(target_rot@np.swapaxes(r[foot],-1,-2)).magnitude()))
        foot_orientation_max=float(np.max(foot_angles))
        passed=angles_ok and speed_ratio<=.81 and fkerror<=1e-8 and rotationerror<=1e-6 and velocity_error<=1e-7 and terrain_same and feet.max()<=.02 and wrists.max()<=.02 and foot_orientation_max<=5 and quaternion_norm_error<=1e-7 and body_linear_error<=1e-7 and body_angular_error<=1e-7 and root_velocity_error<=1e-7 and audit['kinematic_target_gate_pass']
        row=dict(file=path.name,frames=len(q),angles_ok=angles_ok,max_speed_ratio=speed_ratio,fk_max_error_m=fkerror,rotation_max_error_rad=rotationerror,joint_velocity_derivative_error=velocity_error,foot_target_mean_m=float(feet.mean()),foot_target_max_m=float(feet.max()),scaled_wrist_target_mean_m=float(wrists.mean()),scaled_wrist_target_max_m=float(wrists.max()),time_scale=audit['time_scale'],terrain_unchanged=terrain_same,pass_kinematic_gates=bool(passed))
        reports.append(row)
        row['foot_orientation_max_deg']=foot_orientation_max
        row.update(quaternion_norm_error=quaternion_norm_error,body_linear_derivative_error=body_linear_error,body_angular_derivative_error=body_angular_error,root_velocity_consistency_error=root_velocity_error)
        for f in [path,terrain,path.with_name(path.stem+'_audit.json')]:
            manifest[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        print(json.dumps(row),flush=True)
    defaults=np.array([[config['_HU_D04_DEFAULT_JOINT_ANGLES'][n] for n in model.names]])
    p,_,_,_=model.forward(defaults,np.array([[0,0,.91]]),identity)
    summary=dict(status='kinematic_candidate_not_simulation_validated',files=len(reports),expected_files=56,all_kinematic_gates_pass=len(reports)==56 and all(r['pass_kinematic_gates'] for r in reports),frames=sum(r['frames'] for r in reports),foot_target_max_m=max(r['foot_target_max_m'] for r in reports),foot_target_weighted_mean_m=sum(r['foot_target_mean_m']*r['frames'] for r in reports)/sum(r['frames'] for r in reports),max_speed_ratio=max(r['max_speed_ratio'] for r in reports),max_fk_error_m=max(r['fk_max_error_m'] for r in reports),time_scale_range=[min(r['time_scale'] for r in reports),max(r['time_scale'] for r in reports)],scaled_wrist_target_weighted_mean_m=sum(r['scaled_wrist_target_mean_m']*r['frames'] for r in reports)/sum(r['frames'] for r in reports),scaled_wrist_target_max_m=max(r['scaled_wrist_target_max_m'] for r in reports),default_foot_center_z_m={b:float(p[b][0,2]) for b in ['contact_foot_center_L','contact_foot_center_R']},limitations=['No mesh collision, torque feasibility, COM/support, or simulator validation','Source foot reference clearance is retained; ground contact is not certified','Arm reach is morphology-scaled, with residual reported, not an exact source pose match','vel_cmd omitted: only the terrain teacher observation is supported'],per_file=reports,sha256=manifest)
    (dataset/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ['per_file','sha256']}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path)
    args=parser.parse_args();result=validate(args.dataset)
    raise SystemExit(0 if result['all_kinematic_gates_pass'] else 1)