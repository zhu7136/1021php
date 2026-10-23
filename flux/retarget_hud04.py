"""Offline robot-to-robot IK candidates. Does not certify collision/dynamic safety."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from kinematics import ROOT, Model, robot_config

G1_URDF=ROOT/'thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf'
SOURCE=ROOT/'flux/data/motions-low-step-hu-d04'


def solve(data, iterations=100):
    config=robot_config();model=Model();g1=Model(G1_URDF)
    names=config['_HU_D04_DOF_NAMES'];bodies=config['_HU_D04_BODY_NAMES']
    q=np.array(data['joint_pos'][:,7:],copy=True);root=np.array(data['joint_pos'][:,:3],copy=True);quat=data['joint_pos'][:,3:7].copy()
    # HU_D04 elbow flexion uses the opposite sign to G1 in these URDFs.
    for side in ['left','right']:
        q[:,names.index(f'{side}_elbow_joint')]*=-1
    # A single, smooth HU_D04 arm seed avoids selecting different redundant
    # IK branches independently from G1 arm coordinates.
    for i in range(15,29):
        q[:,i]=config['_HU_D04_DEFAULT_JOINT_ANGLES'][names[i]]
    q=np.clip(q,model.lower+1e-5,model.upper-1e-5)
    qseed=q.copy();zero=np.zeros((1,29));ident=np.array([[1.,0,0,0]])
    hp0,hr0,_,_=model.forward(zero,np.zeros((1,3)),ident)
    gp0,gr0,_,_=g1.forward(zero,np.zeros((1,3)),ident)
    feet=['contact_foot_center_L','contact_foot_center_R']
    p,_,_,_=model.forward(q,root,quat)
    offsets=np.mean(np.stack([data['body_pos_w'][:,bodies.index(b),2]-p[b][:,2] for b in feet],axis=1),axis=1)
    root[:,2]+=gaussian_filter1d(offsets,2)
    p,r,_,_=model.forward(q,root,quat)
    targets={};groups=[]
    mapping={feet[0]:'left_foot_contact_point',feet[1]:'right_foot_contact_point'}
    for side,foot,columns in [('left',feet[0],np.arange(0,6)),('right',feet[1],np.arange(6,12))]:
        source_R=Rotation.from_quat(data['body_quat_w'][:,bodies.index(foot),:][:,[1,2,3,0]]).as_matrix()
        targets[foot]=(data['body_pos_w'][:,bodies.index(foot)].copy(),source_R@(gr0[mapping[foot]][0].T@hr0[foot][0]))
        groups.append((foot,columns,None))
    for side,columns in [('left',np.arange(15,22)),('right',np.arange(22,29))]:
        shoulder=f'{side}_shoulder_pitch_link';wrist=f'{side}_wrist_roll_link';elbow=f'{side}_elbow_link'
        # HU_D04's distal wrist is roll; G1's is yaw. Same-name links are
        # different points on the chains, and zero-pose reach is not arm length.
        source_wrist=f'{side}_wrist_yaw_link'
        scale=model.chain_length(shoulder,wrist)/g1.chain_length(shoulder,source_wrist)
        # Preserve shoulder-relative reach direction, scaled for limb length.
        target_p=p[shoulder]+scale*(data['body_pos_w'][:,bodies.index(source_wrist)]-data['body_pos_w'][:,bodies.index(shoulder)])
        source_R=Rotation.from_quat(data['body_quat_w'][:,bodies.index(source_wrist),:][:,[1,2,3,0]]).as_matrix()
        targets[wrist]=(target_p,source_R@(gr0[source_wrist][0].T@hr0[wrist][0]))
        elbow_p=p[shoulder]+scale*(data['body_pos_w'][:,bodies.index(elbow)]-data['body_pos_w'][:,bodies.index(shoulder)])
        groups.append((wrist,columns,(elbow,elbow_p)))
    for step in range(iterations):
        p,r,jp,ja=model.forward(q,root,quat)
        for body,cols,elbow in groups:
            target_p,target_r=targets[body]
            linear,angular=model.jacobian(body,cols,p,jp,ja)
            orientation=Rotation.from_matrix(target_r@np.swapaxes(r[body],-1,-2)).as_rotvec()
            # Arms prioritize morphology-scaled wrist reach; frame orientation
            # and intermediate elbow positions are weak secondary objectives.
            weight=.15 if elbow is None else .006
            J=np.concatenate([linear,weight*angular],axis=1)
            error=np.concatenate([target_p-p[body],weight*orientation],axis=1)
            if elbow is not None:
                # Endpoint position first. G1/HU_D04 elbow frames and wrist
                # axes are different; forcing their orientations is misleading.
                J=linear
                error=target_p-p[body]
            jt=np.swapaxes(J,1,2)
            damping=2e-5
            delta=np.linalg.solve(jt@J+damping*np.eye(len(cols)),(jt@error[:,:,None])[...,0,None])[...,0]
            q[:,cols]+=np.clip(delta,-.15,.15)
            q[:,cols]=np.clip(q[:,cols],model.lower[cols]+1e-5,model.upper[cols]-1e-5)
        # Temporal regularization suppresses frame-wise IK branch jumps.
        q=gaussian_filter1d(q,.7,axis=0,mode='nearest')
        q=np.clip(q,model.lower+1e-5,model.upper-1e-5)
    p,r,_,_=model.forward(q,root,quat)
    foot_errors=np.stack([np.linalg.norm(p[b]-targets[b][0],axis=-1) for b in feet],axis=1)
    foot_angles=np.stack([Rotation.from_matrix(targets[b][1]@np.swapaxes(r[b],-1,-2)).magnitude() for b in feet],axis=1)
    wrist_errors=np.stack([np.linalg.norm(p[b]-targets[b][0],axis=-1) for b,_,e in groups if e is not None],axis=1)
    return model,q,root,quat,dict(foot_target_error_mean_m=float(foot_errors.mean()),foot_target_error_max_m=float(foot_errors.max()),foot_orientation_error_max_deg=float(np.rad2deg(foot_angles).max()),wrist_target_error_mean_m=float(wrist_errors.mean()),wrist_target_error_max_m=float(wrist_errors.max()),root_lift_range_m=[float(offsets.min()),float(offsets.max())],iterations=iterations)


def velocities(rotations,fps):
    # World-frame angular velocity from central rotations, including endpoints.
    n=len(rotations)
    lo=np.maximum(np.arange(n)-1,0);hi=np.minimum(np.arange(n)+1,n-1)
    rel=rotations[hi]@np.swapaxes(rotations[lo],-1,-2)
    return Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(*rel.shape[:-2],3)*fps/(hi-lo).reshape((-1,)+(1,)*(rel.ndim-2))


def export(data, model, q, root, quat, metrics, project_arms=True):
    fps=float(data['fps'][0]);t=np.arange(len(q))/fps
    spline=PchipInterpolator(t,q,axis=0)
    dense=np.linspace(t[0],t[-1],len(q)*4)
    ratio=float(np.max(abs(spline(dense,1))/model.velocity))
    factor=max(1.,ratio/.8)
    if factor>10:
        raise ValueError(f'IK trajectory is too discontinuous: required time scale {factor:.2f}')
    count=int(np.ceil((len(q)-1)*factor))+1
    old=np.linspace(t[0],t[-1],count)
    qnew=spline(old);rootnew=CubicSpline(t,root,axis=0)(old)
    # Cubic overshoot must never silently violate joint limits.
    if np.any(qnew<model.lower-1e-7) or np.any(qnew>model.upper+1e-7):
        raise ValueError('Interpolated IK crosses joint limits; refusing output')
    quatnew=Slerp(t,Rotation.from_quat(quat[:,[1,2,3,0]]))(old).as_quat()[:,[3,0,1,2]]
    # Resolve distal arms on the dense output time grid, so interpolating joint
    # coordinates cannot silently create large Cartesian reach errors.
    bodies=robot_config()['_HU_D04_BODY_NAMES'];g1=Model(G1_URDF)
    for iteration in range(100 if project_arms else 0):
        p,r,jp,ja=model.forward(qnew,rootnew,quatnew)
        for side,cols in [('left',np.arange(15,22)),('right',np.arange(22,29))]:
            shoulder=f'{side}_shoulder_pitch_link';wrist=f'{side}_wrist_roll_link';source_wrist=f'{side}_wrist_yaw_link'
            scale=model.chain_length(shoulder,wrist)/g1.chain_length(shoulder,source_wrist)
            delta=data['body_pos_w'][:,bodies.index(source_wrist)]-data['body_pos_w'][:,bodies.index(shoulder)]
            target=p[shoulder]+scale*PchipInterpolator(t,delta,axis=0)(old)
            jac,_=model.jacobian(wrist,cols,p,jp,ja);jt=jac.transpose(0,2,1)
            dq=np.linalg.solve(jt@jac+2e-5*np.eye(len(cols)),jt@(target-p[wrist])[:,:,None])[...,0]
            qnew[:,cols]=np.clip(qnew[:,cols]+np.clip(dq,-.1,.1),model.lower[cols]+1e-5,model.upper[cols]-1e-5)
        qnew[:,15:]=gaussian_filter1d(qnew[:,15:],.5,axis=0,mode='nearest')
    # A second uniform time scaling accounts for dense-grid IK corrections.
    fine_t=np.arange(len(qnew))/fps
    ratio2=float((abs(PchipInterpolator(fine_t,qnew,axis=0)(np.linspace(fine_t[0],fine_t[-1],4*len(qnew)),1))/model.velocity).max())
    if ratio2>.8:
        count2=int(np.ceil((len(qnew)-1)*ratio2/.8))+1
        if (count2-1)/(len(q)-1)>10:
            raise ValueError('Dense IK requires more than 10x time scaling')
        dense_t=np.linspace(fine_t[0],fine_t[-1],count2)
        qnew=PchipInterpolator(fine_t,qnew,axis=0)(dense_t)
        rootnew=CubicSpline(fine_t,rootnew,axis=0)(dense_t)
        quatnew=Slerp(fine_t,Rotation.from_quat(quatnew[:,[1,2,3,0]]))(dense_t).as_quat()[:,[3,0,1,2]]
        count=count2
    p,r,_,_=model.forward(qnew,rootnew,quatnew)
    phase=np.linspace(t[0],t[-1],count)
    final_wrists=[]
    for side in ['left','right']:
        shoulder=f'{side}_shoulder_pitch_link';wrist=f'{side}_wrist_roll_link';source_wrist=f'{side}_wrist_yaw_link'
        delta=data['body_pos_w'][:,bodies.index(source_wrist)]-data['body_pos_w'][:,bodies.index(shoulder)]
        scale=model.chain_length(shoulder,wrist)/g1.chain_length(shoulder,source_wrist)
        target=p[shoulder]+scale*PchipInterpolator(t,delta,axis=0)(phase)
        final_wrists.append(np.linalg.norm(p[wrist]-target,axis=-1))
    metrics['final_wrist_target_max_m']=float(np.max(final_wrists))
    config=robot_config();bodies=config['_HU_D04_BODY_NAMES']
    positions=np.stack([p[b] for b in bodies],axis=1)
    rotations=np.stack([r[b] for b in bodies],axis=1)
    bodyquat=Rotation.from_matrix(rotations.reshape(-1,3,3)).as_quat().reshape(count,32,4)[:,:,[3,0,1,2]]
    linear=np.gradient(positions,1/fps,axis=0,edge_order=2)
    angular=velocities(rotations,fps)
    qvel=np.gradient(qnew,1/fps,axis=0,edge_order=2)
    if np.max(abs(qvel)/model.velocity)>.81:
        raise ValueError('Time scaling did not meet velocity margin')
    result=dict(fps=np.array([fps]),joint_names=np.array(model.names),body_names=np.array(bodies),joint_pos=np.concatenate([rootnew,quatnew,qnew],axis=1),joint_vel=np.concatenate([linear[:,0],angular[:,0],qvel],axis=1),body_pos_w=positions,body_quat_w=bodyquat,body_lin_vel_w=linear,body_ang_vel_w=angular)
    # vel_cmd is a source-controller auxiliary; no verified retargeted semantics.
    # Omit it rather than copy stale commands across a changed time base.
    metrics.update(time_scale=(count-1)/(len(q)-1),frames=count,max_joint_speed_ratio=float(np.max(abs(qvel)/model.velocity)),source_frames=len(q))
    return result,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern',default='forward_walk1.2s_step0.5s_motion.npz')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=150)
    parser.add_argument('--skip-existing',action='store_true',help='Resume without replacing already generated candidates')
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--limit',type=int,default=1)
    args=parser.parse_args()
    if args.output.resolve()==SOURCE.resolve():
        parser.error('Refusing to overwrite recovered sources')
    args.output.mkdir(parents=True,exist_ok=True)
    paths=sorted(SOURCE.glob(args.pattern))[args.offset:args.offset+args.limit]
    if not paths:
        parser.error('No matching source motions')
    for path in paths:
        output=args.output/path.name
        if output.exists():
            if args.skip_existing:
                continue
            raise FileExistsError(f'Refusing to overwrite {output}')
        with np.load(path,allow_pickle=False) as source:
            data={k:source[k] for k in source.files}
        model,q,root,quat,metrics=solve(data,args.iterations)
        print(json.dumps({'stage':'ik_solved',**metrics}),flush=True)
        try:
            result,metrics=export(data,model,q,root,quat,metrics)
        except ValueError as error:
            rejected=dict(source=path.name,status='rejected_not_training_data',reason=str(error),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_grid_metrics=metrics)
            rejection=args.output/(path.stem+'_rejected.json')
            if not rejection.exists():
                rejection.write_text(json.dumps(rejected,indent=2)+'\n')
            print(json.dumps(rejected),flush=True)
            continue
        metrics.update(source=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status='candidate_not_simulation_validated')
        # Final Cartesian targets are rechecked by validate_hud04.py after
        # dense-grid arm projection; source-grid wrist errors are diagnostic.
        metrics['kinematic_target_gate_pass'] = metrics['foot_target_error_max_m'] <= .02 and metrics['foot_orientation_error_max_deg'] <= 5 and metrics['final_wrist_target_max_m'] <= .02
        np.savez_compressed(output,**result)
        terrain=path.with_name(path.name.replace('_motion.npz','_terrain.npy'))
        shutil.copyfile(terrain,args.output/terrain.name)
        (args.output/(path.stem+'_audit.json')).write_text(json.dumps(metrics,indent=2)+'\n')
        print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    main()