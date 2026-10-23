"""Bounded sequential arm IK with temporal warm starts, for CPU preprocessing."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class Arm:
    def __init__(self, model, side):
        self.model=model
        self.shoulder=f'{side}_shoulder_pitch_link'
        self.body=f'{side}_wrist_roll_link'
        parents={j['child']:j for j in model.joints}
        chain=[];body=self.body
        while True:
            joint=parents[body];chain.append(joint)
            if body==self.shoulder:
                break
            body=joint['parent']
        self.chain=chain[::-1];self.base=self.chain[0]['parent']
        self.cols=np.array([model.indices[j['name']] for j in self.chain])
        self.lower=model.lower[self.cols]+1e-6
        self.upper=model.upper[self.cols]-1e-6

    def forward(self, x, base_pos, base_rot):
        p=base_pos.copy();r=base_rot.copy();origins=[];axes=[]
        for angle,joint in zip(x,self.chain):
            p=p+r@joint['xyz'];r=r@joint['R']
            origins.append(p.copy());axes.append(r@joint['axis'])
            r=r@Rotation.from_rotvec(angle*joint['axis']).as_matrix()
        jac=np.cross(np.array(axes),p-np.array(origins)).T
        return p,jac


def sequential_arms(model,g1,data,q,root,quat,bodies,regularization=.003):
    """Preserve exact v7 endpoint targets; change the solver, not acceptance gates."""
    q=q.copy();p,r,_,_=model.forward(q,root,quat);diagnostics={}
    for side in ['left','right']:
        arm=Arm(model,side);source_wrist=f'{side}_wrist_yaw_link'
        scale=model.chain_length(arm.shoulder,arm.body)/g1.chain_length(arm.shoulder,source_wrist)
        target=p[arm.shoulder]+scale*(data['body_pos_w'][:,bodies.index(source_wrist)]-data['body_pos_w'][:,bodies.index(arm.shoulder)])
        previous=np.clip(q[0,arm.cols],arm.lower,arm.upper)
        residuals=[];evaluations=0
        for i in range(len(q)):
            cached_x=None;cached_p=None;cached_j=None
            def evaluate(x):
                nonlocal cached_x,cached_p,cached_j
                if cached_x is None or not np.array_equal(cached_x,x):
                    cached_p,cached_j=arm.forward(x,p[arm.base][i],r[arm.base][i]);cached_x=x.copy()
                return cached_p,cached_j
            def fun(x):
                point,_=evaluate(x)
                return np.r_[point-target[i],regularization*(x-previous)]
            def jac(x):
                _,matrix=evaluate(x)
                return np.vstack([matrix,regularization*np.eye(len(x))])
            fit=least_squares(fun,previous,jac=jac,bounds=(arm.lower,arm.upper),max_nfev=60,gtol=1e-9,ftol=1e-9,xtol=1e-9)
            q[i,arm.cols]=fit.x;previous=fit.x
            residuals.append(float(np.linalg.norm(fit.fun[:3])));evaluations+=fit.nfev
        diagnostics[side]={'max_source_grid_hand_error_m':max(residuals),'mean_source_grid_hand_error_m':float(np.mean(residuals)),'function_evaluations':evaluations,'max_source_grid_joint_step_rad':float(np.max(np.abs(np.diff(q[:,arm.cols],axis=0))))}
    return q,diagnostics