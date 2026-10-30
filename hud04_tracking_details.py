"""Observe native BadTracking sub-checks once; never refresh or relax them."""
import torch

METHODS=('bad_ref_pos','bad_ref_ori','bad_motion_body_pos','bad_object_pos','bad_object_ori')

def serial(value):
    return value.detach().cpu().tolist()

def tracking_inputs(target,command):
    names=list(target.body_names_to_track)
    indices=target.bad_motion_body_pos_body_indexes
    desired=command.body_pos_relative_w[:,indices]
    actual=command.robot_body_pos_w[:,indices]
    error=torch.linalg.vector_norm(desired-actual,dim=-1)
    z_only=any(c.__name__ in ('BadTrackingBodyPositionZOnly','BadTrackingZOnly') for c in type(target).__mro__)
    effective_error=(desired-actual).abs()[...,-1] if z_only else error
    selected=[names[int(i)] for i in indices.detach().cpu().tolist()]
    return {
        'native_term_class':type(target).__module__+'.'+type(target).__qualname__,
        'reference_frame':serial(command.time_steps),
        'ref_pos_w':serial(command.ref_pos_w),'robot_ref_pos_w':serial(command.robot_ref_pos_w),
        'ref_quat_xyzw':serial(command.ref_quat_w),'robot_ref_quat_xyzw':serial(command.robot_ref_quat_w),
        'root_position_error_l2_m':serial(torch.linalg.vector_norm(command.ref_pos_w-command.robot_ref_pos_w,dim=-1)),
        'root_position_error_z_m':serial((command.ref_pos_w-command.robot_ref_pos_w).abs()[...,-1]),
        'reference_position_threshold':float(target.bad_ref_pos_threshold),
        'reference_gravity_z_difference_threshold':float(target.bad_ref_ori_threshold),
        'body_position_threshold_m':float(target.bad_motion_body_pos_threshold),
        'tracked_end_effector_names':selected,'body_error_mode':'z_only' if z_only else 'l2',
        'cached_body_targets_w':serial(desired),'robot_bodies_w':serial(actual),
        'body_error_l2_m':serial(error),'body_error_effective_m':serial(effective_error),
        'note':'Targets are native cached values read without refreshing; actual component booleans are recorded separately.'}

class TrackingProxy:
    def __init__(self,target,values,details):
        self.target=target;self.values=values;self.details=details
    def __getattr__(self,name):return getattr(self.target,name)
    def __call__(self,*args,**kwargs):
        target=self.target;restores=[];components={};detail={}
        def wrap(name,bound):
            def observed(*a,**kw):
                result=bound(*a,**kw)
                components[name]=serial(result)
                return result
            return observed
        try:
            for name in METHODS:
                if not hasattr(target,name):continue
                original=getattr(target,name)
                restores.append((name,name in target.__dict__,target.__dict__.get(name)))
                setattr(target,name,wrap(name,original))
            result=target(*args,**kwargs)
            # Observation failure is explicitly recorded, never changes a native
            # termination result or turns an invalid run into a valid one.
            try:
                env=args[0] if args else target.env
                detail=tracking_inputs(target,env.command_manager.get_state('motion_command'))
            except Exception as exc:
                detail={'observation_error_type':type(exc).__name__}
            detail['native_component_results']=components
            self.details(detail);self.values(serial(result))
            return result
        finally:
            for name,was_local,value in reversed(restores):
                if was_local:setattr(target,name,value)
                else:delattr(target,name)