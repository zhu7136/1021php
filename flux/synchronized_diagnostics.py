"""Opt-in, read-only first-episode telemetry for the frozen IsaacSim evaluation.

Physics wrappers call the native functions exactly once and consume no RNG.
Implicit-PD torque reports are estimates, NOT measured solver joint efforts.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

FEET=('left_ankle_roll_link','right_ankle_roll_link')

def snapshot(value,shape,name):
    if not torch.is_tensor(value) or tuple(value.shape)!=tuple(shape):
        raise ValueError('Diagnostic tensor shape: '+name)
    if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite diagnostic: '+name)
    return value.detach().clone()

def serial(value):
    if torch.is_tensor(value):
        value=value.detach().cpu()
        if value.is_floating_point():value=value.to(torch.float64).round(decimals=6)
        return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value

class SynchronizedDiagnostics:
    """Strict 32-env/128-step adapter, installed after FirstEpisodeCapture.

    Control rows: after termination checks, before reward/reset/command advance.
    Substep rows: q/qd before native simulation, q/qd/contact after it, target
    and actuator report used for that substep. A push after the boundary is an
    event affecting the NEXT control step, never blamed on the previous one.
    """
    def __init__(self,env,motion,stats,directory):
        if env.num_envs!=32 or env._update_tasks_before_termination:
            raise ValueError('Requires 32-env IsaacSim post-termination command ordering')
        self.env,self.motion,self.stats=env,motion,stats
        self.n=32;self.j=len(env.robot_config.dof_names)
        self.joint_names=list(env.robot_config.dof_names)
        self.ankles=[i for i,n in enumerate(self.joint_names) if n in (
            'left_ankle_pitch_joint','left_ankle_roll_joint',
            'right_ankle_pitch_joint','right_ankle_roll_joint')]
        if len(self.ankles)!=4:raise ValueError('Expected four ankle joints')
        sim=env.simulator
        ids=list(sim.dof_ids)
        if len(set(ids))!=self.j or [sim._robot.joint_names[i] for i in ids]!=self.joint_names:
            raise ValueError('Native DOF mapping mismatch')
        self.body_names=list(sim.body_names)
        self.feet=[self.body_names.index(n) for n in FEET]
        self.tracked=motion.tracked_body_indexes
        self.tracked_names=[self.body_names[int(i)] for i in self.tracked]
        terms=[t for t in env.action_manager._term_instances.values() if hasattr(t,'_requested_position_targets')]
        if len(terms)!=1:raise ValueError('One protected position-target term required')
        self.action=terms[0]
        self.decimation=int(sim.simulator_config.sim.control_decimation_steps)
        self.physics_dt=1./float(sim.simulator_config.sim.fps)
        if not 1<=self.decimation<=16 or abs(self.physics_dt*self.decimation-float(env.dt))>1e-8:
            raise ValueError('Diagnostic physics/control clock mismatch')
        self.contact_period=float(sim.contact_sensor.cfg.update_period)
        if not 0<=self.contact_period<=self.physics_dt+1e-8:
            raise ValueError('Contact sensor is slower than physics; do not label stale data as substep-synchronous')
        self.steps=0;self.samples=0;self.subsamples=0;self.substep=0
        self.in_physics=False;self.before=None;self.active_ids=None
        self.alive=torch.zeros(self.n,dtype=torch.bool,device=env.device)
        self.pending=[];self.event_count=0;self.missing=set();self.closed=False
        self.originals=[]
        directory=Path(directory);directory.mkdir(exist_ok=True)
        if list(directory.glob('sync_*')):raise ValueError('Fresh diagnostic output required')
        self.directory=directory
        self.bytes_written=0
        self.sinks={name:(directory/('sync_'+name+'.jsonl')).open('x',buffering=1)
                    for name in ['control_steps','physics_substeps','push_events']}
        metadata=dict(schema_version=1,diagnostic_only=True,num_envs=self.n,
            joint_names=[self.joint_names[i] for i in self.ankles],tracked_body_names=self.tracked_names,
            contact_body_names=list(FEET),coordinate_frame='world',
            error_sign='actual_minus_reference',control_dt_s=float(env.dt),
            physics_dt_s=self.physics_dt,control_decimation=self.decimation,
            contact_sensor_update_period_s=self.contact_period,
            control_sample_phase='post_termination_pre_reward_pre_reset_pre_command_advance',
            target_frame='native body_pos_relative_w, aligned in reference-body XY/yaw',
            torque_source='IsaacLab Articulation.data.applied_torque / computed_torque',
            torque_semantics='implicit-PD actuator estimate; NOT measured solver joint effort',
            measured_solver_joint_effort_available=False,
            float_serialization='rounded to 6 decimals in native units; validation uses unrounded tensors',
            raw_jsonl_bytes_cap=16*1024*1024,
            contact_semantics='net world force on ankle-roll link, NOT contact point or terrain-only force',
            contact_flag='norm(net_force)>1N diagnostic convention; not a termination condition',
            slip_proxy='ankle-link horizontal speed when force norm>1N; NOT contact-point slip measurement',
            unavailable=['measured solver joint effort','contact-point velocity','reference stance/contact annotation'])
        self.write_json('schema.json',metadata)
        try:self.install()
        except Exception:self.close();raise

    def write_json(self,name,value):
        with (self.directory/('sync_'+name)).open('x') as f:json.dump(serial(value),f,allow_nan=False,indent=2)

    def emit(self,name,row):
        line=json.dumps(serial(row),allow_nan=False,separators=(',',':'))+'\n'
        self.bytes_written+=len(line.encode())
        if self.bytes_written>16*1024*1024:raise ValueError('Diagnostic output byte budget exceeded; no truncation')
        self.sinks[name].write(line)

    def read(self,value,shape,name):return snapshot(value,shape,name)

    def optional_joint_report(self,name):
        value=getattr(self.env.simulator._robot.data,name,None)
        if value is None:
            self.missing.add(name);return None
        value=self.read(value,(self.n,len(self.env.simulator._robot.joint_names)),name)
        return value[:,self.env.simulator.dof_ids][self.active_ids][:,self.ankles]

    def reference(self):
        m=self.motion;b=len(self.tracked_names)
        return dict(motion_frame=self.read(m.time_steps,(self.n,),'motion_frame'),
            motion_id=self.read(m.motion_ids,(self.n,),'motion_id'),
            reference_q_rad=self.read(m.joint_pos,(self.n,self.j),'reference_q'),
            reference_qd_rad_s=self.read(m.joint_vel,(self.n,self.j),'reference_qd'),
            aligned_reference_body_xyz_m=self.read(m.body_pos_relative_w,(self.n,b,3),'aligned_body_reference'),
            raw_reference_body_xyz_m=self.read(m.body_pos_w,(self.n,b,3),'raw_body_reference'),
            reference_torso_xyz_m=self.read(m.ref_pos_w,(self.n,3),'reference_torso'))

    def select(self,row):
        return {k:(v[self.active_ids][:,self.ankles] if k in ('reference_q_rad','reference_qd_rad_s')
                   else v[self.active_ids]) for k,v in row.items()}

    def replace(self,obj,name,fn):
        original=getattr(obj,name);self.originals.append((obj,name,original));setattr(obj,name,fn(original))

    def install(self):
        def physics(original):
            def call(*args,**kwargs):
                if not self.stats.active:return original(*args,**kwargs)
                if self.steps>=128:raise ValueError('Diagnostic step budget exceeded')
                mask=self.stats.mask
                if mask is None or mask.shape!=(self.n,):raise ValueError('First-episode mask required')
                self.active_ids=mask.nonzero().flatten();self.alive=mask.clone()
                self.before=self.reference();self.substep=0
                self.input_push_ids=self.pending;self.pending=[]
                self.in_physics=True
                try:result=original(*args,**kwargs)
                finally:self.in_physics=False
                if self.substep!=self.decimation:raise ValueError('Incomplete physics substep capture')
                return result
            return call
        def simulate(original):
            def call(*args,**kwargs):
                if not self.in_physics:return original(*args,**kwargs)
                sim=self.env.simulator;i=self.active_ids;n=self.n;j=self.j
                q=self.read(sim.dof_pos,(n,j),'q_before')[i]
                qd=self.read(sim.dof_vel,(n,j),'qd_before')[i]
                target=self.read(self.action._applied_position_targets,(n,j),'applied_target')[i]
                requested=self.read(self.action._requested_position_targets,(n,j),'requested_target')[i]
                native=sim._robot.data.joint_pos_target[:,sim.dof_ids][i]
                if not torch.equal(target,native):raise ValueError('Native target mismatch')
                result=original(*args,**kwargs);self.substep+=1
                if self.substep>self.decimation:raise ValueError('Too many physics substeps')
                forces=self.read(sim.contact_sensor.data.net_forces_w,
                    (n,len(sim.contact_sensor.body_names),3),'contact_force')
                sensor_ids=[sim.contact_sensor.body_names.index(name) for name in FEET]
                force=forces[i][:,sensor_ids]
                row=dict(step=self.steps+1,substep=self.substep,active_envs=i,
                    time_start_s=self.steps*float(self.env.dt)+(self.substep-1)*self.physics_dt,
                    time_end_s=self.steps*float(self.env.dt)+self.substep*self.physics_dt,
                    q_before_rad=q[:,self.ankles],qd_before_rad_s=qd[:,self.ankles],
                    q_after_rad=self.read(sim.dof_pos,(n,j),'q_after')[i][:,self.ankles],
                    qd_after_rad_s=self.read(sim.dof_vel,(n,j),'qd_after')[i][:,self.ankles],
                    requested_target_rad=requested[:,self.ankles],applied_target_rad=target[:,self.ankles],
                    actuator_applied_torque_estimate_nm=self.optional_joint_report('applied_torque'),
                    actuator_computed_torque_estimate_nm=self.optional_joint_report('computed_torque'),
                    measured_solver_joint_effort_nm=None,
                    foot_net_force_w_n=force,foot_net_force_over_1n=force.norm(dim=-1)>1.)
                self.emit('physics_substeps',row);self.subsamples+=len(i)
                return result
            return call
        def check(original):
            def call(*args,**kwargs):
                result=original(*args,**kwargs)
                if self.stats.active:self.observe()
                return result
            return call
        def push(original):
            def call(env_ids,*args,**kwargs):
                # Preserve native randomization calls/IDs/RNG even for excluded envs.
                log_ids=env_ids[self.alive[env_ids]] if self.stats.active else env_ids[:0]
                before=self.env.simulator.robot_root_states[log_ids,7:13].detach().clone()
                result=original(env_ids,*args,**kwargs)
                if len(log_ids):
                    after=self.env.simulator.robot_root_states[log_ids,7:13].detach().clone()
                    self.event_count+=1;event_id=self.event_count;self.pending.append(event_id)
                    self.emit('push_events',dict(event_id=event_id,active_envs=log_ids,
                        after_control_step=self.steps,affects_control_step=self.steps+1,
                        boundary_time_s=self.steps*float(self.env.dt),
                        motion_frame=self.motion.time_steps[log_ids],motion_id=self.motion.motion_ids[log_ids],
                        native_returned=True,root_velocity_before=before,root_velocity_after=after,
                        root_velocity_delta=after-before,
                        units=['m/s','m/s','m/s','rad/s','rad/s','rad/s']))
                return result
            return call
        self.replace(self.env,'_physics_step',physics)
        self.replace(self.env.simulator,'simulate_at_each_physics_step',simulate)
        self.replace(self.env,'_check_termination',check)
        self.replace(self.env,'_push_robots',push)

    @torch.no_grad()
    def observe(self):
        if self.before is None or self.substep!=self.decimation:
            raise ValueError('Control boundary without matching physics samples')
        self.steps+=1;env=self.env;sim=env.simulator;i=self.active_ids;n=self.n;j=self.j
        ref=self.reference()
        for key in ref:
            if not torch.equal(ref[key][i],self.before[key][i]):
                raise ValueError('Reference advanced before diagnostic boundary: '+key)
        actual=self.read(self.motion.robot_body_pos_w,(n,len(self.tracked_names),3),'body_actual')
        error=actual-ref['aligned_reference_body_xyz_m']
        body_vel=self.read(self.motion.robot_body_lin_vel_w,(n,len(self.tracked_names),3),'body_velocity')
        contact=self.read(sim.contact_forces,(n,len(self.body_names),3),'mapped_contact')[:,self.feet][i]
        feet_vel=self.read(sim._rigid_body_vel,(n,len(self.body_names),3),'foot_velocity')[:,self.feet][i]
        q=self.read(sim.dof_pos,(n,j),'q')[i];qd=self.read(sim.dof_vel,(n,j),'qd')[i]
        applied=self.read(self.action._applied_position_targets,(n,j),'target')[i]
        requested=self.read(self.action._requested_position_targets,(n,j),'requested')[i]
        self.emit('control_steps',dict(step=self.steps,seconds=self.steps*float(env.dt),active_envs=i,
            **self.select(ref),actual_body_xyz_m=actual[i],body_error_xyz_m=error[i],
            body_error_norm_m=error[i].norm(dim=-1),actual_body_velocity_w_m_s=body_vel[i],
            torso_error_xyz_m=(self.motion.robot_ref_pos_w-self.motion.ref_pos_w)[i],
            q_rad=q[:,self.ankles],qd_rad_s=qd[:,self.ankles],
            requested_target_rad=requested[:,self.ankles],applied_target_rad=applied[:,self.ankles],
            q_minus_applied_target_rad=(q-applied)[:,self.ankles],
            q_minus_reference_rad=(q-ref['reference_q_rad'][i])[:,self.ankles],
            foot_net_force_w_n=contact,foot_net_force_over_1n=contact.norm(dim=-1)>1.,
            foot_link_velocity_w_m_s=feet_vel,foot_link_horizontal_speed_m_s=feet_vel[:,:,:2].norm(dim=-1),
            input_push_event_ids=self.input_push_ids,
            native_termination_flags={k:v[i] for k,v in self.stats.terms.items()},done=env.reset_buf[i].bool()))
        if self.steps==1:
            static={}
            for name in ['default_dof_pos','default_dof_pos_base']:
                value=getattr(env,name,None)
                if value is not None:static[name]=value.detach().clone()
            static.update(joint_names=self.joint_names,torque_limits_nm=sim.torque_limits,
                velocity_limits_rad_s=sim.dof_vel_limits,
                note='default_dof_pos offset is a target/rest-pose offset; not a measured sensor bias')
            self.write_json('static_state.json',static)
        self.samples+=len(i)
        self.alive=self.stats.mask&~env.reset_buf.bool()
        self.before=None

    def close(self):
        if self.closed:return
        for obj,name,original in reversed(self.originals):setattr(obj,name,original)
        for sink in self.sinks.values():sink.close()
        self.closed=True

    def finish(self):
        self.close()
        if self.steps!=128 or self.subsamples!=self.samples*self.decimation:
            raise ValueError('Incomplete synchronized diagnostic capture')
        result=dict(control_steps=self.steps,first_episode_env_steps=self.samples,
            physics_env_substeps=self.subsamples,push_events=self.event_count,
            missing_actuator_reports=sorted(self.missing),measured_solver_joint_effort_available=False,
            no_training=True,gpu_validation='not implied by CPU tests')
        self.write_json('coverage.json',result)
        return result