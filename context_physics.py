"""Procedural ground and native physics context control; zero robot assets."""
import faulthandler
import gc
import math
import weakref

def run_physics(event,DT,STEPS):
    import isaaclab.sim as sim_utils
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT,device='cuda:0',render_interval=4))
    event('physics_context_created')
    sim_utils.GroundPlaneCfg().func('/World/Ground',sim_utils.GroundPlaneCfg())
    sim.reset()
    event('context_control_initialized',robot_loaded=False,external_assets_loaded=False,ground_created=True)
    for step in range(STEPS):
        sim.step(render=False)
        current=float(sim.current_time)
        if not math.isfinite(current):raise ValueError('Nonfinite simulation clock')
        if step in (0,15,31,63):event('physics_progress',steps=step+1,sim_time=current,no_robot=True)
    event('physics_steps_completed',steps=STEPS,simulated_seconds=STEPS*DT,
        sim_time_finite=True,no_robot=True,ppo_updates=0)
    event('physics_cleanup_started',phase_seconds=30)
    faulthandler.enable();faulthandler.dump_traceback_later(5,repeat=True)
    try:
        event('physics_stop_started')
        from stop_guard_candidate import guarded_stop
        guarded_stop(sim,event)
        event('physics_stop_returned')
        event('physics_callbacks_clear_started');sim.clear_all_callbacks()
        event('physics_callbacks_clear_returned')
        event('physics_instance_clear_started');sim.clear_instance()
        event('physics_instance_clear_returned')
        singleton_cleared=sim_utils.SimulationContext.instance() is None
        event('singleton_cleared_observed',is_none=singleton_cleared)
        if not singleton_cleared:raise ValueError('Public singleton still active after clear_instance')
        event('physics_cleanup_returned')
    finally:faulthandler.cancel_dump_traceback_later()
    return weakref.ref(sim)

def observe_scope(physics,event):
    ref=physics()
    if not isinstance(ref,weakref.ReferenceType):raise ValueError('Expected a weak measurement only')
    event('physics_scope_returned')
    event('physics_gc_started')
    faulthandler.enable();faulthandler.dump_traceback_later(5,repeat=True)
    try:count=gc.collect()
    finally:faulthandler.cancel_dump_traceback_later()
    # Object liveness is observed, not a claimed framework exit contract.
    event('physics_gc_returned',unreachable_collected=count,sim_owner_released=ref() is None,
        liveness_is_diagnostic_only=True)