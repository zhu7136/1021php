"""One bounded native 31-joint physics diagnostic; no policy/training/install."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import sys
import time
import traceback
import zipfile

PREFIX = 'HUD04_R10_EVENT '
ARCHIVE_SHA = 'ee5aeac12664ba16e7a5298642c88aeb9e544246331f3524f57d73f88d52c433'
STEPS = 64
DT = .005
HEAD_MAX_RAD = .05

def event(event_name, **fields):
    print(PREFIX + json.dumps({'event': event_name, **fields}, sort_keys=True), flush=True)

def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or not p.parts:
        raise ValueError('Unsafe input path')
    return p

def extract_assets(root, output):
    archive = root / 'input.zip'
    if archive.stat().st_size > 20_000_000 or hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('Historical input archive hash mismatch')
    rows = json.loads((root / 'input-manifest.json').read_text())['files']
    expected = {r['path']: r for r in rows if r['path'].startswith('asset/')}
    if len(expected) != 39:
        raise ValueError('Expected 39 immutable robot asset files')
    output.mkdir(exist_ok=False)
    found = set()
    with zipfile.ZipFile(archive) as z:
        if len(z.infolist()) > 100:
            raise ValueError('Input member limit')
        for info in z.infolist():
            p = safe_member(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Input symlink')
            # The existing input ZIP has an input/ prefix. Only exact manifest assets
            # are extracted, never overlay code or the motion/any pickle.
            key = str(p.relative_to('input')) if p.parts[0] == 'input' else str(p)
            if key not in expected:
                continue
            if key in found:
                raise ValueError('Duplicate asset')
            row = expected[key]
            if info.file_size != row['bytes'] or info.file_size > 30_000_000:
                raise ValueError('Asset size mismatch')
            data = z.read(info)
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Asset hash mismatch')
            dest = output / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            found.add(key)
    if found != set(expected):
        raise ValueError('Missing robot asset')
    return output / 'asset/usd/HU_D04_01.usd'

def resource_ok(device):
    # This is a NEW diagnostic-family contract, not a retroactive r7 pass or a
    # resolution of the catalog's different model label. No training authorization.
    return (device['count'] == 1 and device['name'] in
            ('NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090 D', 'NVIDIA GeForce RTX 4090D')
            and 23 * 2**30 <= device['total_memory_bytes'] <= 25 * 2**30
            and (device['major'], device['minor']) == (8, 9))

def child():
    from runtime_versions import check_runtime_versions
    event('versions_verified', evidence=check_runtime_versions(), python=sys.version.split()[0])
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    args.device = 'cuda:0'
    args.enable_cameras = False
    app = None
    try:
        event('app_launch_started')
        app = AppLauncher(args).app
        event('app_launch_returned')
        import carb.settings
        settings = carb.settings.get_settings()
        settings.set_bool('/app/fastShutdown', False)
        if settings.get('/app/fastShutdown') is not False:
            raise ValueError('fastShutdown readback failed')
        event('fast_shutdown_verified', value=False)
        import torch
        p = torch.cuda.get_device_properties(0)
        device = dict(name=p.name, total_memory_bytes=p.total_memory,
                      major=p.major, minor=p.minor, count=torch.cuda.device_count())
        event('resource_contract', **device, passed=resource_ok(device),
              kind='4090_family_24GB_diagnostic_only', catalog_name='4090D',
              catalog_exact_name_match='4090D' in p.name.replace(' ', ''),
              catalog_label_discrepancy_resolved=False)
        if not resource_ok(device):
            raise ValueError('Diagnostic GPU capacity/family contract failed')
        event('physics_import_started')
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from hud04_robot import (SDK_JOINT_NAMES, HEAD_JOINT_NAMES, POLICY_JOINT_NAMES,
                                 build_hud04_implicit_actuators, default_position, resolve_policy_indices)
        event('physics_import_returned')
        root = Path(__file__).resolve().parent
        usd = root / ('assets-' + os.environ['HUD04_OWNED_TASK']) / 'asset/usd/HU_D04_01.usd'
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device='cuda:0', render_interval=4))
        event('physics_context_created')
        sim_utils.GroundPlaneCfg().func('/World/Ground', sim_utils.GroundPlaneCfg())
        cfg = ArticulationCfg(
            prim_path='/World/HuD04',
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., .9046898903895483),
                joint_pos={n: default_position(n) for n in SDK_JOINT_NAMES}, joint_vel={'.*': 0.}),
            actuators=build_hud04_implicit_actuators(), soft_joint_pos_limit_factor=.9)
        robot = Articulation(cfg)
        event('robot_spawn_requested')
        sim.reset()
        robot.reset()
        robot.update(DT)
        names = tuple(robot.joint_names)
        indices = resolve_policy_indices(names)
        heads = tuple(names.index(n) for n in HEAD_JOINT_NAMES)
        if robot.num_instances != 1 or bool(robot.is_fixed_base):
            raise ValueError('Expected exactly one floating-base HuD04')
        event('robot_initialized', joint_names=names, physical_joints=len(names),
              policy_joints=len(indices), policy_names=POLICY_JOINT_NAMES,
              head_indices=heads, policy_indices=indices, floating_base=True, robot_loaded=True)
        root_state = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        targets = robot.data.default_joint_pos.clone()
        targets[:, list(heads)] = 0.
        robot.write_joint_state_to_sim(targets, torch.zeros_like(targets))
        robot.reset()
        head_max = 0.
        root_z_min = float('inf')
        for step in range(STEPS):
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)
            for tensor in [robot.data.joint_pos, robot.data.joint_vel, robot.data.root_state_w]:
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError('Nonfinite native physics state')
            head_max = max(head_max, float(robot.data.joint_pos[:, list(heads)].abs().max()))
            root_z_min = min(root_z_min, float(robot.data.root_pos_w[:, 2].min()))
            if step in (0, 15, 31, 63):
                event('physics_progress', steps=step + 1, head_max_abs_rad=head_max,
                      root_z_min_m=root_z_min, states_finite=True)
        event('physics_steps_completed', steps=STEPS, simulated_seconds=STEPS * DT,
              states_finite=True, head_zero_target=True, head_max_abs_rad=head_max,
              head_threshold_rad=HEAD_MAX_RAD, head_short_probe_passed=head_max <= HEAD_MAX_RAD,
              root_z_min_m=root_z_min, root_z_final_m=float(robot.data.root_pos_w[0, 2]),
              ppo_updates=0, locomotion_or_reference_feasibility_claimed=False)
        if head_max > HEAD_MAX_RAD:
            raise ValueError('Short-probe head displacement exceeds preregistered 0.05rad threshold')
        event('physics_cleanup_started', phase_seconds=30)
        faulthandler.enable()
        faulthandler.dump_traceback_later(5, repeat=True)
        try:
            event('physics_stop_started')
            from stop_guard_candidate import guarded_stop
            guarded_stop(sim,event)
            event('physics_stop_returned')
            event('physics_callbacks_clear_started')
            sim.clear_all_callbacks()
            event('physics_callbacks_clear_returned')
            event('physics_instance_clear_started')
            sim.clear_instance()
            event('physics_instance_clear_returned')
            event('physics_cleanup_returned')
        finally:
            faulthandler.cancel_dump_traceback_later()
    except BaseException as error:
        event('child_failure', error_type=type(error).__name__, error_message=str(error))
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise
    finally:
        if app is not None:
            event('close_started')
            faulthandler.enable()
            faulthandler.dump_traceback_later(5, repeat=True)
            try:
                import carb.settings
                carb.settings.get_settings().set_bool('/app/fastShutdown', False)
                app.close(wait_for_replicator=False)
                event('close_returned')
            finally:
                faulthandler.cancel_dump_traceback_later()

def accepted(events, returncode, reason):
    rows = {r['event']: r for r in events}
    required = ['versions_verified', 'app_launch_returned', 'fast_shutdown_verified',
                'resource_contract', 'physics_context_created', 'robot_initialized',
                'physics_steps_completed', 'physics_cleanup_started', 'physics_stop_started',
                'stop_guard_source_verified', 'stop_interactive_wait_disabled', 'guarded_stop_returned',
                'stop_interactive_wait_guard_restored', 'physics_stop_returned', 'physics_callbacks_clear_started', 'physics_callbacks_clear_returned',
                'physics_instance_clear_started', 'physics_instance_clear_returned',
                'physics_cleanup_returned', 'close_started', 'close_returned']
    if reason != 'child_exited' or returncode != 0 or 'child_failure' in rows or any(n not in rows for n in required):
        return False
    r, s = rows['robot_initialized'], rows['physics_steps_completed']
    names = [r['event'] for r in events]
    return (rows['resource_contract'].get('passed') is True and
            rows['fast_shutdown_verified'].get('value') is False and
            r.get('physical_joints') == 31 and r.get('policy_joints') == 29 and
            r.get('floating_base') is True and s.get('steps') == STEPS and
            s.get('states_finite') is True and s.get('head_short_probe_passed') is True and
            rows['stop_guard_source_verified'].get('callback_body_sha256') ==
            'd1dc5a0460a8af6df91690d020d5ccd8023a9c578f28a080efb9a83fb57bf93c' and
            rows['stop_interactive_wait_disabled'].get('value') is True and
            rows['stop_interactive_wait_guard_restored'].get('value') is False and
            all(names.index(a) < names.index(b) for a,b in zip(required, required[1:])))

def supervise(command, cwd, env, out, seconds, cleanup_seconds=30):
    out.mkdir(exist_ok=False)
    end = time.monotonic() + seconds
    start = time.monotonic()
    events, offset, pending, close_end = [], 0, b'', None
    code, reason, proc = None, 'not_started', None
    def scan():
        nonlocal offset, pending, close_end
        with (out / 'child.log').open('rb') as stream:
            stream.seek(offset)
            chunk = stream.read(1024 * 1024)
            offset += len(chunk)
        pending += chunk
        lines = pending.split(b'\n')
        pending = lines.pop()
        if len(pending) > 2 * 1024 * 1024:
            raise ValueError('Oversized owned log line')
        for line in lines:
            if line.startswith(PREFIX.encode()):
                row = json.loads(line[len(PREFIX):])
                events.append(row)
                if len(events) > 80:
                    raise ValueError('Event limit')
                print(PREFIX + json.dumps(row, sort_keys=True), flush=True)
                # One shared cleanup/close deadline: reaching close does NOT reset it.
                if row.get('event') in ('physics_cleanup_started', 'close_started') and close_end is None:
                    close_end = min(end - 5, time.monotonic() + cleanup_seconds)
    def interrupted(*_):
        raise InterruptedError('Parent termination')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out / 'child.log').open('xb') as log:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                scan()
                if (out / 'child.log').stat().st_size > 32 * 1024 * 1024:
                    reason = 'log_limit'; break
                code = proc.poll()
                if code is not None:
                    reason = 'child_exited'; scan(); break
                if time.monotonic() >= min(end - 5, close_end or end - 5):
                    reason = ('cleanup_close_timeout' if any(r['event']=='physics_cleanup_started' for r in events)
                              else 'close_timeout') if close_end else 'total_timeout'; break
                time.sleep(.1)
    except BaseException as error:
        reason = type(error).__name__
    finally:
        if proc is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if proc.poll() is not None: break
                try: os.killpg(proc.pid, sig)
                except ProcessLookupError: pass
                try: proc.wait(timeout=2)
                except subprocess.TimeoutExpired: pass
            code = proc.poll()
        signal.signal(signal.SIGTERM, previous)
    result = dict(passed=accepted(events, code, reason), reason=reason, returncode=code,
                  events=events, elapsed_seconds=round(time.monotonic() - start, 3), ppo_updates=0,
                  verified_current31_checkpoint=False)
    if not result['passed']:
        with (out / 'child.log').open('rb') as f:
            f.seek(max(0, (out / 'child.log').stat().st_size - 16000))
            print('HUD04_R10_OWNED_LOG_TAIL\n' + f.read(16000).decode(errors='replace'), flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    event('parent_result', **{k: v for k, v in result.items() if k != 'events'})
    return result

def main():
    started = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument('--child', action='store_true')
    p.add_argument('--execute', action='store_true')
    for name in ('commit', 'task-id', 'thread-id', 'turn-id'): p.add_argument('--' + name)
    args = p.parse_args()
    if args.child: return child()
    if not args.execute:
        event('read_only', execution_requested=False, gpu_started=False); return
    if not all([args.commit, args.task_id, args.thread_id, args.turn_id]) or args.task_id == 'PENDING':
        raise ValueError('Task/provenance binding required')
    root = Path(__file__).resolve().parent
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=5).strip() != args.commit:
        raise ValueError('Source commit mismatch')
    for row in json.loads((root / 'manifest.json').read_text())['files']:
        f = root / safe_member(row['path'])
        if f.is_symlink() or f.stat().st_size != row['bytes'] or hashlib.sha256(f.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Immutable input/source mismatch')
    usd = extract_assets(root, root / ('assets-' + args.task_id))
    event('assets_verified', files=39, archive_sha256=ARCHIVE_SHA, usd=str(usd), no_motion_loaded=True)
    python = next((f for f in [Path('/isaac-sim/python.sh'), Path('/workspace/isaaclab/_isaac_sim/python.sh')]
                   if f.is_file() and os.access(f, os.X_OK) and (f.parent / 'kit/python/bin/python3').is_file()), None)
    if python is None: raise ValueError('Vendor Python unavailable')
    keep = {'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME', 'ISAAC_PATH', 'ISAACSIM_PATH', 'ISAACLAB_PATH',
            'CARB_APP_PATH', 'EXP_PATH', 'HOME', 'USER', 'TMPDIR', 'DISPLAY', 'LANG', 'ACCEPT_EULA', 'PRIVACY_CONSENT'}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(PYTHONUNBUFFERED='1', PYTHONHASHSEED='42', WANDB_MODE='disabled', CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HUD04_OWNED_TASK=args.task_id)
    event('parent_started', task_id=args.task_id, commit=args.commit, worker_seconds=360,
          automatic_retries=0, physics_steps_planned=STEPS, ppo_updates=0,
          provenance={'thread_id': args.thread_id, 'turn_id': args.turn_id})
    remaining = 360 - (time.monotonic() - started)
    if remaining < 40: raise TimeoutError('Insufficient physics diagnostic budget')
    result = supervise([str(python), str(Path(__file__).resolve()), '--child'],
                       root, env, root / ('diagnostic-' + args.task_id), remaining)
    if not result['passed']: raise SystemExit(1)

if __name__ == '__main__': main()