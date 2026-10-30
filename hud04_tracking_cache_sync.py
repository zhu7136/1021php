"""Evaluation-only cache scheduling repair; reuse the pinned native pose math.

No copied simulator implementation, time advancement, physics reset, or relaxed
termination. The exact native target-cache block is extracted at runtime and
called just before the unchanged native termination check.
"""
import ast
import copy
import hashlib
import inspect
import textwrap
from pathlib import Path

NATIVE_SOURCE_SHA256='4acfdb0d14e2877a1503bfb4508c6751db97d064ff270e99bdf74f489fe45d0a'

def assigned(node,name):
    return isinstance(node,ast.Assign) and any(ast.unparse(t)==name for t in node.targets)

def extract_cache_block(source,namespace):
    """Only reuse the native pose-cache block, excluding clocks and sampler."""
    tree=ast.parse(textwrap.dedent(source))
    step=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    start=next(i for i,n in enumerate(step.body) if assigned(n,'use_root'))
    end=next(i for i,n in enumerate(step.body) if assigned(n,'self.body_pos_relative_w'))
    body=copy.deepcopy(step.body[start:end+1])
    for node in ast.walk(ast.Module(body=body,type_ignores=[])):
        if isinstance(node,(ast.AugAssign,ast.For,ast.While,ast.If,ast.With,ast.Try,ast.Return,ast.Raise)):
            raise ValueError('Unexpected native target-cache block')
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self':
                    if target.attr not in ('body_pos_relative_w','body_quat_relative_w'):
                        raise ValueError('Target block mutates unrelated native state')
    expected={'self.body_pos_relative_w','self.body_quat_relative_w'}
    found={ast.unparse(t) for n in body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Attribute)}
    if found!=expected:raise ValueError('Both native pose caches must be refreshed')
    f=ast.FunctionDef(name='refresh_targets',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='self')],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]))
    scope=dict(namespace)
    exec(compile(module,'<pinned-native-tracking-cache>','exec'),scope)
    return scope['refresh_targets']

def native_refresh():
    from holosoma.managers.command.terms.wbt import MotionCommand
    path=Path(inspect.getsourcefile(MotionCommand.step))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=NATIVE_SOURCE_SHA256:
        raise ValueError('Native MotionCommand source changed; refusing unreviewed patch')
    return extract_cache_block(inspect.getsource(MotionCommand.step),MotionCommand.step.__globals__)

def install_cache_sync(env,refresh):
    if env.num_envs!=1 or not env.is_evaluating:raise ValueError('Single-environment evaluation only')
    if hasattr(env,'hud04_cache_sync'):raise RuntimeError('Cache-sync already installed')
    original=env._check_termination
    state={'mode':'refresh_before_native_termination','checks':0,'native_source_sha256':NATIVE_SOURCE_SHA256,
        'reference_time_advanced_by_patch':False,'thresholds_changed':False}
    def synchronized():
        if not env.is_evaluating:raise PermissionError('Cache scheduling patch is not approved for training')
        command=env.command_manager.get_state('motion_command')
        before=command.time_steps.clone()
        refresh(command)
        if not command.time_steps.equal(before):raise RuntimeError('Cache refresh changed reference time')
        state['checks']+=1
        return original()
    env.hud04_cache_sync=state
    env._check_termination=synchronized
    return state