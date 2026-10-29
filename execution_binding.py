"""Validate one task-specific private approval against an immutable public template."""
from copy import deepcopy
import re


def bind_execution(template, requested, task_id, scope):
    if not isinstance(task_id,str) or not re.fullmatch(r'TASK_[0-9_]+',task_id):
        raise ValueError('Invalid task identity')
    if (template.get('scope')!=scope or template.get('task_id') is not None or
        template.get('publication_only') is not True or
        template.get('plan',{}).get('execution_approved') is not False or
        template.get('approval',{}).get('confirmed_by_user') is not False):
        raise ValueError('Unapproved public template required')
    agent=requested.get('plan',{}).get('_wb_agent',{})
    if (not isinstance(agent,dict) or set(agent)!={'thread_id','turn_id'} or
        not all(isinstance(v,str) and 0<len(v)<=200 for v in agent.values())):
        raise ValueError('Current conversation provenance required')
    expected=deepcopy(template)
    expected['task_id']=task_id
    expected['publication_only']=False
    expected['approval'].update(confirmed_by_user=True,max_tasks=1)
    expected['plan'].update(execution_approved=True,_wb_agent=agent)
    gate='compute_budget_requires_fresh_user_approval'
    if expected['plan']['execution_gates'].count(gate)!=1:
        raise ValueError('Fresh compute approval gate required')
    expected['plan']['execution_gates'].remove(gate)
    risk=expected.get('engineering_validation',{})
    unresolved=risk.get('unresolved_gates',[])
    remaining=expected['plan']['execution_gates']
    if len(unresolved)!=len(remaining) or set(unresolved)!=set(remaining):
        raise ValueError('Engineering approval must preserve every unresolved native gate')
    risk['unresolved_gates']=list(remaining)
    if requested!=expected:
        raise ValueError('Only current task approval and provenance may change')
    return deepcopy(requested)