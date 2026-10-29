"""Exact scope of the separately approved 2026-10-29 bounded continuation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

CHECKPOINT_SHA256='e295f4acb723bb93e0cb8946b49c81e242dcaf73a84c5dc981486f1725ce1b15'
CHECKPOINT_BYTES=45006507
SCOPE={'max_tasks':1,'max_training_processes':1,'num_envs':10,'training_updates':1000,
       'save_interval':250,'worker_seconds_max':1800,'automatic_retries':0,'fresh_student':False,'resume_completed_updates':100}
VALIDATION={'purpose':'authorized_bounded_optimizer_continuation','max_additional_updates':1000,
    'start_completed_updates':100,'source_checkpoint_sha256':CHECKPOINT_SHA256,
    'shutdown_fault_accepted_not_hidden':True,'trajectory_exact_resume':False}


def make_template(base):
    value=deepcopy(base)
    if value['plan']['execution_approved'] is not False or value['publication_only'] is not True:raise ValueError('Unapproved base required')
    value['scope']=deepcopy(SCOPE)
    value['plan'].update(additional_updates=1000,save_interval=250,resume_policy='strict optimizer continuation from verified 100-update checkpoint; trajectory reset, not exact replay')
    value['approval'].update(additional_updates=1000)
    # A prepared template is not compute authorization. These switches become
    # true only in a separately approved, task-specific binding.
    value['engineering_validation'].update(long_training_authorized=False,resume_authorized=False)
    value['continuation_validation']=deepcopy(VALIDATION)
    value['resume_sha256']=CHECKPOINT_SHA256
    return value


def bind(template,requested,task):
    if template['scope']!=SCOPE or template['continuation_validation']!=VALIDATION:raise ValueError('Continuation scope identity')
    if not re.fullmatch(r'TASK_[0-9_]+',task):raise ValueError('Task identity')
    if template['publication_only'] is not True or template['task_id'] is not None or template['plan']['execution_approved'] is not False or template['approval']['confirmed_by_user'] is not False:raise ValueError('Unapproved template required')
    agent=requested.get('plan',{}).get('_wb_agent',{})
    if set(agent)!={'thread_id','turn_id'} or not all(isinstance(v,str) and 0<len(v)<=200 for v in agent.values()):raise ValueError('Current provenance')
    expected=deepcopy(template);expected.update(task_id=task,publication_only=False)
    expected['plan'].update(execution_approved=True,_wb_agent=agent)
    expected['approval'].update(confirmed_by_user=True,max_tasks=1)
    expected['engineering_validation'].update(long_training_authorized=True,resume_authorized=True)
    gate='compute_budget_requires_fresh_user_approval'
    if expected['plan']['execution_gates'].count(gate)!=1:raise ValueError('Fresh approval gate')
    expected['plan']['execution_gates'].remove(gate)
    remaining=expected['plan']['execution_gates'];risk=expected['engineering_validation']
    if len(risk['unresolved_gates'])!=len(remaining) or set(risk['unresolved_gates'])!=set(remaining):raise ValueError('Preserve native risks')
    risk['unresolved_gates']=list(remaining)
    if requested!=expected:raise ValueError('Continuation authorization escalation')
    return deepcopy(requested)


def validate_outputs(out,request):
    from .formal_core import continuation_contract,contract_digest
    out=Path(out).resolve();result=json.loads((out/'formal-native-result.json').read_text())
    restored=json.loads((out/'formal-resume-verification.json').read_text())
    if restored!={'checkpoint_sha256':CHECKPOINT_SHA256,'start_completed_updates':100,'strict_model_restore':True,'optimizer_restore_and_readback':True,'rng_restore':True,'trajectory_exact_resume':False,'_wb_agent':request['plan']['_wb_agent']}:raise ValueError('Strict resume evidence')
    added=result['additional_updates_completed']
    if type(added) is not int or not 1<=added<=1000 or result['completed_updates']!=100+added:raise ValueError('Resume update bounds')
    if result.get('stop_reason') not in ['max_updates','worker_soft_deadline'] or (added!=1000 and result.get('stop_reason')!='worker_soft_deadline'):raise ValueError('Unexplained early stop')
    if result['_wb_agent']!=request['plan']['_wb_agent'] or result['continuation_contract_sha256']!=contract_digest(continuation_contract(request['plan'])):raise ValueError('Output contract/provenance')
    if not all(result.get(k) is True for k in ['teachers_frozen_verified','checkpoint_model_readback_verified','student_changed','depth_backbone_changed']):raise ValueError('Learning evidence')
    path=(out/result['checkpoint']).resolve()
    if not path.is_relative_to(out) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=result['checkpoint_sha256']:raise ValueError('Final checkpoint integrity')
    progress=list(out.rglob('formal_progress.jsonl'))
    if len(progress)!=1:raise ValueError('One progress stream')
    rows=[json.loads(line) for line in progress[0].read_text().splitlines() if line]
    if [r['additional_updates_completed'] for r in rows]!=list(range(1,added+1)) or [r['completed_updates'] for r in rows]!=list(range(101,101+added)):raise ValueError('Contiguous optimizer updates')
    return result