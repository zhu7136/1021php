"""Isolated276 continuation, preserving all existing source and budget guards."""
from flux.formal_contract import execute
from flux import g1_resume224_contract as prior
from flux import g1_lowstep_contract as fresh
from flux.g1_resume276_input import SOURCE_TASK,SOURCE_RUN,CHECKPOINT_SHA256

def validate_scope(scope):
    for key,value in {'fresh_training':False,'resume_source_task':SOURCE_TASK,'resume_source_sha256':CHECKPOINT_SHA256}.items():
        if type(scope.get(key)) is not type(value) or scope[key]!=value:
            raise ValueError('Invalid276 continuation scope '+key)
    checked=dict(scope);checked['fresh_training']=True
    fresh.validate_scope(checked)
    return scope

def retarget(text):
    return (text.replace('224','276').replace('r1f8gw5h','wj7an49u')
            .replace('source_iteration=841','source_iteration=1718')
            .replace('source_completed_updates=842','source_completed_updates=1720'))

def teacher_source(root):return retarget(prior.teacher_source(root))
def bootstrap_source(root):return retarget(prior.bootstrap_source(root))
def worker_source(root):return retarget(prior.worker_source(root))