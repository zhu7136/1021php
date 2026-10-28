"""Verify learning separately from true simulator close/return evidence."""
import hashlib,json,math,os
from pathlib import Path,PurePosixPath
SOURCE_SHA='aa543b193b8502142d03d4cbdaba0cb10a0909f2140c8c6a9e8d58b682e7cf1a'
FLAGS=('actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite')
FILES=('learning_state.json','native_reload.json','progress.json','updates.jsonl','new_process_resume.json','capacity.json')
RECEIPT='learning-complete-before-close.json'

def require(ok,reason):
    if not ok:raise ValueError(reason)

def file_at(root,name):
    p=PurePosixPath(name)
    require(not p.is_absolute() and bool(p.parts) and '..' not in p.parts,'unsafe evidence path')
    f=root.joinpath(*p.parts)
    require(not any(root.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)),'symlink evidence')
    require(f.resolve().is_relative_to(root.resolve()) and f.is_file(),'missing evidence: '+name)
    return f

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def inspect_learning(root):
    root=Path(root);data={n:json.loads(file_at(root,n).read_text()) for n in FILES if n.endswith('.json')}
    learning=data['learning_state.json'];reload=data['native_reload.json'];progress=data['progress.json']
    resume=data['new_process_resume.json'];capacity=data['capacity.json'];count=learning['completed_updates'];start=128
    require(all(resume.get(k) is True for k in ['native_new_process_load_verified','actor_critic_and_optimizers_exact','learning_rates_exact']),'resume not verified')
    require(resume.get('source_checkpoint_sha256')==SOURCE_SHA and resume.get('start_iteration')==start,'wrong source checkpoint')
    require(type(count) is int and 1<=count<=100000,'invalid update count')
    require(capacity.get('effective_num_envs')==4096 and capacity.get('samples_per_update')==98304,'effective environment mismatch')
    require(learning.get('effective_num_envs')==4096,'learning environment mismatch')
    require(all(learning.get(k) is True for k in FLAGS),'invalid learning flags')
    require(learning.get('same_learner_reload') is True,'same-learner reload not verified')
    require(reload.get('native_load_passed') is True and reload.get('completed_updates')==count,'invalid reload report')
    require(set(reload.get('checked_modules',[])) >= {'actor','critic','actor_obs_normalizer','critic_obs_normalizer'},'incomplete module reload')
    require(set(reload.get('checked_optimizers',[])) >= {'actor_optimizer','critic_optimizer'},'incomplete optimizer reload')
    require(set(reload.get('checked_scalars',[])) >= {'current_learning_iteration','actor_learning_rate','critic_learning_rate'},'incomplete scalar reload')
    require(reload.get('iteration')==start+count-1,'reload iteration mismatch')
    rows=[json.loads(l) for l in file_at(root,'updates.jsonl').read_text().splitlines()]
    require(len(rows)==count and [r['update'] for r in rows]==list(range(1,count+1)),'non-contiguous updates')
    require(all(type(r['update']) is int and r['iteration']==start+r['update']-1 and r['num_envs']==4096 for r in rows),'invalid update index/environment')
    for row in rows:
        losses=[v for k,v in row.items() if k.startswith('Loss/')]
        require(bool(losses) and all(type(v) in (int,float) and math.isfinite(v) for v in losses),'non-finite losses')
    require(progress.get('completed_updates')==count and progress.get('last_iteration_index')==start+count-1,'progress mismatch')
    checkpoint=file_at(root,reload['path']);require(checkpoint.name=='model_smoke_final.pt','unexpected checkpoint name')
    require(sha(checkpoint)==learning['checkpoint_sha256'],'checkpoint hash mismatch')
    hashes={n:sha(file_at(root,n)) for n in FILES};hashes[reload['path']]=sha(checkpoint)
    return {'completed_updates':count,'checkpoint_sha256':learning['checkpoint_sha256'],'effective_num_envs':4096,'evidence_sha256':hashes}

def write_learning_receipt(root,task_id):
    root=Path(root);require(isinstance(task_id,str) and task_id.startswith('TASK_'),'missing task identity')
    result=inspect_learning(root)
    receipt={'schema_version':1,'phase':'learning_complete_before_simulator_close','task_id':task_id,**result,'simulation_close_verified':False,'train_return_verified':False}
    with (root/RECEIPT).open('x') as stream:json.dump(receipt,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    return receipt

def validate_worker_result(root,exit_code,task_id):
    require(type(exit_code) is int and exit_code==0,'nonzero native exit')
    root=Path(root);learning=inspect_learning(root);receipt=json.loads(file_at(root,RECEIPT).read_text())
    require(receipt.get('schema_version')==1 and receipt.get('phase')=='learning_complete_before_simulator_close','invalid receipt schema')
    require(receipt.get('task_id')==task_id and all(receipt.get(k)==v for k,v in learning.items()),'receipt evidence mismatch')
    returned=False
    if (root/'train-returned.json').exists():returned=json.loads(file_at(root,'train-returned.json').read_text()).get('train_returned') is True
    close_returned=False
    if (root/'simulation-close-returned.json').exists():close_returned=json.loads(file_at(root,'simulation-close-returned.json').read_text()).get('simulation_close_returned') is True
    return {**learning,'native_learning_passed':True,'native_exit_code':0,'train_return_verified':returned,
            'simulation_close_return_verified':close_returned,'lifecycle_status':'train_returned' if returned else 'clean_process_exit_without_train_return',
            'new_process_resume_verified':True,'teacher_converged':False}