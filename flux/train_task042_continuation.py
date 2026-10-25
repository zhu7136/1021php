"""One approved task042 continuation: native201 -> at most401,32env,200updates/300s,no eval/retry."""
from pathlib import Path
import hashlib,json,sys,urllib.parse,urllib.request
import train_protected_resume as resume

SOURCE_RUN='323f06da'
SOURCE_SHA='b1d604f8502289630dfa1a53ae1d8d626d7de5ad43d082e4f0b6bfcdeb5c4394'
SOURCE_BYTES=10345087
SOURCE_ITERATION=201
MAX_UPDATES=200

def validate_source_url(url):
    p=urllib.parse.urlsplit(url);q=urllib.parse.parse_qs(p.query)
    expected='/wandb-production.appspot.com/zhul136-university-of-science-and-technology-of-china/HuD04_WBT/323f06da/resume_trained_checkpoint.pt'
    if (p.scheme!='https' or p.hostname!='storage.googleapis.com' or p.port not in (None,443)
        or p.username or p.password or p.fragment or p.path!=expected
        or not(q.get('X-Goog-Signature') or q.get('Signature'))):
        raise ValueError('Only exact TASK042 native201 signed source allowed')

def fetch_inputs(url):
    validate_source_url(url)
    from reliable_delivery import NoRedirect
    directory=resume.OUT/'inputs';directory.mkdir(exist_ok=False)
    target=directory/'model_final.pt'
    with urllib.request.build_opener(NoRedirect()).open(url,timeout=45) as response,target.open('xb') as sink:
        if response.status!=200:raise ValueError('Source HTTP status differs')
        count=0;digest=hashlib.sha256()
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            count+=len(chunk)
            if count>SOURCE_BYTES:raise ValueError('Source exceeds exact byte bound')
            sink.write(chunk);digest.update(chunk)
    if count!=SOURCE_BYTES or digest.hexdigest()!=SOURCE_SHA:raise ValueError('TASK042 source hash/size mismatch')
    evidence=dict(source_task='TASK_20261025_042',source_run=SOURCE_RUN,source_iteration=SOURCE_ITERATION,
        source_sha256=SOURCE_SHA,bytes=count)
    resume.base.save(resume.OUT/'input_verification.json',evidence)
    resume.emit('task042_source_download_verified',**evidence)

def summarize():
    result=json.loads((resume.OUT/'training/summary.json').read_text())
    updates=result['completed_updates'];final=result['final_checkpoint']['iteration']
    if (not isinstance(updates,int) or not 1<=updates<=MAX_UPDATES or final!=SOURCE_ITERATION+updates
        or result['source_iteration']!=SOURCE_ITERATION or result['source_sha256']!=SOURCE_SHA
        or result['training_seconds']>300 or not result['restore']['actor_critic_optimizer_exact']):
        raise ValueError('Task042 source/update/time/restore contract violated')
    result.update(purpose='bounded_continuation_from_TASK_20261025_042',source_task='TASK_20261025_042',
        source_run=SOURCE_RUN,model_evaluations=0,num_envs=32,policy_quality_evaluated=False,
        teacher_converged=False,controller_checkpoint_download_validation_required=True)
    resume.base.save(resume.OUT/'summary.json',result)
    resume.emit('task042_bounded_continuation_saved',completed_updates=updates,iteration=final,model_evaluations=0,
        policy_quality_evaluated=False,sha256=result['final_checkpoint']['sha256'])

def configure():
    if getattr(resume,'_task042_configured',False):raise RuntimeError('Configure once per process')
    resume.OUT=resume.ROOT/'outputs/task042_continuation'
    resume.SOURCE_ITERATION=SOURCE_ITERATION;resume.SOURCE_SHA=SOURCE_SHA
    resume.MAX_UPDATES=MAX_UPDATES;resume.TRAIN_SECONDS=300;resume.WORKER_SECONDS=1800
    resume.ARMS=(('training',0.),);resume.SEEDS=()
    resume.summarize=summarize;resume.frozen.fetch_inputs=fetch_inputs
    resume._task042_configured=True

def main():
    if '--launch-envelope' in sys.argv:
        from launch_envelope import arguments,decode
        if sys.argv.count('--launch-envelope')!=1:raise ValueError('Duplicate envelope')
        index=sys.argv.index('--launch-envelope')
        if index+1>=len(sys.argv):raise ValueError('Missing envelope')
        token=sys.argv[index+1];validate_source_url(decode(token)['input_url'])
        sys.argv[index:index+2]=arguments(token)
    configure()
    resume.main(child_entrypoint=Path(__file__))

if __name__=='__main__':main()