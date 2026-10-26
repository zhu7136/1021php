"""Frozen G1 task283 evaluation contract. No simulator or training imports."""
import ast
import hashlib
from collections import Counter
from pathlib import Path

SOURCE_TASK='TASK_20261026_283'
SOURCE_RUN='9fe07g0l'
SOURCE_ITERATION=2594
CHECKPOINT_BYTES=10346235
CHECKPOINT_SHA256='416f68e33c18bb39a354f8de66277cb77967b3ac9bbb6022ee631262442174f2'
SEEDS=(42,43,44)
WORKER_SECONDS=1800
PAIR_COUNT=56
VARIANTS=64

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def validate_scope(scope):
    required={'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
              'gpu':'1x4090D-24G','worker_seconds_max':1800,'training_updates':0,
              'source_run':SOURCE_RUN,'checkpoint_sha256':CHECKPOINT_SHA256,
              'dataset':'motions-low-step','pairs':56,'terrain_variants':64,
              'repeats':3,'output_upload_probe_passed':True}
    for k,v in required.items():
        if type(scope.get(k)) is not type(v) or scope[k]!=v:raise ValueError('Invalid evaluation scope '+k)
    if scope.get('seeds')!=list(SEEDS) or not scope.get('code_commit'):raise ValueError('Missing eval identity')
    return scope

def scheduled_reset_source(source):
    """Change only native reset's random motion assignment, fail on source drift."""
    tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MotionCommand')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='reset')
    import textwrap
    text=textwrap.dedent('\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno]))
    old='self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)'
    if text.count(old)!=1:raise ValueError('Pinned native reset assignment changed')
    return text.replace(old,'self.motion_ids[env_ids] = self.eval_motion_ids[env_ids]')

def classify(failed,timeout,completed):
    if failed:return 'failed'
    if timeout:return 'truncated'
    if completed:return 'completed'
    return None

def summarize(rows):
    if len(rows)!=168 or len({(r['clip_index'],r['seed']) for r in rows})!=168:
        raise ValueError('Expected unique56x3 results including untested')
    c=Counter(r['status'] for r in rows)
    if set(c)-{'completed','failed','truncated','not_tested'}:raise ValueError('Unknown result status')
    resolved=c['completed']+c['failed']
    return {'planned_trials':168,'completed':c['completed'],'failed':c['failed'],
            'truncated':c['truncated'],'not_tested':c['not_tested'],
            'tested':168-c['not_tested'],'resolved':resolved,
            'completion_fraction_of_planned':c['completed']/168,
            'completion_rate_resolved_only':c['completed']/resolved if resolved else None,
            'full_coverage':c['not_tested']==0,'training_updates':0,
            'metric_is_clip_tracking_completion_not_independent_stair_goal':True}