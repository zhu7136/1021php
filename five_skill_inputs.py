"""CPU-only fixed, held-out inputs and exact scope for a single frozen screen."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np

SKILLS=['locomotion','low-step','high-step','low-climb','high-climb']
SCOPE={'max_tasks':1,'gpu_num':1,'worker_seconds_max':900,'automatic_retries':0,
       'training_updates':0,'preview_seconds':20,'max_policy_steps':1000,
       'num_envs':1,'skills':SKILLS,'max_clips':5}
MANIFEST_SHA='8e0f3ef89479ddf39e66b4697389ea5d42305a501f9116d52f909652e3161f8e'
CHECKPOINT_SHA='e4e92fd80abacc3023339dd379d48e3f7eee64eb63ff84928cf0c6e5a795d3af'
CHECKPOINT_BYTES=45006507

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def check_scope(scope):
    if not isinstance(scope,dict) or set(scope)!=set(SCOPE):raise ValueError('Scope keys')
    for key,expected in SCOPE.items():
        if type(scope[key]) is not type(expected) or scope[key]!=expected:raise ValueError('Scope value')
    return True

def check_config(c,task_id):
    check_scope(c['scope'])
    expected_approval={'confirmed_by_user':True,'approved_at_utc':'2026-10-30T10:43:16Z',
        'max_tasks':1,'gpu_num':1,'worker_seconds_max':900,'training_updates':0,'automatic_retries':0}
    if c.get('approval')!=expected_approval:raise ValueError('Explicit bounded approval')
    if c.get('task_id')!=task_id or not isinstance(task_id,str) or not task_id.startswith('TASK_'):
        raise ValueError('Task binding')
    if (c.get('native_child')!='hud04_eval_child.py' or c.get('evaluation_role')!='student'
        or type(c.get('evaluation_seed')) is not int or c['evaluation_seed']!=0
        or c.get('same_state_action_probe') is not False
        or type(c.get('source_completed_updates')) is not int or c['source_completed_updates']!=3686):
        raise ValueError('Frozen student-only scope')
    opts=c.get('depth_reset_diagnostics')
    if (opts!={'fix_pending_reset':True,'max_events':64} or type(opts.get('fix_pending_reset')) is not bool
        or type(opts.get('max_events')) is not int):
        raise ValueError('Explicit guarded depth required')
    if (c['checkpoint']['sha256']!=CHECKPOINT_SHA or type(c['checkpoint']['bytes']) is not int
        or c['checkpoint']['bytes']!=CHECKPOINT_BYTES):raise ValueError('Source checkpoint identity')
    return True

def case_deadline(started,now,cases_left=1):
    """Share one 900-second worker cap; reserve 90 seconds for results."""
    if type(cases_left) is not int or not 1<=cases_left<=5:raise ValueError('Remaining cases')
    available=started+810-now
    stop=min(started+810,now+170,now+available/cases_left)
    return stop if stop-now>=100 else None

def choose_cases(manifest):
    rows=manifest['cases']
    train={r['motion_sha256'] for r in rows if r['split']=='train'}
    heldout={r['motion_sha256'] for r in rows if r['split']=='eval'}
    if train & heldout:raise ValueError('Train/eval motion leakage')
    selected=[]
    for route,skill in enumerate(SKILLS):
        candidates=[r for r in rows if r['split']=='eval' and r['skill']==skill and r['teacher_index']==route]
        if not candidates:raise ValueError('Missing held-out skill')
        row=min(candidates,key=lambda r:r['source_case_index'])
        start,end=row['output_frame_range']
        if any(type(v) is not int for v in (start,end,row['frames'])) or not 0<=start<end or end-start!=row['frames']:
            raise ValueError('Case frame range')
        selected.append(row)
    return selected

def isolate(data_root,output):
    data_root=Path(data_root);output=Path(output)
    if sha(data_root/'split-manifest.json')!=MANIFEST_SHA:raise ValueError('Split manifest identity')
    manifest=json.loads((data_root/'split-manifest.json').read_text())
    if sha(data_root/'eval-motion.npz')!=manifest['files']['eval-motion.npz']['sha256']:
        raise ValueError('Eval source hash')
    selected=choose_cases(manifest)
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    with np.load(data_root/'eval-motion.npz',allow_pickle=False) as source:
        arrays={key:source[key] for key in source.files}
    frame_count=len(arrays['motion_idxs'])
    for row in selected:
        start,end=row['output_frame_range'];route=row['teacher_index'];skill=row['skill']
        if end>frame_count:raise ValueError('Case exceeds evaluated data')
        clip={key:value.copy() if key in ('fps','joint_names','body_names') else value[start:end].copy()
              for key,value in arrays.items()}
        if (not bool(clip['motion_ends'][-1]) or clip['motion_ends'][:-1].any()
            or not (clip['motion_idxs']==route).all()):raise ValueError('Clip boundary/route mismatch')
        for key,value in clip.items():
            if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():raise ValueError('Nonfinite clip data')
        fps=float(np.asarray(clip['fps']).reshape(-1)[0])
        if not math.isfinite(fps) or fps<=0:raise ValueError('FPS')
        target=output/(skill+'.npz');np.savez(target,**clip)
        # Preserve every world-coordinate array and the original terrain mesh.
        # Flat locomotion must not be rejected for lacking a raised obstacle.
        roots=clip['joint_pos'][:,:3];lo=roots.min(axis=0);hi=roots.max(axis=0)
        center=(lo+hi)/2;distance=max(4.,float(max(hi[0]-lo[0],hi[1]-lo[1]))*1.1)
        rows.append({'skill':skill,'route':route,'case':row,'motion_file':str(target),
            'motion_sha256':sha(target),'fps':fps,
            'camera':{'type':'fixed','position':(center+np.array([0.,-distance,distance*.65])).tolist(),
                      'target':(center+np.array([0.,0.,-.25])).tolist()},
            'reference_root_bounds_m':[lo.tolist(),hi.tolist()],
            'world_coordinates_unchanged':True,'obstacle_success_certified':False})
    (output/'five-skill-inputs.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    return rows