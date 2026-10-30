"""Deterministic post-hoc numeric summaries; never called from the control loop."""
import json
import math

SEGMENTS=((1,3),(4,20),(21,50),(51,85),(86,100),(101,150),(151,384),(385,1000))


def aggregate(rows):
    valid=[r for r in rows if r.get('comparison_valid')]
    result={'steps':len(rows),'compared_steps':len(valid),
            'expert_invalid_steps':len(rows)-len(valid),
            'zero_depth_steps':sum(r['depth_stats']['zero_fraction']==1 for r in rows)}
    if not valid:return result
    n=len(valid)
    s=[r['student_actions'] for r in valid];t=[r['teacher_actions'] for r in valid]
    ds=[sum((a[j]-b[j])**2 for a,b in zip(s,t)) for j in range(29)]
    ss=sum(x*x for a in s for x in a);tt=sum(x*x for a in t for x in a)
    dot=sum(x*y for a,b in zip(s,t) for x,y in zip(a,b))
    result.update(action_rmse=math.sqrt(sum(ds)/(n*29)),
      teacher_rms=math.sqrt(tt/(n*29)),student_rms=math.sqrt(ss/(n*29)),
      relative_rmse=math.sqrt(sum(ds)/tt) if tt else None,
      pooled_cosine=dot/math.sqrt(ss*tt) if ss*tt else None,
      per_joint_rmse=[math.sqrt(x/n) for x in ds],
      per_joint_bias=[sum(a[j]-b[j] for a,b in zip(s,t))/n for j in range(29)],
      per_joint_teacher_rms=[math.sqrt(sum(a[j]**2 for a in t)/n) for j in range(29)],
      per_joint_student_rms=[math.sqrt(sum(a[j]**2 for a in s)/n) for j in range(29)])
    return result


def compact_events(rows):
    if len(rows)>1000:raise ValueError('Action probe row count exceeded')
    for step,r in enumerate(rows,1):
        if (r.get('schema')!='hud04-same-state-action-probe-v1' or r.get('step')!=step
                or r.get('teacher_actions_applied') is not False):
            raise ValueError('Action probe record contract')
        for key in ('student_actions','teacher_actions'):
            if key=='teacher_actions' and not r.get('comparison_valid'):continue
            if len(r[key])!=29 or not all(type(x) in (int,float) and math.isfinite(x) for x in r[key]):
                raise ValueError('Action probe record contract')
    segments=[]
    for start,end in SEGMENTS:
        selected=[r for r in rows if start<=r['step']<=end]
        if selected:segments.append({'start_step':start,'end_step':end,**aggregate(selected)})
    summary={'event':'hud04_compact_action_summary','schema':'hud04-compact-action-v1',
       'teacher_actions_applied':False,'action_units':'raw_policy_outputs_before_native_scaling',
       'aggregation_scope':'all_saved_probe_rows_not_log_sample','all_rows':aggregate(rows),
       'segments':segments,'full_rows':len(rows),'logged_rows':min(len(rows),100)}
    if len(rows)<=100:sample=rows
    else:
        indices={round(i*(len(rows)-1)/99) for i in range(100)}
        sample=[r for i,r in enumerate(rows) if i in indices]
    result=[]
    for r in sample:
        record={'event':'hud04_compact_action_step','step':r['step'],'reference_frame':r['reference_frame'],
          'comparison_valid':r['comparison_valid'],'route':r['route'],
          'depth_rms':r['depth_stats']['rms'],'depth_zero_fraction':r['depth_stats']['zero_fraction'],
          'actor_rms':r['actor_stats']['rms'],'actor_term_rms':{k:v['rms'] for k,v in r['actor_terms'].items()},
          'action_rmse':r['action_rmse'],
          'student_action_rms':r['student_action_stats']['rms'],
          'teacher_action_rms':r['teacher_action_stats']['rms'] if r['comparison_valid'] else None}
        result.append(record)
    result.append(summary)
    # Entire recovery output, not just one record, stays below the API's observed
    # 200 KiB tail cap, leaving room for episode reports and depth initialization.
    encoded=[json.dumps(r,allow_nan=False,separators=(',',':')) for r in result]
    if sum(len(s.encode())+1 for s in encoded)>110000:
        raise ValueError('Compact action log exceeds bound')
    return encoded