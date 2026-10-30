"""Fixed obstacle preview; reuse the original native lifecycle, never training."""
import json
from pathlib import Path
import offline_video_child as original

base_configuration=original.configuration

def configuration(req):
    # Required before resolving any observation term: the overlay and upstream
    # config_values directories are separate portions of the same package.
    from student_adapter.runtime_imports import verify_overlay_resolution
    verify_overlay_resolution(req['bundle'])
    # Build the original, fully hash-checked formal data configuration first.
    from student_adapter.formal_entry import build_formal_config
    from student_adapter.native_child import hydrate
    data=Path(req['data_root'])
    raw=build_formal_config(json.loads((data/'locomotion-teacher-config.json').read_text()),
        plan=req['source_plan'],bundle=Path(req['bundle']),data_root=data,teachers=req['teachers'],
        output=Path(req['output']),resume=req['checkpoint'])
    row=req['obstacle_case']
    import hashlib
    if hashlib.sha256(Path(row['motion_file']).read_bytes()).hexdigest()!=row['motion_sha256']:raise ValueError('Isolated clip changed')
    raw['algo']['_target_']='obstacle_video_algo.HuD04ObstacleVideo'
    raw['algo']['config']['num_learning_iterations']=1
    raw['training'].update(num_envs=1,headless=True,name='hud04-fixed-'+row['skill']+'-preview')
    for group in ['setup_terms','reset_terms','step_terms']:
        raw['command'][group]['motion_command']['func']='obstacle_video_command:HuD04FixedObstacleCommand'
    motion=raw['command']['setup_terms']['motion_command']['params']['motion_config']
    motion.update(motion_file=row['motion_file'],motion_dir='',shard_motions=False,
        use_adaptive_timesteps_sampler=False,resample_on_motion_end=False,start_at_timestep_zero_prob=1.,
        freeze_at_timestep_zero_prob=0.,default_pose_transition_strategy='none',
        enable_default_pose_prepend=False,enable_default_pose_append=False)
    raw['logger']={'type':'disabled','base_dir':req['output'],'headless_recording':True,
        'video':{'enabled':True,'interval':1000000,'width':960,'height':540,'playback_rate':1.,
            'output_format':'mp4','save_dir':str(Path(req['output'])/'videos'),'upload_to_wandb':False,
            'show_command_overlay':False,'record_env_id':0,'use_recording_thread':False,
            'camera':row['camera'],'vertical_fov':55.}}
    return hydrate(raw)

if __name__=='__main__':
    original.configuration=configuration
    original.main()
