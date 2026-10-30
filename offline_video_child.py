"""Native simulator entrypoint; bounded inference only, no training invocation."""
import faulthandler,json,os,sys
from pathlib import Path

def configuration(req):
    if req['scope']['training_updates']!=0 or req['scope']['automatic_retries']!=0:raise PermissionError('Invalid scope')
    from student_adapter.runtime_imports import verify_overlay_resolution
    verify_overlay_resolution(req['bundle'])
    from student_adapter.formal_entry import build_formal_config
    data=Path(req['data_root'])
    template=json.loads((data/'locomotion-teacher-config.json').read_text())
    raw=build_formal_config(template,plan=req['source_plan'],bundle=Path(req['bundle']),data_root=data,
        teachers=req['teachers'],output=Path(req['output']),resume=req['checkpoint'])
    raw['algo']['_target_']='offline_video_algo.HuD04OfflineVideo'
    raw['algo']['config']['num_learning_iterations']=1
    raw['logger']={'type':'disabled','base_dir':req['output'],'headless_recording':True,
       'video':{'enabled':True,'interval':1000000,'width':640,'height':360,'playback_rate':1.,
         'output_format':'h264','save_dir':str(Path(req['output'])/'videos'),
         'upload_to_wandb':False,'show_command_overlay':False,'record_env_id':0,'use_recording_thread':False}}
    raw['training']['headless']=True
    from student_adapter.native_child import hydrate
    cfg=hydrate(raw)
    return cfg

def main():
    faulthandler.enable(all_threads=True)
    req=json.loads(Path(os.environ['HUD04_VIDEO_REQUEST']).read_text())
    cfg=configuration(req)
    os.environ['HUD04_ARIA_THREAD_ID']=req['_wb_agent']['thread_id']
    os.environ['HUD04_ARIA_TURN_ID']=req['_wb_agent']['turn_id']
    from holosoma.utils.sim_utils import setup_simulator_imports
    from holosoma.utils.eval_utils import init_sim_imports
    setup_simulator_imports(cfg)
    from holosoma import train_agent
    from student_adapter import formal_lifecycle
    app=None;out=Path(req['output'])
    report={'task_id':req['task_id'],'close_started':False,'close_returned':False,'preview_only':True}
    # Reuse only the native factory/setup/load/close path. The algorithm target's
    # overridden learn() is a bounded no-grad student inference loop, not training.
    sys.argv=[sys.argv[0]]
    try:
        app=init_sim_imports(cfg)
        formal_lifecycle.configure(app,out,report)
        context=type('ExistingAppContext',(),{'simulation_app':app})()
        train_agent.train(cfg,training_context=context)
        report['preview_returned']=True;formal_lifecycle.write_json(out/'formal-lifecycle.json',report)
    finally:
        if app is not None:formal_lifecycle.close(app,out,report)
    report['child_returned']=True;formal_lifecycle.write_json(out/'formal-lifecycle.json',report)

if __name__=='__main__':main()
