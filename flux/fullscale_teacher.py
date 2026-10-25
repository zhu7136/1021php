"""Prepared4096env native20k teacher configuration; not an auto-starting Flux job.

Uses the pinned native PPO,not the20update capacity subclass. Worker walltime,
resume source,resource choice and output-retrieval supervisor must be set by
the separate full-scale dispatcher before submitting a paid long run.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def build_config(data,output,checkpoint=None):
    from wbt_training.config_values.experiment import g1_wbt_terrain_ref_noscale
    cfg=g1_wbt_terrain_ref_noscale
    return replace(cfg,
        training=replace(cfg.training,headless=True,num_envs=4096,
            checkpoint=str(checkpoint.resolve()) if checkpoint else None,
            teacher_checkpoint=None,wandb_path=None,registry_name=data.resolve().as_uri(),
            project='HuD04_WBT',name='php-teacher-fullscale-4096'),
        logger=replace(cfg.logger,mode='offline',base_dir=str(output),
            project='HuD04_WBT',entity='zhul136-university-of-science-and-technology-of-china',
            name='php-teacher-fullscale-4096',video=replace(cfg.logger.video,enabled=False)),
        algo=replace(cfg.algo,config=replace(cfg.algo.config,
            num_learning_iterations=20000,save_interval=500)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--approval-file',type=Path,required=True)
    args=parser.parse_args()
    approval=json.loads(args.approval_file.read_text())
    if (approval.get('purpose')!='fullscale_4096_native_20000'
        or approval.get('gpu_start_authorized') is not True
        or approval.get('num_envs')!=4096
        or type(approval.get('worker_seconds')) is not int
        or not 1800<=approval['worker_seconds']<=86400
        or approval.get('external_bounded_supervisor_verified') is not True
        or approval.get('output_retrieval_verified') is not True):
        raise ValueError('Full-scale dispatcher,walltime and output retrieval must be configured first')
    from flux.preflight import runtime_checks
    runtime_checks(args.data)
    from wbt_training.utils.isaacsim_clock import use_php_isaacsim
    from holosoma.utils.sim_utils import setup_simulator_imports
    from holosoma.train_agent import train
    cfg=build_config(args.data,args.output,args.checkpoint)
    os.environ['WANDB_MODE']='offline'
    import wandb
    original=wandb.init
    def init(*a,**kw):
        config=dict(kw.get('config') or {})
        if approval.get('aria_provenance'):config['_wb_agent']=approval['aria_provenance']
        config.update(record_role='native_training',training_phase='fullscale_4096_native_20000')
        kw['config']=config
        return original(*a,**kw)
    wandb.init=init
    cfg=use_php_isaacsim(cfg)
    sys.argv=[sys.argv[0]]
    setup_simulator_imports(cfg)
    train(cfg)

if __name__=='__main__':main()