"""CPU v8 trial: previous-frame bounded arm IK, unchanged Cartesian gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from kinematics import Model,robot_config
from retarget_hud04 import SOURCE,G1_URDF,solve,export
from continuous_arms import sequential_arms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--names',nargs='+')
    selection.add_argument('--all',action='store_true',help='Process all 56 recovered source motions into a separate directory')
    selection.add_argument('--failed-v7',type=Path)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--limit',type=int,default=56)
    parser.add_argument('--skip-existing',action='store_true')
    parser.add_argument('--regularization',type=float,default=.003)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(SOURCE.resolve()):
        parser.error('Refusing recovered source directory')
    if args.regularization<=0 or args.limit<1 or args.offset<0:
        parser.error('regularization/limit must be positive; offset must be nonnegative')
    names=args.names
    if args.all:
        names=[path.name for path in SOURCE.glob('*_motion.npz')]
    if args.failed_v7:
        report=json.loads((args.failed_v7/'validation.json').read_text())
        names=[r['file'] for r in report['per_file'] if not r['pass_kinematic_gates']]
        names += [json.loads(p.read_text())['source'] for p in args.failed_v7.glob('*_rejected.json')]
    if not names:
        parser.error('Supply explicit names, --all, or --failed-v7')
    names=sorted(set(names))[args.offset:args.offset+args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    import wandb
    provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if os.environ.get(v)}
    with wandb.init(entity='zhul136-university-of-science-and-technology-of-china',project='HuD04_WBT',job_type='cpu-retarget-validation',name=f'hud04-sequential-arm-ik-{args.offset}',tags=['aria','cpu-only','retarget-v8'],config={'_wb_agent':provenance,'regularization':args.regularization,'names':names,'gpu_training':False,'solver':'bounded_TRF_previous_frame','source_target_semantics':'unchanged_v7_distal_hand_chain_scale'}) as run:
        successes=0;attempted=0
        for name in names:
            if Path(name).name!=name:
                raise ValueError('Motion names must be plain basenames')
            path=SOURCE/name;output=args.output/name
            if output.exists():
                if args.skip_existing:
                    continue
                raise FileExistsError(output)
            with np.load(path,allow_pickle=False) as loaded:
                data={k:loaded[k] for k in loaded.files}
            model,q,root,quat,metrics=solve(data,150)
            q,diag=sequential_arms(model,Model(G1_URDF),data,q,root,quat,robot_config()['_HU_D04_BODY_NAMES'],args.regularization)
            for key in ['wrist_target_error_mean_m','wrist_target_error_max_m']:
                metrics['initial_dls_'+key]=metrics.pop(key)
            metrics['wrist_target_error_mean_m']=float(np.mean([info['mean_source_grid_hand_error_m'] for info in diag.values()]))
            metrics['wrist_target_error_max_m']=max(info['max_source_grid_hand_error_m'] for info in diag.values())
            metrics['sequential_arms']=diag
            attempted+=1
            try:
                result,metrics=export(data,model,q,root,quat,metrics,project_arms=False)
            except ValueError as error:
                rejected={'source':name,'status':'rejected_not_training_data','reason':str(error),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metrics':metrics}
                (args.output/(path.stem+'_rejected.json')).write_text(json.dumps(rejected,indent=2)+'\n')
                print(json.dumps(rejected),flush=True)
                run.log({'clip_index':attempted,'export_ok':0})
                continue
            metrics.update(source=name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status='candidate_not_simulation_validated',solver='sequential_bounded_TRF_v8',kinematic_target_gate_pass=metrics['foot_target_error_max_m']<=.02 and metrics['foot_orientation_error_max_deg']<=5 and metrics['final_wrist_target_max_m']<=.02)
            np.savez_compressed(output,**result)
            terrain=path.with_name(path.name.replace('_motion.npz','_terrain.npy'))
            shutil.copyfile(terrain,args.output/terrain.name)
            (args.output/(path.stem+'_audit.json')).write_text(json.dumps(metrics,indent=2)+'\n')
            print(json.dumps(metrics),flush=True)
            successes+=bool(metrics['kinematic_target_gate_pass'])
            run.log({'clip_index':attempted,'export_ok':1,'source_grid_foot_max_m':metrics['foot_target_error_max_m'],'final_wrist_target_max_m':metrics['final_wrist_target_max_m'],'time_scale':metrics['time_scale'],'max_joint_speed_ratio':metrics['max_joint_speed_ratio']})
        run.summary.update({'attempted':attempted,'preliminary_passed':successes,'gpu_training_started':False,'independent_readback_required':True})
        print(json.dumps({'run_id':run.id,'run_url':run.url,'attempted':attempted,'preliminary_passed':successes}),flush=True)


if __name__=='__main__':
    main()