"""ONE authorized GPU update with a uniquely named,platform-downloadable native checkpoint."""
from pathlib import Path
import json,sys
import train_delivery_smoke as smoke
resume=smoke.resume

def configure():
    if getattr(resume,'_portable_configured',False):raise RuntimeError('Configure exactly once per process')
    smoke.configure()
    resume.OUT=resume.ROOT/'outputs/portable_smoke'
    original_setup=resume.setup;original_train=resume.train_worker;captured={}
    def setup(arm,run_path):
        algo,motion=original_setup(arm,run_path)
        if arm=='training':captured['algo']=algo
        return algo,motion
    def train_worker(run_path):
        original_train(run_path)
        from portable_checkpoint import publish
        directory=resume.OUT/'training'
        result=json.loads((directory/'summary.json').read_text())
        receipt=publish(captured['algo'],directory/'model_final.pt',result,run_path,resume.OUT/'platform_exports')
        result['portable_checkpoint']=receipt
        resume.base.save(directory/'summary.json',result)
        resume.emit('portable_checkpoint_ready',**receipt)
    resume.setup=setup;resume.train_worker=train_worker;resume._portable_configured=True

def main():
    if '--launch-envelope' in sys.argv:
        from launch_envelope import arguments
        if sys.argv.count('--launch-envelope')!=1:raise ValueError('Duplicate launch envelope')
        index=sys.argv.index('--launch-envelope')
        if index+1>=len(sys.argv):raise ValueError('Missing launch envelope')
        sys.argv[index:index+2]=arguments(sys.argv[index+1])
    configure()
    # Re-enter this SAME one-update wrapper in the runtime child.
    resume.main(child_entrypoint=Path(__file__))

if __name__=='__main__':main()