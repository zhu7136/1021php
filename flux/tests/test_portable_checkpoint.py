"""CPU contracts for unique export,preserved optimizer,and no training/budget expansion."""
import copy,hashlib,json,pathlib,subprocess,sys,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import portable_checkpoint as portable
import torch

class Algo:
    def __init__(self):
        self.actor=torch.nn.Linear(3,2);self.critic=torch.nn.Linear(3,1)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=1e-5)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=1e-5)
        for module,opt in ((self.actor,self.actor_optimizer),(self.critic,self.critic_optimizer)):
            module(torch.ones(2,3)).sum().backward();opt.step();opt.zero_grad()
        self.current_learning_iteration=201
    def load(self,path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        for module,key in ((self.actor,'actor_model_state_dict'),(self.critic,'critic_model_state_dict'),
                           (self.actor_optimizer,'actor_optimizer_state_dict'),(self.critic_optimizer,'critic_optimizer_state_dict')):
            module.load_state_dict(d[key])
        self.current_learning_iteration=d['iter']

class PortableTests(unittest.TestCase):
    def fixture(self):
        root=pathlib.Path(tempfile.mkdtemp(prefix='portable-test-',dir='/workspace'))
        algo=Algo()
        data=dict(iter=201,experiment_config={'native':True},actor_model_state_dict=algo.actor.state_dict(),
            critic_model_state_dict=algo.critic.state_dict(),actor_optimizer_state_dict=algo.actor_optimizer.state_dict(),
            critic_optimizer_state_dict=algo.critic_optimizer.state_dict(),additional_native_field={'keep':[1,2]})
        source=root/'model_final.pt';torch.save(data,source)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        summary=dict(source_iteration=200,source_sha256=portable.SOURCE_SHA,completed_updates=1,last_iteration=201,transitions=768,
            restore={'actor_critic_optimizer_exact':True},final_checkpoint=dict(iteration=201,sha256=digest,
            native_save_reload_verified=True,actor_critic_optimizer_iteration_exact=True))
        return root,algo,source,data,summary
    def test_filename_matches_observed_platform_convention_not_input_name(self):
        self.assertEqual(portable.filename('team/project/a1b2c3d4'),'model_00201.pt')
        self.assertNotEqual(portable.filename('team/project/a1b2c3d4'),'model_final.pt')
    def test_invalid_run_path_is_rejected(self):
        for value in ('team/project/../../bad','team/project/not-eight','../project/a1b2c3d4','team/project/a1b2c3d4?token=x'):
            with self.assertRaises(ValueError):portable.filename(value)
    def test_export_preserves_all_native_fields_and_full_optimizer(self):
        root,algo,source,data,summary=self.fixture();before=source.read_bytes()
        receipt=portable.publish(algo,source,summary,'team/project/a1b2c3d4',root/'exports')
        exported=torch.load(receipt['path'],map_location='cpu',weights_only=True)
        self.assertTrue(portable.equal({k:v for k,v in exported.items() if k!='_aria_delivery'},data))
        self.assertEqual(source.read_bytes(),before)
        self.assertEqual(exported['_aria_delivery']['run_path'],'team/project/a1b2c3d4')
        self.assertTrue(exported['_aria_delivery']['regenerated_from_iteration200'])
        self.assertFalse(exported['_aria_delivery']['old_task032_checkpoint_recovered'])
        self.assertEqual(receipt['sha256'],hashlib.sha256(pathlib.Path(receipt['path']).read_bytes()).hexdigest())
    def test_existing_export_is_never_overwritten(self):
        root,a,p,d,s=self.fixture();r=portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
        original=pathlib.Path(r['path']).read_bytes()
        with self.assertRaises(FileExistsError):portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
        self.assertEqual(pathlib.Path(r['path']).read_bytes(),original)
    def test_wrong_update_bounds_rejected_before_export(self):
        for field,value in (('source_iteration',201),('completed_updates',2),('last_iteration',200),('transitions',769)):
            root,a,p,d,s=self.fixture();s[field]=value
            with self.assertRaises(ValueError):portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
            self.assertFalse((root/'exports/model_00201.pt').exists())
    def test_native_hash_mismatch_rejected(self):
        root,a,p,d,s=self.fixture();s['final_checkpoint']['sha256']='0'*64
        with self.assertRaises(ValueError):portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
    def test_native_loader_failure_never_exposes_pt_export(self):
        root,a,p,d,s=self.fixture()
        with patch.object(a,'load',side_effect=RuntimeError('synthetic native failure')):
            with self.assertRaises(RuntimeError):portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
        self.assertFalse((root/'exports/model_00201.pt').exists())
    def test_native_reload_state_mismatch_is_rejected(self):
        root,a,p,d,s=self.fixture()
        def corrupt(path):a.current_learning_iteration=200
        with patch.object(a,'load',side_effect=corrupt):
            with self.assertRaises(ValueError):portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
        self.assertFalse((root/'exports/model_00201.pt').exists())
    def test_exported_provenance_contains_no_capabilities(self):
        root,a,p,d,s=self.fixture();r=portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
        text=json.dumps(r['provenance'])
        for token in ('https://','upload_url','api_key','Signature','password'):self.assertNotIn(token,text)
        self.assertEqual(r['provenance']['model_evaluations'],0)
    def test_entrypoint_import_is_stdlib_only(self):
        code=f'import sys;sys.path.insert(0,{str(ROOT)!r});import train_portable_smoke;assert "torch" not in sys.modules'
        r=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_controller_accepts_only_current_full_native_export(self):
        root,a,p,d,s=self.fixture();r=portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
        v=portable.verify(r['path'],'team/project/a1b2c3d4',r['sha256'])
        self.assertEqual(v['iteration'],201);self.assertTrue(v['optimizer_state_present'])
        with self.assertRaises(ValueError):portable.verify(r['path'],'team/project/ffffffff')
        with self.assertRaises(ValueError):portable.verify(r['path'],'team/project/a1b2c3d4','0'*64)
    def test_controller_rejects_input_iteration200(self):
        root,a,p,d,s=self.fixture();d['iter']=200;torch.save(d,p)
        with self.assertRaises(ValueError):portable.verify(p,'team/project/a1b2c3d4')
    def test_controller_rejects_nonfinite_optimizer(self):
        root,a,p,d,s=self.fixture();r=portable.publish(a,p,s,'team/project/a1b2c3d4',root/'exports')
        q=pathlib.Path(r['path']);data=torch.load(q,map_location='cpu',weights_only=True)
        next(iter(data['actor_optimizer_state_dict']['state'].values()))['exp_avg'].fill_(float('nan'));torch.save(data,q)
        with self.assertRaises(ValueError):portable.verify(q,'team/project/a1b2c3d4')
    def test_wrapper_retains_one_update_no_eval_and_same_time_bounds(self):
        code=f'import sys;sys.path.insert(0,{str(ROOT)!r});import train_portable_smoke as p;p.configure();r=p.resume;assert r.MAX_UPDATES==1 and r.ARMS==(("training",0.),) and r.SEEDS==() and r.WORKER_SECONDS==1800 and r.TRAIN_SECONDS==300 and r.OUT.name=="portable_smoke"'
        r=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_runtime_child_is_same_portable_wrapper(self):
        import train_portable_smoke as entry
        with patch.object(entry,'configure'),patch.object(entry.resume,'main') as call,patch.object(sys,'argv',['portable']):
            entry.main();self.assertEqual(pathlib.Path(call.call_args.kwargs['child_entrypoint']).name,'train_portable_smoke.py')

if __name__=='__main__':unittest.main()