"""CPU integration: pinned PPO save/load AND pinned native logging-directory guard.

save_checkpoint_artifact below is verbatim from amazon-far/holosoma revision
4a0bf04cde5930af5ed59da797568d32be972525, agents/modules/logging_utils.py
(full upstream file SHA256 4ad7864ec212bedbc8601da373992f0b7281e175260c781d5ff7b31ca5b2f22e).
No GPU modules or network are imported. Only W&B upload is a no-op recorder.
"""
import ast,hashlib,os,sys,tempfile,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'flux'),str(ROOT/'flux/tests')]
import test_fresh_pilot as native
import checkpoint_contract as contract
import train_protected_resume as resume
logger=NS(info=lambda *args:None)

def save_checkpoint_artifact(self, state_dict: dict[str, object], path: str) -> None:
    if not path.startswith(self.log_dir):
        raise ValueError(f"Path {path} is not in the logging directory {self.log_dir}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Saving checkpoint to {path}")
    torch.save(state_dict, path)
    self.save_to_wandb(path)

class ResumeSaveDirectoryTests(unittest.TestCase):
    def fixture(self,root):
        helper=native.FreshTests();algo=helper.make_algo(True)
        algo.current_learning_iteration=200
        contract.attach_metadata(algo,helper.config())
        training=Path(root)/'training';training.mkdir()
        calls=[]
        logging=NS(log_dir=str(training),save_to_wandb=calls.append)
        logging.save_checkpoint_artifact=types.MethodType(save_checkpoint_artifact,logging)
        algo.logging_helper=logging
        return algo,training,calls

    def test_original_inputs_path_fails_native_guard_before_file_write(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            algo,out,calls=self.fixture(d)
            inputs=Path(d)/'inputs';inputs.mkdir()
            source=inputs/'model_final.pt';source.write_bytes(b'original immutable input')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'not in the logging directory'):
                contract.roundtrip(algo,inputs/'resume_preflight.pt')
            self.assertFalse((inputs/'resume_preflight.pt').exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            self.assertEqual(calls,[])
            self.assertEqual(algo.current_learning_iteration,200)

    def test_corrected_preflight_native_roundtrip_preserves_optimizer_iteration(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            algo,out,calls=self.fixture(d)
            result=contract.roundtrip(algo,out/'resume_preflight.pt')
            self.assertTrue(result['actor_critic_optimizer_iteration_exact'])
            self.assertEqual(result['iteration'],200)
            self.assertTrue(algo.actor_optimizer.state)
            self.assertEqual(calls,[str(out/'resume_preflight.pt')])

    def test_final_checkpoint_also_obeys_native_guard(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            algo,out,calls=self.fixture(d)
            algo.current_learning_iteration=201
            result=contract.roundtrip(algo,out/'model_final.pt')
            self.assertEqual(result['iteration'],201)
            self.assertTrue(result['native_save_reload_verified'])
            self.assertEqual(calls,[str(out/'model_final.pt')])

    def test_existing_preflight_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as d:
            algo,out,calls=self.fixture(d)
            path=out/'resume_preflight.pt';path.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):contract.roundtrip(algo,path)
            self.assertEqual(path.read_bytes(),b'preserve');self.assertEqual(calls,[])

    def test_both_worker_roundtrip_paths_are_under_training_log_directory(self):
        fn=next(n for n in ast.parse(Path(resume.__file__).read_text()).body
                if isinstance(n,ast.FunctionDef) and n.name=='train_worker')
        calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='roundtrip']
        self.assertEqual(len(calls),2)
        for call in calls:
            path=call.args[1]
            self.assertIsInstance(path,ast.BinOp)
            self.assertEqual(ast.unparse(path.left),'out')
            self.assertIn(ast.literal_eval(path.right),('resume_preflight.pt','model_final.pt'))
        text=ast.unparse(fn)
        self.assertIn("out = OUT / 'training'",text)
        self.assertLess(text.index('preflight = roundtrip'),text.index('SOURCE_ITERATION + 1'))
        self.assertLess(text.index('preflight = roundtrip'),text.index('trained = train'))

    def test_approved_training_eval_and_input_contracts_unchanged(self):
        self.assertEqual((resume.SOURCE_ITERATION,resume.MAX_UPDATES,resume.TRAIN_SECONDS,resume.WORKER_SECONDS),(200,200,300,1800))
        self.assertEqual(len(resume.ARMS),7)
        self.assertEqual(resume.SEEDS,(1402,2402,3402))
        self.assertEqual(resume.SOURCE_SHA,'3dd4286a4b5e825f8e224f476cc5cc510a375c77a9ffabd168f518d946c680f1')

if __name__=='__main__':unittest.main()