"""CPU-only Python control-flow audit; not a simulator teardown test."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parent
main=next(n for n in ast.parse((ROOT/'source/three_teacher_child.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Try))
code=compile(ast.fix_missing_locations(ast.Module(body=main.body[index:],type_ignores=[])),'child_exit_audit_control_flow','exec')

class ChildExitAuditTests(unittest.TestCase):
    def execute(self,error=None,close_error=None):
        report={'train_returned':False,'close_returned':False};events=[]
        def train(*a,**k):
            events.append('train')
            if error is not None:raise error
        def write(out,r):events.append(('write',dict(r)))
        def close(app,out,r):
            events.append('close')
            if close_error is not None:raise close_error
            r['close_returned']=True
        scope={'app':object(),'cfg':object(),'out':ROOT,'report':report,'json':json,
            'init_sim_imports':lambda cfg:object(),
            'train_agent':SimpleNamespace(train=train),
            'close_audit':SimpleNamespace(configure=lambda app,out,r:None,write=write,close=close)}
        with redirect_stdout(io.StringIO()):
            try:exec(code,scope)
            except BaseException as exc:return report,events,exc
        return report,events,None
    def test_system_exit_is_recorded_and_propagated(self):
        r,e,x=self.execute(SystemExit(1));self.assertIsInstance(x,SystemExit);self.assertEqual(x.code,1)
        self.assertEqual(r['train_exception_type'],'SystemExit');self.assertEqual(r['train_system_exit_code'],1)
        self.assertFalse(r['train_returned']);self.assertTrue(r['close_returned']);self.assertNotIn('child_returned',r)
    def test_exception_is_not_hidden_by_close(self):
        r,e,x=self.execute(ValueError('fixture'));self.assertIsInstance(x,ValueError)
        self.assertEqual(r['train_exception_type'],'ValueError');self.assertTrue(r['close_returned'])
    def test_normal_python_path_records_both_returns(self):
        r,e,x=self.execute();self.assertIsNone(x)
        self.assertTrue(all(r[k] for k in ('train_returned','close_returned','child_returned')))
    def test_close_exception_prevents_success_marker(self):
        r,e,x=self.execute(close_error=RuntimeError('fixture close failed'))
        self.assertIsInstance(x,RuntimeError);self.assertFalse(r['close_returned']);self.assertNotIn('child_returned',r)

if __name__=='__main__':unittest.main(verbosity=2)