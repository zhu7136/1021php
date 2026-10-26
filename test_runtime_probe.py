import json,os
from pathlib import Path
import sys,tempfile,unittest
from runtime_probe import PREFIX,supervise

class SupervisorTests(unittest.TestCase):
    def run_child(self,variant):
        root=Path(tempfile.mkdtemp(prefix='r6-cpu-',dir='/workspace/hud04_r6'))
        names=['image_dependency_inventory','gpu_properties_recorded','resource_contract',
               'versions_verified','app_launch_returned','fast_shutdown_verified','cuda_verified',
               'empty_app_updates_completed','close_started','close_returned']
        if variant=='missing_close':names.pop()
        if variant=='hang_close':names.pop()
        if variant=='failure':names.insert(0,'child_failure')
        if variant=='total_timeout':names=[]
        rows=[{'event':n,'value':False,'updates':8,'passed':True} for n in names]
        if variant=='bad_setting':next(r for r in rows if r['event']=='fast_shutdown_verified')['value']=True
        if variant=='resource_mismatch':next(r for r in rows if r['event']=='resource_contract')['passed']=False
        if variant=='missing_inventory':rows=[r for r in rows if r['event']!='image_dependency_inventory']
        code='import json,time;rows='+repr(rows)+'\nfor row in rows:print('+repr(PREFIX)+'+json.dumps(row),flush=True)\n'
        if variant in {'hang_close','hang_after_close','total_timeout'}:code+='time.sleep(60)\n'
        return supervise([sys.executable,'-c',code],cwd=root,
            env={'PATH':os.environ.get('PATH',''),'PYTHONUNBUFFERED':'1'},out=root/'result',seconds=5.35,close_seconds=.2)
    def test_success(self):self.assertTrue(self.run_child('success')['passed'])
    def test_no_close_return_even_zero_exit_is_failure(self):self.assertFalse(self.run_child('missing_close')['passed'])
    def test_failed_setting_is_failure(self):self.assertFalse(self.run_child('bad_setting')['passed'])
    def test_recorded_failure_cannot_be_hidden(self):self.assertFalse(self.run_child('failure')['passed'])
    def test_resource_mismatch_cannot_be_hidden(self):self.assertFalse(self.run_child('resource_mismatch')['passed'])
    def test_inventory_required(self):self.assertFalse(self.run_child('missing_inventory')['passed'])
    def test_close_timeout_reaped(self):
        r=self.run_child('hang_close');self.assertFalse(r['passed']);self.assertEqual(r['reason'],'close_timeout');self.assertIsNotNone(r['returncode'])
    def test_after_close_hang_still_fails(self):self.assertFalse(self.run_child('hang_after_close')['passed'])
    def test_total_timeout_reaped(self):
        r=self.run_child('total_timeout');self.assertFalse(r['passed']);self.assertEqual(r['reason'],'total_timeout');self.assertIsNotNone(r['returncode'])

if __name__=='__main__':unittest.main(verbosity=2)
