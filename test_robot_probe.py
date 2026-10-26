import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from robot_probe import PREFIX, STEPS, accepted, event, resource_ok, safe_member, supervise, extract_assets
from hud04_robot import SDK_JOINT_NAMES, POLICY_JOINT_NAMES, HEAD_JOINT_NAMES, resolve_policy_indices, calibration, default_position

def fixture():
    names = ['versions_verified', 'app_launch_returned', 'fast_shutdown_verified', 'resource_contract',
             'physics_context_created', 'robot_initialized', 'physics_steps_completed',
             'physics_cleanup_returned', 'close_started', 'close_returned']
    rows = [{'event': n} for n in names]
    for r in rows:
        if r['event'] == 'fast_shutdown_verified': r['value'] = False
        if r['event'] == 'resource_contract': r['passed'] = True
        if r['event'] == 'robot_initialized': r.update(physical_joints=31, policy_joints=29, floating_base=True)
        if r['event'] == 'physics_steps_completed': r.update(steps=STEPS, states_finite=True, head_short_probe_passed=True)
    return rows

class Tests(unittest.TestCase):
    def test_read_only(self):
        r = subprocess.run([sys.executable, str(Path(__file__).with_name('robot_probe.py'))], capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 0)
        self.assertFalse(json.loads(r.stdout.removeprefix(PREFIX))['gpu_started'])
    def test_event_name(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out): event('gpu', name='NVIDIA GeForce RTX 4090')
        self.assertEqual(json.loads(out.getvalue().removeprefix(PREFIX))['event'], 'gpu')
    def test_names(self):
        self.assertEqual(len(SDK_JOINT_NAMES), 31)
        self.assertEqual(len(POLICY_JOINT_NAMES), 29)
        self.assertFalse(set(POLICY_JOINT_NAMES) & set(HEAD_JOINT_NAMES))
    def test_mapping_permutation(self):
        names = tuple(reversed(SDK_JOINT_NAMES))
        self.assertEqual(tuple(names[i] for i in resolve_policy_indices(names)), POLICY_JOINT_NAMES)
    def test_old29_rejected(self):
        with self.assertRaises(ValueError): resolve_policy_indices(POLICY_JOINT_NAMES)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): resolve_policy_indices(SDK_JOINT_NAMES[:-1] + (SDK_JOINT_NAMES[0],))
    def test_head_targets(self):
        for n in HEAD_JOINT_NAMES:
            self.assertEqual(default_position(n), 0.)
            self.assertEqual(calibration(n)[:2], (20., .8))
    def test_gains_defined_all31(self):
        for n in SDK_JOINT_NAMES:
            self.assertTrue(all(x > 0 for x in calibration(n)))
    def test_diagnostic_resource(self):
        self.assertTrue(resource_ok(dict(count=1, name='NVIDIA GeForce RTX 4090', total_memory_bytes=24*2**30, major=8, minor=9)))
    def test_bad_resource(self):
        for change in [dict(count=2), dict(name='CPU_FIXTURE'), dict(total_memory_bytes=8*2**30), dict(major=9)]:
            d = dict(count=1, name='NVIDIA GeForce RTX 4090', total_memory_bytes=24*2**30, major=8, minor=9)
            d.update(change)
            self.assertFalse(resource_ok(d))
    def test_path_escape(self):
        for name in ['../x', '/x', 'a/../../b', 'a\\b']:
            with self.assertRaises(ValueError): safe_member(name)
    def test_pass_fixture(self): self.assertTrue(accepted(fixture(), 0, 'child_exited'))
    def test_missing_close(self): self.assertFalse(accepted(fixture()[:-1], 0, 'child_exited'))
    def test_nonzero_exit(self): self.assertFalse(accepted(fixture(), 1, 'child_exited'))
    def test_hang_even_close_returned(self): self.assertFalse(accepted(fixture(), 0, 'total_timeout'))
    def test_failure_event(self): self.assertFalse(accepted(fixture() + [{'event':'child_failure'}], 0, 'child_exited'))
    def test_finite_and_head(self):
        for key in ['states_finite', 'head_short_probe_passed']:
            rows = fixture()
            next(r for r in rows if r['event']=='physics_steps_completed')[key] = False
            self.assertFalse(accepted(rows, 0, 'child_exited'))
    def test_no_empty_app_shortcut(self):
        rows = [r for r in fixture() if r['event'] != 'robot_initialized']
        self.assertFalse(accepted(rows, 0, 'child_exited'))
    def test_all64_steps(self):
        rows = fixture()
        next(r for r in rows if r['event']=='physics_steps_completed')['steps'] = 8
        self.assertFalse(accepted(rows, 0, 'child_exited'))
    def test_floating_base(self):
        rows = fixture()
        next(r for r in rows if r['event']=='robot_initialized')['floating_base'] = False
        self.assertFalse(accepted(rows, 0, 'child_exited'))
    def test_required_cleanup(self):
        self.assertFalse(accepted([r for r in fixture() if r['event']!='physics_cleanup_returned'], 0, 'child_exited'))
    def test_real_subprocess(self):
        root = Path(tempfile.mkdtemp(prefix='r8-cpu-', dir=str(Path(__file__).parent)))
        code = 'import json; rows=' + repr(fixture()) + '\nfor r in rows: print(' + repr(PREFIX) + '+json.dumps(r),flush=True)'
        with contextlib.redirect_stdout(io.StringIO()):
            r = supervise([sys.executable, '-c', code], root, {'PATH':os.environ.get('PATH','')}, root/'out', 6)
        self.assertTrue(r['passed'])
    def test_real_timeout(self):
        root = Path(tempfile.mkdtemp(prefix='r8-cpu-', dir=str(Path(__file__).parent)))
        with contextlib.redirect_stdout(io.StringIO()):
            r = supervise([sys.executable, '-c', 'import time;time.sleep(60)'], root, {'PATH':os.environ.get('PATH','')}, root/'out', 5.2)
        self.assertFalse(r['passed'])
        self.assertEqual(r['reason'], 'total_timeout')
        self.assertIsNotNone(r['returncode'])
    def test_real_asset_roundtrip(self):
        root = Path(__file__).parent
        out = Path(tempfile.mkdtemp(prefix='r8-assets-', dir=str(root))) / 'extract'
        usd = extract_assets(root, out)
        self.assertTrue(usd.is_file())
        self.assertEqual(len([p for p in (out/'asset').rglob('*') if p.is_file()]),39)
        with self.assertRaises(FileExistsError): extract_assets(root, out)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    Path(__file__).with_name('cpu-test-report.json').write_text(json.dumps(dict(tests_run=result.testsRun, passed=result.wasSuccessful(),
        gpu_executed=False, native_isaacsim_executed=False), indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())