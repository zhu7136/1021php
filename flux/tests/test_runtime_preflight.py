"""CPU-only vendor-launcher/asset regressions, with no simulator launch."""
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime_preflight as runtime
import git_smoke_v8 as smoke


class RuntimePreflightTests(unittest.TestCase):
    def setUp(self):
        base=Path('/workspace/hud04_runtime_preflight_tests'); base.mkdir(exist_ok=True)
        self.root=Path(tempfile.mkdtemp(dir=base))
        self.asset=self.root/'flux/holosoma_overlay/data/robots/hu_d04'
        self.asset.mkdir(parents=True)
        (self.asset/'HU_D04_01_train.urdf').write_text('<robot><link><visual><geometry><mesh filename="one.STL"/></geometry></visual></link></robot>')
        self.mesh=self.asset/'one.STL'; self.mesh.write_bytes(b'\0'*80+struct.pack('<I',1)+b'\0'*50)
        self.launcher=self.root/'isaac/python.sh'; self.launcher.parent.mkdir()
        self.launcher.write_text('#!/bin/sh\nexit 99\n'); self.launcher.chmod(0o700)
        python=self.launcher.parent/'kit/python/bin/python3'; python.parent.mkdir(parents=True)
        python.write_text('fixture, never executed')

    def test_actual_training_meshes_pass_without_lfs(self):
        report=runtime.verify_assets(Path(__file__).resolve().parents[2])
        self.assertEqual(report['isaacsim_urdf_meshes_verified'],32)

    def test_missing_mesh_fails(self):
        with patch.object(Path,'is_file',side_effect=lambda:False):
            with self.assertRaisesRegex(ValueError,'Missing'):runtime.verify_assets(self.root)

    def test_lfs_pointer_fails(self):
        self.mesh.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:fixture\n')
        with self.assertRaisesRegex(ValueError,'LFS pointer'):runtime.verify_assets(self.root)

    def test_truncated_stl_fails(self):
        self.mesh.write_bytes(b'\0'*80+struct.pack('<I',3)+b'\0'*50)
        with self.assertRaisesRegex(ValueError,'STL size'):runtime.verify_assets(self.root)

    def test_default_layout_selects_vendor_script(self):
        with patch.dict(os.environ,{},clear=True),patch.object(runtime,'LAUNCHER_CANDIDATES',(self.launcher,)):
            self.assertEqual(runtime.resolve_isaac_launcher(),self.launcher)

    def test_override_selects_vendor_script(self):
        with patch.dict(os.environ,{'HUD04_ISAAC_PYTHON_SH':str(self.launcher)},clear=True):
            self.assertEqual(runtime.resolve_isaac_launcher(),self.launcher)

    def test_invalid_override_does_not_silently_fall_back(self):
        with patch.dict(os.environ,{'HUD04_ISAAC_PYTHON_SH':'/missing/python.sh'},clear=True), \
             patch.object(runtime,'LAUNCHER_CANDIDATES',(self.launcher,)):
            with self.assertRaisesRegex(RuntimeError,'No verified'):runtime.resolve_isaac_launcher()

    def test_system_python_override_rejected(self):
        with patch.dict(os.environ,{'HUD04_ISAAC_PYTHON_SH':sys.executable},clear=True):
            with self.assertRaisesRegex(RuntimeError,'no system-Python fallback'):runtime.resolve_isaac_launcher()

    def test_script_without_bundled_interpreter_rejected(self):
        other=self.root/'wrong/python.sh';other.parent.mkdir();other.write_text('#!/bin/sh\n');other.chmod(0o700)
        with patch.dict(os.environ,{'HUD04_ISAAC_PYTHON_SH':str(other)},clear=True):
            with self.assertRaisesRegex(RuntimeError,'No verified'):runtime.resolve_isaac_launcher()

    def test_runtime_failure_never_installs_or_trains(self):
        output=io.StringIO()
        with patch.object(smoke,'ROOT',self.root),patch('source_preflight.verify_sources',return_value={}), \
             patch.dict(os.environ,{'HUD04_ISAAC_PYTHON_SH':'/missing/python.sh'},clear=True), \
             patch('offline_dependencies.install') as install,patch('runpy.run_path') as train, \
             redirect_stdout(output),self.assertRaises(RuntimeError):smoke.worker(None,data_parts_dir='unused')
        install.assert_not_called();train.assert_not_called()
        events=[json.loads(x) for x in output.getvalue().splitlines() if x.startswith('{')]
        self.assertEqual(events[-1]['stage'],'runtime_preflight')
        self.assertEqual(events[-1]['event'],'stage_failed')


if __name__=='__main__':unittest.main()