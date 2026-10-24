"""Regression for TASK_20261024_018: incomplete clone must fail before pip."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_preflight import SUBMODULE_URLS, verify_sources
import git_smoke_v8 as smoke


class SourcePreflightTests(unittest.TestCase):
    def setUp(self):
        base=Path('/workspace/hud04_source_preflight_tests'); base.mkdir(exist_ok=True)
        self.root=Path(tempfile.mkdtemp(dir=base))
        self.mapping=''.join(f'[submodule "{p}"]\n\tpath = {p}\n\turl = {u}\n' for p,u in SUBMODULE_URLS.items())
        (self.root/'.gitmodules').write_text(self.mapping)
        source=self.root/'thirdparty/holosoma/src/holosoma/holosoma/train_agent.py'
        source.parent.mkdir(parents=True); source.write_text('# fixture\n')
        (self.root/'flux').mkdir()
        (self.root/'flux/holosoma-base.json').write_text(json.dumps({
            'src/holosoma/holosoma/train_agent.py': hashlib.sha256(source.read_bytes()).hexdigest()}))
        self.source=source

    def test_complete_fixture_passes(self):
        self.assertEqual(verify_sources(self.root)['submodule_mappings'],3)

    def test_missing_gmr_mapping_rejected(self):
        start=self.mapping.index('[submodule "humanoid-description"]')
        (self.root/'.gitmodules').write_text(self.mapping[start:])
        with self.assertRaisesRegex(ValueError,'GMR'): verify_sources(self.root)

    def test_wrong_upstream_rejected(self):
        (self.root/'.gitmodules').write_text(self.mapping.replace('limxdynamics/GMR.git','example/wrong.git'))
        with self.assertRaisesRegex(ValueError,'mapping'): verify_sources(self.root)

    def test_incomplete_clone_rejected(self):
        with patch.object(Path,'is_file',return_value=False):
            with self.assertRaisesRegex(ValueError,'Missing pinned Holosoma'): verify_sources(self.root)

    def test_changed_pinned_source_rejected(self):
        self.source.write_text('# wrong revision\n')
        with self.assertRaisesRegex(ValueError,'changed pinned'): verify_sources(self.root)

    def test_source_failure_never_assembles_installs_or_trains(self):
        self.source.write_text('# wrong revision\n')
        output=io.StringIO()
        with patch.object(smoke,'ROOT',self.root), patch('git_parts_v8.assemble_parts') as assemble, \
             patch('offline_dependencies.install') as install, patch('runpy.run_path') as train, \
             redirect_stdout(output), self.assertRaises(ValueError):
            smoke.worker(None,data_parts_dir='unused')
        assemble.assert_not_called(); install.assert_not_called(); train.assert_not_called()
        events=[json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
        self.assertEqual(events[-1]['stage'],'source_preflight')
        self.assertEqual(events[-1]['event'],'stage_failed')

    def test_actual_checkout_has_all_gitlink_mappings(self):
        import configparser, subprocess
        root=Path(__file__).resolve().parents[2]
        config=configparser.ConfigParser(); config.read(root/'.gitmodules')
        lines=subprocess.check_output(['git','-C',str(root),'ls-files','--stage'],text=True).splitlines()
        paths=[line.split('\t',1)[1] for line in lines if line.startswith('160000 ')]
        self.assertEqual(set(paths),set(SUBMODULE_URLS))
        for path in paths:
            self.assertEqual(config[f'submodule "{path}"']['path'],path)
            self.assertEqual(config[f'submodule "{path}"']['url'],SUBMODULE_URLS[path])


if __name__=='__main__': unittest.main()