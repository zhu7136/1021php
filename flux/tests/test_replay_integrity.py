"""Actual CPU codec checks and failure-path tests; no simulator, network or GPU."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

FLUX=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(FLUX))
import replay_integrity as integrity
from test_finite_replay import video_utils

spec=importlib.util.spec_from_file_location('bounded_replay_bootstrap_under_test',FLUX/'git_replay_v8.py')
bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
SHA=bootstrap.EXPECTED_SHA256


def fixture(out):
    """Synthetic artifact fixture, never a model evaluation result."""
    out.mkdir(parents=True)
    video=video_utils.create_video(np.zeros((1,360,640,3),dtype=np.uint8),50,
        out/'videos',output_format='mp4',wandb_logging=False,episode_id=1)
    shapes=dict(root_pos=(1000,3),root_quat_xyzw=(1000,4),body_pos=(1000,3,3),
        joint_pos=(1000,29),joint_vel=(1000,29),contact_force_norm=(1000,3),
        root_tilt_rad=(1000,),root_clearance_proxy_m=(1000,),reset=(1000,),
        timeout=(1000,),actions=(1000,29))
    arrays={key:np.zeros(shape) for key,shape in shapes.items()}
    arrays['root_quat_xyzw'][:,3]=1
    np.savez_compressed(out/'trajectory.npz',**arrays)
    summary=dict(steps=1000,policy_steps=999,initialization_zero_action_steps=1,
        num_envs=1,training_updates=0,terrain_variants=32,checkpoint_sha256=SHA,
        checkpoint_unchanged=True,actor_weights_unchanged=True,joint_mapping_runtime_verified=True,
        all_states_and_actions_finite=True,resets=0,timeouts=0,action_abs_max=0,
        body_names=['foot','pelvis','hand'],videos=[video.relative_to(out).as_posix()])
    (out/'summary.json').write_text(json.dumps(summary))
    (out/'joint_mapping.json').write_text(json.dumps([
        dict(policy_index=i,joint='joint_'+str(i),isaac_index=28-i) for i in range(29)]))
    (out/'events.jsonl').write_text('')
    (out/'replay_config.yaml').write_text('synthetic_fixture: true\n')
    (out/'encoder_preflight.json').write_text(json.dumps(dict(output_format='mp4',fixture=True)))
    return summary


def receipt(out):
    bundle=out.parent/'replay_bundle.zip'
    with zipfile.ZipFile(bundle,'w') as target:
        for path in out.rglob('*'):
            if path.is_file():target.write(path,path.relative_to(out).as_posix())
    result=dict(status='uploaded',bytes=bundle.stat().st_size,sha256=integrity.file_digest(bundle))
    (out.parent/'replay_bundle_receipt.json').write_text(json.dumps(result))


class CodecTests(unittest.TestCase):
    def test_real_pinned_codec_needs_no_ffmpeg_even_on_one_frame_reset(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory, \
                patch.object(video_utils.subprocess,'run',side_effect=FileNotFoundError('ffmpeg')) as process:
            result=integrity.preflight_mp4(video_utils.create_video,Path(directory)/'codec')
            self.assertEqual([item['frames'] for item in result['tests']],[1,3])
            self.assertFalse(result['ffmpeg_required'])
            process.assert_not_called()

    def test_no_file_from_encoder_fails(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            with self.assertRaisesRegex(RuntimeError,'no file'):
                integrity.preflight_mp4(lambda *a,**kw:None,Path(directory)/'codec')

    def test_undecodable_file_fails(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            path=Path(directory)/'broken.mp4';path.write_bytes(b'not a video')
            with self.assertRaises(RuntimeError):integrity.inspect_mp4(path)

    def test_wrong_number_of_frames_fails(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            path=video_utils.create_video(np.zeros((1,360,640,3),dtype=np.uint8),50,
                Path(directory),output_format='mp4',wandb_logging=False)
            with self.assertRaises(RuntimeError):integrity.inspect_mp4(path,expected_frames=3)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir='/workspace')
        self.addCleanup(self.directory.cleanup)
        self.out=Path(self.directory.name)/'outputs'/'replay'
        self.summary=fixture(self.out)

    def test_sealed_complete_outputs_pass_including_zero_resets(self):
        integrity.complete_replay(self.out,SHA)
        self.assertEqual(integrity.verify_completion(self.out,SHA)['steps'],1000)

    def test_partial_replay_never_counts_as_complete(self):
        with self.assertRaises(RuntimeError):integrity.verify_completion(self.out,SHA)

    def test_false_step_count_rejected(self):
        self.summary['steps']=1
        (self.out/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaisesRegex(RuntimeError,'steps'):integrity.complete_replay(self.out,SHA)

    def test_non_readonly_summary_rejected(self):
        self.summary['actor_weights_unchanged']=False
        (self.out/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaises(RuntimeError):integrity.complete_replay(self.out,SHA)

    def test_nonfinite_trajectory_rejected_even_if_summary_claims_finite(self):
        with np.load(self.out/'trajectory.npz') as saved:
            arrays={key:saved[key] for key in saved.files}
        arrays['actions'][3,0]=np.nan
        np.savez_compressed(self.out/'trajectory.npz',**arrays)
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):integrity.complete_replay(self.out,SHA)

    def test_truncated_trajectory_rejected(self):
        with np.load(self.out/'trajectory.npz') as saved:
            arrays={key:saved[key] for key in saved.files}
        arrays['actions']=arrays['actions'][:-1]
        np.savez_compressed(self.out/'trajectory.npz',**arrays)
        with self.assertRaisesRegex(RuntimeError,'shape'):integrity.complete_replay(self.out,SHA)

    def test_missing_video_rejected(self):
        self.summary['videos']=[]
        (self.out/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaisesRegex(RuntimeError,'videos'):integrity.complete_replay(self.out,SHA)

    def test_temporary_video_rejected(self):
        self.summary['videos']=['videos/temp_raw.mp4']
        (self.out/'videos/temp_raw.mp4').write_bytes(b'fixture')
        (self.out/'summary.json').write_text(json.dumps(self.summary))
        with self.assertRaisesRegex(RuntimeError,'temporary'):integrity.complete_replay(self.out,SHA)

    def test_post_validation_file_change_rejected_by_parent(self):
        integrity.complete_replay(self.out,SHA)
        (self.out/'events.jsonl').write_text('{"fixture":"changed"}\n')
        with self.assertRaisesRegex(RuntimeError,'changed'):integrity.verify_completion(self.out,SHA)

    def test_upload_receipt_requires_exact_hash(self):
        integrity.complete_replay(self.out,SHA);receipt(self.out)
        self.assertEqual(integrity.verify_upload_receipt(self.out)['status'],'uploaded')
        with (self.out.parent/'replay_bundle.zip').open('ab') as target:target.write(b'x')
        with self.assertRaises(RuntimeError):integrity.verify_upload_receipt(self.out)

    def test_parent_rejects_missing_upload_receipt(self):
        integrity.complete_replay(self.out,SHA)
        with self.assertRaises(RuntimeError):integrity.verify_upload_receipt(self.out)

    def test_reset_event_mismatch_rejected(self):
        (self.out/'events.jsonl').write_text('{"event":"pre_reset","step":0}\n')
        with self.assertRaisesRegex(RuntimeError,'resets'):integrity.complete_replay(self.out,SHA)


class BootstrapTests(unittest.TestCase):
    def test_child_failure_writes_diagnostics_and_attempts_one_upload(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            root=Path(directory);out=root/'outputs/replay'
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'OUT',out), \
                    patch.object(sys,'argv',['bootstrap','--runtime-child','--upload-url',
                        'https://storage.googleapis.com/test']), \
                    patch.object(bootstrap,'worker',side_effect=RuntimeError('fixture')), \
                    patch.object(bootstrap,'upload_bundle') as upload, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'fixture'):bootstrap.main()
            self.assertEqual(json.loads((out/'failure.json').read_text())['exception_type'],'RuntimeError')
            self.assertFalse((out/'completion.json').exists())
            upload.assert_called_once()

    def test_upload_success_persists_hash_receipt_after_acknowledgment(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            root=Path(directory);out=root/'outputs/replay';fixture(out)
            response=Mock(status=200)
            context=Mock();context.__enter__=Mock(return_value=response);context.__exit__=Mock(return_value=False)
            opener=Mock();opener.open.return_value=context
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'OUT',out), \
                    patch.object(bootstrap.urllib.request,'build_opener',return_value=opener), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(bootstrap.upload_bundle('https://storage.googleapis.com/test',{}))
            self.assertEqual(integrity.verify_upload_receipt(out)['status'],'uploaded')
            opener.open.assert_called_once()

    def test_failed_upload_has_no_success_receipt_and_no_retry(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            root=Path(directory);out=root/'outputs/replay';fixture(out)
            response=Mock(status=500)
            context=Mock();context.__enter__=Mock(return_value=response);context.__exit__=Mock(return_value=False)
            opener=Mock();opener.open.return_value=context
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'OUT',out), \
                    patch.object(bootstrap.urllib.request,'build_opener',return_value=opener):
                with self.assertRaises(RuntimeError):bootstrap.upload_bundle('https://storage.googleapis.com/test',{})
            self.assertFalse((out.parent/'replay_bundle_receipt.json').exists())
            opener.open.assert_called_once()

    def invoke_parent(self,after_child):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            root=Path(directory);out=root/'outputs/replay'
            runtime=types.SimpleNamespace(resolve_isaac_launcher=lambda:Path('/fake/isaac/python.sh'))
            diagnostics=types.SimpleNamespace(wait_for_child=lambda *a:after_child(out))
            output=io.StringIO()
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'OUT',out), \
                    patch.object(sys,'argv',['bootstrap','--upload-url','https://storage.googleapis.com/test']), \
                    patch.dict(sys.modules,{'runtime_preflight':runtime,'startup_diagnostics':diagnostics}), \
                    patch.object(bootstrap.subprocess,'Popen') as popen,contextlib.redirect_stdout(output):
                try:
                    bootstrap.main()
                    code=0
                except SystemExit as exc:code=exc.code
            return code,output.getvalue(),popen.call_count

    def test_child_exit_zero_with_no_result_is_failure_not_success(self):
        code,output,calls=self.invoke_parent(lambda out:0)
        self.assertEqual(code,1);self.assertEqual(calls,1)
        self.assertIn('replay_completion_rejected',output)
        self.assertNotIn('replay_entrypoint_complete',output)

    def test_child_exit_zero_with_partial_result_is_failure(self):
        def child(out):
            fixture(out)
            return 0
        code,output,calls=self.invoke_parent(child)
        self.assertEqual(code,1);self.assertEqual(calls,1)
        self.assertNotIn('replay_entrypoint_complete',output)

    def test_valid_outputs_but_no_upload_receipt_is_failure(self):
        def child(out):
            fixture(out);integrity.complete_replay(out,SHA)
            return 0
        code,output,calls=self.invoke_parent(child)
        self.assertEqual(code,1)
        self.assertNotIn('replay_entrypoint_complete',output)

    def test_verified_artifacts_and_acknowledged_upload_are_success(self):
        def child(out):
            fixture(out);integrity.complete_replay(out,SHA);receipt(out)
            return 0
        code,output,calls=self.invoke_parent(child)
        self.assertEqual(code,0);self.assertEqual(calls,1)
        self.assertIn('replay_artifacts_verified',output)
        self.assertIn('replay_entrypoint_complete',output)

    def test_stale_output_refused_before_any_launch(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as directory:
            root=Path(directory);out=root/'outputs/replay';out.mkdir(parents=True)
            with patch.object(bootstrap,'ROOT',root),patch.object(bootstrap,'OUT',out), \
                    patch.object(sys,'argv',['bootstrap','--upload-url','https://storage.googleapis.com/test']), \
                    patch.object(bootstrap.subprocess,'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError,'stale'):bootstrap.main()
            popen.assert_not_called()


if __name__=='__main__':unittest.main()