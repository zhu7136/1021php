"""Reproduce nested shell quoting at CPU scale,using synthetic scoped URLs only."""
import json,pathlib,re,subprocess,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import launch_envelope as env
import reliable_delivery as delivery
import train_delivery_smoke as smoke

class EnvelopeTests(unittest.TestCase):
    def inputs(self):
        path='test-team/HuD04_WBT/1234abcd'
        def url(run,file):return 'https://storage.googleapis.com/wandb-production.appspot.com/'+run+'/'+file+'?X-Goog-Signature=synthetic&X-Goog-Date=20261025T031109Z&X-Goog-Expires=3600'
        outputs={k:dict(put_url=url(path,v)+'&method=put',get_url=url(path,v)+'&method=get' if k=='probe' else None,headers={}) for k,v in delivery.FILES.items()}
        return path,url('test-team/HuD04_WBT/94728bf7','protected_pilot_bundle.zip'),dict(schema_version=1,run_path=path,outputs=outputs)
    def test_roundtrip_preserves_every_url_byte(self):
        path,url,manifest=self.inputs();data=env.decode(env.encode(path,url,manifest))
        self.assertEqual((data['run_path'],data['input_url'],data['delivery_manifest']),(path,url,manifest))
    def test_entire_command_has_no_quote_dollar_ampersand_or_shell_metacharacters(self):
        cmd=env.start_script(*self.inputs())
        self.assertRegex(cmd,r'^gm-run 1021php/flux/train_delivery_smoke.py --launch-envelope [A-Za-z0-9_-]+$')
        self.assertFalse(re.search(r'''[&'"$;|<>`\\\n]''',cmd))
    def test_single_and_double_quoted_platform_assignment_keep_command_intact(self):
        cmd=env.start_script(*self.inputs())
        for quote in ("'",'"'):
            script='START_SCRIPT='+quote+cmd+quote+'\nprintf "%s" "$START_SCRIPT"'
            p=subprocess.run(['/bin/sh','-c',script],capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout,cmd);self.assertEqual(p.stderr,'')
    def test_repeated_shell_roundtrips_keep_encoded_payload_intact(self):
        cmd=env.start_script(*self.inputs());original=cmd
        for quote in ("'",'"',"'"):
            p=subprocess.run(['/bin/sh','-c','START_SCRIPT='+quote+cmd+quote+';printf "%s" "$START_SCRIPT"'],capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,0);cmd=p.stdout
        self.assertEqual(cmd,original);self.assertEqual(env.decode(cmd.split()[-1])['input_url'],self.inputs()[1])
    def test_decoded_data_only_becomes_python_argv(self):
        args=env.arguments(env.encode(*self.inputs()));self.assertEqual(args[0],'--wandb-run-path')
        self.assertEqual(args[3],self.inputs()[1]);self.assertEqual(json.loads(args[5]),self.inputs()[2])
    def test_bad_tokens_rejected_without_echoing_input(self):
        for token in ('',"x'",'x&y','$(echo secret)','a'*64001,'not-valid-json'):
            with self.subTest(length=len(token)),self.assertRaises(ValueError) as cm:env.decode(token)
            self.assertNotIn('secret',str(cm.exception))
    def test_foreign_run_or_source_is_rejected(self):
        path,url,manifest=self.inputs()
        for bad in (url.replace('94728bf7','aaaaaaaa'),url.replace('storage.googleapis.com','example.org')):
            with self.assertRaises(ValueError):env.encode(path,bad,manifest)
        manifest['run_path']='test-team/HuD04_WBT/abcd1234'
        with self.assertRaises(ValueError):env.encode(path,url,manifest)
    def test_wrapper_expands_before_dispatch(self):
        token=env.encode(*self.inputs());argv=['train_delivery_smoke.py','--task_id=TASK_SYNTHETIC','--launch-envelope',token]
        with patch.object(sys,'argv',argv),patch.object(smoke,'configure'),patch.object(smoke.resume,'main') as fn:
            smoke.main();fn.assert_called_once();self.assertNotIn('--launch-envelope',sys.argv)
            self.assertIn('--task_id=TASK_SYNTHETIC',sys.argv);self.assertEqual(sys.argv[sys.argv.index('--input-url')+1],self.inputs()[1])
    def test_missing_or_duplicate_token_rejected_before_dispatch(self):
        for argv in (['smoke','--launch-envelope'],['smoke','--launch-envelope','a','--launch-envelope','b']):
            with patch.object(sys,'argv',argv),patch.object(smoke.resume,'main') as fn:
                with self.assertRaises(ValueError):smoke.main()
                fn.assert_not_called()

if __name__=='__main__':unittest.main()