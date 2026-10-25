"""CPU-only deterministic transport/worker regressions. No real cloud writes or GPU."""
import ast,datetime,hashlib,io,json,pathlib,socket,ssl,sys,tempfile,types,unittest,urllib.error
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'flux'))
import reliable_delivery as rd
import train_protected_resume as resume
import prepare_delivery_manifest as controller

RUN='team/project/testrun1'
def manifest():
    outputs={}
    for key,name in rd.FILES.items():
        url='https://storage.googleapis.com/wandb-production.appspot.com/'+RUN+'/'+name
        outputs[key]=dict(put_url=url+'?X-Goog-Signature=private-put',get_url=url+'?X-Goog-Signature=private-get',headers={})
    return dict(schema_version=1,run_path=RUN,outputs=outputs)

class Response(io.BytesIO):
    status=200
    headers={}

class Store:
    def __init__(self):self.data={};self.calls=[];self.failures=[];self.corrupt=False
    def open(self,request,timeout):
        self.calls.append((request.get_method(),request.full_url.split('?')[0]))
        if self.failures:
            fail=self.failures.pop(0)
            if fail:raise fail
        object_path=request.full_url.split('?')[0]
        if request.get_method()=='PUT':self.data[object_path]=request.data;return Response()
        return Response(b'corrupt' if self.corrupt else self.data[object_path])

class ReliableDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/workspace');self.addCleanup(self.tmp.cleanup)
        self.root=pathlib.Path(self.tmp.name);self.store=Store();self.events=[];self.sleeps=[]
    def delivery(self,m=None):
        return rd.Delivery(m or manifest(),self.root,lambda event,**kw:self.events.append(dict(event=event,**kw)),
            open_url=self.store.open,sleep=self.sleeps.append)
    def file(self,name='checkpoint.pt',data=b'new native checkpoint bytes'):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
    def test_probe_put_get_checks_download_bytes_before_ready(self):
        d=self.delivery();r=d.preflight()
        self.assertTrue(d.preflight_verified);self.assertTrue(r['download_sha256_verified'])
        self.assertEqual([x[0] for x in self.store.calls],['PUT','GET'])
        self.assertTrue((self.root/'delivery_receipts.json').is_file())
    def test_no_file_delivery_without_successful_preflight(self):
        with self.assertRaisesRegex(RuntimeError,'preflight'):self.delivery().send_file('checkpoint',self.file())
        self.assertEqual(self.store.calls,[])
    def test_checkpoint_put_get_sha_and_local_file_retained(self):
        d=self.delivery();d.preflight();p=self.file();r=d.send_file('checkpoint',p)
        self.assertEqual(r['sha256'],hashlib.sha256(p.read_bytes()).hexdigest());self.assertTrue(p.exists())
        self.assertEqual(self.store.data[next(k for k in self.store.data if k.endswith('.pt'))],p.read_bytes())
    def test_put_success_get_mismatch_is_failure_not_success(self):
        d=self.delivery();d.preflight();self.store.corrupt=True;p=self.file()
        with self.assertRaises(rd.DeliveryError) as cm:d.send_file('checkpoint',p)
        self.assertEqual(cm.exception.diagnostic['category'],'hash_mismatch');self.assertTrue(p.exists())
        self.assertFalse(any(r['output']=='checkpoint' for r in d.receipts));self.assertEqual(self.sleeps,[])
    def test_http403_not_retried_and_signed_url_never_logged(self):
        d=self.delivery();self.store.failures=[urllib.error.HTTPError(manifest()['outputs']['probe']['put_url'],403,'secret detail',{},None)]
        with self.assertRaises(rd.DeliveryError) as cm:d.preflight()
        self.assertEqual(len(self.store.calls),1);self.assertFalse(d.preflight_verified)
        text=str(cm.exception)+json.dumps(self.events)
        self.assertNotIn('private-put',text);self.assertNotIn('secret detail',text)
        self.assertEqual(cm.exception.diagnostic['http_status'],403)
    def test_transient_http503_retried_only_once(self):
        self.store.failures=[urllib.error.HTTPError('secret',503,'secret',{},None)]
        d=self.delivery();d.preflight();self.assertTrue(d.preflight_verified);self.assertEqual(self.sleeps,[1])
        self.assertEqual(len(self.store.calls),3)
    def test_repeated_transient_failure_stops_after_two_attempts(self):
        self.store.failures=[TimeoutError('signed secret'),TimeoutError('signed secret')]
        with self.assertRaises(rd.DeliveryError):self.delivery().preflight()
        self.assertEqual(len(self.store.calls),2);self.assertEqual(self.sleeps,[1])
    def test_permanent_dns_failure_not_retried(self):
        self.store.failures=[urllib.error.URLError(socket.gaierror(socket.EAI_NONAME,'secret hostname'))]
        with self.assertRaises(rd.DeliveryError) as cm:self.delivery().preflight()
        self.assertEqual(cm.exception.diagnostic['category'],'dns');self.assertEqual(len(self.store.calls),1)
    def test_tls_failure_classified_without_secret_message(self):
        e=urllib.error.URLError(ssl.SSLError('private signed URL'))
        r=rd.error_details(e);self.assertEqual(r['category'],'tls');self.assertNotIn('private',json.dumps(r))
        self.assertFalse(rd.transient(e))
    def test_redirects_are_rejected_and_not_retried(self):
        self.assertIsNone(rd.NoRedirect().redirect_request(None,None,302,'','', 'https://other.example'))
        self.store.failures=[urllib.error.HTTPError('secret',302,'redirect',{},None)]
        with self.assertRaises(rd.DeliveryError):self.delivery().preflight()
        self.assertEqual(len(self.store.calls),1)
    def test_get_transient_failure_reputs_identical_bytes(self):
        self.store.failures=[None,TimeoutError('secret')]
        self.delivery().preflight();self.assertEqual([m for m,_ in self.store.calls],['PUT','GET','PUT','GET'])
        self.assertEqual(self.sleeps,[1])
    def test_reject_non_gcs_destination(self):
        m=manifest();m['outputs']['probe']['put_url']='https://other.example/file?Signature=secret'
        with self.assertRaises(ValueError):self.delivery(m)
    def test_reject_other_run_or_output_destination(self):
        m=manifest();m['outputs']['checkpoint']['put_url']=m['outputs']['checkpoint']['put_url'].replace('/testrun1/','/another/')
        with self.assertRaises(ValueError):self.delivery(m)
    def test_reject_api_credentials_in_headers(self):
        m=manifest();m['outputs']['probe']['headers']['Authorization']='secret'
        with self.assertRaises(ValueError):self.delivery(m)
    def test_reject_header_injection(self):
        m=manifest();m['outputs']['probe']['headers']['Content-Type']='x\nAuthorization: secret'
        with self.assertRaises(ValueError):self.delivery(m)
    def test_all_independent_destinations_required(self):
        m=manifest();m['outputs'].pop('checkpoint')
        with self.assertRaises(ValueError):self.delivery(m)
    def test_get_capability_must_be_distinct_from_put(self):
        m=manifest();m['outputs']['probe']['get_url']=m['outputs']['probe']['put_url']
        with self.assertRaises(ValueError):self.delivery(m)
    def test_reject_unsigned_url(self):
        m=manifest();m['outputs']['probe']['get_url']=m['outputs']['probe']['get_url'].split('?')[0]
        with self.assertRaises(ValueError):self.delivery(m)
    def test_symlink_and_outside_output_root_rejected(self):
        d=self.delivery();d.preflight();p=self.file();link=self.root/'link.pt';link.symlink_to(p)
        with self.assertRaises(ValueError):d.send_file('checkpoint',link)
        with self.assertRaises(ValueError):d.send_file('checkpoint',ROOT/'flux/train_protected_resume.py')
    def test_local_oversized_output_rejected_before_network(self):
        d=self.delivery();d.preflight();p=self.file(data=b'x'*100)
        before=len(self.store.calls)
        with patch.object(rd,'MAX_BYTES',10):
            with self.assertRaises(ValueError):d.send_file('checkpoint',p)
        self.assertEqual(len(self.store.calls),before)
    def test_remote_oversized_download_rejected(self):
        d=self.delivery();d.preflight();p=self.file(data=b'abc');self.store.corrupt=True
        with patch.object(rd,'MAX_BYTES',4):
            with self.assertRaises(rd.DeliveryError):d.send_file('checkpoint',p)
    def test_checkpoint_and_summary_uploaded_before_evaluation(self):
        d=self.delivery();d.preflight();self.file('training/model_final.pt');self.file('training/summary.json',b'{}')
        d.stage_complete('training',self.root/'training')
        self.assertEqual([r['output'] for r in d.receipts],['probe','checkpoint','training_summary'])
        self.file('before-seed1402/evaluation.json',b'{"source_iteration":200}')
        d.stage_complete('before-seed1402',self.root/'before-seed1402')
        self.assertEqual(d.receipts[-1]['output'],'before-seed1402')
    def test_final_bundle_failure_keeps_verified_checkpoint_and_local_files(self):
        d=self.delivery();d.preflight();p=self.file('training/model_final.pt');self.file('training/summary.json',b'{}')
        d.stage_complete('training',p.parent)
        self.store.failures=[urllib.error.HTTPError('secret',403,'secret',{},None)]
        with self.assertRaises(rd.DeliveryError):d.finish()
        self.assertTrue(p.is_file());self.assertTrue((self.root/'protected_resume_bundle.zip').is_file())
        self.assertTrue(any(r['output']=='checkpoint' for r in d.receipts))
    def test_finish_without_preflight_never_retries_upload(self):
        d=self.delivery();d.finish();self.assertEqual(self.store.calls,[])
    def test_failed_preflight_prevents_bootstrap_and_gpu_process_in_actual_main(self):
        out=self.root/'fresh-output';m=manifest()
        self.store.failures=[urllib.error.HTTPError('secret',403,'secret',{},None)]
        original=rd.Delivery
        def ctor(manifest,root,emit):return original(manifest,root,emit,open_url=self.store.open,sleep=self.sleeps.append)
        argv=['worker','--input-url','https://unused.example','--wandb-run-path',RUN,'--delivery-manifest-json',json.dumps(m)]
        with patch.object(resume,'OUT',out),patch.object(sys,'argv',argv),patch.object(rd,'Delivery',side_effect=ctor),patch.object(resume.base,'bootstrap') as boot,patch.object(resume.subprocess,'Popen') as popen,patch.object(resume.signal,'signal'),patch.object(resume.signal,'alarm'),patch.object(resume,'emit'):
            with self.assertRaisesRegex(RuntimeError,'incomplete'):resume.main()
            boot.assert_not_called();popen.assert_not_called()
        failure=json.loads((out/'failure.json').read_text())
        self.assertEqual(failure['http_status'],403);self.assertNotIn('private',json.dumps(failure))
    def test_legacy_upload_only_mode_rejected_before_work(self):
        argv=['worker','--input-url','unused','--upload-url','unused','--wandb-run-path',RUN]
        with patch.object(resume,'OUT',self.root/'absent'),patch.object(sys,'argv',argv),patch.object(resume.base,'bootstrap') as boot:
            with self.assertRaisesRegex(ValueError,'legacy'):resume.main()
            boot.assert_not_called()
    def test_main_orders_preflight_before_bootstrap_and_per_stage_before_summarize(self):
        fn=next(n for n in ast.parse(pathlib.Path(resume.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        text=ast.unparse(fn)
        self.assertLess(text.index('delivery.preflight()'),text.index('base.bootstrap()'))
        self.assertLess(text.index('delivery.stage_complete'),text.index('summarize()'))
        self.assertNotIn('base.seal(',text)

class ManifestControllerTests(unittest.TestCase):
    def checkpoint_fixture(self,temp,iteration=201,optimizers=True):
        import torch
        state=dict(actor_model_state_dict={'weight':torch.ones(512)},critic_model_state_dict={'weight':torch.zeros(512)},
            actor_optimizer_state_dict=dict(state={1:{'exp_avg':torch.ones(512)}},param_groups=[{'lr':1e-5}]),
            critic_optimizer_state_dict=dict(state={1:{'exp_avg':torch.zeros(512)}},param_groups=[{'lr':1e-5}]),
            experiment_config={'saved':True},iter=iteration)
        if not optimizers:state['actor_optimizer_state_dict']['state']={}
        path=pathlib.Path(temp)/'checkpoint.pt';torch.save(state,path)
        return path,hashlib.sha256(path.read_bytes()).hexdigest()
    def test_downloaded_native_checkpoint_gate_verifies_iteration_and_optimizer_presence(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as temp:
            p,sha=self.checkpoint_fixture(temp);r=controller.verify_checkpoint_file(p,sha,201)
            self.assertTrue(r['optimizer_state_present']);self.assertTrue(r['download_sha256_verified'])
    def test_downloaded_checkpoint_gate_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as temp:
            p,_=self.checkpoint_fixture(temp)
            with self.assertRaisesRegex(ValueError,'SHA256'):controller.verify_checkpoint_file(p,'0'*64,201)
    def test_downloaded_checkpoint_gate_rejects_copied_source_and_wrong_iteration(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as temp:
            p,sha=self.checkpoint_fixture(temp,iteration=200)
            with self.assertRaisesRegex(ValueError,'NEW'):controller.verify_checkpoint_file(p,sha,200)
            with self.assertRaisesRegex(ValueError,'iteration'):controller.verify_checkpoint_file(p,sha,201)
    def test_downloaded_checkpoint_gate_rejects_missing_optimizer(self):
        with tempfile.TemporaryDirectory(dir='/workspace') as temp:
            p,sha=self.checkpoint_fixture(temp,optimizers=False)
            with self.assertRaisesRegex(ValueError,'Optimizer'):controller.verify_checkpoint_file(p,sha,201)
    def test_v4_capability_ttl(self):
        now=datetime.datetime(2026,10,25,3,tzinfo=datetime.timezone.utc)
        self.assertEqual(controller.remaining_seconds('https://storage.googleapis.com/x?X-Goog-Date=20261025T030000Z&X-Goog-Expires=3600',now),3600)
    def test_legacy_expiry_ttl(self):
        now=datetime.datetime(2026,10,25,3,tzinfo=datetime.timezone.utc)
        self.assertEqual(controller.remaining_seconds('https://storage.googleapis.com/x?Expires='+str(int(now.timestamp())+3600),now),3600)
    def test_missing_expiry_rejected(self):
        with self.assertRaises(ValueError):controller.remaining_seconds('https://storage.googleapis.com/x?Signature=secret')
    def fake_api(self,registered=False):
        from unittest.mock import MagicMock
        api=MagicMock();m=manifest()
        api.run.return_value.files.return_value=[types.SimpleNamespace(name=rd.FILES['checkpoint'],size=10)] if registered else []
        api.run.return_value.upload_file.return_value.direct_url=m['outputs']['probe']['get_url']
        api.run.return_value.file.side_effect=lambda name:types.SimpleNamespace(direct_url=next(v['get_url'] for k,v in m['outputs'].items() if rd.FILES[k]==name))
        api._service_api.execute_graphql.return_value=dict(createRunFiles=dict(runID='opaque',uploadHeaders=['Content-Type: application/octet-stream'],files=[dict(name=rd.FILES[k],uploadUrl=v['put_url'],directUrl=v['get_url']) for k,v in m['outputs'].items()]))
        return api
    def test_controller_uses_server_get_capabilities_and_no_credentials(self):
        api=self.fake_api()
        with patch.object(controller,'remaining_seconds',return_value=3500):m=controller.prepare(api,RUN)
        self.assertEqual(set(m['outputs']),set(rd.FILES))
        self.assertEqual(m['outputs']['probe']['headers'],{'Content-Type':'application/octet-stream'})
        api.run.return_value.file.assert_not_called()
        self.assertIn('directUrl',api._service_api.execute_graphql.call_args.args[0])
    def test_preexisting_output_registration_blocks_mutation(self):
        api=self.fake_api(registered=True)
        with self.assertRaisesRegex(ValueError,'history'):controller.prepare(api,RUN)
        api._service_api.execute_graphql.assert_not_called()
    def test_backend_unfiltered_bookkeeping_files_do_not_block_output_preparation(self):
        api=self.fake_api();api.run.return_value.files.return_value=[types.SimpleNamespace(name='config.yaml')]
        with patch.object(controller,'remaining_seconds',return_value=3500):m=controller.prepare(api,RUN)
        self.assertEqual(m['run_path'],RUN)
    def test_registered_but_unwritten_output_can_receive_fresh_capability(self):
        api=self.fake_api();api.run.return_value.files.return_value=[types.SimpleNamespace(name=rd.FILES['probe'],size=0)]
        with patch.object(controller,'remaining_seconds',return_value=3500):m=controller.prepare(api,RUN)
        self.assertEqual(m['run_path'],RUN)
    def test_unsigned_new_checkpoint_get_is_pending_not_claimed_downloaded(self):
        api=self.fake_api()
        entries=api._service_api.execute_graphql.return_value['createRunFiles']['files']
        for f in entries:f['directUrl']=f['directUrl'].split('?')[0]
        with patch.object(controller,'remaining_seconds',return_value=3500):m=controller.prepare(api,RUN)
        self.assertIsNone(m['outputs']['checkpoint']['get_url'])
        self.assertIsNotNone(m['outputs']['probe']['get_url'])
        api.run.return_value.upload_file.assert_called_once()
    def test_worker_pending_get_requires_matching_storage_md5(self):
        m=manifest();m['outputs']['checkpoint']['get_url']=None
        with tempfile.TemporaryDirectory(dir='/workspace') as temp:
            store=Store();d=rd.Delivery(m,temp,lambda *a,**kw:None,open_url=store.open,sleep=lambda x:None)
            d.preflight();p=pathlib.Path(temp)/'test.pt';p.write_bytes(b'checkpoint')
            with self.assertRaises(rd.DeliveryError):d.send_file('checkpoint',p)
            with patch.object(Response,'headers',{'ETag':hashlib.md5(b'checkpoint').hexdigest()}):r=d.send_file('checkpoint',p)
            self.assertTrue(r['storage_md5_verified']);self.assertFalse(r['download_sha256_verified'])
            self.assertTrue(r['controller_download_required'])
    def test_short_lifetime_blocks_submission_manifest(self):
        with patch.object(controller,'remaining_seconds',return_value=30):
            with self.assertRaisesRegex(RuntimeError,'lifetime'):controller.prepare(self.fake_api(),RUN)
    def test_incomplete_registration_rejected(self):
        api=self.fake_api();api._service_api.execute_graphql.return_value['createRunFiles']['files'].pop()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):controller.prepare(api,RUN)

if __name__=='__main__':unittest.main()