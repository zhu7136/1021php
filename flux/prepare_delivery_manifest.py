"""Controller-only W&B signed-capability preparation. Never send API keys to a worker.

Call only for a newly created bookkeeping run. This registers output filenames;
it does not start Flux, initialize simulation, or upload a model. Do not log or
commit the returned manifest. Signed capabilities expire and must be minted
again for a separately authorized task, not replayed from old payloads.
"""
from __future__ import annotations
import datetime,hashlib,json,pathlib,tempfile,urllib.parse
from reliable_delivery import Delivery,FILES

def remaining_seconds(url,now=None):
    now=now or datetime.datetime.now(datetime.timezone.utc)
    q=urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    if 'X-Goog-Date' in q and 'X-Goog-Expires' in q:
        at=datetime.datetime.strptime(q['X-Goog-Date'][0],'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc)
        return (at+datetime.timedelta(seconds=int(q['X-Goog-Expires'][0]))-now).total_seconds()
    if 'Expires' in q:return int(q['Expires'][0])-now.timestamp()
    raise ValueError('Signed capability expiry required')

def prepare(api,run_path,*,minimum_ttl=2400):
    entity,project,run=run_path.split('/')
    source=api.run(run_path)
    # The SDK/backend may return bookkeeping files even with a names filter.
    # Check exact names locally; config.yaml must not block a fresh output probe.
    if any(f.name in set(FILES.values())-{FILES['probe']} and f.size>0 for f in source.files(names=list(FILES.values()),per_page=20)):
        raise ValueError('Nonempty outputs already exist; use a fresh bookkeeping run, never overwrite history')
    mutation='''mutation DeliveryFiles($entity:String!,$project:String!,$run:String!,$files:[String!]!){
      createRunFiles(input:{entityName:$entity,projectName:$project,runName:$run,files:$files}){
        runID uploadHeaders files{name uploadUrl directUrl}}}'''
    response=api._service_api.execute_graphql(mutation,dict(entity=entity,project=project,run=run,files=list(FILES.values())))['createRunFiles']
    if not response['runID'] or {f['name'] for f in response['files']}!=set(FILES.values()):
        raise RuntimeError('Output capability registration incomplete')
    headers={}
    for h in response['uploadHeaders']:
        name,value=h.split(':',1);headers[name.strip()]=value.strip()
    put={f['name']:f['uploadUrl'] for f in response['files']}
    get={f['name']:f['directUrl'] for f in response['files']}
    # GCS GET URLs for not-yet-uploaded objects are unsigned. Seed ONLY the
    # JSON probe (never a dummy .pt), then let W&B sign its now-existing object.
    with tempfile.TemporaryDirectory(prefix='delivery-probe-',dir='/workspace') as temp:
        path=pathlib.Path(temp)/FILES['probe']
        path.write_text(json.dumps(dict(kind='controller_output_probe_seed',run_path=run_path),sort_keys=True))
        uploaded=source.upload_file(str(path),root=temp)
        get[FILES['probe']]=uploaded.direct_url
    outputs={}
    for key,name in FILES.items():
        # Read directly from createRunFiles: before PUT, Run.file(name) may
        # return an unrelated bookkeeping file or omit an as-yet-unwritten object.
        get_url=get[name]
        q=urllib.parse.parse_qs(urllib.parse.urlsplit(get_url).query)
        if not (q.get('X-Goog-Signature') or q.get('Signature')):get_url=None
        if key=='probe' and get_url is None:raise RuntimeError('W&B probe upload not yet visible; do not start GPU')
        if remaining_seconds(put[name])<minimum_ttl or (get_url is not None and remaining_seconds(get_url)<minimum_ttl):
            raise RuntimeError('Capability lifetime too short for worker bound')
        outputs[key]=dict(put_url=put[name],get_url=get_url,headers=dict(headers))
    result=dict(schema_version=1,run_path=run_path,outputs=outputs)
    # Validate scope, host, names and absence of credential headers without network calls.
    Delivery(result,'.',lambda *a,**k:None)
    return result

def verify_checkpoint_file(path,expected_sha256,expected_iteration):
    """Controller-only evidence gate. This does not claim policy quality or simulator equivalence."""
    from reliable_delivery import MAX_BYTES
    path=pathlib.Path(path)
    if not path.is_file() or path.is_symlink() or not 1024<=path.stat().st_size<=MAX_BYTES:
        raise ValueError('Native checkpoint absent or outside byte bound')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=expected_sha256:raise ValueError('Downloaded checkpoint SHA256 mismatch')
    if not isinstance(expected_iteration,int) or expected_iteration<=200:
        raise ValueError('Must verify a NEW checkpoint iteration, not copied iteration200 input')
    import torch
    data=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(data,dict) or data.get('iter')!=expected_iteration:raise ValueError('Downloaded iteration mismatch')
    for key in ('actor_model_state_dict','critic_model_state_dict'):
        if not isinstance(data.get(key),dict) or not data[key]:raise ValueError('Model state missing')
    for key in ('actor_optimizer_state_dict','critic_optimizer_state_dict'):
        state=data.get(key)
        if not isinstance(state,dict) or not state.get('state') or not state.get('param_groups'):
            raise ValueError('Optimizer state missing')
    if not isinstance(data.get('experiment_config'),dict):raise ValueError('Native saved config missing')
    return dict(download_sha256_verified=True,iteration=expected_iteration,sha256=digest,
        bytes=path.stat().st_size,optimizer_state_present=True,native_config_present=True)

def download_checkpoint(api,run_path,root,expected_sha256,expected_iteration):
    from reliable_delivery import MAX_BYTES
    root=pathlib.Path(root);root.mkdir(parents=True,exist_ok=True)
    name=FILES['checkpoint']
    if (root/name).exists():raise FileExistsError('Never overwrite prior downloaded checkpoint')
    file=api.run(run_path).file(name)
    if file.name!=name or not 1024<=file.size<=MAX_BYTES:raise ValueError('New remote checkpoint not yet registered')
    downloaded=pathlib.Path(file.download(root=str(root)).name)
    if downloaded.resolve()!=(root/name).resolve():raise ValueError('Download filename differs')
    return verify_checkpoint_file(downloaded,expected_sha256,expected_iteration)