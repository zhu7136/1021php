"""Deliver only verified G1 Low-speed step data; never reuse locomotion weights."""
import hashlib,json,sys,tarfile,urllib.parse,urllib.request
from pathlib import Path
from flux.preflight import static_checks as base_static,HOLOSOMA_COMMIT,ROOT

def static_checks(expected_commit=None):
    result=base_static(expected_commit)
    component=json.loads((ROOT/'release-assets.json').read_text())['components']['motions-low-step']
    result.update(dataset='motions-low-step',dataset_archive_sha256=component['archive']['sha256'],
                  declared_dataset_files=len(component['files']),dataset_extracted_bytes=sum(x['size'] for x in component['files']))
    return result

def fetch_data(control,destination):
    if set(control)!={'url','sha256','bytes'} or type(control['bytes']) is not int or not 1<=control['bytes']<=60*1024**2:
        raise ValueError('Invalid low-step input control')
    url=urllib.parse.urlsplit(control['url'])
    if url.scheme!='https' or url.hostname!='storage.googleapis.com':
        raise ValueError('Only scoped W&B storage input URLs accepted')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    archive=destination.parent/'low-step-input.tar.gz'
    h=hashlib.sha256();remaining=control['bytes']
    with urllib.request.urlopen(control['url'],timeout=45) as response,archive.open('xb') as out:
        if urllib.parse.urlsplit(response.url).hostname!='storage.googleapis.com':raise ValueError('Unexpected input redirect')
        while remaining:
            block=response.read(min(1048576,remaining))
            if not block:raise ValueError('Truncated low-step input')
            h.update(block);out.write(block);remaining-=len(block)
        if response.read(1):raise ValueError('Oversized low-step input')
    if h.hexdigest()!=control['sha256']:raise ValueError('Low-step archive hash mismatch')
    component=json.loads((ROOT/'release-assets.json').read_text())['components']['motions-low-step']
    expected={r['path']:r for r in component['files']};seen=set()
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected low-step archive member')
            row=expected[member.name]
            if member.size!=row['size']:raise ValueError('Low-step member size mismatch')
            path=destination/member.name
            if not path.resolve().is_relative_to(destination.resolve()):raise ValueError('Unsafe data path')
            path.parent.mkdir(parents=True,exist_ok=True)
            data=source.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Low-step member hash mismatch')
            with path.open('xb') as output:output.write(data)
            seen.add(member.name)
    if seen!=set(expected):raise ValueError('Low-step files missing')
    return destination

def runtime_checks(data):
    import importlib.util,subprocess
    from flux.runtime_versions import check_runtime_versions
    result=check_runtime_versions()
    for module in ('tyro','wandb','onnx','tensorboard','trimesh','scipy','holosoma'):
        if importlib.util.find_spec(module) is None:raise ValueError('Missing module: '+module)
    submodule=ROOT/'thirdparty/holosoma'
    if subprocess.check_output(['git','-C',str(submodule),'rev-parse','HEAD'],text=True).strip()!=HOLOSOMA_COMMIT:
        raise ValueError('Unexpected Holosoma version')
    import holosoma
    if not Path(holosoma.__file__).resolve().is_relative_to(submodule.resolve()):raise ValueError('Unpinned Holosoma import')
    sys.path.insert(0,str(ROOT/'scripts'))
    from download_assets import read_component,verify
    component=read_component(ROOT/'release-assets.json','motions-low-step')
    problems=verify(component,data)
    if problems:raise ValueError('Low-step data contract failed')
    from wbt_training.utils.registry import validate_bundle
    validate_bundle(data)
    return {**result,'dataset':'motions-low-step','data_verified':True,'declared_files':len(component['files'])}