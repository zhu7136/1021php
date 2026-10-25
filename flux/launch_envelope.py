"""Shell-safe capability transport,NOT encryption. Never log or commit live tokens."""
import base64,json,re
from urllib.parse import urlsplit,parse_qs

MAX_TOKEN_BYTES=64000

def validate(data):
    if not isinstance(data,dict) or set(data)!={'schema_version','run_path','input_url','delivery_manifest'} or data['schema_version']!=1:
        raise ValueError('Invalid launch envelope schema')
    path=data['run_path']
    if not isinstance(path,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[a-z0-9]{8}',path):
        raise ValueError('Invalid launch run path')
    from reliable_delivery import Delivery
    manifest=data['delivery_manifest']
    if not isinstance(manifest,dict) or manifest.get('run_path')!=path:raise ValueError('Envelope run differs')
    Delivery(manifest,'.',lambda *a,**k:None)
    url=data['input_url']
    if not isinstance(url,str) or len(url)>8192:raise ValueError('Invalid source capability')
    p=urlsplit(url);q=parse_qs(p.query)
    expected='/wandb-production.appspot.com/'+path.rsplit('/',1)[0]+'/94728bf7/protected_pilot_bundle.zip'
    if (p.scheme!='https' or p.hostname!='storage.googleapis.com' or p.port not in (None,443)
        or p.username or p.password or p.fragment or p.path!=expected
        or not (q.get('X-Goog-Signature') or q.get('Signature'))):
        raise ValueError('Source capability outside declared source bundle')
    return data

def encode(run_path,input_url,delivery_manifest):
    data=validate(dict(schema_version=1,run_path=run_path,input_url=input_url,delivery_manifest=delivery_manifest))
    token=base64.urlsafe_b64encode(json.dumps(data,separators=(',',':'),sort_keys=True).encode()).decode().rstrip('=')
    if len(token)>MAX_TOKEN_BYTES:raise ValueError('Launch envelope too large')
    return token

def decode(token):
    if not isinstance(token,str) or not 1<=len(token)<=MAX_TOKEN_BYTES or not re.fullmatch(r'[A-Za-z0-9_-]+',token):
        raise ValueError('Launch envelope must use shell-safe URL-base64 alphabet')
    try:
        raw=base64.b64decode(token+'='*((-len(token))%4),altchars=b'-_',validate=True)
        data=json.loads(raw)
    except (ValueError,UnicodeError) as exc:
        raise ValueError('Invalid encoded launch envelope') from None
    return validate(data)

def arguments(token):
    data=decode(token)
    # Return a Python argv array;never eval/source/shell-execute any decoded capability.
    return ['--wandb-run-path',data['run_path'],'--input-url',data['input_url'],
        '--delivery-manifest-json',json.dumps(data['delivery_manifest'],separators=(',',':'))]

def start_script(run_path,input_url,delivery_manifest):
    return 'gm-run 1021php/flux/train_delivery_smoke.py --launch-envelope '+encode(run_path,input_url,delivery_manifest)