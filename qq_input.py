"""Allow the user-approved QQ input host; preserve GCS result upload rules.

QQ link expiry is opaque and is NOT claimed to be verified. HTTP errors fail
closed; the caller's 120s deadline always applies. No redirects or retries.
"""
from urllib.parse import urlsplit,parse_qs
from scoped_transport import validate_capability as gcs_validate

def validate_capability(cap,method):
    u=urlsplit(cap.get('url',''))
    if u.hostname=='storage.googleapis.com':return gcs_validate(cap,method)
    if set(cap)!={'url','headers','object_path','method'} or method!='GET' or cap['method']!='GET':
        raise ValueError('Input GET capability required')
    if (u.scheme!='https' or u.hostname!='multimedia.qfile.qq.com' or u.port not in (None,443)
            or u.username or u.password or u.fragment or u.path!='/download'
            or cap['object_path']!=u.path or cap['headers']!={}):
        raise ValueError('Unapproved QQ input capability')
    q=parse_qs(u.query,keep_blank_values=True)
    expected={'appid','client_type','client_ver','fileid','fldc','rkey','filename'}
    if set(q)!=expected or any(len(v)!=1 or not v[0] for v in q.values()):
        raise ValueError('Unexpected query shape')
    if q['appid']!=['14901'] or q['filename']!=['private-trial-input.zip'] or q['client_type']!=['web']:
        raise ValueError('Wrong input selection')
    return u,float('inf')  # unknown provider expiry, not an infinite-lived link