"""One explicitly pinned Flux OSS checkpoint capability, never a generic URL fetcher."""
import math
import time
from urllib.parse import urlsplit,parse_qs

FLUX_OSS_HOST='limx-gradmotion.oss-cn-beijing.aliyuncs.com'


def validate_flux_checkpoint(cap,method):
 if method!='GET' or set(cap)!={'url','headers','object_path','method'} or cap['method']!='GET' or cap['headers']!={}:raise ValueError('Single GET capability only')
 u=urlsplit(cap['url'])
 if u.scheme!='https' or u.hostname!=FLUX_OSS_HOST or u.port not in (None,443) or u.username or u.password or u.fragment or u.path!=cap['object_path'] or not u.path.startswith('/') or not u.path.endswith('.pt'):raise ValueError('Exact Flux OSS checkpoint host/path')
 q=parse_qs(u.query,keep_blank_values=True)
 if not {'Expires','Signature','OSSAccessKeyId'}<=set(q) or any(len(v)!=1 or not v[0] for v in q.values()):raise ValueError('Signed OSS capability required')
 if set(q)-{'Expires','Signature','OSSAccessKeyId','security-token','response-content-disposition','response-content-type'}:raise ValueError('Unexpected signed URL parameters')
 expiry=float(q['Expires'][0])
 if not math.isfinite(expiry) or expiry<=time.time():raise ValueError('Expired object capability')
 return u,expiry