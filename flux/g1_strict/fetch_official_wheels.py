"""Dependency-free, hash-pinned official-wheel fetch candidate. Never installs code."""
import argparse,hashlib,http.client,json,re,time,signal
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse,unquote

MAX_TOTAL=128*1024*1024
def load_manifest(path,pin):
 raw=Path(path).read_bytes()
 if not re.fullmatch('[0-9a-f]{64}',pin) or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('manifest pin mismatch')
 manifest=json.loads(raw);rows=manifest['files']
 if manifest['scope']!='frozen-overlay-33' or len(rows)!=33:raise ValueError('wrong scope')
 total=0;seen=set()
 for row in rows:
  name=row['filename'];u=urlparse(row['url']);size=row['bytes']
  if not re.fullmatch('[A-Za-z0-9._+-]+\\.whl',name) or name in seen:raise ValueError('unsafe or duplicate filename')
  if u.scheme!='https' or u.hostname!='files.pythonhosted.org' or u.port not in [None,443] or u.username or u.password or u.query or u.fragment or not u.path.startswith('/packages/') or unquote(u.path.rsplit('/',1)[-1])!=name:raise ValueError('not canonical official wheel URL')
  if type(size) is not int or not 0<size<=MAX_TOTAL or not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('invalid bounds or digest')
  seen.add(name);total+=size
 if total>MAX_TOTAL:raise ValueError('aggregate limit exceeded')
 return manifest

def remaining(deadline):
 delta=deadline-time.monotonic()
 if delta<=0:raise TimeoutError('absolute deadline expired')
 return min(20,delta)

def download(row,directory,deadline,connection_factory=http.client.HTTPSConnection):
 u=urlparse(row['url']);connection=connection_factory(u.hostname,timeout=remaining(deadline));count=0;hash_=hashlib.sha256()
 try:
  connection.request('GET',u.path,headers={'Accept-Encoding':'identity','User-Agent':'hud04-static-provenance/1'})
  response=connection.getresponse()
  if response.status!=200:raise ValueError('unexpected HTTP status; redirects are forbidden')
  length=response.getheader('Content-Length')
  if length is not None and int(length)!=row['bytes']:raise ValueError('declared size mismatch')
  if response.getheader('Content-Encoding') not in [None,'identity']:raise ValueError('unexpected encoding')
  target=directory/(row['filename']+'.part')
  with target.open('xb') as file:
   while True:
    timeout=remaining(deadline)
    if connection.sock is not None:connection.sock.settimeout(timeout)
    chunk=response.read(min(65536,row['bytes']-count+1))
    if not chunk:break
    count+=len(chunk)
    if count>row['bytes']:raise ValueError('stream exceeds size limit')
    hash_.update(chunk);file.write(chunk)
  if count!=row['bytes'] or hash_.hexdigest()!=row['sha256']:raise ValueError('wheel digest mismatch')
  # New, non-existing target in a freshly created output directory, no overwrite.
  final=directory/row['filename']
  if final.exists() or final.is_symlink():raise ValueError('refuse output replacement')
  target.rename(final)
  return dict(filename=row['filename'],bytes=count,sha256=hash_.hexdigest(),http_status=200,official_get_verified=True)
 finally:connection.close()

@contextmanager
def absolute_budget(seconds):
 start=time.monotonic();old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
 def expired(*_):raise TimeoutError('absolute download deadline expired')
 signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,min(seconds,old_timer[0]) if old_timer[0]>0 else seconds)
 try:yield
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
  if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(0.000001,old_timer[0]-(time.monotonic()-start)),old_timer[1])

def fetch(manifest,output,budget,connection_factory=http.client.HTTPSConnection):
 with absolute_budget(budget):return _fetch(manifest,output,budget,connection_factory)

def _fetch(manifest,output,budget,connection_factory=http.client.HTTPSConnection):
 if not 1<=budget<=300:raise ValueError('budget outside bounded candidate scope')
 deadline=time.monotonic()+budget;output=Path(output).absolute()
 if any(p.is_symlink() for p in [output,*output.parents]):raise ValueError('symlink output path')
 output.mkdir(exist_ok=False);directory=output/'wheels';directory.mkdir();records=[]
 for row in manifest['files']:
  record=download(row,directory,deadline,connection_factory);records.append(record);print(json.dumps(record),flush=True)
  (output/'partial-receipt.json').write_text(json.dumps(dict(complete=False,records=records),indent=2)+'\n')
 result=dict(complete=True,records=records,total_bytes=sum(r['bytes'] for r in records),installed=False,native_code_executed=False,worker_reachability_proven=False)
 (output/'download-receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--output');parser.add_argument('--budget-seconds',type=int,default=110);args=parser.parse_args()
 manifest=load_manifest(args.manifest,args.manifest_sha256)
 if not args.execute:print(json.dumps(dict(execution_requested=False,network_requested=False,files=len(manifest['files']),installed=False)));return
 if not args.output:parser.error('--execute requires a new --output directory')
 print(json.dumps({k:v for k,v in fetch(manifest,args.output,args.budget_seconds).items() if k!='records'}),flush=True)
if __name__=='__main__':main()