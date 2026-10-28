"""Bounded local Git-part assembly. No network, credentials, GPU or training."""
import hashlib,json,re,stat,time
from pathlib import Path
MAX_BYTES=512*1024**2
CHUNK=1024**2

def digest_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
 return h.hexdigest()

def assemble(root,destination,*,expected_sha256,expected_bytes,absolute_deadline):
 root=Path(root);destination=Path(destination)
 def check():
  if time.monotonic()>=absolute_deadline:raise TimeoutError('Assembly deadline')
 def regular(p):
  if not stat.S_ISREG(p.lstat().st_mode):raise ValueError('Nonregular part/manifest')
 check()
 if root.is_symlink() or not root.is_dir():raise ValueError('Invalid part directory')
 if destination.exists() or destination.is_symlink():raise FileExistsError('Fresh output required')
 if not re.fullmatch('[0-9a-f]{64}',expected_sha256) or type(expected_bytes) is not int or not 0<expected_bytes<=MAX_BYTES:raise ValueError('Invalid pinned archive')
 mp=root/'parts-manifest.json';regular(mp)
 if mp.stat().st_size>65536:raise ValueError('Manifest too large')
 m=json.loads(mp.read_text())
 if m.get('schema_version')!=1 or m.get('archive_sha256')!=expected_sha256 or m.get('archive_bytes')!=expected_bytes:raise ValueError('Manifest/pin mismatch')
 rows=m['parts']
 if not isinstance(rows,list) or not 1<=len(rows)<=32:raise ValueError('Invalid part count')
 expected_names=[f'input.part{i:03d}' for i in range(len(rows))]
 if [r['path'] for r in rows]!=expected_names:raise ValueError('Wrong part order/name')
 if {p.name for p in root.iterdir()}!=set(expected_names)|{'parts-manifest.json'}:raise ValueError('Missing/extra part files')
 total=0
 for row in rows:
  check();p=root/row['path'];regular(p)
  if type(row['bytes']) is not int or not 0<row['bytes']<=48*1024**2 or p.stat().st_size!=row['bytes']:raise ValueError('Invalid part size')
  if not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('Invalid part hash')
  h=hashlib.sha256()
  with p.open('rb') as f:
   head=f.read(128)
   if head.startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('LFS pointer rejected')
   h.update(head)
   for b in iter(lambda:f.read(CHUNK),b''):check();h.update(b)
  if h.hexdigest()!=row['sha256']:raise ValueError('Part hash mismatch')
  total+=row['bytes']
 if total!=expected_bytes:raise ValueError('Part total mismatch')
 check();h=hashlib.sha256()
 with destination.open('xb') as out:
  for row in rows:
   with (root/row['path']).open('rb') as f:
    for b in iter(lambda:f.read(CHUNK),b''):check();out.write(b);h.update(b)
 if h.hexdigest()!=expected_sha256 or destination.stat().st_size!=expected_bytes:raise ValueError('Assembled archive mismatch')
 return {'parts_verified':len(rows),'archive_bytes':expected_bytes,'archive_sha256':h.hexdigest()}