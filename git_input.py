"""Verify ordinary Git chunks and stream-assemble a pinned local ZIP; no network."""
import hashlib,json,re,time
from pathlib import Path,PurePosixPath
def assemble(root,manifest,kind,destination,*,size,sha256,absolute_deadline):
 root=Path(root).resolve(strict=True);destination=Path(destination)
 data=json.loads(Path(manifest).read_text());row=data['archives'][kind]
 if data['schema_version']!=1 or row['bytes']!=size or row['sha256']!=sha256 or not 0<size<512*1024**2 or not re.fullmatch('[0-9a-f]{64}',sha256):raise ValueError('Archive binding failed')
 total=0;full=hashlib.sha256();seen=set()
 with destination.open('xb') as out:
  for i,part in enumerate(row['parts']):
   p=PurePosixPath(part['path'])
   if not re.fullmatch(r'git-inputs/(base|locomotion|high-step)/part-[0-9]{3}\.bin',part['path']) or p.is_absolute() or '..' in p.parts or part['path'] in seen or part['index']!=i or not 0<part['bytes']<=32*1024**2:raise ValueError('Invalid ordered part')
   seen.add(part['path']);source=root.joinpath(*p.parts)
   if source.is_symlink() or not source.resolve(strict=True).is_relative_to(root) or source.stat().st_size!=part['bytes']:raise ValueError('Missing, pointer, or invalid part')
   chunkhash=hashlib.sha256();n=0
   with source.open('rb') as f:
    first=True
    for b in iter(lambda:f.read(1024*1024),b''):
     if time.monotonic()>=absolute_deadline:raise TimeoutError('Assembly deadline')
     if first and b.startswith(b'version https://git-lfs.github.com/spec/'):raise ValueError('LFS pointer rejected')
     first=False;chunkhash.update(b);full.update(b);out.write(b);n+=len(b);total+=len(b)
   if n!=part['bytes'] or chunkhash.hexdigest()!=part['sha256']:raise ValueError('Part hash mismatch')
 if total!=size or full.hexdigest()!=sha256:raise ValueError('Complete ZIP hash mismatch')
 return {'bytes':total,'sha256':sha256,'parts_verified':len(seen),'hash_verified':True,'transport':'ordinary_git_local_parts','automatic_retries':0}