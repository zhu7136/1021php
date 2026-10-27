"""Allowlisted transport evidence only: never retain scoped URLs or argv."""
import json,re,time
from pathlib import Path

CODES=frozenset(('invalid_download_contract','download_path_already_exists','download_cancelled',
 'download_deadline_exceeded','range_response_required','wrong_content_range','wrong_content_length',
 'encoded_range_rejected','strong_etag_required','object_changed','truncated_range','oversized_range',
 'invalid_range_probe','assembled_size_mismatch','input_hash_mismatch','transfer_or_local_error'))

def safe_code(value):
 return value if isinstance(value,str) and (value in CODES or re.fullmatch(r'http_[1-5][0-9]{2}',value)) else 'unclassified'

def transfer_evidence(name,source,destination):
 """Copy only a bounded, type-checked subset, never arbitrary exception text."""
 if not re.fullmatch(r'(teacher-[0-4]\.pt|dataset-[0-3]\.tar\.gz)',name):raise ValueError('Invalid input name')
 row={'name':name,'diagnostics_available':False}
 p=Path(source)
 if p.is_file() and not p.is_symlink() and p.stat().st_size<=128*1024:
  try:v=json.loads(p.read_text())
  except (ValueError,UnicodeError):v={}
  if isinstance(v,dict):
   row['diagnostics_available']=True
   for k in ('expected_bytes','workers','chunk_bytes','deadline_seconds','socket_timeout_seconds','task_retries','transfer_retries'):
    if type(v.get(k)) is int and v[k]>=0:row[k]=v[k]
   if type(v.get('elapsed_seconds')) in (int,float) and 0<=v['elapsed_seconds']<=1800:row['elapsed_seconds']=v['elapsed_seconds']
   row['published_verified']=v.get('published_verified') is True
   row['failure_code']=None if v.get('failure') is None else safe_code(v['failure'])
 out=Path(destination);out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(row,indent=2));return row

def monitored_download(name,control,source_dir,evidence_dir,download,emit):
 started=time.monotonic();root=Path(source_dir);diag=root/(name+'.json')
 emit('five_distill_input_begin',name=name,expected_bytes=control['bytes'])
 try:
  p=download(control['url'],root/name,control['bytes'],control['sha256'],diag)
 except BaseException:
  row=transfer_evidence(name,diag,Path(evidence_dir)/(name+'.json'))
  emit('five_distill_input_failed',**row)
  raise
 else:
  row=transfer_evidence(name,diag,Path(evidence_dir)/(name+'.json'))
  emit('five_distill_input_verified',**row,wall_seconds=round(time.monotonic()-started,3))
  return p