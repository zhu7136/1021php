"""Small tail-safe, content-addressed receipts; no wheels or credentials embedded."""
import hashlib,json
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def build(rows,records,installed,imports,rehashed,manifest_sha256):
 expected={r['filename']:r for r in rows}
 if len(expected)!=33 or len(records)!=33 or len(rehashed)!=33:raise ValueError('receipt scope mismatch')
 if installed['returncode']!=0 or imports['returncode']!=0:raise ValueError('subprocess unsuccessful')
 for collection in [records,rehashed]:
  if len({r['filename'] for r in collection})!=33 or set(r['filename'] for r in collection)!=set(expected):raise ValueError('receipt names differ')
  for r in collection:
   e=expected[r['filename']]
   if r['sha256']!=e['sha256'] or r.get('received_bytes',r.get('bytes'))!=e['bytes']:raise ValueError('receipt hash/size differs')
 for r in records:
  if r['passed'] is not True or r['http_status']!=200 or r['automatic_retries']!=0:raise ValueError('download failed or retried')
 result=dict(schema=1,manifest_sha256=manifest_sha256,downloads=records,post_install_rehashes=rehashed,installation={k:installed[k] for k in ['returncode','log_bytes','log_sha256']},import_process={k:imports[k] for k in ['returncode','log_bytes','log_sha256']})
 if len(canonical(result))>60000:raise ValueError('audit tail exceeds bounded size')
 return result,hashlib.sha256(canonical(result)).hexdigest()