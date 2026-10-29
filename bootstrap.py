"""Reuse pinned public inputs; keep learned checkpoints private; one bounded attempt."""
import argparse,base64,hashlib,json,os,sys,time
from pathlib import Path
import fresh_bootstrap as base
from scoped_transport import transfer,validate_capability

def main():
 p=argparse.ArgumentParser(add_help=False)
 p.add_argument('--source-cap',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--source-bytes',required=True,type=int)
 extra,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
 source=json.loads(base64.urlsafe_b64decode(extra.source_cap));validate_capability(source,'GET')
 assert len(extra.source_sha)==64 and 0<extra.source_bytes<20*1024**2
 original_stage=base.stage;original_pack=base.pack;installed=[False]
 def stage(command,log,env,hard,stop_file=None,soft=None):
  if not installed[0]:
   work=Path(log).parent;code=work/'payload/teacher-code';assert (code/'teacher_algo.py').exists()
   (code/'fresh_algo.py').write_bytes((code/'teacher_algo.py').read_bytes())
   root=Path(__file__).resolve().parent
   (code/'teacher_algo.py').write_bytes((root/'continuation_algo.py').read_bytes())
   (code/'teacher_child.py').write_bytes((root/'continuation_child.py').read_bytes())
   transfer(source,code/'continuation-source.pt',method='GET',size=extra.source_bytes,sha256=extra.source_sha,absolute_deadline=min(hard-120,time.monotonic()+100))
   installed[0]=True
  env['CONTINUATION_SOURCE_SHA']=extra.source_sha
  # Reserve 150 seconds beyond the old soft stop for post-training diagnostics.
  return original_stage(command,log,env,hard,stop_file=stop_file,soft=(soft-150 if stop_file else soft))
 def pack(work,state):
  state.update(fresh_training=False,source_checkpoint_sha256=extra.source_sha,source_checkpoint_private=True)
  out=work/'native-output'
  if (out/'new_process_resume.json').exists():state['native_resume']=json.loads((out/'new_process_resume.json').read_text())
  for phase in ['before','after']:
   if (out/('evaluation-'+phase+'.json')).exists():state['evaluation_'+phase]=json.loads((out/('evaluation-'+phase+'.json')).read_text())
  state['rollout_quality_certified']=False
  return original_pack(work,state)
 base.stage=stage;base.pack=pack
 return base.main()
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:
  print(json.dumps({'event':'continuation_bootstrap_failure','error_type':type(e).__name__,'automatic_retry':False}),flush=True);sys.exit(1)