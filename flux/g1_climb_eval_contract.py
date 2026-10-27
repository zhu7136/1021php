"""Frozen evaluation of pinned final climb checkpoints, one <=600s worker each."""
import os,re,hashlib
from pathlib import Path
ROOT=Path(__file__).parent
SOURCES={
 'low':('TASK_20261027_123','iodb0n0v',1345,'e82625606bcf622e4c0950a95526dbce7b6a7bca35463fdabf1f2a196caed028'),
 'high':('TASK_20261027_118','od4r78u5',1323,'c2ce9b74df347ce8818e936dd94794c78910d33f34924e2d1ecc9713f745d7be')}
def kind():
 value=os.environ.get('PHP_G1_CLIMB_KIND')
 if value not in SOURCES:raise ValueError('Explicit low/high climb kind required')
 return value
def numbers(text,mapping):
 return re.sub(r'\b('+ '|'.join(map(str,mapping))+r')\b',lambda m:str(mapping[int(m[0])]),text)
def replace(text,old,new):
 if old not in text:raise ValueError('Evaluation source anchor missing')
 return text.replace(old,new)
def protocol_source():
 task,run,iteration,sha=SOURCES[kind()]
 text=(ROOT/'g1_eval283_protocol.py').read_text()
 for old,new in [("SOURCE_TASK='TASK_20261026_283'",f'SOURCE_TASK={task!r}'),("SOURCE_RUN='9fe07g0l'",f'SOURCE_RUN={run!r}'),('SOURCE_ITERATION=2594',f'SOURCE_ITERATION={iteration}'),('416f68e33c18bb39a354f8de66277cb77967b3ac9bbb6022ee631262442174f2',sha),('motions-low-step','motions-'+kind()+'-climb-76')]:text=replace(text,old,new)
 text=numbers(text,{1800:600,56:40,168:120})
 return text.replace('unique56x3','unique40x3').replace('task283','pinned climb')
def input_source():
 text=(ROOT/'g1_resume276_input.py').read_text()
 start=text.index("SOURCE_TASK=");end=text.index('\ndef digest',start)
 text=text[:start]+'from flux.g1_climb_eval_protocol import SOURCE_TASK,SOURCE_RUN,SOURCE_ITERATION,CHECKPOINT_BYTES,CHECKPOINT_SHA256\n'+text[end:]
 text=text.replace('source276_final','source_climb_final')
 start=text.index('def fetch_checkpoint(')
 text=text[:start]+'''def fetch_checkpoint(control,destination):
    validate_control(control)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    path=destination/'source_climb_final.pt'
    from flux.parallel_download import download
    download(control['url'],path,CHECKPOINT_BYTES,CHECKPOINT_SHA256,destination.parent/'checkpoint-download.json')
    return path
'''
 return text
def teacher_source():
 text=(ROOT/'g1_eval283_teacher.py').read_text()
 text=replace(text,'from flux.g1_eval283_protocol import *','from flux.g1_climb_eval_protocol import *')
 text=replace(text,'from flux.g1_lowstep_input import runtime_checks','from flux.g1_climb_input import runtime_checks')
 text=numbers(text,{56:40,168:120,3584:2560})
 text=text.replace('56x64','40x64').replace('56env','40env').replace('eval283','climb_eval').replace('task283','climb')
 text=replace(text,'    # Explicit, reproducible regenerated terrain seed',"    episode_seconds=max(12.0,max(lengths)/50.0+1.0)\n    # Explicit, reproducible regenerated terrain seed")
 text=replace(text,'max_episode_length_s=12.0','max_episode_length_s=episode_seconds')
 text=text.replace("'episode_timeout_seconds':12.0","'episode_timeout_seconds':episode_seconds")
 text=text.replace('episode timeout10s->12s to cover longest10.76s clip; failure thresholds unchanged','episode timeout extended only if needed to cover full source clips; failure thresholds unchanged')
 return text
def worker_source():
 text=(ROOT/'run_g1_eval283.py').read_text()
 for old,new in [('g1_eval283_protocol','g1_climb_eval_protocol'),('g1_eval283_input','g1_climb_eval_input'),('g1_lowstep_input','g1_climb_input'),('g1_eval283_teacher.py','g1_climb_eval_teacher.py'),('lowstep-input-b64','climb-input-b64'),('a.lowstep_input_b64','a.climb_input_b64'),('motions-low-step','motions-'+kind()+'-climb-76')]:text=replace(text,old,new)
 text=numbers(text,{1800:600,1680:540,1640:540,1795:595,1200:360,1500:480,56:40,168:120})
 text=replace(text,'from flux.g1_climb_input import static_checks,fetch_data','from flux.g1_climb_eval_data import static_checks,fetch_data')
 return text.replace('eval283','climb_eval')