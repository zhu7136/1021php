"""Credential-free, strict training JSONL -> platform stdout metric relay.

Only completed-update metrics are relayed. W&B ingestion is performed outside
the GPU worker; no W&B account key is handed to the simulator.
"""
import hashlib
import json
import math
from pathlib import Path
import re

LOSS_KEYS={'behavior','warmup_active','valid_fraction','optimizer_minibatches','skipped_all_invalid_minibatches'}


def make_event(row,task_id,agent):
 if not re.fullmatch(r'TASK_[0-9_]+',task_id):raise ValueError('Task identity')
 if set(agent)!={'thread_id','turn_id'} or not all(isinstance(v,str) and 0<len(v)<=200 for v in agent.values()):raise ValueError('Provenance')
 for key in ['completed_updates','additional_updates_completed']:
  if type(row[key]) is not int or row[key]<1:raise ValueError('Completed update counter')
 if row['completed_updates']<row['additional_updates_completed'] or row.get('skills_evaluated') is not False:raise ValueError('Metric role')
 losses=row['losses']
 if set(losses)!=LOSS_KEYS or any(type(v) not in (int,float) or not math.isfinite(v) for v in losses.values()):raise ValueError('Known finite scalar losses only')
 if not 0<=losses['valid_fraction']<=1 or losses['behavior']<0:raise ValueError('Loss scale')
 return {'schema':'hud04-training-metric-v1','event':'training_update_completed','task_id':task_id,
  '_wb_agent':agent,'completed_updates':row['completed_updates'],
  'additional_updates_completed':row['additional_updates_completed'],'losses':dict(losses),'skills_evaluated':False}


def metric_values(event):
 checked=make_event(event,event['task_id'],event['_wb_agent'])
 if event!=checked:raise ValueError('Unexpected event fields')
 return {'train/completed_updates':event['completed_updates'],
  'train/additional_updates_completed':event['additional_updates_completed'],
  **{'train/'+k:v for k,v in event['losses'].items()}}


def parse_events(text,task_id,agent):
 found={}
 for line in text.splitlines():
  marker='{"schema": "hud04-training-metric-v1"'
  pos=line.find(marker)
  if pos<0:continue
  event,_=json.JSONDecoder().raw_decode(line[pos:])
  if event.get('task_id')!=task_id or event.get('_wb_agent')!=agent:raise ValueError('Foreign metric event')
  metric_values(event);step=event['completed_updates']
  if step in found and found[step]!=event:raise ValueError('Conflicting duplicate step')
  found[step]=event
 return [found[k] for k in sorted(found)]


class ProgressMirror:
 def __init__(self,output,task_id,agent,emit=print):
  self.output=Path(output);self.task_id=task_id;self.agent=agent;self.emit=emit
  self.path=None;self.offset=0;self.pending=b'';self.last=0
 def __call__(self):
  if self.path is None:
   paths=list(self.output.rglob('formal_progress.jsonl'))
   if not paths:return
   if len(paths)!=1:raise ValueError('Ambiguous progress stream')
   self.path=paths[0]
  if self.path.is_symlink() or not self.path.resolve().is_relative_to(self.output.resolve()):raise ValueError('Unsafe progress file')
  with self.path.open('rb') as f:
   f.seek(self.offset);data=f.read(65536);self.offset=f.tell()
  self.pending+=data
  while b'\n' in self.pending:
   line,self.pending=self.pending.split(b'\n',1)
   if not line or len(line)>4096:raise ValueError('Progress record bound')
   event=make_event(json.loads(line),self.task_id,self.agent)
   if event['additional_updates_completed']!=self.last+1:raise ValueError('Noncontiguous completed updates')
   self.emit(json.dumps(event,allow_nan=False),flush=True);self.last+=1
  if len(self.pending)>4096:raise ValueError('Unterminated progress record bound')