"""Bounded output recovery via platform logs when the archive PUT fails."""
import json,re
from pathlib import Path

def emit_case_diagnostics(caseout):
    caseout=Path(caseout)
    for name in ['replay-diagnostics.json','video-evidence.json','formal-lifecycle.json']:
        path=caseout/name
        if not path.is_file() or path.stat().st_size>1024*1024:continue
        try:
            report=json.loads(path.read_text())
            samples=report.pop('samples',None)
            if samples:
                report['selected_samples']=[samples[i] for i in sorted({0,len(samples)//2,len(samples)-1})]
                report['stored_sample_count']=len(samples)
            print(json.dumps({'event':'replay_file_fallback','file':name,'data':report}),flush=True)
        except Exception as e:
            print(json.dumps({'event':'replay_fallback_error','file':name,'error_type':type(e).__name__}),flush=True)
    path=caseout/'native.log'
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-24000));text=stream.read(24000).decode(errors='replace')
        lines=[]
        for line in text.splitlines():
            if re.search(r'(?i)(authorization|password|api.?key|secret|access.?token|--config|output_cap|input_cap)',line):
                lines.append('[sensitive diagnostic line omitted]');continue
            lines.append(re.sub(r'https?://[^\s\"<>]+','[URL_REDACTED]',line))
        print('REPLAY_NATIVE_LOG_TAIL_BEGIN',flush=True)
        print('\n'.join(lines),flush=True)
        print('REPLAY_NATIVE_LOG_TAIL_END',flush=True)
