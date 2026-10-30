"""Bounded numeric log recovery; no simulator, network, or task submission."""
import json
import traceback
from pathlib import Path


def diagnostic_error(exc, stage, attempted_step=None):
    # Never serialize exception arguments, locals, URLs or full filesystem paths.
    known_messages = {
        'Teacher observation route mismatch', 'Observation shape contract',
        'Student action shape', 'Nonfinite action probe input', 'Teacher action shape/finite',
        'Probe step scope', 'Evaluation deadline', 'Wrong actor observation contract',
        'Wrong expert/case route', 'Invalid action output',
        'Expected exactly one native termination check per control step',
        'Native done/termination masks disagree', 'Model changed during inference',
        'Action probe log size exceeded', 'Action probe row count exceeded',
        'Action probe record contract', 'Action probe metadata size exceeded',
        'Action probe metadata must be an object',
    }
    message = str(exc)
    return {'type': type(exc).__name__, 'stage': stage, 'attempted_policy_step': attempted_step,
            'message': message if message in known_messages else 'Exception message withheld; inspect private native log',
            'message_withheld': message not in known_messages,
            'frames': [{'file': Path(frame.filename).name, 'function': frame.name, 'line': frame.lineno}
                       for frame in traceback.extract_tb(exc.__traceback__)[-8:]]}


def emit_action_probe_diagnostics(caseout, *, compact=False):
    if type(compact) is not bool:raise ValueError('Action probe record contract')
    caseout = Path(caseout)
    for filename, event in [('action-probe-contract.json', 'hud04_action_probe_contract'),
                            ('same-state-action-summary.json', 'hud04_same_state_action_summary')]:
        path = caseout / filename
        if path.is_file():
            if path.stat().st_size > 65536:
                raise ValueError('Action probe metadata size exceeded')
            metadata = json.loads(path.read_text())
            if not isinstance(metadata, dict):
                raise ValueError('Action probe metadata must be an object')
            print(json.dumps({**metadata, 'event': event}, allow_nan=False), flush=True)
    path = caseout / 'same-state-actions.jsonl'
    if not path.is_file():
        return
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('Action probe log size exceeded')
    lines = path.read_text().splitlines()
    if len(lines) > 1000:
        raise ValueError('Action probe row count exceeded')
    # Validate the full bounded file before emitting it. These records never
    # replace the caller's skill/case object or change its native return code.
    records = []
    for step, line in enumerate(lines, 1):
        record = json.loads(line)
        if (not isinstance(record, dict)
                or record.get('schema') != 'hud04-same-state-action-probe-v1'
                or record.get('step') != step
                or record.get('teacher_actions_applied') is not False):
            raise ValueError('Action probe record contract')
        records.append(json.dumps({**record, 'event': 'hud04_same_state_action'}, allow_nan=False))
    if compact:
        from hud04_compact_action_log import compact_events
        records=compact_events([json.loads(line) for line in lines])
    for record in records:
        print(record, flush=True)


def emit_depth_reset_diagnostics(caseout):
    """Recover only fixed-schema bounded numeric events from a child log."""
    path=Path(caseout)/'native.log'
    if not path.is_file():
        return
    if path.stat().st_size>16*1024*1024:
        raise ValueError('Depth native log exceeds recovery bound')
    base={'event','schema','sequence','host_elapsed_ns','kind','fix_pending_reset',
          'capture_calls','completed_captures'}
    optional={'pending_reset','reference_frame','reset_count','history_write','output_all_zero'}
    kinds={'constructed','reset_before','reset_after','native_reset_failed','capture_before',
           'capture_after','native_capture_failed','observation_before','observation_after',
           'native_observation_failed','event_limit'}
    records=[]
    for line in path.read_bytes().splitlines():
        if not line.startswith(b'{"event":"hud04_depth_reset_trace"'):
            continue
        if len(line)>4096 or len(records)>=512:
            raise ValueError('Depth event recovery bound')
        record=json.loads(line)
        if (not base<=set(record) or not set(record)<=base|optional
                or record['schema']!='hud04-depth-reset-trace-v1'
                or record['event']!='hud04_depth_reset_trace'
                or record['kind'] not in kinds or type(record['fix_pending_reset']) is not bool
                or record['sequence']!=len(records)+1):
            raise ValueError('Depth event recovery schema')
        for key in ['sequence','host_elapsed_ns','capture_calls','completed_captures','reset_count','reference_frame']:
            if key in record and not (key=='reference_frame' and record[key] is None):
                if type(record[key]) is not int or record[key]<0:
                    raise ValueError('Depth event integer contract')
        for key in ['pending_reset','history_write','output_all_zero']:
            if key in record and not (key=='output_all_zero' and record[key] is None):
                if type(record[key]) is not bool:
                    raise ValueError('Depth event boolean contract')
        records.append(json.dumps(record,allow_nan=False,separators=(',',':')))
    for record in records:
        print(record,flush=True)