"""Absolute deadlines measured from the original worker start, never reset."""
WORKER_SECONDS=1800
INPUT_DEADLINE=600
INSTALL_DEADLINE=720
GPU_PREFLIGHT_DEADLINE=750
LATEST_TRAIN_START=750
TRAIN_SOFT_DEADLINE=1650
TRAIN_HARD_DEADLINE=1710
RESULT_UPLOAD_DEADLINE=1790
MIN_TRAIN_WINDOW=900

def deadlines(worker_start,now):
    if now<worker_start or now>=worker_start+WORKER_SECONDS:
        raise TimeoutError('Invalid or exhausted original worker budget')
    return {k:worker_start+v for k,v in {
        'input':INPUT_DEADLINE,'install':INSTALL_DEADLINE,'gpu':GPU_PREFLIGHT_DEADLINE,
        'train_soft':TRAIN_SOFT_DEADLINE,'train_hard':TRAIN_HARD_DEADLINE,
        'result':RESULT_UPLOAD_DEADLINE,'worker':WORKER_SECONDS}.items()}

def require_training_window(worker_start,now):
    ends=deadlines(worker_start,now)
    if now>worker_start+LATEST_TRAIN_START or ends['train_soft']-now<MIN_TRAIN_WINDOW:
        raise TimeoutError('Insufficient reserved training window')
    return ends