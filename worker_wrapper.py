"""Shorten, never restart, the audited r15 worker budget."""
import math
import sys
import time

if __name__=='__main__':
    if sys.argv[1]!='--outer-deadline':raise ValueError('Outer deadline required')
    end=float(sys.argv[2])
    remaining=end-time.monotonic()-45
    if not math.isfinite(end) or remaining<90:raise TimeoutError('Insufficient native worker budget')
    if sys.argv[3]!='--native-source':raise ValueError('Native source required')
    sys.path.insert(0,sys.argv[4])
    import worker
    worker.WORKER_SECONDS=min(900,remaining)
    sys.argv=[sys.argv[0],*sys.argv[5:]]
    sys.exit(worker.main())