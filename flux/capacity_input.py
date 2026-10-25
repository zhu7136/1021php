"""Immutable task044 source, isolated from prior128-env continuation contracts."""
from pathlib import Path
from flux import formal_input

# Reuse the audited byte-bounded reader without modifying prior module globals.
_source=Path(formal_input.__file__).read_text()
_changes={
    'ARCHIVE_BYTES = 30751697':'ARCHIVE_BYTES = 31656303',
    'b5ff326aad43ae9c654d419e07f3e9af55d5ad88267e3513b5833f752e735dc7':'49322c3816f4df7f639b9d765c06c30dbb5a51b1454f294971e6fb8e371bd961',
    '6b1bab179456d475af5c6cd223c36fc5e5c45f4abb13a7db31e7865ae2082bc9':'0f6f936da4971de02a9fcb897e46df4b92ece61934655da974bf9bc374b91965',
    '20261025_034855':'20261025_050843',
    "SOURCE_TASK = 'TASK_20261025_033'":"SOURCE_TASK = 'TASK_20261025_044'",
    'SOURCE_ITERATION = 955':'SOURCE_ITERATION = 2260',
    'MAX_UPDATES = 10000':'MAX_UPDATES = 20',
}
for _old,_new in _changes.items():
    if _source.count(_old)!=1:raise ValueError('Capacity input source contract changed')
    _source=_source.replace(_old,_new,1)
exec(compile(_source,__file__,'exec'),globals())