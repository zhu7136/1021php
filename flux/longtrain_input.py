"""Immutable independently verified task045 native final checkpoint."""
from pathlib import Path
from flux import formal_input
_source=Path(formal_input.__file__).read_text()
_changes={
    'ARCHIVE_BYTES = 30751697':'ARCHIVE_BYTES = 28676205',
    'b5ff326aad43ae9c654d419e07f3e9af55d5ad88267e3513b5833f752e735dc7':'e6b9720239c70bbce59c8250b08621e1b5cd4a4fdfdfd3a2e7424b88fd331194',
    '6b1bab179456d475af5c6cd223c36fc5e5c45f4abb13a7db31e7865ae2082bc9':'b6599f50fbef9e31e068bcd1bd1b001797bb6376f18b78c757596a7d861b5c3a',
    '20261025_034855-php-locomotion-teacher-formal-30m-locomotion':'20261025_055005-php-teacher-capacity-4096-20updates-locomotion',
    "SOURCE_TASK = 'TASK_20261025_033'":"SOURCE_TASK = 'TASK_20261025_045'",
    'SOURCE_ITERATION = 955':'SOURCE_ITERATION = 2279',
    'MAX_UPDATES = 10000':'MAX_UPDATES = 20000',
}
for _a,_b in _changes.items():
    if _source.count(_a)!=1:raise ValueError('Longtrain source contract changed')
    _source=_source.replace(_a,_b,1)
exec(compile(_source,__file__,'exec'),globals())