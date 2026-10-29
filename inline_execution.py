"""Small private task configuration, not a signed URL or credential channel.

Hash identity is an integrity check, not authorization. The unchanged native
task/template binding remains responsible for authorization and scope checks.
"""
import base64
import hashlib
import json
import re

MAX_BYTES=16384


def decode_inline(encoded,sha256):
    if not isinstance(encoded,str) or not 0<len(encoded)<=4*((MAX_BYTES+2)//3):
        raise ValueError('Inline configuration length')
    if not isinstance(sha256,str) or not re.fullmatch('[0-9a-f]{64}',sha256):
        raise ValueError('Inline configuration hash format')
    try:raw=base64.b64decode(encoded.encode('ascii'),altchars=b'-_',validate=True)
    except (ValueError,UnicodeError):raise ValueError('Inline configuration encoding') from None
    if not 0<len(raw)<=MAX_BYTES or hashlib.sha256(raw).hexdigest()!=sha256:
        raise ValueError('Inline configuration hash mismatch')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate configuration key')
            result[key]=value
        return result
    def constant(_):raise ValueError('Nonfinite configuration value')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    if not isinstance(value,dict):raise ValueError('Configuration object required')
    return raw,value
