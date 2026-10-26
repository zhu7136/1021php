"""One strict, padded standard-Base64 JSON-object wire format for CLI controls."""
import base64
import binascii
import json

MAX_BYTES = 8192


def encode_control(value):
    if not isinstance(value, dict):
        raise ValueError('Control parameter must be an object')
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > MAX_BYTES:
        raise ValueError('Oversized control parameter')
    return base64.b64encode(raw).decode('ascii')


def decode_control(value):
    if not isinstance(value, str) or len(value) > 4*((MAX_BYTES+2)//3) or len(value) % 4:
        raise ValueError('Invalid control encoding')
    try:
        decoded = base64.b64decode(value, validate=True)
        if len(decoded) > MAX_BYTES or base64.b64encode(decoded).decode('ascii') != value:
            raise ValueError('Invalid control encoding')
        result = json.loads(decoded)
    except (binascii.Error, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError('Invalid control encoding') from None
    if not isinstance(result, dict):
        raise ValueError('Control parameter must be an object')
    return result