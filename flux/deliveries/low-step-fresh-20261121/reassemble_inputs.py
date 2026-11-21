"""Verify and reconstruct Git shards locally; no network or training calls."""
import argparse
import json
from pathlib import Path
from local_shards import reassemble

MANIFEST_SHA256 = '24e92ac169f998acd280b670d48b7bae7f1d0dcc914ac5a276574d72c4a91e54'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True,
                   help='New directory only; existing outputs are never overwritten.')
    a = p.parse_args()
    receipt = reassemble(Path(__file__).resolve().parent / 'inputs', a.destination,
                         manifest_sha256=MANIFEST_SHA256)
    print(json.dumps({'event':'git_inputs_reassembled', 'training_started':False, **receipt}, indent=2))