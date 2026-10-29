"""CPU-only reconstruction/verification. Never install dependencies or train."""
import argparse
import json
from pathlib import Path
import time
from bounded_training_trial import unpack
from local_trial_inputs import materialize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='New directory; must not already exist')
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    pins=materialize(root/'inputs',output,manifest_sha256=args.manifest_sha256,
                     absolute_deadline=time.monotonic()+600,unpack_archive=unpack)
    execution=json.loads((output/'payload/execution.json').read_text())
    if execution['plan']['execution_approved'] is not False or execution['task_id'] is not None:
        raise ValueError('Expected publication-only, unapproved template')
    print(json.dumps({'inputs_verified':True,'training_started':False,'inputs':pins},indent=2))


if __name__=='__main__':main()