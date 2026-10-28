"""QQ-input adapter around the unchanged bounded native resume worker."""
import json,sys
import bootstrap
import transfer_diagnostics
from qq_input import validate_capability

bootstrap.validate_capability=validate_capability
transfer_diagnostics.validate_capability=validate_capability

if __name__=='__main__':
    try:sys.exit(bootstrap.main())
    except Exception as e:
        print(json.dumps({'event':'bootstrap_failure','error_type':type(e).__name__}),flush=True)
        sys.exit(1)