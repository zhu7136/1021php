"""Local-only depth reset candidate; not installed in any runtime/config.

The pinned native depth term fills all latency slots on its next mutating
capture when reset_episodes is true. Empty or partial subsequent reset calls
must not erase an earlier request before that capture consumes it.

Use only as a mixin before the pinned native depth term in an explicitly
reviewed integration. No imports of the simulator, cameras, or native package.
This module performs no installation, patching, task submission, or I/O.
"""


class PendingDepthResetMixin:
    def reset(self, env_ids=None):
        if env_ids is None:
            self.reset_episodes[:] = True
        elif env_ids.numel() > 0:
            self.reset_episodes[env_ids] = True
        # Do not clear other pending resets. The original __call__ consumes
        # and clears the flags only after acquiring the processed frame.