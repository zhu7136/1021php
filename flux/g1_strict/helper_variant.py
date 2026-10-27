"""Hash-pinned runtime variant; never edits frozen source or USD methods."""
import hashlib,inspect

SOURCE_SHA='16c60deb4d5fb77a7f524e02085039e15b95209c56dbcfd1a722f6ada9eb9801'
PATCH='''            # Replace with a no-op version
            def noop_close_stage(self, *args, **kwargs):
                logger.info("Skipping close_stage() to avoid hanging")
                return True

            # Apply the patch
            context_class.close_stage = noop_close_stage
            logger.info("Successfully patched close_stage method")'''

def make_variant(original,before_close):
    source=inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:raise ValueError('Unpinned original helper')
    if source.count(PATCH)!=1 or source.count('        simulation_app.close(wait_for_replicator=False)')!=1:
        raise ValueError('Unexpected helper body')
    changed=source.replace(PATCH,'            # USD close_stage deliberately left unchanged.',1)
    changed=changed.replace('        simulation_app.close(wait_for_replicator=False)',
        '        _hud04_before_close()\n        simulation_app.close(wait_for_replicator=False)',1)
    bindings=dict(original.__globals__);bindings['_hud04_before_close']=before_close
    exec(compile(changed,'<hud04-r31-helper-without-usd-patch>','exec'),bindings)
    return bindings['close_simulation_app'],dict(original_source_sha256=SOURCE_SHA,
        variant_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),usd_patch_removed=True,
        full_original_module_globals_used=True)