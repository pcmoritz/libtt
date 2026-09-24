"""Diagnostic-only IR export; does not change graph optimizations."""
import os
if os.environ.get('LIBTT_AUDIT_EXPORT'):
    import jax
    _original_jit = jax.jit
    def _audit_jit(*args, **kwargs):
        options = kwargs.get('compiler_options')
        if options and ('enable_trace' in options or 'optimization_level' in options):
            kwargs['compiler_options'] = dict(options, export_path=os.environ['LIBTT_AUDIT_EXPORT'])
        return _original_jit(*args, **kwargs)
    jax.jit = _audit_jit
