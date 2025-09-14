import rp
import sys, threading, linecache, os

# Global toggle for debug_print behavior
_DEBUG_PRINT_ENABLED = True

def set_debug_print_enabled(enabled: bool) -> None:
    """Enable or disable debug_print globally at runtime.

    When disabled, all calls to debug_print become no-ops.
    """
    global _DEBUG_PRINT_ENABLED
    _DEBUG_PRINT_ENABLED = bool(enabled)

def debug_print(*args, **kwargs):
    """Styled debug print that can be disabled via set_debug_print_enabled."""
    if not _DEBUG_PRINT_ENABLED:
        return
    # Ensure our style is applied while respecting caller kwargs
    kwargs.setdefault("style", "blue cyan italic")
    return rp.fansi_print(*args, **kwargs)

def enable_line_tracing(include_libs=True):
    """Enable super-verbose line-level tracing. Prints every executed line.

    include_libs: if False, restrict to files under repo root (same dir as this file).
    """
    repo_root = os.path.abspath(os.path.dirname(__file__))
    repo_root = repo_root if repo_root.endswith(os.sep) else repo_root + os.sep

    def tracer(frame, event, arg):
        if event != "line":
            return tracer
        filename = frame.f_code.co_filename
        # Avoid tracing this file to prevent recursion
        if filename.endswith("ryan_utils.py"):
            return tracer
        if not include_libs:
            # Only trace files under the same repository root
            if not os.path.abspath(filename).startswith(repo_root):
                return tracer
        try:
            src = linecache.getline(filename, frame.f_lineno).rstrip()
        except Exception:
            src = ""
        fn_name = frame.f_code.co_name
        try:
            debug_print(f"TRACE LINE {filename}:{frame.f_lineno} in {fn_name} -> {src}")
        except Exception:
            # Last-resort print if rp hiccups
            print(f"TRACE LINE {filename}:{frame.f_lineno} in {fn_name} -> {src}")
        return tracer

    sys.settrace(tracer)
    threading.settrace(tracer)
    debug_print("Line-level tracing enabled (this will be VERY noisy)")
