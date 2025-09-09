from functools import partial
import rp
import sys, threading, linecache, os

debug_print = partial(rp.fansi_print, style="blue cyan italic")

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
