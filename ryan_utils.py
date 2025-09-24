import rp
import sys, threading, linecache, os
import time

# Track start time for relative timestamps
_START_TIME = time.time()

# Global rank information
import torch.distributed as dist

def _get_rank_info():
    """Get current rank info, checking if distributed is initialized each time"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return None, None

# Global list of ranks that are allowed to print debug messages
_DEBUG_PRINT_RANKS = None

def set_debug_print_ranks(ranks_str: str) -> None:
    """Set which ranks are allowed to print debug messages.

    Args:
        ranks_str: Comma-separated rank numbers, 'all', or 'silent'
                  Examples: '0', '0,1,2', 'all', 'silent'
    """
    global _DEBUG_PRINT_RANKS

    if ranks_str.lower() == 'silent' or ranks_str.strip() == '':
        _DEBUG_PRINT_RANKS = []
    elif ranks_str.lower() == 'all':
        _DEBUG_PRINT_RANKS = None  # None means all ranks can print
    else:
        _DEBUG_PRINT_RANKS = [int(rank.strip()) for rank in ranks_str.split(',') if rank.strip()]


def debug_print(*args, **kwargs):
    """Styled debug print with rank-based filtering."""
    # Get current rank info dynamically
    current_rank, _ = _get_rank_info()

    # Check rank if using distributed training and ranks are specified
    if current_rank is not None and _DEBUG_PRINT_RANKS is not None and current_rank not in _DEBUG_PRINT_RANKS:
        return

    # Build prefix with timestamp and rank (if applicable)
    elapsed = time.time() - _START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

    if current_rank is not None:
        prefix = f"[{timestamp}|R{current_rank}]"
    else:
        prefix = f"[{timestamp}]"

    # Prepend prefix to the first argument
    if args:
        args = (f"{prefix} {args[0]}",) + args[1:]
    else:
        args = (prefix,)

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
