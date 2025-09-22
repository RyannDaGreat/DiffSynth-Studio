# Local Claude Instructions

## Focus on WAN Models Only

Only work on WAN video models and related code. Do NOT modify or spend time on:
- FLUX models
- Qwen Image models
- Other model types

The priority is WAN video generation and training.

## No Defensive Programming

Do NOT add try/except blocks with fallbacks. Do NOT handle "edge cases" with default values. This code is for Ryan's machine only, not for a general audience.

Bad:
```python
try:
    import torch.distributed as dist
    if dist.is_available():
        rank = dist.get_rank()
    else:
        rank = 0  # fallback
except ImportError:
    rank = 0  # fallback
```

Good:
```python
import torch.distributed as dist
rank = dist.get_rank()
```

If something is missing or misconfigured, let it crash loudly. Silent failures waste debugging time.

## Why Claude Defaults to Defensive Programming

Claude is trained to write "robust" code for a general audience, which leads to:
- Excessive try/except blocks with fallbacks
- Handling every possible edge case
- Prioritizing "keeps running" over "fails fast"
- Adding complexity to handle broken environments

This is terrible for personal/research code where you want crashes to surface real problems quickly.