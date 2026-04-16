from collections import Counter

import torch
from torch import nn

def check_model(model: nn.Module, show_stats: bool=True):
    assert isinstance(model, nn.Module)
    print("-"*80)

    # 1. nn.Module 信息
    print(f"> Info of `nn.Module`: ")
    print(f"  - Module name: {model.__class__.__name__}")
    print(f"  - Children modules: {[m.__class__.__name__ for m in model.children()]}")
    print(f"  - Children parameters: {[name for name, _ in model.named_parameters(recurse=False)]}")
    print(f"  - Training: {model.training}")

    # 2. nn.Parameter 信息
    params = dict(model.named_parameters())
    params_trainable = {k: v for k, v in params.items() if v.requires_grad}
    params_frozen = {k: v for k, v in params.items() if not v.requires_grad}
    num_elements = sum(p.numel() for _, p in params.items())
    num_trainable_elements = sum(p.numel() for _, p in params.items() if p.requires_grad)
    num_frozen_elements = sum(p.numel() for _, p in params.items() if not p.requires_grad)
    dtypes = Counter(str(p.dtype) for _, p in params.items())
    devices = Counter(str(p.device) for _, p in params.items())
    # params_flattened = None
    print(f"> Info of `nn.Parameters`: ")
    print(f"  - Total: {len(params)} - {num_elements / 1e9:.3f} B ({num_elements:,})")
    print(f"  - Trainable: {len(params_trainable)} - {num_trainable_elements / 1e9:.3f} B ({num_trainable_elements:,})")
    print(f"  - Frozen: {len(params_frozen)} - {num_frozen_elements / 1e9:.3f} B ({num_frozen_elements:,})")
    print(f"  - Types:")
    for dtype, num in dtypes.items():
        print(f"        `{dtype}`: {num}")
    print(f"  - Devices:")
    for device, num in devices.items():
        print(f"        `{device}`: {num}")

    reverse_params = {v: k for k, v in params.items()}
    params_ = dict(model.named_parameters(remove_duplicate=False))
    params_shared = {}
    for k, v in params_.items():
        if v in reverse_params:
            if reverse_params[v] in params_shared:
                params_shared[reverse_params[v]].append(k)
            else:
                params_shared[reverse_params[v]] = []
    params_shared = {k: v for k, v in params_shared.items() if len(v) > 0}
    print(f"  - Shared:")
    for k, v in params_shared.items():
        print(f"        `{k}`: {v}")
    if params_shared == {}:
        print(f"        No shared.")

    params_flattened = torch.cat([p.data.flatten().cpu() for _, p in params.items()]) if show_stats and "meta" not in devices else None
    print(f"  - Stats:")
    if isinstance(params_flattened, torch.Tensor):
        print(f"        min: {params_flattened.min().item():.8g}")
        print(f"        max: {params_flattened.max().item():.8g}")
        print(f"        mean: {params_flattened.mean().item():.8g}")
        print(f"        std: {params_flattened.std().item():.8g}")
    else:
        print(f"        No data.")

    print("-"*80)
