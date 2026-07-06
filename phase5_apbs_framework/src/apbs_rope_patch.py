from __future__ import annotations

import math
import types
from dataclasses import dataclass

import torch


@dataclass
class APBSConfig:
    method: str
    gamma: float = 0.3
    target_length: int = 16384
    train_context_length: int = 4096
    base: float = 10000.0


def apbs_position_weight(position_ids: torch.Tensor, target_length: int) -> torch.Tensor:
    """Piecewise middle-window weight: 0 at ends, 1 in 40%-60%, linear ramps."""
    p = position_ids.to(torch.float32) / max(float(target_length - 1), 1.0)
    zeros = torch.zeros_like(p)
    ones = torch.ones_like(p)
    left_ramp = (p - 0.30) / 0.10
    right_ramp = (0.70 - p) / 0.10
    return torch.where(
        (p >= 0.40) & (p <= 0.60),
        ones,
        torch.where(
            (p >= 0.30) & (p < 0.40),
            torch.clamp(left_ramp, 0.0, 1.0),
            torch.where((p > 0.60) & (p <= 0.70), torch.clamp(right_ramp, 0.0, 1.0), zeros),
        ),
    )


def ntk_base(base: float, target_length: int, train_context_length: int, rotary_dim: int) -> float:
    if target_length <= train_context_length:
        return float(base)
    exponent = rotary_dim / max(rotary_dim - 2, 1)
    return float(base * (target_length / train_context_length) ** exponent)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _build_forward(module, cfg: APBSConfig):
    original_forward = module.forward
    rotary_dim = int(module.inv_freq.numel() * 2)
    base = float(getattr(module, "base", cfg.base))
    if hasattr(module, "rope_kwargs") and isinstance(module.rope_kwargs, dict):
        base = float(module.rope_kwargs.get("base", base))

    def patched_forward(self, x, position_ids=None, *args, **kwargs):
        if cfg.method == "baseline":
            return original_forward(x, position_ids, *args, **kwargs)

        if position_ids is None:
            seq_len = x.shape[-2]
            position_ids_local = torch.arange(seq_len, device=x.device).unsqueeze(0)
        else:
            position_ids_local = position_ids

        device = x.device
        dtype = torch.float32
        dim_idx = torch.arange(0, rotary_dim, 2, device=device, dtype=dtype)
        exponent = dim_idx / rotary_dim

        global_base = ntk_base(base, cfg.target_length, cfg.train_context_length, rotary_dim)
        if cfg.method == "ntk":
            effective_base = torch.full_like(position_ids_local, global_base, dtype=dtype, device=device)
        elif cfg.method == "apbs":
            w = apbs_position_weight(position_ids_local.to(device), cfg.target_length)
            effective_base = global_base * (1.0 + cfg.gamma * w)
        else:
            raise ValueError(f"Unknown RoPE method: {cfg.method}")

        inv_freq = 1.0 / (effective_base.unsqueeze(-1) ** exponent)
        freqs = position_ids_local.to(device=device, dtype=dtype).unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin

    return types.MethodType(patched_forward, module)


def apply_rope_patch(model, cfg: APBSConfig) -> int:
    """Patch rotary embedding modules in-place. Returns the number of patched modules."""
    if cfg.method == "baseline":
        return 0

    patched = 0
    for module in model.modules():
        name = module.__class__.__name__.lower()
        if "rotary" in name and hasattr(module, "inv_freq") and callable(getattr(module, "forward", None)):
            module.forward = _build_forward(module, cfg)
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "No rotary embedding modules were patched. Check the model architecture or update apbs_rope_patch.py."
        )
    return patched


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Utility kept for local experiments; HF model code usually provides its own version."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

