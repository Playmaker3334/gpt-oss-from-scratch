"""
Rotary Position Embeddings (RoPE) with YaRN extension for GPT-OSS
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings with YaRN (Yet another RoPE extensioN) support
    Enables context length scaling beyond training length
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
        yarn_scale: float = 1.0,
        yarn_original_max_position: int = 8192,
        yarn_beta_slow: float = 1.0,
        yarn_beta_fast: float = 32.0
    ):
        """
        Args:
            dim: Dimension of embeddings (head_dim)
            max_position_embeddings: Maximum sequence length
            base: Base for computing frequencies
            device: Device to place buffers
            scaling_factor: Linear scaling factor
            yarn_scale: YaRN scaling factor
            yarn_original_max_position: Original training length
            yarn_beta_slow: YaRN slow ramp parameter
            yarn_beta_fast: YaRN fast ramp parameter
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.yarn_scale = yarn_scale
        self.yarn_original_max_position = yarn_original_max_position
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_beta_fast = yarn_beta_fast
        
        # Compute inverse frequencies with NTK-aware scaling
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache for positions
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.float32
        )
        
    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute inverse frequencies with YaRN scaling"""
        # Standard RoPE frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        
        # Apply YaRN scaling if needed
        if self.yarn_scale > 1.0:
            # Compute wavelengths
            wavelengths = 2 * math.pi / inv_freq
            
            # YaRN: interpolate between original and scaled frequencies
            ratio = self.yarn_original_max_position / wavelengths
            
            # Ramp function for smooth interpolation
            ramp = torch.clip(ratio - self.yarn_beta_slow, 0, 1)
            ramp = ramp * (1 - 1 / ((ratio / self.yarn_beta_slow) ** 4))
            ramp = ramp * (ratio < self.yarn_beta_fast)
            
            # Apply ramped scaling
            inv_freq = inv_freq / (self.yarn_scale ** ramp)
            
        return inv_freq
    
    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        """Precompute cos and sin values for positions"""
        self.max_seq_len_cached = seq_len
        
        # Create position indices
        t = torch.arange(seq_len, device=device, dtype=dtype)
        
        # Apply linear scaling if needed
        if self.scaling_factor != 1.0:
            t = t / self.scaling_factor
            
        # Compute frequencies for each position
        freqs = torch.outer(t, self.inv_freq)
        
        # Create cos and sin embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensor
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            seq_len: Sequence length (if different from cached)
            position_ids: Custom position indices
            
        Returns:
            Tuple of (cos, sin) embeddings
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
            
        if position_ids is None:
            # Use default sequential positions
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype)
            )
        else:
            # Use custom positions
            return (
                self.cos_cached[position_ids].to(dtype=x.dtype),
                self.sin_cached[position_ids].to(dtype=x.dtype)
            )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine embeddings
        sin: Sine embeddings
        position_ids: Position indices
        unsqueeze_dim: Dimension to unsqueeze cos/sin
        
    Returns:
        Tuple of rotated (query, key) tensors
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply rotation using complex number multiplication
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class YaRNScaledRotaryEmbedding(RotaryEmbedding):
    """
    YaRN-specific implementation with temperature scaling
    Used for extreme context length extensions
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        scaling_factor: float = 1.0,
        yarn_temperature: float = 1.0,
        **kwargs
    ):
        self.yarn_temperature = yarn_temperature
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
            **kwargs
        )
        
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply YaRN-scaled rotary embeddings with temperature"""
        cos, sin = super().forward(x, seq_len, position_ids)
        
        # Apply temperature scaling for attention scores
        if self.yarn_temperature != 1.0:
            cos = cos * self.yarn_temperature
            sin = sin * self.yarn_temperature
            
        return cos, sin