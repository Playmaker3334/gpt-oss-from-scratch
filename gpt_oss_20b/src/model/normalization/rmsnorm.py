"""
Root Mean Square Layer Normalization for GPT-OSS
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Computes: x / sqrt(mean(x^2) + eps) * gamma
    More efficient than LayerNorm as it doesn't compute mean or bias
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            hidden_size: Dimension of the hidden states
            eps: Small constant for numerical stability
            device: Device to place the parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate RMS
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        # Apply learned scaling
        return x * self.weight
    
    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


class FusedRMSNorm(nn.Module):
    """
    Fused RMSNorm implementation for better performance
    Uses single kernel for normalization and scaling
    """
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fused RMS normalization
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Store original dtype for mixed precision training
        input_dtype = x.dtype
        
        # Compute in float32 for numerical stability
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        
        # Cast back to original dtype and apply weight
        x = x.to(input_dtype)
        return x * self.weight