"""
SwiGLU (Swish-Gated Linear Unit) feedforward module for GPT-OSS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: Swish-Gated Linear Unit
    Formula: SwiGLU(x) = Swish(xW) ⊙ (xV)
    More expressive than standard FFN with fewer parameters
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",  # Swish activation
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden dimension (typically 4x hidden_size)
            hidden_act: Activation function (silu/swish for SwiGLU)
            bias: Whether to use bias in linear projections
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Three projections for SwiGLU
        # W and V for gating, and output projection
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size,
            bias=bias, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size,
            bias=bias, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size,
            bias=bias, device=device, dtype=dtype
        )
        
        # Activation function
        if hidden_act == "silu" or hidden_act == "swish":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        elif hidden_act == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation
        
        Args:
            x: Input tensor of shape (..., hidden_size)
            
        Returns:
            Output tensor of shape (..., hidden_size)
        """
        # Gate path with activation
        gate = self.act_fn(self.gate_proj(x))
        
        # Value path (no activation)
        up = self.up_proj(x)
        
        # Element-wise multiplication (gating)
        intermediate = gate * up
        
        # Output projection
        output = self.down_proj(intermediate)
        
        return output
    
    def forward_chunked(
        self,
        x: torch.Tensor,
        chunk_size: int = 65536
    ) -> torch.Tensor:
        """
        Forward pass with chunking for memory efficiency
        Useful for very long sequences
        
        Args:
            x: Input tensor
            chunk_size: Size of chunks to process
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        if seq_len <= chunk_size:
            return self.forward(x)
            
        # Process in chunks
        output_chunks = []
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk = x[:, i:end_idx]
            output_chunks.append(self.forward(chunk))
            
        return torch.cat(output_chunks, dim=1)


class GeGLU(nn.Module):
    """
    GeGLU variant: GELU-Gated Linear Unit
    Alternative to SwiGLU using GELU activation
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size,
            bias=bias, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size,
            bias=bias, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size,
            bias=bias, device=device, dtype=dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class FusedSwiGLU(nn.Module):
    """
    Fused SwiGLU implementation for better performance
    Combines gate and up projections into single operation
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Fused gate and up projection
        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size,
            bias=bias, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size,
            bias=bias, device=device, dtype=dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single matrix multiplication for both gate and up
        gate_up = self.gate_up_proj(x)
        
        # Split into gate and up components
        gate, up = gate_up.chunk(2, dim=-1)
        
        # Apply activation and gating
        intermediate = F.silu(gate) * up
        
        # Output projection
        return self.down_proj(intermediate)