"""
Expert module for Mixture of Experts in GPT-OSS
"""

import torch
import torch.nn as nn
from typing import Optional
from ..feedforward.swiglu import SwiGLU, FusedSwiGLU


class Expert(nn.Module):
    """
    Single expert in the MoE layer
    Each expert is a SwiGLU feedforward network
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        use_fused: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden dimension for FFN
            hidden_act: Activation function
            use_fused: Whether to use fused SwiGLU implementation
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        
        if use_fused:
            self.ffn = FusedSwiGLU(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                bias=False,
                device=device,
                dtype=dtype
            )
        else:
            self.ffn = SwiGLU(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                bias=False,
                device=device,
                dtype=dtype
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.ffn(x)


class ExpertParallel(nn.Module):
    """
    Parallelized expert computation for efficiency
    Processes multiple experts simultaneously
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            num_experts: Number of experts
            hidden_size: Input/output dimension
            intermediate_size: Hidden dimension
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Batched weight matrices for all experts
        # More efficient than separate Linear layers
        self.gate_proj = nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size,
                device=device, dtype=dtype
            )
        )
        self.up_proj = nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size,
                device=device, dtype=dtype
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                num_experts, intermediate_size, hidden_size,
                device=device, dtype=dtype
            )
        )
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters with scaled normal distribution"""
        std = (2 / (5 * self.hidden_size)) ** 0.5
        nn.init.normal_(self.gate_proj, mean=0.0, std=std)
        nn.init.normal_(self.up_proj, mean=0.0, std=std)
        nn.init.normal_(self.down_proj, mean=0.0, std=std)
        
    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through selected experts
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            expert_indices: Which expert to use for each token
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # (batch * seq, hidden)
        expert_indices_flat = expert_indices.view(-1)  # (batch * seq)
        
        # Gather expert weights
        gate_w = self.gate_proj[expert_indices_flat]  # (batch * seq, hidden, inter)
        up_w = self.up_proj[expert_indices_flat]
        down_w = self.down_proj[expert_indices_flat]
        
        # Batched matrix multiplication
        gate = torch.bmm(x_flat.unsqueeze(1), gate_w).squeeze(1)
        up = torch.bmm(x_flat.unsqueeze(1), up_w).squeeze(1)
        
        # SwiGLU activation
        intermediate = torch.nn.functional.silu(gate) * up
        
        # Output projection
        output = torch.bmm(intermediate.unsqueeze(1), down_w.transpose(-1, -2)).squeeze(1)
        
        return output.view(batch_size, seq_len, hidden_size)