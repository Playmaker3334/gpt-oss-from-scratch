import torch
import torch.nn as nn
from typing import Optional
from ..feedforward.swiglu import SwiGLU, FusedSwiGLU


class Expert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        use_fused: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
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
        return self.ffn(x)


class ExpertParallel(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
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
        
        self.reset_parameters()
        
    def reset_parameters(self):
        std = (2 / (5 * self.hidden_size)) ** 0.5
        nn.init.normal_(self.gate_proj, mean=0.0, std=std)
        nn.init.normal_(self.up_proj, mean=0.0, std=std)
        nn.init.normal_(self.down_proj, mean=0.0, std=std)
        
    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        expert_indices_flat = expert_indices.view(-1)
        
        gate_w = self.gate_proj[expert_indices_flat]
        up_w = self.up_proj[expert_indices_flat]
        down_w = self.down_proj[expert_indices_flat]
        
        gate = torch.bmm(x_flat.unsqueeze(1), gate_w).squeeze(1)
        up = torch.bmm(x_flat.unsqueeze(1), up_w).squeeze(1)
        
        intermediate = torch.nn.functional.silu(gate) * up
        
        output = torch.bmm(intermediate.unsqueeze(1), down_w).squeeze(1)
        
        return output.view(batch_size, seq_len, hidden_size)