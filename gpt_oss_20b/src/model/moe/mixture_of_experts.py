import torch
import torch.nn as nn
from typing import Optional, Tuple
from .expert import Expert, ExpertParallel
from .router import Router, TokenChoiceRouter


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        aux_loss_coef: float = 0.001,
        jitter_noise: float = 0.0,
        use_expert_parallel: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        
        self.router = TokenChoiceRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            aux_loss_coef=aux_loss_coef,
            jitter_noise=jitter_noise,
            device=device,
            dtype=dtype
        )
        
        if use_expert_parallel:
            self.experts = ExpertParallel(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=device,
                dtype=dtype
            )
        else:
            self.experts = nn.ModuleList([
                Expert(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    device=device,
                    dtype=dtype
                )
                for _ in range(num_experts)
            ])
            
        self.use_expert_parallel = use_expert_parallel
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_losses: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        router_weights, selected_experts, lb_loss, z_loss = self.router(
            hidden_states.float(), training=self.training
        )
        
        if self.use_expert_parallel:
            expert_output = self._forward_parallel(
                hidden_states, router_weights, selected_experts
            )
        else:
            expert_output = self._forward_efficient(
                hidden_states, router_weights, selected_experts
            )
            
        if output_router_losses:
            return expert_output, (lb_loss, z_loss)
        else:
            return expert_output, None
            
    def _forward_parallel(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        output = torch.zeros_like(hidden_states)
        
        for k in range(self.num_experts_per_token):
            expert_idx = selected_experts[:, :, k]
            expert_weight = router_weights[:, :, k:k+1]
            
            expert_out = self.experts(hidden_states, expert_idx)
            
            output += expert_weight * expert_out
            
        return output
    
    def _forward_efficient(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue
            
            expert_tokens = hidden_states[expert_mask]
            
            if expert_tokens.numel() == 0:
                continue
                
            expert_output = self.experts[expert_idx](expert_tokens)
            
            for k in range(self.num_experts_per_token):
                weight_mask = selected_experts[:, :, k] == expert_idx
                if weight_mask.any():
                    weights = router_weights[:, :, k:k+1]
                    output[weight_mask] += weights[weight_mask] * expert_output[weight_mask[expert_mask]]
        
        return output
    
    def _forward_standard(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        output = torch.zeros_like(hidden_states_flat)
        
        for expert_idx in range(self.num_experts):
            expert_mask_flat = torch.zeros(batch_size * seq_len, dtype=torch.bool, device=hidden_states.device)
            expert_weights_sum = torch.zeros(batch_size * seq_len, 1, device=hidden_states.device)
            
            for k in range(self.num_experts_per_token):
                mask = (selected_experts[:, :, k] == expert_idx).view(-1)
                expert_mask_flat |= mask
                weights = router_weights[:, :, k].view(-1, 1)
                expert_weights_sum[mask] += weights[mask]
            
            if not expert_mask_flat.any():
                continue
                
            expert_input = hidden_states_flat[expert_mask_flat]
            expert_output = self.experts[expert_idx](expert_input)
            output[expert_mask_flat] += expert_weights_sum[expert_mask_flat] * expert_output
        
        return output.view(batch_size, seq_len, hidden_size)
    
    def balance_load(self, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        ideal_tokens = tokens_per_expert.sum() / self.num_experts
        variance = ((tokens_per_expert - ideal_tokens) ** 2).mean()
        return variance


class SparseMoE(MixtureOfExperts):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_losses: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        router_weights, selected_experts, lb_loss, z_loss = self.router(
            hidden_states.float(), training=self.training
        )
        
        dispatch_mask = torch.zeros(
            batch_size * seq_len, self.num_experts,
            device=hidden_states.device
        )
        
        for k in range(self.num_experts_per_token):
            expert_idx = selected_experts[:, :, k].flatten()
            token_idx = torch.arange(
                batch_size * seq_len,
                device=hidden_states.device
            )
            dispatch_mask[token_idx, expert_idx] = router_weights[:, :, k].flatten()
            
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        output = torch.zeros_like(hidden_states_flat)
        
        for expert_id in range(self.num_experts):
            expert_mask = dispatch_mask[:, expert_id] > 0
            if not expert_mask.any():
                continue
                
            expert_input = hidden_states_flat[expert_mask]
            expert_weights = dispatch_mask[expert_mask, expert_id:expert_id+1]
            
            if self.use_expert_parallel:
                expert_output = self.experts.experts[expert_id](expert_input)
            else:
                expert_output = self.experts[expert_id](expert_input)
                
            output[expert_mask] += expert_weights * expert_output
            
        output = output.view(batch_size, seq_len, hidden_size)
        
        if output_router_losses:
            return output, (lb_loss, z_loss)
        else:
            return output, None