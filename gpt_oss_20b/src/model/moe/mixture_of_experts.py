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
        
        hidden_states = torch.clamp(hidden_states, min=-10, max=10)
        
        router_weights, selected_experts, lb_loss, z_loss = self.router(
            hidden_states.float(), training=self.training
        )
        
        if self.use_expert_parallel:
            expert_output = self._forward_parallel(
                hidden_states, router_weights, selected_experts
            )
        else:
            expert_output = self._forward_standard(
                hidden_states, router_weights, selected_experts
            )
            
        expert_output = torch.clamp(expert_output, min=-10, max=10)
            
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
    
    def _forward_standard(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        output_flat = output.view(-1, hidden_size)
        router_weights_flat = router_weights.view(-1, self.num_experts_per_token)
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_token)
        
        for token_idx in range(hidden_states_flat.size(0)):
            token_hidden = hidden_states_flat[token_idx]
            
            for k in range(self.num_experts_per_token):
                expert_idx = selected_experts_flat[token_idx, k].item()
                expert_weight = router_weights_flat[token_idx, k]
                
                expert_output = self.experts[expert_idx](token_hidden.unsqueeze(0))
                
                output_flat[token_idx] += expert_weight * expert_output.squeeze(0)
                
        return output_flat.view(batch_size, seq_len, hidden_size)
    
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
        
        hidden_states = torch.clamp(hidden_states, min=-10, max=10)
        
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
        output = torch.clamp(output, min=-10, max=10)
        
        if output_router_losses:
            return output, (lb_loss, z_loss)
        else:
            return output, None