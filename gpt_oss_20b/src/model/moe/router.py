import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Router(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 4,
        aux_loss_coef: float = 0.01,
        jitter_noise: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.aux_loss_coef = aux_loss_coef
        self.jitter_noise = jitter_noise
        
        self.gate = nn.Linear(
            hidden_size, num_experts,
            bias=False, device=device, dtype=dtype
        )
        
        nn.init.trunc_normal_(self.gate.weight, std=0.01)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        router_logits = self.gate(hidden_states.float())
        router_logits = torch.clamp(router_logits, min=-10, max=10)
        
        if training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
            
        router_weights, selected_experts = self._top_k_routing(
            router_logits, self.num_experts_per_token
        )
        
        load_balancing_loss = self._load_balancing_loss(
            router_logits, selected_experts, batch_size * seq_len
        )
        
        router_z_loss = self._router_z_loss(router_logits)
        
        return router_weights, selected_experts, load_balancing_loss, router_z_loss
    
    def _top_k_routing(
        self,
        logits: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
        topk_weights = F.softmax(topk_logits.float(), dim=-1)
        return topk_weights, topk_indices
    
    def _load_balancing_loss(
        self,
        logits: torch.Tensor,
        selected_experts: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.float().reshape(-1, self.num_experts)
        tokens_per_expert = expert_mask.sum(dim=0) / num_tokens
        
        router_probs = F.softmax(logits.float(), dim=-1)
        router_probs = router_probs.reshape(-1, self.num_experts)
        router_prob_per_expert = router_probs.mean(dim=0)
        
        loss = self.num_experts * torch.sum(
            tokens_per_expert * router_prob_per_expert
        )
        
        return loss * self.aux_loss_coef
    
    def _router_z_loss(
        self,
        logits: torch.Tensor,
        z_loss_coef: float = 0.01
    ) -> torch.Tensor:
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        return z_loss * z_loss_coef


class TokenChoiceRouter(Router):
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().forward(hidden_states, training)


class ExpertChoiceRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_capacity: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        self.gate = nn.Linear(
            hidden_size, num_experts,
            bias=False, device=device, dtype=dtype
        )
        nn.init.trunc_normal_(self.gate.weight, std=0.01)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        
        router_logits = self.gate(hidden_states.float())
        router_logits = torch.clamp(router_logits, min=-10, max=10)
        router_probs = F.softmax(router_logits, dim=-1)
        
        dispatch_mask = torch.zeros_like(router_probs)
        
        for expert_idx in range(self.num_experts):
            expert_scores = router_probs[:, :, expert_idx].flatten()
            top_indices = torch.topk(
                expert_scores, 
                min(self.expert_capacity, expert_scores.size(0))
            ).indices
            
            mask = torch.zeros_like(expert_scores)
            mask[top_indices] = 1.0
            dispatch_mask[:, :, expert_idx] = mask.view(batch_size, seq_len)
            
        dispatch_weights = dispatch_mask * router_probs
        dispatch_weights = dispatch_weights / (
            dispatch_weights.sum(dim=-1, keepdim=True) + 1e-10
        )
        
        return dispatch_weights, dispatch_mask