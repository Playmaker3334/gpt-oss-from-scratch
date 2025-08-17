"""
Router module for Mixture of Experts in GPT-OSS
Implements Top-K routing with load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Router(nn.Module):
    """
    Learned linear router for expert selection
    Uses Top-K routing with softmax-after-topk strategy
    """
    
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
        """
        Args:
            hidden_size: Input dimension
            num_experts: Total number of experts
            num_experts_per_token: Number of experts to route to (Top-K)
            aux_loss_coef: Coefficient for auxiliary load balancing loss
            jitter_noise: Noise for exploration during training
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.aux_loss_coef = aux_loss_coef
        self.jitter_noise = jitter_noise
        
        # Linear projection to expert logits
        self.gate = nn.Linear(
            hidden_size, num_experts,
            bias=False, device=device, dtype=dtype
        )
        
        # Initialize with truncated normal
        nn.init.trunc_normal_(self.gate.weight, std=0.02)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            training: Whether in training mode
            
        Returns:
            Tuple of:
                - Expert weights (batch_size, seq_len, num_experts_per_token)
                - Expert indices (batch_size, seq_len, num_experts_per_token)
                - Load balancing loss
                - Router z-loss for stability
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # (batch, seq, num_experts)
        
        # Add jitter noise during training for exploration
        if training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
            
        # Top-K routing
        router_weights, selected_experts = self._top_k_routing(
            router_logits, self.num_experts_per_token
        )
        
        # Compute auxiliary losses
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
        """
        Perform top-k routing with softmax-after-topk
        
        Args:
            logits: Router logits (batch, seq, num_experts)
            k: Number of experts to select
            
        Returns:
            Tuple of weights and indices
        """
        # Get top-k values and indices
        topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
        
        # Softmax over selected experts only (softmax-after-topk)
        topk_weights = F.softmax(topk_logits, dim=-1)
        
        return topk_weights, topk_indices
    
    def _load_balancing_loss(
        self,
        logits: torch.Tensor,
        selected_experts: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss
        Encourages uniform distribution of tokens across experts
        
        Args:
            logits: Router logits
            selected_experts: Selected expert indices
            num_tokens: Total number of tokens
            
        Returns:
            Load balancing loss
        """
        # Compute expert fractions (how many tokens go to each expert)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_mask = expert_mask.float().reshape(-1, self.num_experts)
        tokens_per_expert = expert_mask.sum(dim=0) / num_tokens
        
        # Compute router probabilities
        router_probs = F.softmax(logits, dim=-1)
        router_probs = router_probs.reshape(-1, self.num_experts)
        router_prob_per_expert = router_probs.mean(dim=0)
        
        # Load balancing loss encourages uniform distribution
        loss = self.num_experts * torch.sum(
            tokens_per_expert * router_prob_per_expert
        )
        
        return loss * self.aux_loss_coef
    
    def _router_z_loss(
        self,
        logits: torch.Tensor,
        z_loss_coef: float = 1e-3
    ) -> torch.Tensor:
        """
        Router z-loss for numerical stability
        Penalizes large logits to prevent router collapse
        
        Args:
            logits: Router logits
            z_loss_coef: Coefficient for z-loss
            
        Returns:
            Router z-loss
        """
        # Compute log(sum(exp(logits)))
        log_z = torch.logsumexp(logits, dim=-1)
        
        # Squared loss on log partition function
        z_loss = (log_z ** 2).mean()
        
        return z_loss * z_loss_coef


class TokenChoiceRouter(Router):
    """
    Token-choice routing where each token selects its experts
    Standard routing strategy for GPT-OSS
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Token-choice routing: each token independently selects experts
        """
        return super().forward(hidden_states, training)


class ExpertChoiceRouter(nn.Module):
    """
    Expert-choice routing where experts select tokens
    Alternative routing strategy for better load balancing
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_capacity: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            hidden_size: Input dimension
            num_experts: Number of experts
            expert_capacity: Maximum tokens per expert
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        self.gate = nn.Linear(
            hidden_size, num_experts,
            bias=False, device=device, dtype=dtype
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing: experts select top tokens
        
        Args:
            hidden_states: Input tensor
            training: Whether in training mode
            
        Returns:
            Tuple of dispatch and combine tensors
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute affinities
        router_logits = self.gate(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Each expert selects top tokens
        dispatch_mask = torch.zeros_like(router_probs)
        
        for expert_idx in range(self.num_experts):
            expert_scores = router_probs[:, :, expert_idx].flatten()
            top_indices = torch.topk(
                expert_scores, 
                min(self.expert_capacity, expert_scores.size(0))
            ).indices
            
            # Create mask for selected tokens
            mask = torch.zeros_like(expert_scores)
            mask[top_indices] = 1.0
            dispatch_mask[:, :, expert_idx] = mask.view(batch_size, seq_len)
            
        # Normalize dispatch weights
        dispatch_weights = dispatch_mask * router_probs
        dispatch_weights = dispatch_weights / (
            dispatch_weights.sum(dim=-1, keepdim=True) + 1e-10
        )
        
        return dispatch_weights, dispatch_mask