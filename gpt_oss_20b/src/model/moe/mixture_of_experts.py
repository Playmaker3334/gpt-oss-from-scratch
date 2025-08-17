"""
Mixture of Experts layer implementation for GPT-OSS
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .expert import Expert, ExpertParallel
from .router import Router, TokenChoiceRouter


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with Top-K routing
    32 experts for GPT-OSS-20B, 128 for GPT-OSS-120B
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 32,
        num_experts_per_token: int = 4,
        aux_loss_coef: float = 0.01,
        jitter_noise: float = 0.0,
        use_expert_parallel: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            hidden_size: Model hidden dimension
            intermediate_size: FFN intermediate dimension
            num_experts: Total number of experts
            num_experts_per_token: Number of experts per token (Top-K)
            aux_loss_coef: Auxiliary loss coefficient
            jitter_noise: Router exploration noise
            use_expert_parallel: Use parallelized expert computation
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        
        # Router
        self.router = TokenChoiceRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            aux_loss_coef=aux_loss_coef,
            jitter_noise=jitter_noise,
            device=device,
            dtype=dtype
        )
        
        # Experts
        if use_expert_parallel:
            # Parallelized expert computation
            self.experts = ExpertParallel(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=device,
                dtype=dtype
            )
        else:
            # Individual expert modules
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
        """
        Forward pass through MoE layer
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_size)
            output_router_losses: Whether to output auxiliary losses
            
        Returns:
            Tuple of:
                - Output tensor
                - Optional tuple of (load_balancing_loss, router_z_loss)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route tokens to experts
        router_weights, selected_experts, lb_loss, z_loss = self.router(
            hidden_states, training=self.training
        )
        
        # Process through experts
        if self.use_expert_parallel:
            # Efficient parallel processing
            expert_output = self._forward_parallel(
                hidden_states, router_weights, selected_experts
            )
        else:
            # Standard processing
            expert_output = self._forward_standard(
                hidden_states, router_weights, selected_experts
            )
            
        # Return output and optional losses
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
        """
        Parallel expert processing for efficiency
        
        Args:
            hidden_states: Input tensor
            router_weights: Routing weights
            selected_experts: Selected expert indices
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process each expert selection slot
        for k in range(self.num_experts_per_token):
            # Get expert index for this slot
            expert_idx = selected_experts[:, :, k]  # (batch, seq)
            expert_weight = router_weights[:, :, k:k+1]  # (batch, seq, 1)
            
            # Process through selected experts
            expert_out = self.experts(hidden_states, expert_idx)
            
            # Weighted sum
            output += expert_weight * expert_out
            
        return output
    
    def _forward_standard(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard expert processing
        
        Args:
            hidden_states: Input tensor
            router_weights: Routing weights
            selected_experts: Selected expert indices
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        # Flatten for processing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        output_flat = output.view(-1, hidden_size)
        router_weights_flat = router_weights.view(-1, self.num_experts_per_token)
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_token)
        
        # Process each token
        for token_idx in range(hidden_states_flat.size(0)):
            token_hidden = hidden_states_flat[token_idx]
            
            # Process through selected experts
            for k in range(self.num_experts_per_token):
                expert_idx = selected_experts_flat[token_idx, k].item()
                expert_weight = router_weights_flat[token_idx, k]
                
                # Forward through expert
                expert_output = self.experts[expert_idx](token_hidden.unsqueeze(0))
                
                # Weighted sum
                output_flat[token_idx] += expert_weight * expert_output.squeeze(0)
                
        return output_flat.view(batch_size, seq_len, hidden_size)
    
    def balance_load(self, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing metric
        
        Args:
            tokens_per_expert: Number of tokens routed to each expert
            
        Returns:
            Load balancing score (lower is better)
        """
        # Ideal uniform distribution
        ideal_tokens = tokens_per_expert.sum() / self.num_experts
        
        # Compute variance from ideal
        variance = ((tokens_per_expert - ideal_tokens) ** 2).mean()
        
        return variance


class SparseMoE(MixtureOfExperts):
    """
    Sparse MoE variant with conditional computation
    Only computes selected experts, saving computation
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_losses: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Sparse forward pass - only compute selected experts
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route tokens
        router_weights, selected_experts, lb_loss, z_loss = self.router(
            hidden_states, training=self.training
        )
        
        # Create dispatch/combine tensors for sparse computation
        dispatch_mask = torch.zeros(
            batch_size * seq_len, self.num_experts,
            device=hidden_states.device
        )
        
        # Fill dispatch mask
        for k in range(self.num_experts_per_token):
            expert_idx = selected_experts[:, :, k].flatten()
            token_idx = torch.arange(
                batch_size * seq_len,
                device=hidden_states.device
            )
            dispatch_mask[token_idx, expert_idx] = router_weights[:, :, k].flatten()
            
        # Sparse expert computation
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        output = torch.zeros_like(hidden_states_flat)
        
        for expert_id in range(self.num_experts):
            # Find tokens for this expert
            expert_mask = dispatch_mask[:, expert_id] > 0
            if not expert_mask.any():
                continue
                
            # Get tokens for this expert
            expert_input = hidden_states_flat[expert_mask]
            expert_weights = dispatch_mask[expert_mask, expert_id:expert_id+1]
            
            # Compute expert output
            if self.use_expert_parallel:
                expert_output = self.experts.experts[expert_id](expert_input)
            else:
                expert_output = self.experts[expert_id](expert_input)
                
            # Weighted accumulation
            output[expert_mask] += expert_weights * expert_output
            
        output = output.view(batch_size, seq_len, hidden_size)
        
        if output_router_losses:
            return output, (lb_loss, z_loss)
        else:
            return output, None