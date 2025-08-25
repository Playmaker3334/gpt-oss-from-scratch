"""
Transformer block implementation for GPT-OSS
Combines attention, MoE, and normalization layers
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .attention.grouped_query_attention import GroupedQueryAttention
from .moe.mixture_of_experts import MixtureOfExperts
from .normalization.rmsnorm import RMSNorm


class GPTOSSTransformerBlock(nn.Module):
    """
    Single transformer block for GPT-OSS
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int,
        max_position_embeddings: int,
        rope_theta: float = 10000.0,
        use_attention_sinks: bool = False,
        attention_sink_size: int = 0,
        use_sparse_attention: bool = False,
        sparse_window_size: int = 0,
        rms_norm_eps: float = 1e-5,
        aux_loss_coef: float = 0.0,
        router_jitter_noise: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_attention_sinks = use_attention_sinks
        self.attention_sink_size = attention_sink_size
        self.use_sparse_attention = use_sparse_attention
        self.sparse_window_size = sparse_window_size

        # Norms
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)

        # Attention
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            use_attention_sinks=use_attention_sinks,
            attention_sink_size=attention_sink_size,
            use_sparse_attention=use_sparse_attention,
            sparse_window_size=sparse_window_size,
            device=device,
            dtype=dtype
        )

        # MoE / FFN
        self.moe = MixtureOfExperts(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            aux_loss_coef=aux_loss_coef,
            jitter_noise=router_jitter_noise,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_router_losses: bool = False,
    ):
        all_hidden_states = None
        all_attentions = None
        all_router_losses = []

        # Pre-norm and attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache
        )

        attn_hidden = attn_outputs[0]
        hidden_states = residual + attn_hidden  # residual after attention
        hidden_states = torch.clamp(hidden_states, min=-100, max=100)  # Prevenir explosión

        # Pre-norm and MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        moe_output, router_losses = self.moe(
            hidden_states,
            output_router_losses=output_router_losses
        )

        # Residual connection
        hidden_states = residual + moe_output
        hidden_states = torch.clamp(hidden_states, min=-100, max=100)  # Prevenir explosión

        # Prepare outputs in a stable order:
        outputs = (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = attn_outputs[1] if len(attn_outputs) > 1 else None
            outputs += (next_cache,)

        if output_attentions:
            attn_weights = attn_outputs[-1]
            outputs += (attn_weights,)

        if output_router_losses and router_losses is not None:
            all_router_losses.append(router_losses)

        if output_router_losses and all_router_losses:
            total_lb_loss = sum(loss[0] for loss in all_router_losses) / len(all_router_losses)
            total_z_loss = sum(loss[1] for loss in all_router_losses) / len(all_router_losses)
            router_losses = (total_lb_loss, total_z_loss)
        else:
            router_losses = None

        return (
            hidden_states,
            next_cache,
            all_hidden_states,
            all_attentions,
            router_losses
        )


class GPTOSSTransformerStack(nn.Module):
    """
    Stack of transformer blocks for GPT-OSS.
    Encadena N bloques y agrega (opcionalmente) caches, atenciones y pérdidas de ruteo.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int,
        max_position_embeddings: int,
        rope_theta: float = 10000.0,
        use_attention_sinks: bool = False,
        attention_sink_size: int = 0,
        use_sparse_attention: bool = False,
        sparse_window_size: int = 0,
        rms_norm_eps: float = 1e-5,
        aux_loss_coef: float = 0.0,
        router_jitter_noise: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GPTOSSTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                use_attention_sinks=use_attention_sinks,
                attention_sink_size=attention_sink_size,
                use_sparse_attention=use_sparse_attention,
                sparse_window_size=sparse_window_size,
                rms_norm_eps=rms_norm_eps,
                aux_loss_coef=aux_loss_coef,
                router_jitter_noise=router_jitter_noise,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_router_losses: bool = False,
    ):
        all_hidden_states = []
        all_attentions = [] if output_attentions else None
        all_router_losses = [] if output_router_losses else None
        next_caches = [] if use_cache else None

        for i, block in enumerate(self.layers):
            pkv = past_key_values[i] if (past_key_values is not None and i < len(past_key_values)) else None

            block_out = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=pkv,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_router_losses=output_router_losses,
            )

            hidden_states = block_out[0]

            if use_cache:
                next_caches.append(block_out[1])
            if output_attentions:
                all_attentions.append(block_out[3])
            if output_router_losses and block_out[4] is not None:
                all_router_losses.append(block_out[4])

            all_hidden_states.append(hidden_states)

        router_losses = None
        if output_router_losses and all_router_losses:
            lb = torch.stack([l[0] for l in all_router_losses]).mean()
            zl = torch.stack([l[1] for l in all_router_losses]).mean()
            router_losses = (lb, zl)

        return (
            hidden_states,    # último hidden
            next_caches,      # lista de caches por capa (si use_cache)
            all_hidden_states,
            all_attentions,
            router_losses
        )
