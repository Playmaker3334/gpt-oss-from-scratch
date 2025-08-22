"""
Transformer block implementation for GPT-OSS
Combines attention, MoE, and normalization layers
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention.grouped_query_attention import GroupedQueryAttention
from .moe.mixture_of_experts import MixtureOfExperts
from .normalization.rmsnorm import RMSNorm


class GPTOSSTransformerBlock(nn.Module):
    """
    Single transformer block for GPT-OSS
    Pre-norm architecture with GQA and MoE
    """

    def __init__(
        self,
        hidden_size: int = 2880,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        intermediate_size: int = 5760,
        num_experts: int = 32,
        num_experts_per_token: int = 4,
        layer_idx: int = 0,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        use_attention_sinks: bool = True,
        attention_sink_size: int = 4,
        use_sparse_attention: bool = True,
        sparse_window_size: int = 128,
        rms_norm_eps: float = 1e-5,
        aux_loss_coef: float = 0.01,
        router_jitter_noise: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of KV heads for GQA
            intermediate_size: FFN intermediate dimension
            num_experts: Number of MoE experts
            num_experts_per_token: Top-K experts per token
            layer_idx: Layer index for sparse attention pattern
            max_position_embeddings: Maximum sequence length
            rope_theta: RoPE base
            use_attention_sinks: Whether to use attention sinks
            attention_sink_size: Number of sink tokens
            use_sparse_attention: Whether to use sparse attention
            sparse_window_size: Window size for sparse attention
            rms_norm_eps: RMSNorm epsilon
            aux_loss_coef: MoE auxiliary loss coefficient
            router_jitter_noise: Router exploration noise
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        # Pre-norm for attention
        self.input_layernorm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype
        )

        # Grouped Query Attention
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=hidden_size // num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            use_attention_sinks=use_attention_sinks,
            attention_sink_size=attention_sink_size,
            use_sparse_attention=use_sparse_attention,
            sparse_window_size=sparse_window_size,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype
        )

        # Pre-norm for MoE
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype
        )

        # Mixture of Experts feedforward
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
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        output_router_losses: bool = True,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through transformer block

        Returns:
            Tuple containing:
                - Output hidden states
                - Optional attention weights
                - Optional cached KV states
                - Optional MoE losses
        """
        residual = hidden_states

        # Pre-norm and attention
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )

        attn_hidden = attn_outputs[0]
        hidden_states = residual + attn_hidden  # residual after attention

        # Pre-norm and MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        moe_output, router_losses = self.moe(
            hidden_states,
            output_router_losses=output_router_losses
        )

        # Residual connection
        hidden_states = residual + moe_output

        # Prepare outputs in a stable, documented order:
        # (hidden_states, [attn_weights], [cache], [router_losses])
        outputs = (hidden_states,)

        # Optional outputs are appended in a fixed order
        if output_attentions:
            outputs += (attn_outputs[1],)

        if use_cache:
            # cache is always the LAST item returned by self_attn when use_cache=True
            outputs += (attn_outputs[-1],)

        if output_router_losses and router_losses is not None:
            outputs += (router_losses,)

        return outputs


class GPTOSSTransformerStack(nn.Module):
    """
    Stack of transformer blocks for GPT-OSS
    """

    def __init__(
        self,
        num_layers: int = 24,
        hidden_size: int = 2880,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        intermediate_size: int = 5760,
        num_experts: int = 32,
        num_experts_per_token: int = 4,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        use_attention_sinks: bool = True,
        attention_sink_size: int = 4,
        use_sparse_attention: bool = True,
        sparse_window_size: int = 128,
        rms_norm_eps: float = 1e-5,
        aux_loss_coef: float = 0.01,
        router_jitter_noise: float = 0.0,
        gradient_checkpointing: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            num_layers: Number of transformer layers
            Other args: Same as GPTOSSTransformerBlock
        """
        super().__init__()
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing

        # Create transformer blocks
        self.layers = nn.ModuleList([
            GPTOSSTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                layer_idx=i,
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
            for i in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_losses: bool = True,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through all transformer layers

        Returns:
            (final_hidden_states, next_cache, all_hidden_states, all_attentions, router_losses)
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_losses = [] if output_router_losses else None
        next_cache = () if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                # Checkpointing: pass ONLY tensors; capture flags and PKV by closure
                def custom_forward(hs, am, pid):
                    return layer(
                        hs,
                        attention_mask=am,
                        position_ids=pid,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        output_router_losses=output_router_losses,
                        use_cache=use_cache
                    )

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_router_losses=output_router_losses,
                    use_cache=use_cache
                )

            # Unpack deterministically: (hs, [attn], [cache], [router])
            hidden_states = layer_outputs[0]
            idx = 1

            attn_out = None
            cache_out = None
            router_out = None

            if output_attentions:
                attn_out = layer_outputs[idx]
                idx += 1

            if use_cache:
                cache_out = layer_outputs[idx]
                idx += 1

            if output_router_losses and len(layer_outputs) > idx:
                router_out = layer_outputs[idx]

            if use_cache:
                next_cache += (cache_out,)

            if output_attentions:
                all_attentions += (attn_out,)

            if output_router_losses and router_out is not None:
                all_router_losses.append(router_out)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Aggregate router losses across layers (mean)
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
