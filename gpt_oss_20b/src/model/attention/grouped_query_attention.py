"""
Grouped Query Attention (GQA) implementation for GPT-OSS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .rope import apply_rotary_pos_emb, RotaryEmbedding
from .attention_sinks import AttentionSinks


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with 8:1 query-to-KV ratio
    Implements both dense and sparse attention patterns
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 45,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        use_attention_sinks: bool = True,
        attention_sink_size: int = 4,
        use_sparse_attention: bool = True,
        sparse_window_size: int = 128,
        layer_idx: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_heads: Number of query heads (64 for GPT-OSS)
            num_key_value_heads: Number of KV heads (8 for GPT-OSS)
            head_dim: Dimension per head
            max_position_embeddings: Maximum sequence length
            rope_theta: Base for RoPE
            use_attention_sinks: Whether to use attention sinks
            attention_sink_size: Number of sink tokens
            use_sparse_attention: Whether this layer uses sparse attention
            sparse_window_size: Window size for sparse attention
            layer_idx: Layer index for alternating patterns
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        
        # Determine if this layer uses sparse attention (every other layer)
        self.use_sparse = use_sparse_attention and (layer_idx % 2 == 1)
        self.sparse_window_size = sparse_window_size
        
        # Query, Key, Value projections with bias (uncommon but used in GPT-OSS)
        self.q_proj = nn.Linear(
            hidden_size, num_heads * head_dim,
            bias=True, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim,
            bias=True, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim,
            bias=True, device=device, dtype=dtype
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim, hidden_size,
            bias=False, device=device, dtype=dtype
        )
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            device=device
        )
        
        # Attention sinks
        self.attention_sinks = None
        if use_attention_sinks:
            self.attention_sinks = AttentionSinks(
                num_heads=num_heads,
                sink_size=attention_sink_size,
                device=device,
                dtype=dtype
            )
            
    def _get_sparse_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create sparse attention mask with sliding window"""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        
        for i in range(seq_len):
            # Local window
            start = max(0, i - self.sparse_window_size // 2)
            end = min(seq_len, i + self.sparse_window_size // 2 + 1)
            mask[i, start:end] = 0
            
        # Always attend to first tokens (potential sinks)
        if self.attention_sinks:
            mask[:, :self.attention_sinks.sink_size] = 0
            
        return mask
    
    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match number of Q heads for GQA
        
        Args:
            hidden_states: KV tensor of shape (batch, seq_len, num_kv_heads, head_dim)
            
        Returns:
            Repeated tensor of shape (batch, seq_len, num_heads, head_dim)
        """
        batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
        
        if self.num_key_value_groups == 1:
            return hidden_states
            
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, seq_len, num_kv_heads, self.num_key_value_groups, head_dim
        )
        return hidden_states.reshape(batch, seq_len, num_kv_heads * self.num_key_value_groups, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Grouped Query Attention
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position indices for RoPE
            past_key_value: Cached KV states
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache KV states
            
        Returns:
            Tuple of:
                - Output tensor
                - Attention weights (if output_attentions=True)
                - Cached KV states (if use_cache=True)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # Handle KV cache
        if past_key_value is not None:
            cache_k, cache_v = past_key_value
            key_states = torch.cat([cache_k, key_states], dim=2)
            value_states = torch.cat([cache_v, value_states], dim=2)
            
        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None
            
        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states.transpose(1, 2)).transpose(1, 2)
        value_states = self._repeat_kv(value_states.transpose(1, 2)).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply sparse attention mask if needed
        if self.use_sparse:
            sparse_mask = self._get_sparse_attention_mask(
                seq_len, attn_weights.device, attn_weights.dtype
            )
            attn_weights = attn_weights + sparse_mask.unsqueeze(0).unsqueeze(0)
            
        # Apply attention sinks
        if self.attention_sinks is not None:
            attn_weights, sink_weights = self.attention_sinks(
                attn_weights, key_states.shape[2], attention_mask
            )
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
            
        return outputs