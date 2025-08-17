"""
Attention Sinks implementation for GPT-OSS
Stabilizes long-context processing through learned bias logits
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AttentionSinks(nn.Module):
    """
    Attention Sinks: Learned per-head bias logits for stabilizing attention
    These are not actual tokens but learned parameters that allow attention
    heads to "skip" tokens when necessary
    """
    
    def __init__(
        self,
        num_heads: int,
        sink_size: int = 4,
        temperature: float = 1.0,
        learnable: bool = True,
        init_value: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            num_heads: Number of attention heads
            sink_size: Number of sink tokens per head
            temperature: Temperature for sink attention scores
            learnable: Whether sink values are learnable parameters
            init_value: Initial value for sink biases
            device: Device to place parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.num_heads = num_heads
        self.sink_size = sink_size
        self.temperature = temperature
        
        if learnable:
            # Learnable sink biases per head
            self.sink_biases = nn.Parameter(
                torch.zeros(
                    num_heads, sink_size,
                    device=device, dtype=dtype
                ) + init_value
            )
        else:
            # Fixed sink biases
            self.register_buffer(
                "sink_biases",
                torch.zeros(
                    num_heads, sink_size,
                    device=device, dtype=dtype
                ) + init_value
            )
            
        # Optional learnable temperature per head
        self.head_temperatures = nn.Parameter(
            torch.ones(num_heads, device=device, dtype=dtype) * temperature
        )
        
    def forward(
        self,
        attention_scores: torch.Tensor,
        key_length: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention sinks to attention scores
        
        Args:
            attention_scores: Raw attention scores (batch, num_heads, seq_len, key_len)
            key_length: Length of key sequence
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of:
                - Modified attention scores with sinks
                - Sink attention weights for analysis
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Get sink biases for current heads
        sink_biases = self.sink_biases[:num_heads, :self.sink_size]
        
        # Expand sink biases to match batch and sequence dimensions
        sink_biases = sink_biases.unsqueeze(0).unsqueeze(2)  # (1, num_heads, 1, sink_size)
        sink_biases = sink_biases.expand(batch_size, -1, seq_len, -1)
        
        # Apply temperature scaling
        temps = self.head_temperatures[:num_heads].view(1, num_heads, 1, 1)
        sink_biases = sink_biases / temps
        
        # Concatenate sink biases with attention scores
        # Sinks are prepended to the attention scores
        extended_scores = torch.cat([
            sink_biases,
            attention_scores
        ], dim=-1)
        
        # Adjust attention mask if provided
        if attention_mask is not None:
            # Create sink mask (always attend to sinks)
            sink_mask = torch.ones(
                batch_size, 1, seq_len, self.sink_size,
                device=attention_scores.device,
                dtype=attention_mask.dtype
            )
            
            # Concatenate masks
            extended_mask = torch.cat([
                sink_mask,
                attention_mask
            ], dim=-1)
            
            # Apply mask
            extended_scores = extended_scores.masked_fill(
                extended_mask == 0,
                float('-inf')
            )
            
        # Extract sink attention weights for analysis
        sink_weights = torch.softmax(extended_scores[..., :self.sink_size], dim=-1)
        
        return extended_scores, sink_weights
    
    def compute_sink_loss(
        self,
        sink_weights: torch.Tensor,
        target_distribution: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute regularization loss for sink attention
        Encourages sinks to capture uniform or specific attention patterns
        
        Args:
            sink_weights: Sink attention weights from forward pass
            target_distribution: Target distribution for sinks (uniform if None)
            
        Returns:
            Sink regularization loss
        """
        if target_distribution is None:
            # Use uniform distribution as target
            target_distribution = torch.ones_like(sink_weights) / self.sink_size
            
        # KL divergence loss
        loss = torch.nn.functional.kl_div(
            torch.log(sink_weights + 1e-10),
            target_distribution,
            reduction='batchmean'
        )
        
        return loss


class StreamingAttentionSinks(AttentionSinks):
    """
    Attention Sinks optimized for streaming/incremental generation
    Maintains sink state across generation steps
    """
    
    def __init__(self, *args, window_size: int = 1024, **kwargs):
        """
        Args:
            window_size: Size of attention window for streaming
            *args, **kwargs: Arguments for parent AttentionSinks
        """
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.register_buffer("sink_cache", None)
        
    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize sink cache for streaming"""
        self.sink_cache = torch.zeros(
            batch_size, self.num_heads, self.sink_size,
            device=device, dtype=dtype
        )
        
    def update_cache(self, new_sinks: torch.Tensor, step: int):
        """Update sink cache with exponential moving average"""
        if self.sink_cache is None:
            self.sink_cache = new_sinks
        else:
            # Exponential moving average
            alpha = min(1.0, 2.0 / (step + 1))
            self.sink_cache = alpha * new_sinks + (1 - alpha) * self.sink_cache
            
    def get_streaming_mask(self, seq_len: int) -> torch.Tensor:
        """Get attention mask for streaming with sliding window"""
        mask = torch.ones(seq_len, seq_len)
        
        # Apply sliding window
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, :start] = 0
            
        # Always attend to sinks (first tokens)
        mask[:, :self.sink_size] = 1
        
        return mask