"""
Token embeddings for GPT-OSS
"""

import torch
import torch.nn as nn
import math
from typing import Optional
import torch.nn.functional as F  


class TokenEmbedding(nn.Module):
    """
    Token embedding layer for GPT-OSS
    Handles vocabulary of 201,088 tokens (Harmony tokenizer)
    """
    
    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Embedding dimension
            padding_idx: Index of padding token
            max_norm: Max norm for embeddings
            norm_type: Type of norm for max_norm
            scale_grad_by_freq: Scale gradients by token frequency
            sparse: Use sparse gradients
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        
        # Token embedding matrix
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, device=device, dtype=dtype)
        )
        
        # Initialize embeddings
        self.reset_parameters()
        
        # Optional configurations
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
    def reset_parameters(self):
        """Initialize embedding weights"""
        # Standard normal initialization scaled by dimension
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        # Zero out padding token embedding if specified
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
                
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get token embeddings
        
        Args:
            input_ids: Token indices (batch_size, seq_len)
            
        Returns:
            Token embeddings (batch_size, seq_len, hidden_size)
        """
        return F.embedding(
            input_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )
        
    def resize_token_embeddings(self, new_vocab_size: int):
        """
        Resize token embeddings for vocabulary changes
        
        Args:
            new_vocab_size: New vocabulary size
        """
        old_vocab_size = self.vocab_size
        
        if new_vocab_size == old_vocab_size:
            return
            
        # Create new embedding matrix
        new_embeddings = torch.empty(
            new_vocab_size, self.hidden_size,
            device=self.weight.device,
            dtype=self.weight.dtype
        )
        
        # Initialize new embeddings
        nn.init.normal_(new_embeddings, mean=0.0, std=0.02)
        
        # Copy old embeddings
        num_to_copy = min(old_vocab_size, new_vocab_size)
        with torch.no_grad():
            new_embeddings[:num_to_copy, :] = self.weight[:num_to_copy, :]
            
        # Update weight
        self.weight = nn.Parameter(new_embeddings)
        self.vocab_size = new_vocab_size


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (alternative to RoPE)
    Not used in GPT-OSS but included for completeness
    """
    
    def __init__(
        self,
        max_position_embeddings: int,
        hidden_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        
        self.weight = nn.Parameter(
            torch.empty(
                max_position_embeddings, hidden_size,
                device=device, dtype=dtype
            )
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize positional embeddings"""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get positional embeddings
        
        Args:
            position_ids: Position indices
            seq_len: Sequence length
            
        Returns:
            Positional embeddings
        """
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=self.weight.device
            )
            
        return F.embedding(position_ids, self.weight)


class GPTOSSEmbedding(nn.Module):
    """
    Complete embedding module for GPT-OSS
    Combines token embeddings with proper initialization
    """
    
    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        padding_idx: Optional[int] = 200002,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,  # GPT-OSS doesn't use dropout
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension
            padding_idx: Padding token index
            layer_norm_eps: Layer norm epsilon
            dropout: Dropout rate (0 for GPT-OSS)
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        
        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype
        )
        
        # No dropout in GPT-OSS
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Layer norm after embeddings (optional)
        self.layer_norm = nn.LayerNorm(
            hidden_size,
            eps=layer_norm_eps,
            device=device,
            dtype=dtype
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get embeddings for input tokens
        
        Args:
            input_ids: Token indices
            token_type_ids: Token type indices (unused in GPT-OSS)
            
        Returns:
            Embedded representations
        """
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Apply layer norm
        embeddings = self.layer_norm(embeddings)
        
        # Apply dropout (identity for GPT-OSS)
        embeddings = self.dropout(embeddings)
        
        return embeddings
        
    def resize_token_embeddings(self, new_vocab_size: int):
        """Resize token embeddings"""
        self.token_embedding.resize_token_embeddings(new_vocab_size)


