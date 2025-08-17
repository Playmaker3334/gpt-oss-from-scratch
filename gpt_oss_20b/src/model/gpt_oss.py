"""
GPT-OSS Model Implementation
Main model class combining all components
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

from .embeddings.token_embeddings import GPTOSSEmbedding
from .transformer_block import GPTOSSTransformerStack
from .normalization.rmsnorm import RMSNorm
from ..utils.tensor_utils import create_causal_mask


@dataclass
class GPTOSSOutput:
    """Output class for GPT-OSS model"""
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    router_losses: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    loss: Optional[torch.Tensor] = None


class GPTOSSModel(nn.Module):
    """
    GPT-OSS base model without language modeling head
    """
    
    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        num_layers: int = 24,
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
        pad_token_id: int = 200002,
        gradient_checkpointing: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of KV heads for GQA
            intermediate_size: FFN intermediate dimension
            num_experts: Number of MoE experts
            num_experts_per_token: Top-K experts per token
            max_position_embeddings: Maximum sequence length
            rope_theta: RoPE base
            use_attention_sinks: Whether to use attention sinks
            attention_sink_size: Number of sink tokens
            use_sparse_attention: Whether to use sparse attention
            sparse_window_size: Window size for sparse attention
            rms_norm_eps: RMSNorm epsilon
            aux_loss_coef: MoE auxiliary loss coefficient
            router_jitter_noise: Router exploration noise
            pad_token_id: Padding token ID
            gradient_checkpointing: Whether to use gradient checkpointing
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.embed_tokens = GPTOSSEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=pad_token_id,
            device=device,
            dtype=dtype
        )
        
        # Transformer layers
        self.layers = GPTOSSTransformerStack(
            num_layers=num_layers,
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
            gradient_checkpointing=gradient_checkpointing,
            device=device,
            dtype=dtype
        )
        
        # Final layer norm
        self.norm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_losses: bool = True,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through GPT-OSS model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached KV states
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            output_router_losses: Whether to output MoE losses
            use_cache: Whether to cache KV states
            
        Returns:
            Tuple of outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = create_causal_mask(
                seq_len, 
                hidden_states.device,
                hidden_states.dtype
            )
        else:
            # Expand attention mask
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
            
        # Forward through transformer layers
        outputs = self.layers(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_losses=output_router_losses,
            use_cache=use_cache
        )
        
        hidden_states = outputs[0]
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        return (hidden_states,) + outputs[1:]


class GPTOSSForCausalLM(nn.Module):
    """
    GPT-OSS for causal language modeling
    Includes language modeling head for next token prediction
    """
    
    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        num_layers: int = 24,
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
        pad_token_id: int = 200002,
        gradient_checkpointing: bool = False,
        tie_word_embeddings: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            Same as GPTOSSModel plus:
            tie_word_embeddings: Whether to tie input and output embeddings
        """
        super().__init__()
        
        # Base model
        self.model = GPTOSSModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
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
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            device=device,
            dtype=dtype
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(
            hidden_size, vocab_size,
            bias=False, device=device, dtype=dtype
        )
        
        # Tie embeddings if specified
        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.token_embedding.weight
            
        # Initialize lm_head if not tied
        if not tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_losses: bool = True,
        use_cache: bool = False
    ) -> GPTOSSOutput:
        """
        Forward pass for causal language modeling
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached KV states
            labels: Target labels for language modeling
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            output_router_losses: Whether to output MoE losses
            use_cache: Whether to cache KV states
            
        Returns:
            GPTOSSOutput with logits and optional losses
        """
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_losses=output_router_losses,
            use_cache=use_cache
        )
        
        hidden_states = outputs[0]
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add router losses if available
            if output_router_losses and len(outputs) > 4:
                router_losses = outputs[4]
                if router_losses is not None:
                    lb_loss, z_loss = router_losses
                    loss = loss + lb_loss + z_loss
                    
        return GPTOSSOutput(
            logits=logits,
            hidden_states=outputs[2] if output_hidden_states else None,
            attentions=outputs[3] if output_attentions else None,
            past_key_values=outputs[1] if use_cache else None,
            router_losses=outputs[4] if output_router_losses and len(outputs) > 4 else None,
            loss=loss
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Generate text using the model
        
        Args:
            input_ids: Starting input IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            use_cache: Whether to use KV cache
            
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if pad_token_id is None:
            pad_token_id = self.model.pad_token_id
        if eos_token_id is None:
            eos_token_id = 200001  # Default EOS for Harmony tokenizer
            
        # Initialize generation
        generated = input_ids
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated if not use_cache else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_router_losses=False
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply top-k and top-p filtering
            if do_sample:
                filtered_logits = self._top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p
                )
                probs = torch.softmax(filtered_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                
            # Update finished sequences
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            
            # Replace tokens for finished sequences with pad token
            next_tokens[finished] = pad_token_id
            
            # Append to generated
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Stop if all sequences are finished
            if finished.all():
                break
                
        return generated
    
    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:
        """
        Filter logits using top-k and/or top-p (nucleus) filtering
        
        Args:
            logits: Logits distribution
            top_k: Keep only top k tokens
            top_p: Keep top tokens with cumulative probability >= top_p
            filter_value: Value to assign filtered tokens
            
        Returns:
            Filtered logits
        """
        batch_size, vocab_size = logits.shape
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, vocab_size)
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
            
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
            
        return logits