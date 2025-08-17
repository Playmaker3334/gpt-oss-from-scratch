"""
Tensor utilities for GPT-OSS
"""

import torch
from typing import Optional, Tuple


def create_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create causal attention mask
    
    Args:
        seq_len: Sequence length
        device: Device for tensor
        dtype: Data type for tensor
        
    Returns:
        Causal mask tensor
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype),
        diagonal=1
    )
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)


def create_position_ids(
    seq_len: int,
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create position IDs for sequences
    
    Args:
        seq_len: Sequence length
        batch_size: Batch size
        device: Device for tensor
        
    Returns:
        Position IDs tensor
    """
    position_ids = torch.arange(seq_len, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    return position_ids


def create_attention_mask_from_lengths(
    lengths: torch.Tensor,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create attention mask from sequence lengths
    
    Args:
        lengths: Tensor of sequence lengths
        max_length: Maximum sequence length
        device: Device for tensor
        dtype: Data type for tensor
        
    Returns:
        Attention mask tensor
    """
    batch_size = lengths.shape[0]
    mask = torch.arange(max_length, device=device).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return mask.to(dtype)


def pad_sequences(
    sequences: list,
    max_length: Optional[int] = None,
    padding_value: int = 0,
    padding_side: str = "right"
) -> torch.Tensor:
    """
    Pad sequences to same length
    
    Args:
        sequences: List of sequences
        max_length: Maximum length (None for longest)
        padding_value: Value to use for padding
        padding_side: Side to pad ("left" or "right")
        
    Returns:
        Padded sequences tensor
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
        
    padded = []
    for seq in sequences:
        seq_tensor = torch.tensor(seq) if not isinstance(seq, torch.Tensor) else seq
        
        if len(seq_tensor) >= max_length:
            padded.append(seq_tensor[:max_length])
        else:
            pad_length = max_length - len(seq_tensor)
            if padding_side == "left":
                padding = torch.full((pad_length,), padding_value)
                padded_seq = torch.cat([padding, seq_tensor])
            else:
                padding = torch.full((pad_length,), padding_value)
                padded_seq = torch.cat([seq_tensor, padding])
            padded.append(padded_seq)
            
    return torch.stack(padded)


def chunk_sequences(
    sequences: torch.Tensor,
    chunk_size: int,
    overlap: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunk sequences for processing long contexts
    
    Args:
        sequences: Input sequences
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        Tuple of chunked sequences and chunk positions
    """
    batch_size, seq_len = sequences.shape
    
    if seq_len <= chunk_size:
        return sequences.unsqueeze(1), torch.zeros(batch_size, 1, dtype=torch.long)
        
    chunks = []
    positions = []
    
    stride = chunk_size - overlap
    for i in range(0, seq_len - overlap, stride):
        end = min(i + chunk_size, seq_len)
        chunk = sequences[:, i:end]
        
        # Pad if necessary
        if chunk.shape[1] < chunk_size:
            padding = torch.zeros(
                batch_size, chunk_size - chunk.shape[1],
                dtype=sequences.dtype, device=sequences.device
            )
            chunk = torch.cat([chunk, padding], dim=1)
            
        chunks.append(chunk)
        positions.append(torch.full((batch_size,), i, dtype=torch.long))
        
    chunks = torch.stack(chunks, dim=1)
    positions = torch.stack(positions, dim=1)
    
    return chunks, positions


def apply_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply temperature scaling to logits
    
    Args:
        logits: Input logits
        temperature: Temperature value
        
    Returns:
        Scaled logits
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def top_k_filtering(
    logits: torch.Tensor,
    k: int,
    filter_value: float = -float('inf')
) -> torch.Tensor:
    """
    Apply top-k filtering to logits
    
    Args:
        logits: Input logits
        k: Number of top elements to keep
        filter_value: Value for filtered elements
        
    Returns:
        Filtered logits
    """
    if k <= 0:
        return logits
        
    indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value
    return logits


def top_p_filtering(
    logits: torch.Tensor,
    p: float,
    filter_value: float = -float('inf')
) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to logits
    
    Args:
        logits: Input logits
        p: Cumulative probability threshold
        filter_value: Value for filtered elements
        
    Returns:
        Filtered logits
    """
    if p >= 1.0:
        return logits
        
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = filter_value
    
    return logits


def compute_perplexity(
    loss: torch.Tensor
) -> torch.Tensor:
    """
    Compute perplexity from loss
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return torch.exp(loss)


def get_device() -> torch.device:
    """
    Get best available device
    
    Returns:
        Device (cuda if available, else cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "total_billions": total_params / 1e9,
        "trainable_billions": trainable_params / 1e9
    }