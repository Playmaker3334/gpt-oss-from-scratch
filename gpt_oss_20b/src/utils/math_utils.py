"""
Mathematical utilities for GPT-OSS
"""

import torch
import math
from typing import Optional, Tuple


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation function with tanh approximation
    Used in some transformer implementations
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
    ))


def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Swish activation function
    
    Args:
        x: Input tensor
        beta: Beta parameter
        
    Returns:
        Activated tensor
    """
    return x * torch.sigmoid(beta * x)


def cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute cosine similarity between tensors
    
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimension along which to compute similarity
        eps: Small value for numerical stability
        
    Returns:
        Cosine similarity
    """
    x1_norm = torch.norm(x1, p=2, dim=dim, keepdim=True)
    x2_norm = torch.norm(x2, p=2, dim=dim, keepdim=True)
    
    x1_normalized = x1 / (x1_norm + eps)
    x2_normalized = x2 / (x2_norm + eps)
    
    return torch.sum(x1_normalized * x2_normalized, dim=dim)


def compute_entropy(
    probs: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute entropy of probability distribution
    
    Args:
        probs: Probability distribution
        dim: Dimension along which to compute
        eps: Small value for numerical stability
        
    Returns:
        Entropy
    """
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    return entropy


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute KL divergence between distributions
    
    Args:
        p: First distribution
        q: Second distribution
        dim: Dimension along which to compute
        eps: Small value for numerical stability
        
    Returns:
        KL divergence
    """
    kl = torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)), dim=dim)
    return kl


def warmup_cosine_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
    max_lr: float = 1.0
) -> float:
    """
    Cosine learning rate schedule with warmup
    
    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        
    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    elif step < total_steps:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        return min_lr


def linear_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
    max_lr: float = 1.0
) -> float:
    """
    Linear learning rate schedule with warmup
    
    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        
    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        # Linear warmup
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    else:
        # Linear decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max_lr - (max_lr - min_lr) * progress


def compute_grad_norm(
    model: torch.nn.Module,
    norm_type: float = 2.0
) -> float:
    """
    Compute gradient norm for model
    
    Args:
        model: PyTorch model
        norm_type: Type of norm
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def clip_grad_norm(
    model: torch.nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, norm_type
    )


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Focal loss for addressing class imbalance
    
    Args:
        inputs: Predicted logits
        targets: Ground truth labels
        alpha: Weighting factor
        gamma: Focusing parameter
        reduction: Reduction method
        
    Returns:
        Focal loss
    """
    ce_loss = torch.nn.functional.cross_entropy(
        inputs, targets, reduction="none"
    )
    
    p = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - p) ** gamma * ce_loss
    
    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    else:
        return focal_loss


def compute_bleu_score(
    predictions: torch.Tensor,
    references: torch.Tensor,
    max_n: int = 4,
    smoothing: bool = True
) -> float:
    """
    Compute BLEU score for sequence generation
    Simplified version for demonstration
    
    Args:
        predictions: Predicted sequences
        references: Reference sequences
        max_n: Maximum n-gram order
        smoothing: Whether to use smoothing
        
    Returns:
        BLEU score
    """
    # This is a simplified implementation
    # In practice, use sacrebleu or similar library
    
    def get_ngrams(seq, n):
        ngrams = []
        for i in range(len(seq) - n + 1):
            ngrams.append(tuple(seq[i:i+n].tolist()))
        return ngrams
    
    scores = []
    for n in range(1, min(max_n + 1, predictions.shape[1])):
        pred_ngrams = get_ngrams(predictions[0], n)
        ref_ngrams = get_ngrams(references[0], n)
        
        if not pred_ngrams or not ref_ngrams:
            continue
            
        matches = sum(1 for ng in pred_ngrams if ng in ref_ngrams)
        precision = matches / len(pred_ngrams) if pred_ngrams else 0
        
        if smoothing and precision == 0:
            precision = 1 / (2 ** n)
            
        scores.append(precision)
        
    if not scores:
        return 0.0
        
    # Geometric mean
    score = math.exp(sum(math.log(s) for s in scores) / len(scores))
    
    # Brevity penalty
    pred_len = predictions.shape[1]
    ref_len = references.shape[1]
    if pred_len < ref_len:
        brevity_penalty = math.exp(1 - ref_len / pred_len)
    else:
        brevity_penalty = 1.0
        
    return score * brevity_penalty


def exponential_moving_average(
    current: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.999
) -> torch.Tensor:
    """
    Update target with exponential moving average
    
    Args:
        current: Current values
        target: Target values to update
        beta: EMA coefficient
        
    Returns:
        Updated target
    """
    return beta * target + (1 - beta) * current