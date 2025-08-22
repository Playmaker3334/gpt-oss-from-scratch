# ARCHIVO 1: config/model_config.py
# UBICACION: gpt_oss_20b/config/model_config.py
# ESTE ARCHIVO DEFINE LA CONFIGURACION DEL MODELO REDUCIDO PARA KAGGLE

"""
GPT-OSS Model Configuration - Kaggle Version
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTOSSConfig:
    """Configuration for GPT-OSS models - KAGGLE VERSION"""
    
    # Model size parameters
    model_variant: str = "kaggle"
    vocab_size: int = 201088  # Harmony tokenizer size
    hidden_size: int = 768
    num_layers: int = 6
    
    # Attention configuration
    num_attention_heads: int = 12
    num_key_value_heads: int = 3
    head_dim: int = 64
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    yarn_scale: float = 1.0
    yarn_original_max_position: int = 2048
    
    # MoE configuration
    num_experts: int = 8
    num_experts_per_token: int = 2
    intermediate_size: int = 2048
    router_aux_loss_coef: float = 0.01
    router_jitter_noise: float = 0.0
    
    # Attention sinks
    use_attention_sinks: bool = False
    attention_sink_size: int = 4
    
    # Sparse attention pattern
    use_sparse_attention: bool = False
    sparse_attention_window: int = 128
    sparse_attention_interval: int = 2
    
    # Training configuration
    use_dropout: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Normalization
    layer_norm_epsilon: float = 1e-5
    use_rms_norm: bool = True
    
    # Initialization
    initializer_range: float = 0.02
    use_scaled_init: bool = True
    
    # Quantization
    use_mxfp4_quantization: bool = False
    mxfp4_block_size: int = 32
    
    # Tokenization
    bos_token_id: int = 200000
    eos_token_id: int = 200001
    pad_token_id: int = 200002
    
    # Activation
    hidden_act: str = "swiglu"
    
    # Optimization
    gradient_checkpointing: bool = True
    use_cache: bool = True
    
    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
            
    @property
    def total_params(self) -> int:
        """Calculate total parameters"""
        # Embeddings
        params = self.vocab_size * self.hidden_size
        
        # Transformer layers
        per_layer = 0
        
        # Attention (with GQA)
        per_layer += self.hidden_size * self.hidden_size  # Q projection
        per_layer += self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads) * 2  # K, V
        per_layer += self.hidden_size * self.hidden_size  # Output projection
        
        # MoE FFN
        expert_params = 3 * self.hidden_size * self.intermediate_size
        per_layer += expert_params * self.num_experts
        
        # Router
        per_layer += self.hidden_size * self.num_experts
        
        # Layer norms
        per_layer += 2 * self.hidden_size
        
        params += per_layer * self.num_layers
        
        # Final layer norm and output
        params += self.hidden_size
        params += self.hidden_size * self.vocab_size
        
        return params
    
    @property
    def active_params_per_token(self) -> int:
        """Calculate active parameters per token"""
        # Embeddings
        params = self.vocab_size * self.hidden_size
        
        # Per layer active params
        per_layer = 0
        
        # Attention (always active)
        per_layer += self.hidden_size * self.hidden_size  # Q
        per_layer += self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads) * 2  # K, V
        per_layer += self.hidden_size * self.hidden_size  # Output
        
        # Only top-k experts active
        expert_params = 3 * self.hidden_size * self.intermediate_size
        per_layer += expert_params * self.num_experts_per_token
        
        # Router (always active)
        per_layer += self.hidden_size * self.num_experts
        
        # Layer norms (always active)
        per_layer += 2 * self.hidden_size
        
        params += per_layer * self.num_layers
        
        # Final layer norm and output
        params += self.hidden_size
        params += self.hidden_size * self.vocab_size
        
        return params