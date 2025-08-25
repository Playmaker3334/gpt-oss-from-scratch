from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTOSSConfig:
    model_variant: str = "kaggle-mini"
    vocab_size: int = 201088
    hidden_size: int = 512
    num_layers: int = 4
    
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    yarn_scale: float = 1.0
    yarn_original_max_position: int = 512
    
    num_experts: int = 4
    num_experts_per_token: int = 2
    intermediate_size: int = 1024
    aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    
    use_attention_sinks: bool = False
    attention_sink_size: int = 4
    
    use_sparse_attention: bool = False
    sparse_window_size: int = 128
    sparse_attention_interval: int = 2
    
    use_dropout: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    layer_norm_epsilon: float = 1e-5
    rms_norm_eps: float = 1e-5
    use_rms_norm: bool = True
    
    initializer_range: float = 0.01
    use_scaled_init: bool = True
    
    use_mxfp4_quantization: bool = False
    mxfp4_block_size: int = 32
    
    bos_token_id: int = 200000
    eos_token_id: int = 200001
    pad_token_id: int = 200002
    
    hidden_act: str = "silu"
    
    gradient_checkpointing: bool = True
    use_cache: bool = False
    
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
        if hasattr(self, 'rms_norm_eps') and hasattr(self, 'layer_norm_epsilon'):
            if self.use_rms_norm:
                self.layer_norm_epsilon = self.rms_norm_eps
            
    @property
    def total_params(self) -> int:
        params = self.vocab_size * self.hidden_size
        
        per_layer = 0
        
        per_layer += self.hidden_size * self.hidden_size
        per_layer += self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads) * 2
        per_layer += self.hidden_size * self.hidden_size
        
        expert_params = 3 * self.hidden_size * self.intermediate_size
        per_layer += expert_params * self.num_experts
        
        per_layer += self.hidden_size * self.num_experts
        
        per_layer += 2 * self.hidden_size
        
        params += per_layer * self.num_layers
        
        params += self.hidden_size
        params += self.hidden_size * self.vocab_size
        
        return params
    
    @property
    def active_params_per_token(self) -> int:
        params = self.vocab_size * self.hidden_size
        
        per_layer = 0
        
        per_layer += self.hidden_size * self.hidden_size
        per_layer += self.hidden_size * (self.hidden_size // self.num_attention_heads * self.num_key_value_heads) * 2
        per_layer += self.hidden_size * self.hidden_size
        
        expert_params = 3 * self.hidden_size * self.intermediate_size
        per_layer += expert_params * self.num_experts_per_token
        
        per_layer += self.hidden_size * self.num_experts
        
        per_layer += 2 * self.hidden_size
        
        params += per_layer * self.num_layers
        
        params += self.hidden_size
        params += self.hidden_size * self.vocab_size
        
        return params