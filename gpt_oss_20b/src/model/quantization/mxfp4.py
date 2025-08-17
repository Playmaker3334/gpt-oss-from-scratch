"""
MXFP4 Quantization for MoE weights in GPT-OSS
Microscaling 4-bit floating point format with block-wise scaling
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class MXFP4Quantizer:
    """
    MXFP4 (Microscaling FP4) quantization
    Used for MoE projection weights to reduce memory usage
    """
    
    def __init__(
        self,
        block_size: int = 32,
        use_stochastic: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            block_size: Size of blocks for scaling (32 default)
            use_stochastic: Whether to use stochastic rounding
            device: Device for operations
        """
        self.block_size = block_size
        self.use_stochastic = use_stochastic
        self.device = device
        
        # FP4 format: 1 sign, 2 exponent, 1 mantissa
        self.n_bits = 4
        self.n_exp = 2
        self.n_mantissa = 1
        
        # Maximum representable values
        self.max_exp = 2 ** self.n_exp - 1
        self.max_mantissa = 2 ** self.n_mantissa - 1
        
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to MXFP4 format
        
        Args:
            tensor: Input tensor to quantize
            
        Returns:
            Tuple of:
                - Quantized tensor (packed uint8)
                - Scale factors per block
        """
        original_shape = tensor.shape
        tensor = tensor.flatten()
        
        # Pad to multiple of block_size
        pad_size = (self.block_size - len(tensor) % self.block_size) % self.block_size
        if pad_size > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_size))
            
        # Reshape into blocks
        n_blocks = len(tensor) // self.block_size
        tensor_blocks = tensor.view(n_blocks, self.block_size)
        
        # Compute scale per block (max abs value)
        scales = tensor_blocks.abs().max(dim=1)[0]
        scales = scales.clamp(min=1e-10)  # Avoid division by zero
        
        # Normalize by scale
        normalized = tensor_blocks / scales.unsqueeze(1)
        
        # Quantize to FP4
        quantized = self._quantize_to_fp4(normalized)
        
        # Pack two FP4 values into one uint8
        quantized_packed = self._pack_fp4(quantized)
        
        return quantized_packed, scales
    
    def dequantize(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
        original_shape: torch.Size
    ) -> torch.Tensor:
        """
        Dequantize from MXFP4 format
        
        Args:
            quantized: Quantized tensor (packed uint8)
            scales: Scale factors per block
            original_shape: Original tensor shape
            
        Returns:
            Dequantized tensor
        """
        # Unpack from uint8
        unpacked = self._unpack_fp4(quantized)
        
        # Dequantize from FP4
        dequantized = self._dequantize_from_fp4(unpacked)
        
        # Apply scales
        n_blocks = scales.shape[0]
        dequantized = dequantized.view(n_blocks, -1)
        dequantized = dequantized * scales.unsqueeze(1)
        
        # Flatten and reshape to original
        dequantized = dequantized.flatten()
        total_elements = torch.prod(torch.tensor(original_shape)).item()
        dequantized = dequantized[:total_elements]
        
        return dequantized.view(original_shape)
    
    def _quantize_to_fp4(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized values to FP4 representation
        
        Args:
            tensor: Normalized input tensor
            
        Returns:
            FP4 quantized values
        """
        # Get sign
        sign = (tensor < 0).float()
        abs_val = tensor.abs()
        
        # Compute exponent and mantissa
        # FP4 can represent: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        quantized = torch.zeros_like(tensor, dtype=torch.uint8)
        
        # Special case for zero
        zero_mask = abs_val < 0.25
        quantized[zero_mask] = 0
        
        # Quantization levels
        levels = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=tensor.device)
        
        for i, level in enumerate(levels):
            if i == 0:
                mask = (abs_val >= 0.25) & (abs_val < 0.75)
            elif i == len(levels) - 1:
                mask = abs_val >= 5.0
            else:
                prev_level = levels[i-1] if i > 0 else 0.25
                next_level = levels[i+1] if i < len(levels)-1 else 8.0
                mid_prev = (prev_level + level) / 2
                mid_next = (level + next_level) / 2
                mask = (abs_val >= mid_prev) & (abs_val < mid_next)
                
            quantized[mask] = i + 1
            
        # Apply sign
        quantized = quantized | (sign.to(torch.uint8) << 3)
        
        return quantized
    
    def _dequantize_from_fp4(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Convert FP4 values back to float
        
        Args:
            quantized: FP4 quantized values
            
        Returns:
            Dequantized float tensor
        """
        # Extract sign
        sign = ((quantized >> 3) & 1).float() * 2 - 1
        sign[quantized >> 3 == 0] = 1.0
        
        # Extract magnitude
        magnitude = quantized & 0b0111
        
        # Dequantization lookup
        lookup = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            device=quantized.device, dtype=torch.float32
        )
        
        dequantized = lookup[magnitude] * sign
        
        return dequantized
    
    def _pack_fp4(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pack two FP4 values into one uint8
        
        Args:
            tensor: FP4 values
            
        Returns:
            Packed uint8 tensor
        """
        # Ensure even number of elements
        if tensor.shape[-1] % 2 != 0:
            tensor = torch.nn.functional.pad(tensor, (0, 1))
            
        # Pack pairs
        tensor_reshaped = tensor.view(-1, 2)
        packed = (tensor_reshaped[:, 0] << 4) | tensor_reshaped[:, 1]
        
        return packed.to(torch.uint8)
    
    def _unpack_fp4(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Unpack uint8 into two FP4 values
        
        Args:
            packed: Packed uint8 tensor
            
        Returns:
            Unpacked FP4 values
        """
        high = (packed >> 4) & 0b1111
        low = packed & 0b1111
        
        unpacked = torch.stack([high, low], dim=-1).flatten()
        
        return unpacked


class MXFP4Linear(nn.Module):
    """
    Linear layer with MXFP4 quantization
    Drop-in replacement for nn.Linear with 4-bit weights
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias
            block_size: Block size for quantization
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # Initialize with full precision then quantize
        weight = torch.randn(
            out_features, in_features,
            device=device, dtype=dtype or torch.float32
        ) * math.sqrt(2.0 / in_features)
        
        # Quantize weights
        quantizer = MXFP4Quantizer(block_size=block_size, device=device)
        quantized_weight, weight_scales = quantizer.quantize(weight)
        
        # Store quantized weights and scales
        self.register_buffer('quantized_weight', quantized_weight)
        self.register_buffer('weight_scales', weight_scales)
        self.weight_shape = weight.shape
        
        # Bias in full precision
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype or torch.float32)
            )
        else:
            self.register_parameter('bias', None)
            
        self.quantizer = quantizer
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantized weights
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor
        """
        # Dequantize weights on the fly
        weight = self.quantizer.dequantize(
            self.quantized_weight,
            self.weight_scales,
            self.weight_shape
        )
        
        # Standard linear operation
        return torch.nn.functional.linear(input, weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quantized=MXFP4'


def quantize_model_mxfp4(
    model: nn.Module,
    layers_to_quantize: Optional[list] = None,
    block_size: int = 32
) -> nn.Module:
    """
    Quantize specific layers of a model to MXFP4
    
    Args:
        model: Model to quantize
        layers_to_quantize: List of layer names to quantize (None = all linear)
        block_size: Block size for quantization
        
    Returns:
        Quantized model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this layer should be quantized
            if layers_to_quantize is None or any(
                pattern in name for pattern in layers_to_quantize
            ):
                # Only quantize MoE expert weights in GPT-OSS
                if 'expert' in name and ('gate_proj' in name or 'up_proj' in name or 'down_proj' in name):
                    # Get parent module and attribute name
                    parent_name = '.'.join(name.split('.')[:-1])
                    attr_name = name.split('.')[-1]
                    parent = model
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                            
                    # Replace with quantized version
                    quantized = MXFP4Linear(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        block_size=block_size,
                        device=module.weight.device,
                        dtype=module.weight.dtype
                    )
                    
                    setattr(parent, attr_name, quantized)
                    
    return model