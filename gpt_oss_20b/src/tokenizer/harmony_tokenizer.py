"""
Harmony Tokenizer for GPT-OSS
o200k_harmony tokenizer with 201,088 vocabulary size
"""

import json
import os
from typing import List, Optional, Dict, Union
import regex as re
import torch


class HarmonyTokenizer:
    """
    Harmony tokenizer implementation for GPT-OSS
    Uses BPE with special tokens for structured format
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        vocab_size: int = 201088,
        model_max_length: int = 131072,
        padding_side: str = "left",
        use_harmony_format: bool = True
    ):
        """
        Args:
            vocab_file: Path to vocabulary file
            merges_file: Path to merges file
            vocab_size: Vocabulary size
            model_max_length: Maximum sequence length
            padding_side: Side to pad sequences
            use_harmony_format: Whether to use Harmony formatting
        """
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.padding_side = padding_side
        self.use_harmony_format = use_harmony_format
        
        # Special tokens for Harmony format
        self.special_tokens = {
            "<|start|>": 200000,
            "<|end|>": 200001,
            "<|pad|>": 200002,
            "<|message|>": 200003,
            "<|system|>": 200004,
            "<|developer|>": 200005,
            "<|user|>": 200006,
            "<|assistant|>": 200007,
            "<|tool|>": 200008,
            "<|channel|>": 200009,
            "<|thinking|>": 200010,
            "<|output|>": 200011,
        }
        
        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Initialize vocabulary (simplified for demonstration)
        self.vocab = self._initialize_vocab()
        self.merges = self._initialize_merges()
        
        # BPE regex pattern
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        # Cache for BPE
        self.cache = {}
        
    def _initialize_vocab(self) -> Dict[str, int]:
        """Initialize base vocabulary"""
        vocab = dict(self.special_tokens)
        
        # Add byte tokens (256 tokens)
        for i in range(256):
            vocab[chr(i)] = len(vocab)
            
        # Add common tokens (simplified)
        # In practice, this would load from vocab_file
        return vocab
    
    def _initialize_merges(self) -> List[tuple]:
        """Initialize BPE merges"""
        # In practice, this would load from merges_file
        # Simplified version for demonstration
        return []
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate
            padding: Whether to pad
            return_tensors: Return format ('pt' for PyTorch)
            
        Returns:
            Token IDs
        """
        if self.use_harmony_format and add_special_tokens:
            text = self._apply_harmony_format(text)
            
        # Tokenize with BPE
        tokens = self._tokenize(text)
        
        # Convert to IDs
        token_ids = self._convert_tokens_to_ids(tokens)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.special_tokens["<|start|>"]] + token_ids + [self.special_tokens["<|end|>"]]
            
        # Truncation
        if truncation and max_length:
            token_ids = token_ids[:max_length]
            
        # Padding
        if padding and max_length:
            pad_token_id = self.special_tokens["<|pad|>"]
            if self.padding_side == "left":
                token_ids = [pad_token_id] * (max_length - len(token_ids)) + token_ids
            else:
                token_ids = token_ids + [pad_token_id] * (max_length - len(token_ids))
                
        # Convert to tensor if requested
        if return_tensors == "pt":
            return torch.tensor(token_ids, dtype=torch.long)
            
        return token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Convert IDs to tokens
        tokens = self._convert_ids_to_tokens(token_ids)
        
        # Filter special tokens if requested
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
            
        # Join tokens
        text = "".join(tokens)
        
        # Clean up
        if clean_up_tokenization_spaces:
            text = self._clean_up_tokenization(text)
            
        return text
    
    def _apply_harmony_format(self, text: str) -> str:
        """
        Apply Harmony formatting to text
        
        Args:
            text: Input text
            
        Returns:
            Harmony formatted text
        """
        # Simple formatting - in practice this would be more sophisticated
        if not text.startswith("<|"):
            # Assume user message if no role specified
            text = f"<|message|><|user|>{text}<|assistant|>"
            
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using BPE
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        tokens = []
        
        # Split by regex pattern
        for match in self.pat.finditer(text):
            word = match.group()
            
            # Check for special tokens
            if word in self.special_tokens:
                tokens.append(word)
            else:
                # Apply BPE
                word_tokens = self._bpe(word)
                tokens.extend(word_tokens)
                
        return tokens
    
    def _bpe(self, word: str) -> List[str]:
        """
        Apply BPE to a word
        
        Args:
            word: Input word
            
        Returns:
            BPE tokens
        """
        if word in self.cache:
            return self.cache[word]
            
        # Convert to bytes
        word_bytes = word.encode('utf-8')
        tokens = [chr(b) for b in word_bytes]
        
        # Apply merges (simplified)
        # In practice, this would apply learned BPE merges
        
        self.cache[word] = tokens
        return tokens
    
    def _convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Unknown token - use first byte
                ids.append(ord(token[0]) if token else 0)
        return ids
    
    def _convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            elif id < 256:
                tokens.append(chr(id))
            else:
                tokens.append(f"<unk{id}>")
        return tokens
    
    def _clean_up_tokenization(self, text: str) -> str:
        """Clean up tokenization artifacts"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r' ([,.!?;:])', r'\1', text)
        return text.strip()
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts
        
        Args:
            texts: List of texts
            max_length: Maximum sequence length
            padding: Whether to pad
            truncation: Whether to truncate
            return_tensors: Return format
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        max_length = max_length or self.model_max_length
        
        # Encode all texts
        encoded = []
        for text in texts:
            ids = self.encode(
                text,
                max_length=max_length,
                padding=False,
                truncation=truncation,
                return_tensors=None
            )
            encoded.append(ids)
            
        # Find max length in batch
        batch_max_length = max(len(ids) for ids in encoded)
        batch_max_length = min(batch_max_length, max_length)
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        
        pad_token_id = self.special_tokens["<|pad|>"]
        
        for ids in encoded:
            # Truncate if needed
            if len(ids) > batch_max_length:
                ids = ids[:batch_max_length]
                
            # Create attention mask
            mask = [1] * len(ids)
            
            # Pad
            if padding and len(ids) < batch_max_length:
                pad_length = batch_max_length - len(ids)
                if self.padding_side == "left":
                    ids = [pad_token_id] * pad_length + ids
                    mask = [0] * pad_length + mask
                else:
                    ids = ids + [pad_token_id] * pad_length
                    mask = mask + [0] * pad_length
                    
            input_ids.append(ids)
            attention_mask.append(mask)
            
        # Convert to tensors
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration"""
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            "vocab_size": self.vocab_size,
            "model_max_length": self.model_max_length,
            "padding_side": self.padding_side,
            "use_harmony_format": self.use_harmony_format,
            "special_tokens": self.special_tokens
        }
        
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, model_path: str) -> "HarmonyTokenizer":
        """Load tokenizer from saved configuration"""
        config_path = os.path.join(model_path, "tokenizer_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            return cls(**config)
        else:
            # Return default tokenizer
            return cls()