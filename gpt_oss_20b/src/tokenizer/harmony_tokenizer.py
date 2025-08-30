import json
import os
from typing import List, Optional, Dict, Union
import torch

try:
    import tiktoken
except ImportError:
    raise ImportError("Please install tiktoken: pip install tiktoken")


class HarmonyTokenizer:
    """
    Tokenizer wrapper using tiktoken (OpenAI's tokenizer)
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        vocab_size: int = 100277,
        model_max_length: int = 131072,
        padding_side: str = "left",
        use_harmony_format: bool = False,
        encoding_name: str = "cl100k_base"
    ):
        self.model_max_length = model_max_length
        self.padding_side = padding_side
        self.use_harmony_format = use_harmony_format
        self.encoding_name = encoding_name
        
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoder.n_vocab
        
        self.bos_token_id = self.encoder.encode("<|endoftext|>")[0] if "<|endoftext|>" in self.encoder._special_tokens else 0
        self.eos_token_id = self.encoder.encode("<|endoftext|>")[0] if "<|endoftext|>" in self.encoder._special_tokens else 0
        self.pad_token_id = self.vocab_size
        
        self.bos_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        
        self.special_tokens = {
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.pad_token: self.pad_token_id
        }
        
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        if isinstance(text, list):
            text = " ".join(text)
        
        token_ids = self.encoder.encode(text)
        
        if add_special_tokens and self.bos_token_id is not None:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        if truncation and max_length:
            token_ids = token_ids[:max_length]
        
        if padding and max_length:
            if len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                if self.padding_side == "left":
                    token_ids = [self.pad_token_id] * pad_length + token_ids
                else:
                    token_ids = token_ids + [self.pad_token_id] * pad_length
        
        if return_tensors == "pt":
            return torch.tensor(token_ids, dtype=torch.long)
            
        return token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]]
        
        token_ids = [t for t in token_ids if 0 <= t < self.vocab_size]
        
        try:
            text = self.encoder.decode(token_ids)
        except:
            text = ""
            
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        max_length = max_length or self.model_max_length
        
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
        
        batch_max_length = min(max(len(ids) for ids in encoded), max_length) if encoded else 0
        
        input_ids = []
        attention_mask = []
        
        for ids in encoded:
            if len(ids) > batch_max_length:
                ids = ids[:batch_max_length]
            
            mask = [1] * len(ids)
            
            if padding and len(ids) < batch_max_length:
                pad_length = batch_max_length - len(ids)
                if self.padding_side == "left":
                    ids = [self.pad_token_id] * pad_length + ids
                    mask = [0] * pad_length + mask
                else:
                    ids = ids + [self.pad_token_id] * pad_length
                    mask = mask + [0] * pad_length
            
            input_ids.append(ids)
            attention_mask.append(mask)
        
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            "vocab_size": self.vocab_size,
            "model_max_length": self.model_max_length,
            "padding_side": self.padding_side,
            "use_harmony_format": self.use_harmony_format,
            "encoding_name": self.encoding_name
        }
        
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        with open(os.path.join(save_directory, "tokenizer.json"), "w") as f:
            json.dump({
                "tokenizer_class": "HarmonyTokenizer",
                "encoding_name": self.encoding_name,
                "vocab_size": self.vocab_size
            }, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "HarmonyTokenizer":
        config_path = os.path.join(model_path, "tokenizer_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                return cls(**config)
            except:
                return cls()
        else:
            return cls()
