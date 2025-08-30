import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

from .embeddings.token_embeddings import GPTOSSEmbedding
from .transformer_block import GPTOSSTransformerStack
from .normalization.rmsnorm import RMSNorm
from ..utils.tensor_utils import create_causal_mask


@dataclass
class GPTOSSOutput:
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    router_losses: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    loss: Optional[torch.Tensor] = None


class GPTOSSModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        intermediate_size: int = 1024,
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        use_attention_sinks: bool = False,
        attention_sink_size: int = 4,
        use_sparse_attention: bool = False,
        sparse_window_size: int = 128,
        rms_norm_eps: float = 1e-5,
        aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        pad_token_id: int = 200002,
        gradient_checkpointing: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.aux_loss_coef = aux_loss_coef

        self.embed_tokens = GPTOSSEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=pad_token_id,
            device=device,
            dtype=dtype
        )

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

        self.norm = RMSNorm(
            hidden_size,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.num_layers)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            std = 0.02 / math.sqrt(2 * self.num_layers)
            module.weight.data.normal_(mean=0.0, std=std)
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
        batch_size, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        causal = create_causal_mask(
            seq_len,
            hidden_states.device,
            hidden_states.dtype
        )
        if attention_mask is None:
            attn_bias = causal
        else:
            pad = (1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
            attn_bias = causal + pad

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        outputs = self.layers(
            hidden_states=hidden_states,
            attention_mask=attn_bias,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_losses=output_router_losses,
            use_cache=use_cache
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        return (hidden_states,) + outputs[1:]


class GPTOSSForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        intermediate_size: int = 1024,
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        use_attention_sinks: bool = False,
        attention_sink_size: int = 4,
        use_sparse_attention: bool = False,
        sparse_window_size: int = 128,
        rms_norm_eps: float = 1e-5,
        aux_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        pad_token_id: int = 200002,
        gradient_checkpointing: bool = True,
        tie_word_embeddings: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.aux_loss_coef = aux_loss_coef

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

        self.lm_head = nn.Linear(
            hidden_size, vocab_size,
            bias=False, device=device, dtype=dtype
        )

        if tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.token_embedding.weight

        if not tie_word_embeddings:
            std = 0.02 / math.sqrt(2 * num_layers)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

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
        use_cache: bool = False,
        return_dict: bool = True,
    ):
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
        logits = self.lm_head(hidden_states)

        loss = None
        router_losses = outputs[4] if (output_router_losses and len(outputs) > 4) else None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            pad_id = self.model.pad_token_id
            shift_labels = shift_labels.masked_fill(shift_labels == pad_id, -100)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            if router_losses is not None:
                lb_loss, z_loss = router_losses
                if torch.is_tensor(lb_loss):
                    lb_loss = lb_loss.float()
                if torch.is_tensor(z_loss):
                    z_loss = z_loss.float()
                loss = loss + self.aux_loss_coef * lb_loss + 0.01 * z_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return GPTOSSOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[2] if output_hidden_states else None,
            attentions=outputs[3] if output_attentions else None,
            past_key_values=outputs[1] if use_cache else None,
            router_losses=router_losses
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
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if pad_token_id is None:
            pad_token_id = self.model.pad_token_id
        if eos_token_id is None:
            eos_token_id = 200001

        generated = input_ids
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.forward(
                input_ids=generated if not use_cache else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_router_losses=False
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if temperature != 1.0:
                logits = logits / temperature

            if do_sample:
                filtered_logits = self._top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p
                )
                probs = torch.softmax(filtered_logits.float(), dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            next_tokens[finished] = pad_token_id

            generated = torch.cat([generated, next_tokens], dim=-1)

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
        batch_size, vocab_size = logits.shape

        if top_k > 0:
            top_k = min(top_k, vocab_size)
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value

        return logits