# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from aria.model.projector import AriaProjector
from aria.model.vision_encoder import AriaVisionConfig, AriaVisionModel


def find_multiple(n: int, *args: Tuple[int]) -> int:
    k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
    if n % k == 0:
        return n
    return n + k - (n % k)


# TODO remove suplerfluous arg
def prepare_inputs_for_model(inps, max_new_tokens=1):
    # this is because input from lm-eval is 2d
    if inps.dim() > 2:
        raise ValueError(f"Expected input to be of dim 1 or 2, but got {inps.dim()}")

    input_pos = torch.arange(0, inps.numel(), device=inps.device)
    return (inps.view(1, -1), input_pos)


@dataclass
class ModelArgs:
    block_size: int = 16384
    vocab_size: int = 100352
    n_layer: int = 28
    n_head: int = 20
    dim: int = 2560
    intermediate_size: int = 1664
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 5000000
    norm_eps: float = 1e-5
    use_scaled_rope: bool = False
    num_experts: int = 64
    router_topk: int = 6
    num_shared_experts: int = 2
    image_token_index: int = 9

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        self.head_dim = self.dim // self.n_head


# this is a model specific variable that controls whether index_put is used for the kv_cache update,
# it is needed for GPTQ but otherwise attenuates perf so the default is to not use it
use_index_put_for_kv_cache = False


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        if use_index_put_for_kv_cache:
            k_out = torch.ops.aten.index_put_(
                self.k_cache, [None, None, input_pos], k_val
            )
            v_out = torch.ops.aten.index_put_(
                self.v_cache, [None, None, input_pos], v_val
            )
        else:
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(
        self,
        max_batch_size,
        max_seq_length,
        training: bool = False,
        linear_causal_mask=False,
        prompt_length=None,
    ):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype

        self.linear_causal_mask = linear_causal_mask
        if not self.linear_causal_mask:
            self.causal_mask = torch.tril(
                torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
            )
        else:
            assert (
                prompt_length is not None and prompt_length > 1
            ), "need to set prompt_length>1 to use non quadratic causal mask in setup_caches"
            self.causal_mask = torch.zeros(
                1, 1, 1, self.max_seq_length, dtype=torch.bool
            )
            self.causal_mask[:, :, :, :prompt_length] = 1

        if not training:
            for b in self.layers:
                b.attention.kv_cache = KVCache(
                    max_batch_size,
                    max_seq_length,
                    self.config.n_local_heads,
                    head_dim,
                    dtype,
                )
        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            dtype,
            use_scaled=self.config.use_scaled_rope,
        )

    def reset_caches(self):
        """Reset caches.

        The caches used by training stage and inference stage may be different, reset them before switching.
        """
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None

    def forward(
        self,
        idx: Tensor,
        input_pos: Optional[Tensor] = None,
        input_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            idx  (`torch.LongTensor` of shape `(batch_size, seq_length)`):
                Indices of input sequence tokens in the vocabulary.
            input_pos (`torch.LongTensor` of shape `(batch_size, seq_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
                This argument is optional for training mode but required for
                inference mode(when model.setup_caches(training=False) is used).

        Returns:
            Tensor: The output logits tensor.
        """
        assert self.freqs_cis is not None, "Caches must be initialized first"

        if input_pos is None:
            mask = None
            freqs_cis = self.freqs_cis[: idx.shape[1]]
        else:
            if not self.linear_causal_mask:
                mask = self.causal_mask[None, None, input_pos]
            elif (
                len(input_pos) > 1 and self.linear_causal_mask
            ):  # prefill for linear causal mask
                mask = (
                    torch.tril(
                        torch.ones(
                            len(input_pos),
                            self.max_seq_length,
                            dtype=torch.bool,
                            device=input_pos.device,
                        )
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            else:  # decode_one_token for linear causal mask
                self.causal_mask[0, 0, 0, input_pos] = 1
                mask = self.causal_mask
            freqs_cis = self.freqs_cis[input_pos]

        if input_embeds is None:
            x = self.tok_embeddings(idx)
        else:
            x = input_embeds

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits


class TokenDispatcher:
    def __init__(self, config: ModelArgs):
        self.config = config
        self.hidden_states_shape = None
        self.reversed_input_permutation_mapping = None

    def token_permutation(
        self, hidden_states: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        self.hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        flatten_indices = indices.flatten()
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        permuted_tokens = hidden_states.index_select(
            0, sorted_indices // self.config.router_topk
        )
        self.reversed_input_permutation_mapping = sorted_indices
        return permuted_tokens

    def token_unpermutation(
        self, permuted_tokens: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        num_unpermuted_tokens = scores.numel()
        unpermuted_tokens = torch.zeros(
            (num_unpermuted_tokens, permuted_tokens.size(1)),
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        unpermuted_tokens.index_copy_(
            0, self.reversed_input_permutation_mapping, permuted_tokens
        )
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, self.config.router_topk, permuted_tokens.size(1)
        )

        unpermuted_tokens = unpermuted_tokens * scores.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1).type_as(permuted_tokens)
        output = unpermuted_tokens.view(self.hidden_states_shape)
        return output


def sequential_gemm(input, weight, tokens_per_expert):
    num_tokens = input.shape[0]
    out_features = weight.shape[1]
    output = torch.zeros(
        num_tokens, out_features, dtype=input.dtype, device=input.device
    )

    cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
    # Insert zero at the begining for offset index's convenience
    zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
    cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))

    for expert_num in range(weight.shape[0]):
        start = cumsum_num_tokens[expert_num]
        end = cumsum_num_tokens[expert_num + 1]
        tokens = input[start:end]

        out = torch.matmul(tokens, weight[expert_num].T)
        output[start:end] = out
    return output


class ConditionalFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.intermediate_size)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.intermediate_size, config.dim)
        )
        self.token_dispatcher = TokenDispatcher(config)

    def forward(
        self, x: Tensor, expert_indices: Tensor, expert_weights: Tensor
    ) -> Tensor:
        if x.size(0) < 50:
            w1_weights = self.w1[expert_indices]  # [T, A, D, D]
            w3_weights = self.w3[expert_indices]  # [T, A, D, D]
            w2_weights = self.w2[expert_indices]  # [T, A, D, D]
            x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
            x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
            expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
            expert_outs = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        else:
            permuted_tokens = self.token_dispatcher.token_permutation(x, expert_indices)
            tokens_per_expert = torch.histc(
                expert_indices.flatten(),
                bins=self.config.num_experts,
                min=0,
                max=self.config.num_experts - 1,
            )
            x1 = sequential_gemm(permuted_tokens, self.w1, tokens_per_expert)
            x3 = sequential_gemm(permuted_tokens, self.w3, tokens_per_expert)
            up = F.silu(x1) * x3
            down = sequential_gemm(up, self.w2, tokens_per_expert)
            expert_outs = self.token_dispatcher.token_unpermutation(
                down, expert_weights
            )
        return expert_outs


class MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(config)
        self.shared_ffn = FeedForward(
            config.dim, config.intermediate_size * config.num_shared_experts
        )
        self.dim = config.dim
        self.num_activated_experts = config.router_topk

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights, expert_indices = torch.topk(
            scores, self.num_activated_experts, dim=-1
        )  # [T, A], [T, A]
        expert_weights = F.softmax(expert_weights, dim=-1)
        expert_outs = self.cond_ffn(x, expert_indices, expert_weights)
        shared_outs = self.shared_ffn(x)
        return expert_outs + shared_outs


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MOEFeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: Tensor,
        input_pos: Optional[Tensor],
        freqs_cis: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor],
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        if mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    use_scaled: bool = False,
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class Aria(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        vision_config = AriaVisionConfig(
            _flash_attn_2_enabled=True,
            _attn_implementation="flash_attention_2",
            architectures=["AriaVisionModel"],
            hidden_size=1152,
            image_size=980,
            intermediate_size=4304,
            model_type="aria_vision_model",
            num_attention_heads=16,
            num_hidden_layers=27,
            patch_size=14,
            torch_dtype="bfloat16",
        )
        self.vision_tower = AriaVisionModel(vision_config)

        self.multi_modal_projector = AriaProjector(
            patch_to_query_dict={1225: 128, 4900: 256},
            embed_dim=vision_config.hidden_size,
            num_heads=vision_config.num_attention_heads,
            kv_dim=vision_config.hidden_size,
            ff_dim=config.dim,
            output_dim=config.dim,
        )

        self.llm = Transformer(config)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape

        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_image_tokens.max() * (num_image_patches - 1)
        ) + sequence_length
        batch_indices, non_image_indices = torch.where(
            input_ids != self.config.image_token_index
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1)
            - 1
        )
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_image_indices
        ]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = (
            image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )

        return final_embedding

    def prepare_embeddings(self, idx: Tensor, pixel_values: Tensor, pixel_mask: Tensor):
        image_outputs, image_attn_mask = self.vision_tower(pixel_values, pixel_mask)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(
            selected_image_feature, image_attn_mask
        )

        inputs_embeds = self.llm.tok_embeddings(idx)

        n_image_tokens = (idx == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (idx == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            special_image_mask, image_features
        )
        return inputs_embeds

    def forward(
        self,
        idx: Tensor,
        input_pos: Optional[Tensor] = None,
        input_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        return self.llm.forward(idx, input_pos, input_embeds)

    def setup_caches(
        self,
        max_batch_size,
        max_seq_length,
        training: bool = False,
        linear_causal_mask=False,
        prompt_length=None,
    ):
        self.llm.setup_caches(
            max_batch_size, max_seq_length, training, linear_causal_mask, prompt_length
        )
