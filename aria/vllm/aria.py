# Copyright 2024 Rhymes AI. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import math
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import LlamaConfig
from transformers.utils import logging
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, MultiModalConfig
from vllm.distributed import (
    divide,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.inputs import INPUT_REGISTRY, LLMInputs
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput, SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    RMSNorm,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_layers,
    merge_multimodal_embeddings,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.image import cached_get_image_processor
from vllm.multimodal.utils import (
    cached_get_tokenizer,
    repeat_and_pad_placeholder_tokens,
)
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of

from aria.model.configuration_aria import AriaConfig
from aria.model.projector import AriaProjector
from aria.model.vision_encoder import AriaVisionModel

logger = logging.get_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}


class AriaMoELMConfig(LlamaConfig):
    """
    Configuration class for AriaMoE language model.

    This class extends the LlamaConfig to include additional parameters specific to the Mixture of Experts (MoE) architecture.
    """

    model_type = "aria_moe_lm"

    def __init__(
        self,
        moe_intermediate_size: int = 4096,
        moe_num_experts: int = 8,
        moe_topk: int = 2,
        moe_z_loss_coeff: float = 1e-5,
        moe_aux_loss_coeff: float = 1e-3,
        moe_num_shared_experts: int = 2,
        **kwargs,
    ):
        """
        Initialize the AriaMoELMConfig.

        Args:
            moe_intermediate_size (int): The intermediate size for MoE layers. Default is 4096.
            moe_num_experts (int): The number of experts in the MoE layer. Default is 8.
            moe_topk (int): The number of top experts to route to for each token. Default is 2.
            moe_z_loss_coeff (float): The coefficient for the auxiliary z-loss. Default is 1e-5.
            moe_aux_loss_coeff (float): The coefficient for the auxiliary load balancing loss. Default is 1e-3.
            moe_num_shared_experts (int): The number of shared experts. Default is 2.
            **kwargs: Additional keyword arguments to be passed to the parent LlamaConfig.
        """
        super().__init__(**kwargs)
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.moe_num_shared_experts = moe_num_shared_experts


# copied from https://github.com/NVIDIA/Megatron-LM/blob/54f1f78529cbc2b9cddad313e7f9d96ac0420a27/megatron/core/transformer/moe/moe_utils.py#L101-L142
class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
):
    """Calculate the auxiliary loss for better load balacing.
    Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        probs (torch.Tensor): The softmax probs output by the router for each token. [num_tokens, num_experts]
        tokens_per_expert (torch.Tensor): The number of assigned tokens for each expert. [num_experts]

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_tokens = probs.shape[0] * topk
    num_experts = probs.shape[1]

    probs_mean_per_expert = probs.mean(dim=0)
    aux_loss = torch.sum(probs_mean_per_expert * tokens_per_expert) * (
        num_experts / num_tokens * moe_aux_loss_coeff
    )
    return aux_loss


# adapted from https://github.com/NVIDIA/Megatron-LM/blob/54f1f78529cbc2b9cddad313e7f9d96ac0420a27/megatron/core/transformer/moe/router.py#L96-L304
class TopKRouter(nn.Module):
    """
    Top-K Router for Mixture of Experts (MoE) models.

    This router determines which experts should process each token based on the top-k scoring experts.
    It also applies auxiliary losses to encourage load balancing among experts.

    Args:
        config (AriaMoELMConfig): Configuration object containing MoE-related parameters.
    """

    def __init__(self, config: AriaMoELMConfig):
        super().__init__()
        self.config = config

        self.weight = nn.Parameter(
            torch.empty((self.config.moe_num_experts, self.config.hidden_size))
        )
        # FIXME: initialize the weight

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the gating logits for each token-expert pair.

        Args:
            input (torch.Tensor): Input tensor of shape [batch_size * seq_len, hidden_size].

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size * seq_len, num_experts].
        """
        logits = torch.nn.functional.linear(input, self.weight)
        return logits

    def apply_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply z-loss to encourage router logits to remain small for enhanced stability.

        Args:
            logits (torch.Tensor): Router logits.

        Returns:
            torch.Tensor: Logits with z-loss applied.
        """
        z_loss = z_loss_func(logits, self.config.moe_z_loss_coeff)
        logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits

    def apply_aux_loss(
        self,
        logits: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply auxiliary loss for load balancing among experts.

        Args:
            logits (torch.Tensor): Router logits.
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.
            activation (torch.Tensor): Activation values.

        Returns:
            torch.Tensor: Activation with auxiliary loss applied.
        """
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        aux_loss = switch_load_balancing_loss_func(
            probs,
            tokens_per_expert,
            self.config.moe_topk,
            self.config.moe_aux_loss_coeff,
        )
        return MoEAuxLossAutoScaler.apply(activation, aux_loss)

    def routing(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform the routing operation to determine expert assignments.

        Args:
            logits (torch.Tensor): Router logits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - scores: Softmax probabilities for top-k experts.
                - top_indices: Indices of top-k experts for each token.
                - tokens_per_expert: Number of tokens assigned to each expert.
        """
        logits = self.apply_z_loss(logits)

        top_logits, top_indices = torch.topk(logits, k=self.config.moe_topk, dim=1)
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)

        tokens_per_expert = torch.histc(
            top_indices.flatten(),
            bins=self.config.moe_num_experts,
            min=0,
            max=self.config.moe_num_experts - 1,
        )

        scores = self.apply_aux_loss(logits, tokens_per_expert, scores)
        return scores, top_indices, tokens_per_expert

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TopKRouter.

        Args:
            input (torch.Tensor): Input tensor of shape [batch_size * seq_len, hidden_size].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - scores: Softmax probabilities for top-k experts.
                - top_indices: Indices of top-k experts for each token.
                - tokens_per_expert: Number of tokens assigned to each expert.
        """
        logits = self.gating(input)
        logits = logits.view(-1, self.config.moe_num_experts)
        scores, top_indices, tokens_per_expert = self.routing(logits)
        return scores, top_indices, tokens_per_expert


# adapted from https://github.com/NVIDIA/Megatron-LM/blob/54f1f78529cbc2b9cddad313e7f9d96ac0420a27/megatron/core/transformer/moe/token_dispatcher.py#L291-L587
class TokenDispatcher:
    """
    Handles the dispatching and gathering of tokens to and from experts.

    This class is responsible for permuting tokens based on expert assignments and
    unpermuting them after expert processing.

    Args:
        config (AriaMoELMConfig): Configuration object containing MoE-related parameters.
    """

    def __init__(self, config: AriaMoELMConfig):
        self.config = config
        self.hidden_states_shape = None
        self.reversed_input_permutation_mapping = None

    def token_permutation(
        self, hidden_states: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Permute tokens based on expert assignments.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            indices (torch.Tensor): Expert assignment indices.

        Returns:
            torch.Tensor: Permuted tokens.
        """
        self.hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        flatten_indices = indices.flatten()
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        permuted_tokens = hidden_states.index_select(
            0, sorted_indices // self.config.moe_topk
        )
        self.reversed_input_permutation_mapping = sorted_indices
        return permuted_tokens

    def token_unpermutation(
        self, permuted_tokens: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Unpermute tokens and combine expert outputs.

        Args:
            permuted_tokens (torch.Tensor): Tokens after expert processing.
            scores (torch.Tensor): Expert assignment scores.

        Returns:
            torch.Tensor: Unpermuted and combined output.
        """
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
            -1, self.config.moe_topk, permuted_tokens.size(1)
        )

        unpermuted_tokens = unpermuted_tokens * scores.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1).type_as(permuted_tokens)
        output = unpermuted_tokens.view(self.hidden_states_shape)
        return output


def sequential_gemm(input, weight, tokens_per_expert):
    """
    Compute the matrix multiplication (GEMM) for each expert sequentially. This approach is computationally inefficient, especially when dealing with a large number of experts.

    Args:
        input (torch.Tensor): Input tensor of shape (num_tokens, in_features).
        weight (torch.Tensor): Weight tensor of shape (num_experts, in_features, out_features).
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

    Returns:
        torch.Tensor: Output tensor of shape (num_tokens, out_features).
    """
    num_tokens = input.shape[0]
    out_features = weight.shape[-1]
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

        out = torch.matmul(tokens, weight[expert_num])
        output[start:end] = out
    return output


try:
    from grouped_gemm.ops import gmm as experts_gemm

    if os.environ.get("USE_GROUPED_GEMM", "1") == "0":
        logger.warning(
            "environment variable USE_GROUPED_GEMM is set to 0, using sequential GEMM instead."
        )
        experts_gemm = sequential_gemm
except ImportError:
    logger.warning(
        "`grouped_gemm` is not installed, using sequential GEMM, which is slower."
    )
    experts_gemm = sequential_gemm


class GroupedGEMM(nn.Module):
    """
    Grouped GEMM (General Matrix Multiplication) module for efficient expert computation.
    This module utilizes the grouped_gemm library (https://github.com/fanshiqing/grouped_gemm)
    for optimized performance. If the grouped_gemm library is not installed, it gracefully
    falls back to a sequential GEMM implementation, which may be slower but ensures
    functionality.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        groups (int): Number of expert groups.
    """

    def __init__(self, in_features, out_features, groups, tp_dim):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_dim = tp_dim
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(groups, in_features, out_features))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if self.tp_size > 1:
            if self.tp_dim == "row":
                shard_size = self.in_features
                start_idx = self.tp_rank * shard_size
                param.data.copy_(loaded_weight.narrow(1, start_idx, shard_size))
            else:
                ups, gates = [], []
                for g in range(self.groups):
                    up, gate = loaded_weight[g].chunk(2, -1)
                    ups.append(up.chunk(self.tp_size, -1)[self.tp_rank])
                    gates.append(gate.chunk(self.tp_size, -1)[self.tp_rank])
                ups, gates = torch.stack(ups), torch.stack(gates)
                weights = torch.cat([ups, gates], dim=-1)
                param.data.copy_(weights)
        else:
            param.data.copy_(loaded_weight)

    def forward(self, input, tokens_per_expert):
        """
        Perform grouped matrix multiplication.

        Args:
            input (torch.Tensor): Input tensor of shape (num_tokens, in_features).
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor of shape (num_tokens, out_features).
        """
        return experts_gemm(input, self.weight, tokens_per_expert)


class GroupedMLP(nn.Module):
    """
    Grouped MLP module for Mixture of Experts.

    Args:
        config (AriaMoELMConfig): Configuration object for the model.
    """

    def __init__(self, config: AriaMoELMConfig) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.fc1 = GroupedGEMM(
            config.hidden_size,
            divide(config.moe_intermediate_size * 2, tp_size),
            config.moe_num_experts,
            "col",
        )
        self.fc2 = GroupedGEMM(
            divide(config.moe_intermediate_size, tp_size),
            config.hidden_size,
            config.moe_num_experts,
            "row",
        )

        def glu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        # Manually hook the forward pass to perform tensor model parallel all-reduce
        if tp_size > 1:
            self.register_forward_hook(
                lambda _, __, output_parallel: tensor_model_parallel_all_reduce(
                    output_parallel
                )
            )

        self.activation_func = glu

    def forward(self, permuted_tokens, tokens_per_expert):
        """
        Forward pass of the Grouped MLP.

        Args:
            permuted_tokens (torch.Tensor): Permuted input tokens.
            tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        tokens_per_expert = tokens_per_expert.cpu()
        fc1_output = self.fc1(permuted_tokens, tokens_per_expert)
        fc1_output = self.activation_func(fc1_output)
        fc2_output = self.fc2(fc1_output, tokens_per_expert)
        return fc2_output


class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer for the AriaMoE model.

    This layer implements the MoE mechanism, which routes input tokens to different experts
    based on a routing algorithm, processes them through the experts, and then combines
    the outputs.

    Args:
        config (AriaMoELMConfig): Configuration object for the MoE layer.
    """

    def __init__(
        self,
        config: AriaMoELMConfig,
        quant_config: Optional[QuantizationConfig],
        lora_config: Optional[LoRAConfig],
    ) -> None:
        super().__init__()

        self.router = TopKRouter(config)
        self.token_dispatcher = TokenDispatcher(config)
        self.experts = GroupedMLP(config)
        self.shared_experts = LlamaMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.moe_num_shared_experts,
            "silu",
            quant_config=quant_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE Layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor after passing through the MoE layer.

        Process:
        1. Route tokens to experts using the router.
        2. Permute tokens based on routing decisions.
        3. Process tokens through experts.
        4. Unpermute and combine expert outputs.
        5. Add shared expert output to the final result.
        """
        scores, indices, tokens_per_expert = self.router(hidden_states)

        permuted_tokens = self.token_dispatcher.token_permutation(
            hidden_states, indices
        )

        expert_output = self.experts(permuted_tokens, tokens_per_expert)

        output = self.token_dispatcher.token_unpermutation(expert_output, scores)

        shared_expert_output = self.shared_experts(hidden_states)
        output += shared_expert_output
        return output


class MoEDecoderLayer(LlamaDecoderLayer):
    """
    Custom Decoder Layer for the AriaMoE model which modifies the standard `LlamaDecoderLayer` by
    replacing the traditional MLP with a Mixture of Experts (MoE) Layer.

    Args:
        config (LlamaConfig): Configuration object for the layer.
        layer_idx (int): Index of the current layer in the model.
    """

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MoELayer(config, quant_config=quant_config, lora_config=lora_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class AriaMoELMModel(LlamaModel):
    """
    Custom LlamaModel for the AriaMoE model which modifies the standard LlamaModel by
    replacing the `LlamaDecoderLayer` with `MoEDecoderLayer`.

    This model implements a Mixture of Experts (MoE) approach, where each layer contains
    multiple expert networks that specialize in different aspects of the input.

    Args:
        config (LlamaConfig): Configuration object for the model.
    """

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MoEDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()


class AriaMoELMForCausalLM(LlamaForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm",
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        nn.Module.__init__(self)

        self.config = config
        self.lora_config = lora_config

        self.model = AriaMoELMModel(
            config, cache_config, quant_config, lora_config=lora_config, prefix="model"
        )
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config
                    else lora_config.lora_vocab_padding_size
                ),
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size, config.vocab_size, logit_scale
            )
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()


def build_mm_projector(config: AriaConfig):
    """
    Builds and returns an AriaProjector instance based on the provided configuration.

    Args:
        config (AriaConfig): The configuration object containing necessary parameters.

    Returns:
        AriaProjector: An instance of the AriaProjector class.
    """
    return AriaProjector(
        patch_to_query_dict=config.projector_patch_to_query_dict,
        embed_dim=config.vision_config.hidden_size,
        num_heads=config.vision_config.num_attention_heads,
        kv_dim=config.vision_config.hidden_size,
        ff_dim=config.text_config.hidden_size,
        output_dim=config.text_config.hidden_size,
    )


def _select_best_resolution(
    img_width: int, img_height: int, target_ratios: List[List[int]], patch_size: int
):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        img_width: the original widths of images.
        img_height: the original heights of images.
        target_ratios (2d numpy array): dimension size (M,2)
        patch_size (int): image patch size

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """

    aspect_ratio = img_width / img_height
    best_ratio_diff = float("inf")
    best_ratio_w, best_ratio_h = 1, 1
    area = np.int32(img_height) * np.int32(img_height)
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio_w, best_ratio_h = ratio[0], ratio[1]
        elif (
            ratio_diff == best_ratio_diff
            and area > 0.5 * patch_size * patch_size * ratio[0] * ratio[1]
        ):
            best_ratio_w, best_ratio_h = ratio[0], ratio[1]

    return best_ratio_w, best_ratio_h


def split_image(
    image: Image.Image,
    split_image: bool,
    split_ratio: List[List[int]] = [
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [1, 6],
        [1, 7],
        [1, 8],
        [2, 4],
        [2, 3],
        [2, 2],
        [2, 1],
        [3, 1],
        [3, 2],
        [4, 1],
        [4, 2],
        [5, 1],
        [6, 1],
        [7, 1],
        [8, 1],
    ],
    patch_size: int = 980,
) -> List[Image.Image]:
    """
    Split image into multiple patches

    Args:
        image (PIL.Image): Input image.
        split_image (bool): Whether to split the image into patches.
        split_ratio (2d numpy array): dimension size (M,2)
        patch_size (int): image patch size

    Returns:
        List[PIL.Image]: List of splitted images.
    """
    if split_image:
        ratio_width, ratio_height = _select_best_resolution(
            image.width, image.height, split_ratio, patch_size
        )
        resize_width = patch_size * ratio_width
        resize_height = patch_size * ratio_height
        blocks = ratio_width * ratio_height
        resized_img = image.resize((resize_width, resize_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (resize_width // patch_size)) * patch_size,
                (i // (resize_width // patch_size)) * patch_size,
                ((i % (resize_width // patch_size)) + 1) * patch_size,
                ((i // (resize_width // patch_size)) + 1) * patch_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if len(processed_images) != 1:
            processed_images.insert(0, image)
        return processed_images
    else:
        return [image]


def get_max_multimodal_tokens(ctx):
    return max(ctx.model_config.hf_config.image_size2tokens.values())


def input_mapper_for_aria(ctx, data):
    """
    This is almost same with _default_input_mapper from vllm.multimodal.image.py.
    Args:
        ctx (ModelExecutorContext): The context object containing necessary parameters.
        data (Union[Image.Image, torch.Tensor, List[Union[Image.Image, torch.Tensor]]]): The input data to be processed.
    The only different is we would like to support runtime max_image_size adjustment.
    """
    model_config = ctx.model_config
    max_image_size = getattr(model_config.multimodal_config, "max_image_size", 980)

    # PIL image
    if isinstance(data, Image.Image) or is_list_of(data, Image.Image):
        image_processor = cached_get_image_processor(
            model_config.model, trust_remote_code=model_config.trust_remote_code
        )
        if image_processor is None:
            raise RuntimeError(
                "No HuggingFace processor is available " "to process the image object"
            )
        try:
            batch_data = image_processor.preprocess(
                data, max_image_size=max_image_size, return_tensors="pt"
            ).data
            batch_data.pop("num_crops")
        except Exception:
            logger.error("Failed to process image (%s)", data)
            raise

        return MultiModalInputs(batch_data)

    # Image embedding
    elif isinstance(data, torch.Tensor) or is_list_of(data, torch.Tensor):
        return MultiModalInputs({"image_embeds": data})

    raise TypeError(f"Invalid image type: {type(data)}")


def input_processor(ctx, llm_inputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    # if it is pure text input, use it as is
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config

    tokenizer = cached_get_tokenizer(model_config.tokenizer)
    hf_config = model_config.hf_config

    # prepare image tokens, the max_image_size is used to determine the number of patch_size for every image
    max_image_size = multi_modal_data.pop("max_image_size", 980)
    _split_image = multi_modal_data.pop("split_image", False)

    assert isinstance(max_image_size, int) or isinstance(
        max_image_size, float
    ), "max_image_size should be float or int"
    images = (
        multi_modal_data["image"]
        if isinstance(multi_modal_data["image"], list)
        else [multi_modal_data["image"]]
    )
    num_crops = []
    splitted_images = []
    for image in images:
        splitted_image = split_image(image, _split_image, patch_size=max_image_size)
        splitted_images.extend(splitted_image)
        num_crops.append(len(splitted_image))
    max_image_size = [max_image_size] * len(images)
    # reassign the image because we might split them into mini-patches
    multi_modal_data["image"] = splitted_images

    # Mapping the image patch size to the corresponding number of tokens for each image
    image_feature_sizes = []
    for image_size, num_crop in zip(max_image_size, num_crops):
        assert (
            image_size in hf_config.image_size2tokens
        ), f"Invalid image size: {image_size}, available options: {list(hf_config.image_size2tokens.keys())}"
        image_feature_sizes.append(hf_config.image_size2tokens[image_size] * num_crop)

    # Set up the max_image_size and split_image in the RuntimeContext for the image processor
    # TODO: Supports dynamic image size support
    setattr(model_config.multimodal_config, "max_image_size", max(max_image_size))

    new_prompt, new_token_ids = repeat_and_pad_placeholder_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        placeholder_token_id=hf_config.image_token_index,
        repeat_count=image_feature_sizes,
    )

    return LLMInputs(
        prompt=new_prompt,
        prompt_token_ids=new_token_ids,
        multi_modal_data=multi_modal_data,
    )


# adapted from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_multimodal_tokens)
@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_aria)
@INPUT_REGISTRY.register_input_processor(input_processor)
class AriaForConditionalGeneration(nn.Module, SupportsMultiModal):
    """
    Aria model for conditional generation tasks.

    This model combines a vision tower, a multi-modal projector, and a language model
    to perform tasks that involve both image and text inputs.
    """

    def __init__(
        self,
        config: AriaConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # prepare the image_size to tokens mapping for the image preprocess, see input_processor
        setattr(
            config,
            "image_size2tokens",
            {
                int(math.sqrt(k) * config.vision_config.patch_size): v
                for k, v in config.projector_patch_to_query_dict.items()
            },
        )
        self.config = config
        self.vision_tower = AriaVisionModel(config.vision_config)
        self.multi_modal_projector = build_mm_projector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AriaMoELMModel(
            config.text_config, cache_config, quant_config
        )
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size,
            quant_config=quant_config,
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, self.vocab_size, logit_scale
        )
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ):
        # 1. Extra the input embeddings
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        pixel_values = kwargs.get("pixel_values", None)
        pixel_mask = kwargs.get("pixel_mask", None)

        # 2. Merge text and images
        if pixel_values is not None:
            pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:]).to(
                torch.bfloat16
            )
            pixel_mask = pixel_mask.view(-1, *pixel_mask.shape[-2:])
            image_outputs, image_attn_mask = self.vision_tower(
                pixel_values,
                pixel_mask=pixel_mask,
            )
            selected_image_feature = image_outputs.last_hidden_state

            image_features = self.multi_modal_projector(
                selected_image_feature, attn_mask=image_attn_mask
            )

            inputs_embeds = inputs_embeds.to(image_features.dtype)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, image_features, self.config.image_token_index
            )

        hidden_states = self.language_model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            None,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            shard_id = None
            # Because we used the origin hf vit and vision projector, we cound keep the weight in the sharded shape.
            # Only for the language model part needs to adjust the weight loading.
            if "language_model" in name:
                for param_name, weight_name, _shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    shard_id = _shard_id
                    break

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is not None:
                weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader(param, loaded_weight)
