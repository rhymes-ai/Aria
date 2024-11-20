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
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import LlamaConfig
from transformers.utils import logging
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.inputs import INPUT_REGISTRY, token_inputs
from vllm.model_executor.layers.fused_moe import fused_moe
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

from .vision_encoder import AriaVisionModel

from aria.model.configuration_aria import AriaConfig
from aria.model.projector import AriaProjector
# from aria.model.vision_encoder import AriaVisionModel

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


class Experts(nn.Module):
    def __init__(self, config: AriaMoELMConfig):
        super().__init__()
        self.config = config

        self.router_weight = nn.Parameter(
            torch.empty((self.config.moe_num_experts, self.config.hidden_size))
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor model parallel size {self.tp_size} is greater than the number of experts {config.moe_num_experts}"
            )

        self.w1 = nn.Parameter(
            torch.empty(
                (
                    config.moe_num_experts,
                    config.moe_intermediate_size * 2 // self.tp_size,
                    config.hidden_size,
                )
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                (
                    config.moe_num_experts,
                    config.hidden_size,
                    config.moe_intermediate_size // self.tp_size,
                )
            )
        )
        set_weight_attrs(self.router_weight, {"weight_loader": self.weight_loader})
        set_weight_attrs(self.w1, {"weight_loader": self.weight_loader})
        set_weight_attrs(self.w2, {"weight_loader": self.weight_loader})

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ):
        if shard_id == "router":
            param.data.copy_(loaded_weight)
        elif shard_id == "w1":
            if self.tp_size > 1:
                # the shape of loaded_weight is (num_experts, hidden_size, 2 * moe_intermediate_size)
                up, gate = loaded_weight.chunk(2, dim=-1)
                up_current_rank = up.chunk(self.tp_size, dim=-1)[self.tp_rank]
                gate_current_rank = gate.chunk(self.tp_size, dim=-1)[self.tp_rank]
                up_and_gate = torch.cat(
                    [up_current_rank, gate_current_rank], dim=-1
                ).transpose(1, 2)
                param.data.copy_(up_and_gate)
            else:
                param.data.copy_(loaded_weight.transpose(1, 2))
        else:
            if self.tp_size > 1:
                # the shape of loaded_weight is (num_experts, moe_intermediate_size, hidden_size)
                down_current_rank = loaded_weight.chunk(self.tp_size, dim=1)[
                    self.tp_rank
                ]
                param.data.copy_(down_current_rank.transpose(1, 2))
            else:
                param.data.copy_(loaded_weight.transpose(1, 2))

    def forward(self, hidden_states):
        router_output = torch.nn.functional.linear(hidden_states, self.router_weight)

        def custom_routing_function(hidden_states, router_output, topk, renormalize):
            top_logits, top_indices = torch.topk(
                router_output, k=self.config.moe_topk, dim=1
            )
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32)
            return scores, top_indices.to(torch.int32)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        final_hidden_states = fused_moe(
            hidden_states,
            self.w1,
            self.w2,
            router_output,
            self.config.moe_topk,
            False,
            inplace=True,
            custom_routing_function=custom_routing_function,
        )
        final_hidden_states = final_hidden_states.view(hidden_states_shape)
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states


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
        self.config = config

        self.experts = Experts(config)
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
        """

        shared_expert_output = self.shared_experts(hidden_states)
        sparse_expert_output = self.experts(hidden_states)

        return sparse_expert_output + shared_expert_output


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

        # FIXME(zhoufan): this is a hack to avoid the error: AttributeError: 'AriaMoELMModel' object has no attribute 'do_not_compile'.
        self.do_not_compile = True

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
    area = np.int32(img_width) * np.int32(img_height)
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

    new_prompt, new_token_ids, ranges = repeat_and_pad_placeholder_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        placeholder_token_id=hf_config.image_token_index,
        repeat_count=image_feature_sizes,
    )

    return token_inputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data=multi_modal_data,
        # multi_modal_placeholders={"image": ranges},
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
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

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
            selected_image_feature, image_attn_mask = self.vision_tower(
                pixel_values,
                pixel_mask=pixel_mask,
            )

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
            ("experts.router_weight", "router.weight", "router"),
            ("experts.w1", "experts.fc1.weight", "w1"),
            ("experts.w2", "experts.fc2.weight", "w2"),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            shard_id = None
            # Because we used the origin hf vit and vision projector, we cound keep the weight in the sharded shape.
            # Only for the language model part needs to adjust the weight loading.
            if "language_model" in name or "vision_tower" in name:
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
