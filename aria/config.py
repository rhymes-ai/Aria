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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trl import ModelConfig
from trl.commands.cli_utils import SFTScriptArguments


@dataclass
class AriaModelConfig(ModelConfig):
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "The path to the tokenizer."},
    )
    peft_model_path: str = field(
        default=None,
        metadata={"help": "The path to the PEFT model."},
    )
    freeze_vit: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the ViT model."},
    )
    freeze_projector: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the projector."},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the LLM model."},
    )
    freeze_llm_layers: List[int] = field(
        default=None,
        metadata={"help": "The indices of the LLM layers to freeze."},
    )
    moe_z_loss_coeff: float = field(
        default=1e-5,
        metadata={"help": "The coefficient for the z loss."},
    )
    moe_aux_loss_coeff: float = field(
        default=1e-3,
        metadata={"help": "The coefficient for the auxiliary loss."},
    )
    max_image_size: int = field(
        default=980,
        metadata={
            "help": "The maximum size of the image after processing before being passed to the vision encoder.",
            "choices": [490, 980],
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.max_image_size not in [490, 980]:
            raise ValueError("max_image_size must be either 490 or 980")


@dataclass
class AriaSFTScriptArguments(SFTScriptArguments):
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": ("Datasets and their proportions to be used for training ift/rl.")
        },
    )
