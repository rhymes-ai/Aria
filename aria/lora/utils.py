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

from typing import List

from transformers.utils import logging

from aria.config import AriaModelConfig

logger = logging.get_logger(__name__)


def get_lora_target_modules(
    model_named_modules: List[str], config: AriaModelConfig
) -> List[str]:
    """
    This function identifies and return the target modules for LoRA based on the config.
    """
    lora_target_modules = []

    for key in model_named_modules:
        if config.freeze_vit and "vision_tower" in key:
            continue
        if config.freeze_projector and "multi_modal_projector" in key:
            continue
        if config.freeze_llm and "language_model" in key:
            continue

        def should_be_frozen(key):
            if not config.freeze_llm_layers:
                return False
            for freeze_layer in config.freeze_llm_layers:
                if f"language_model.model.layers.{freeze_layer}." in key:
                    return True
            return False

        if should_be_frozen(key):
            continue

        for module in config.lora_target_modules:
            if module in key:
                logger.info(
                    f"Adding {key} to lora target modules, as it contains {module}"
                )
                lora_target_modules.append(key)
                break
    return lora_target_modules
