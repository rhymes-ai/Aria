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

import torch
from peft import PeftConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoTokenizer
from trl import (
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.commands.cli_utils import TrlParser

from aria.config import AriaModelConfig, AriaSFTScriptArguments
from aria.data import apply_chat_template_and_tokenize, mix_datasets
from aria.load_video import load_video
from aria.lora.layers import GroupedGemmLoraLayer
from aria.lora.utils import get_lora_target_modules
from aria.model import (
    AriaForConditionalGeneration,
    AriaVisionProcessor,
    GroupedGEMM,
    MoEAuxLossAutoScaler,
)


def setup_model_and_tokenizer(model_config):
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AriaForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )

    model.set_moe_z_loss_coeff(model_config.moe_z_loss_coeff)
    model.set_moe_aux_loss_coeff(model_config.moe_aux_loss_coeff)

    if model_config.freeze_vit:
        model.freeze_vit()
    if model_config.freeze_projector:
        model.freeze_projector()
    if model_config.freeze_llm:
        model.freeze_llm()

    if model_config.use_peft:
        model = setup_peft(model, model_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_path, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    processor = AriaVisionProcessor(max_image_size=model_config.max_image_size)

    return model, tokenizer, processor


def setup_peft(model, model_config):
    if model_config.peft_model_path is not None:
        peft_config = PeftConfig.from_pretrained(model_config.peft_model_path)
        model = PeftModel.from_pretrained(
            model, model_config.peft_model_path, is_trainable=True
        )
    else:
        # Override the LoRA target modules based on the config
        # This supports freezing layers and ensures LoRA is not applied to frozen layers
        # The target modules are determined dynamically based on the model's structure and freezing configuration
        named_modules = [key for key, _ in model.named_modules()]
        target_modules = get_lora_target_modules(named_modules, model_config)
        model_config.lora_target_modules = target_modules

        peft_config = get_peft_config(model_config)

        custom_module_mapping = {GroupedGEMM: GroupedGemmLoraLayer}
        peft_config._register_custom_module(custom_module_mapping)

        # Disable autocast_adapter_dtype to maintain bfloat16 precision for GroupedGemm compatibility
        # Setting it to True would cast LoRA layers to fp32, which is unsupported by GroupedGemm
        model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        model.print_trainable_parameters()
    return model


def collate_fn(
    examples,
    tokenizer,
    processor,
    split_image: bool = False,
    max_seq_length: int = 1024,
):
    images = []
    messages = []
    for example in examples:
        # Process video input:
        # 1. Extract frames from the video
        # 2. Convert video messages to image messages
        #
        # Example transformation:
        # Input:
        # {
        #     "video": {
        #         "path": "path/to/video",
        #         "num_frames": 3
        #     },
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"text": "Hello, how are you?", "type": "text"},
        #                 {"text": None, "type": "video"}
        #             ]
        #         }
        #     ]
        # }
        #
        # Output:
        # [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"text": "Hello, how are you?", "type": "text"},
        #             {"text": None, "type": "image"},
        #             {"text": None, "type": "image"},
        #             {"text": None, "type": "image"}
        #         ]
        #     }
        # ]
        if example["video"]:
            frames = load_video(
                example["video"]["path"], example["video"]["num_frames"]
            )
            images.extend(frames)
            actual_num_frames = len(
                frames
            )  # The main purpose is to prevent the load from entering 0 frames due to file damage.
            for message in example["messages"]:
                for cont_idx, cont in enumerate(message["content"]):
                    if cont["type"] == "video":
                        del message["content"][cont_idx]
                        for img_i in range(actual_num_frames):
                            insert_item = {
                                "text": None,
                                "type": "image",
                            }
                            message["content"].insert(cont_idx + img_i, insert_item)
            messages.append(example["messages"])
        else:
            if example["images"]:
                images.extend(example["images"])
            messages.append(example["messages"])

    if images:
        images = [
            Image.open(image).convert("RGB") if isinstance(image, str) else image
            for image in images
        ]
        image_inputs = processor(images, split_image=split_image)

        batch = apply_chat_template_and_tokenize(
            messages,
            tokenizer,
            iter(image_inputs.pop("num_crops")),
            max_length=max_seq_length,
        )

        batch.update(image_inputs)
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    else:  # text-only
        batch = apply_chat_template_and_tokenize(
            messages,
            tokenizer,
            max_length=max_seq_length,
        )

    return batch


def main():
    parser = TrlParser((AriaSFTScriptArguments, SFTConfig, AriaModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.dataset_text_field = ""  # need a dummy field
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    model, tokenizer, processor = setup_model_and_tokenizer(model_config)

    raw_datasets = mix_datasets(sft_script_args.dataset_mixer)
    train_dataset = raw_datasets[sft_script_args.dataset_train_split]
    eval_dataset = raw_datasets[sft_script_args.dataset_test_split]

    if eval_dataset is None:
        training_args.eval_strategy = "no"

    MoEAuxLossAutoScaler.set_loss_scale(1 / training_args.gradient_accumulation_steps)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=lambda examples: collate_fn(
            examples,
            tokenizer,
            processor,
            model_config.split_image,
            training_args.max_seq_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    processor.save_pretrained(training_args.output_dir)

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
