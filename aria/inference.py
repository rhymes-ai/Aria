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

import argparse

import torch
from peft import PeftConfig, PeftModel
from PIL import Image

from aria.lora.layers import GroupedGemmLoraLayer
from aria.model import AriaForConditionalGeneration, AriaProcessor, GroupedGEMM


def parse_arguments():
    parser = argparse.ArgumentParser(description="Aria Inference Script")
    parser.add_argument(
        "--base_model_path", required=True, help="Path to the base model"
    )
    parser.add_argument("--peft_model_path", help="Path to the PEFT model (optional)")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model")
    parser.add_argument(
        "--max_image_size",
        type=int,
        help="Maximum size of the image to be processed",
        default=980,
    )
    return parser.parse_args()


def load_model(base_model_path, peft_model_path=None):
    model = AriaForConditionalGeneration.from_pretrained(
        base_model_path, device_map="auto", torch_dtype=torch.bfloat16
    )

    if peft_model_path:
        peft_config = PeftConfig.from_pretrained(peft_model_path)
        custom_module_mapping = {GroupedGEMM: GroupedGemmLoraLayer}
        peft_config._register_custom_module(custom_module_mapping)
        model = PeftModel.from_pretrained(
            model,
            peft_model_path,
            config=peft_config,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )

    return model


def prepare_input(image_path, prompt, processor: AriaProcessor, max_image_size):
    image = Image.open(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": prompt, "type": "text"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        max_image_size=max_image_size,
    )

    return inputs


def inference(
    image_path,
    prompt,
    model: AriaForConditionalGeneration,
    processor: AriaProcessor,
    max_image_size,
):
    inputs = prepare_input(image_path, prompt, processor, max_image_size)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
            do_sample=True,
            temperature=0.9,
        )
        result = processor.batch_decode(output, skip_special_tokens=True)

    return result


def main():
    args = parse_arguments()
    # if the tokenizer is not put in the same folder as the model, we need to specify the tokenizer path
    processor = AriaProcessor.from_pretrained(
        args.base_model_path, tokenizer_path=args.tokenizer_path
    )
    model = load_model(args.base_model_path, args.peft_model_path)

    result = inference(
        args.image_path, args.prompt, model, processor, args.max_image_size
    )
    print(result)


if __name__ == "__main__":
    main()
