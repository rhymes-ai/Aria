# Aria Model Inference Methods

This document outlines three different approaches for performing inference with the Aria model, a multimodal AI capable of processing both text and images.

## 1. Basic Inference with Hugging Face Transformers

This method utilizes the Hugging Face Transformers library, ideal for quick starts and basic usage.

### How to Use:
```python
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_id_or_path = "rhymes-ai/Aria"

model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)

image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"

image = Image.open(requests.get(image_path, stream=True).raw)

messages = [
    {
        "role": "user",
        "content": [
            {"text": None, "type": "image"},
            {"text": "what is the image?", "type": "text"},
        ],
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt")
inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        stop_strings=["<|im_end|>"],
        tokenizer=processor.tokenizer,
        do_sample=True,
        temperature=0.9,
    )
    output_ids = output[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(output_ids, skip_special_tokens=True)

print(result)
```

## 2. Inference with LoRA Support

This method uses a Python script to run inference, supporting model fine-tuning with LoRA. It offers more flexibility and control over the inference process, especially when working with fine-tuned models.

### How to Use:
```bash
python aria/inference.py \
    --base_model_path /path/to/base/model \
    --tokenizer_path /path/to/tokenizer \
    --image_path /path/to/image.png \
    --prompt "Your prompt here" \
    --max_image_size 980 \
    --peft_model_path /path/to/peft/model  # Optional, for fine LoRA fine-tuned models
```

For more details, please refer to the script's help documentation:
```bash
python aria/inference.py --help
```

## 3. High-Performance Inference with vLLM

This method leverages vLLM for high-performance inference, particularly useful for scenarios requiring parallel processing or handling multiple requests.

### Install vLLM:
```bash
pip install -e .[vllm]
```

### How to Use:
```python
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.model_executor.models import _MULTIMODAL_MODELS

from aria.vllm.aria import AriaForConditionalGeneration

ModelRegistry.register_model(
    "AriaForConditionalGeneration", AriaForConditionalGeneration
)
_MULTIMODAL_MODELS["AriaForConditionalGeneration"] = (
    "aria",
    "AriaForConditionalGeneration",
)


def main():
    llm = LLM(
        model="rhymes-ai/Aria",
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 256},
        enforce_eager=True,
        trust_remote_code=True,
        skip_tokenizer_init=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "rhymes-ai/Aria", trust_remote_code=True, use_fast=False
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Compare Image 1 and image 2, tell me about the differences between image 1 and image 2.\nImage 1\n",
                },
                {"type": "image"},
                {"type": "text", "text": "\nImage 2\n"},
                {"type": "image"},
            ],
        }
    ]

    message = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    outputs = llm.generate(
        {
            "prompt_token_ids": message,
            "multi_modal_data": {
                "image": [
                    Image.open("assets/princess1.jpg"),
                    Image.open("assets/princess2.jpg"),
                ],
                "max_image_size": 980,  # [Optional] The max image patch size, default `980`
                "split_image": True,  # [Optional] whether to split the images, default `False`
            },
        },
        sampling_params=SamplingParams(max_tokens=200, top_k=1),
    )

    for o in outputs:
        generated_tokens = o.outputs[0].token_ids
        print(tokenizer.decode(generated_tokens))


if __name__ == "__main__":
    main()

```
