# Aria

[😊 Hugging Face](#) | 
[📄 Paper](#) | 
[📰 Blog](#) | 
[📚 Tutorial](#) | 
[💻 Demo](#) | 
[🌐 Website](#) | 


## Introduction
Aria is the first open MoE model that is natively multimodal. It features SoTA performance on OCR and video understanding tasks, competitve performance on language and coding tasks, and fast inference speed with merely 3.9B activated parameters per token. 

| Category                            | Benchmark               | Aria  | Pixtral 12B | Llama3 8B | Llama3-V 8B | GPT-4V | GPT-4o mini | GPT-4o | Gemini-1.5 Flash | Gemini-1.5 Pro |
|-------------------------------------|-------------------------|-------|-------------|-----------|-------------|--------|-------------|--------|------------------|----------------|
| **Knowledge(Multimodal)**                  | MMMU              | 54.9  | 52.5        | -         | 49.6        | 56.4   | 59.4        | 69.1   | 56.1             | 62.2           |
| **Math(Multimodal)**                    | MathVista   | 66.1  | 58.0        | -         | -           | -      | 54.7        | 63.8   | 58.4             | 63.9           |
| **Document**       | DocQA            | 92.6  | 90.7           | -         | 84.4        | 88.4   | -           | 92.8  | 89.9             | 93.1           |
| **Chart**               | ChartQA           | 86.4  | 81.8        | -         | 78.7        | 78.4   | -           | 85.7   | 85.4             | 87.2           |
| **Scene Text**                                      | TextVQA         | 81.1  | -           | -         | 78.2        | 78.0      | -           | -      | 78.7                | 78.7              |
| **General Visual QA**               | MMBench-1.1             | 80.3  | -           | -         | -           | 79.8   | 76.0        | 82.2   | -                | 73.9           |
| **Video Understanding**        | LongVideoBench  | 65.3  | 47.4           | -      | -           | 60.7   | 58.8        | 66.7      | 62.4                | 64.4              |
| **Knowledge(Language)**        | MMLU (5-shot)           | 73.3  | 69.2        | 69.4      | -        | 86.4   | -           | 89.1   | 78.9             | 85.9           |
| **Math(Language)**                      | MATH              | 50.8  | 48.1        | 51.9         | -        | -      | 70.2           | 76.6   | -            | -           |
| **Reasoning(Language)**                                    | ARC Challenge           | 91.0  | -           | 83.4         | -        | -      | 96.4           | 96.7      | -                | -              |
| **Coding**                          | HumanEval               | 73.2  | 72.0        | 72.6      | -        | 67.0   | 87.2        | 90.2   | 74.3             | 84.1           |


## News

## Quick Start

### Installation

```bash
pip install -e .
# or install with dev dependencies if you want to contribute to the project
pip install -e .[dev] 

pip install flash-attn --no-build-isolation
```

### Inference

The total number of parameters in Aria is about 25B, it can be loaded in one A100 (80GB) GPU with bfloat16 precision.

Performing inference is simple with the Hugging Face ecosystem:

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

We offer additional inference methods, such as utilizing [VLLM](https://github.com/vllm-project/vllm) for enhanced performance. For comprehensive details, please refer to [docs/inference.md](docs/inference.md).

### Fine-tuning

Aria supports fine-tuning through methods like LoRA (Low-Rank Adaptation) and full parameter tuning. For detailed instructions and code samples on how to fine-tune Aria, please refer to [docs/finetune.md](docs/finetune.md).
