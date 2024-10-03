# Aria

[😊 Hugging Face](#) | 
[📄 Paper](#) | 
[📚 Blog](#) | 
[🌐 WebDemo](#) 


## Introduction
Aria is a multimodal native MoE model. It features:
- State-of-the-art performance on various multimodal and language tasks, especially in video and document understanding;
- Long multimodal context window of 64K tokens;
- 3.9B activated parameters per token, enabling fast inference speed and low fine-tuning cost.
  
<!-- 
| Category                            | Benchmark               | Aria  | Pixtral 12B | Llama3.2 11B | Llama3-V 8B | GPT-4V | GPT-4o mini | GPT-4o | Gemini-1.5 Flash | Gemini-1.5 Pro |
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
-->

## News
- 2024.10.10: We release Aria!

## Quick Start

### Installation

```bash
pip install -e .
# or install with dev dependencies if you want to contribute to the project
pip install -e .[dev] 

pip install flash-attn --no-build-isolation
```

### Inference

Aria has 25.3B total parameters, it can be loaded in one A100 (80GB) GPU with bfloat16 precision.

Here is a code snippet to show you how to use Aria with Hugging Face Transformers.

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

### Cookbook
Checkout these [inference examples](https://github.com/rhymes-ai/Aria/tree/main/inference/notebooks) that demonstrate how to use Aria on various applications such as chart understanding, PDF reading, video understanding, etc.

## Fine-tuning

We offer both LoRA fine-tuning and full parameter tuning, using various dataset types:
- Single-image datasets
- Multi-image datasets
- Video datasets

For a quick try, visit the [examples](./examples) folder and choose one of the fine-tuning examples.

### Prepare dataset
Please refer to [custom_dataset.md](custom_dataset.md) for how to prepare your dataset.

### Fine-tune with LoRA

After preparing your dataset, follow these steps to fine-tune Aria using LoRA:

1. Open the configuration file `recipes/config_lora.yaml`. Locate the `dataset_mixer` section and update it with your dataset paths:

```yaml
dataset_mixer:
  "path/to/dataset1": 1
  "path/to/dataset2": 0.5
  "path/to/dataset3": 2
```

> **Note on dataset mixing:** Aria supports combining multiple datasets with different sampling rates. In the example above:
> - `dataset1` will be used entirely (weight 1)
> - `dataset2` will use 50% of its data (weight 0.5)
> - `dataset3` will be used twice (weight 2)

2. Start the fine-tuning process by running the following command on one A100 (80GB) or H100 (80GB) GPU:

```bash
python aria/train.py --config recipes/config_lora.yaml
```

3. For multi-GPU training, use the [`accelerate`](https://huggingface.co/docs/accelerate/index) library:

```bash
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml aria/train.py --config recipes/config_lora.yaml --num_processes [number_of_gpus]
```

   - Choose from pre-configured accelerate settings in `recipes/accelerate_configs/`
   - Adjust the `--num_processes` argument to match your available GPUs
   - For custom configurations, refer to the [accelerate documentation](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
  
4. Inference with the fine-tuned model:

See [inference with LoRA support](inference.md#2-inference-with-lora-support) for how to inference with the fine-tuned model.

### Full parameter fine-tuning

Everything is the same as the LoRA fine-tuning process, except for the configuration file `recipes/config_full.yaml`.

Full parameter tuning consumes more GPU memory, thus multiple GPUs are required. The following command has been tested on 8 A100 (80GB) GPUs.

```bash
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml aria/train.py --config recipes/config_full.yaml
```

If you encounter out-of-memory errors, try reducing the `per_device_train_batch_size` in the config file. Adjust the `gradient_accumulation_steps` accordingly to maintain the effective training batch size.

```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
```

Memory consumption varies across datasets. Generally, more memory is required for multi-image and video datasets. Adjust the `deepspeed_config` parameters to optimize memory consumption, such as using `zero_stage` 3 and offloading parameters and optimizer to the CPU.

```yaml
deepspeed_config:
  gradient_accumulation_steps: auto
  gradient_clipping: auto
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 3
```

## Citation
If you find our work helpful, please consider citing.
```
@article{aria,
  title={},
  author={},
  year={2024},
  journal={}
}
```


