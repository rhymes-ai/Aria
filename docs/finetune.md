# Fine-tuning

Aria offers the following fine-tuning methods:
- LoRA (Low-Rank Adaptation)
- Full parameter tuning

It also supports various dataset types:
- Single-image datasets
- Multi-image datasets
- Video datasets

For a quick try, visit the [examples](../examples) folder and choose one of the fine-tuning examples.

## Prepare dataset
Please refer to [custom_dataset.md](custom_dataset.md) for how to prepare your dataset.

## Fine-tune with LoRA

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

2. Start the fine-tuning process by running on one A100 (80GB) or H100 (80GB) GPU:

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

## Fine-tune with full parameter tuning

Everything is the same as the LoRA fine-tuning process, except for the configuration file `recipes/config_full.yaml`.

Full parameter tuning consumes more GPU memory, and multiple GPUs are required. We have tested it on 8 A100 (80GB) GPUs.

```bash
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml aria/train.py --config recipes/config_full.yaml
```

If you encounter out-of-memory errors, try reducing the `per_device_train_batch_size` in the config file. Adjust the `gradient_accumulation_steps` accordingly to maintain the effective training batch size.

```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
```

Memory consumption varies with the dataset. Generally, more memory is required for multi-image and video datasets. Adjust the `deepspeed_config` parameters to fit your GPU memory, such as using `zero_stage` 3 and offloading parameters and optimizer to the CPU.

```yaml
deepspeed_config:
  gradient_accumulation_steps: auto
  gradient_clipping: auto
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero_stage: 3
```
