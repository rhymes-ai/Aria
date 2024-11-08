Accelerate generation speed with [gpt-fast](https://github.com/pytorch-labs/gpt-fast)

## Downloading Weights

```bash
export MODEL_REPO=rhymes-ai/Aria
python scripts/download.py --repo_id $MODEL_REPO
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL_REPO
```

## Generate Text

```bash
python generate.py checkpoints/rhymes-ai/Aria/model.pth --compile --apply_regional_compilation --prompt "What is the meaning of life?"
```