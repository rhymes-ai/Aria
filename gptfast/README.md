Accelerate generation speed with [gpt-fast](https://github.com/pytorch-labs/gpt-fast)

## Downloading Weights

```bash
export MODEL_REPO=rhymes-ai/Aria
python scripts/download.py --repo_id $MODEL_REPO
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$MODEL_REPO
```

## Benchmark

```bash
python benchmark.py --compile
```

### Performance Results (Single H100 GPU)

| Mode    | Performance (tokens/s) |
|---------|----------------------:|
| Base    | 25.2                  |
| Compile | 130.0                 |

## Chat Interface

### Running the Chat

To start the chat interface, run:

```bash
python -m gptfast.chat
```

### Available Commands

The chat interface supports the following commands:

- `help` - Display all available commands
- `quit` - Exit the chat
- `reset` - Clear the chat history
- `image` - Start a chat with an image (supports local paths and URLs)

### Examples

Basic chat:
```bash
You: Hello! Who are you?
Assistant: I am Aria, an AI assistant...

You: What can you help me with?
Assistant: I can help you with various tasks...
```

Chat with images:
```bash
You: image
Enter image path or URL: https://example.com/cat.jpg
Enter your message about the image: What do you see in this image?
Assistant: I can see a cat...
```