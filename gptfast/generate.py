# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import contextlib
import itertools

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
import torch._dynamo.config
import torch._inductor.config
from model import Aria, ModelArgs, Transformer
from PIL import Image
from torch.nn.attention import SDPBackend
from transformers import AutoProcessor, AutoTokenizer


def get_model_size_in_bytes(model, ignore_embeddings=False):
    """
    Returns the model size in bytes. The option to ignore embeddings
    is useful for models with disproportionately large embeddings compared
    to other model parameters that get quantized/sparsified.
    """

    def flat_size(tensor):
        if hasattr(tensor, "__tensor_flatten__"):
            size = 0
            # 0th element is a list of attributes that
            # hold tensors
            for attr_name in tensor.__tensor_flatten__()[0]:
                sub_tensor = getattr(tensor, attr_name)
                size += flat_size(sub_tensor)
            return size
        else:
            return tensor.numel() * tensor.element_size()

    model_size = 0
    for name, child in model.named_children():
        if not (isinstance(child, torch.nn.Embedding) and ignore_embeddings):
            for p in itertools.chain(
                child.parameters(recurse=False), child.buffers(recurse=False)
            ):
                model_size += flat_size(p)
            model_size += get_model_size_in_bytes(child, ignore_embeddings)
    return model_size


from model import ModelArgs


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    input_embeds: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, input_embeds)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.nn.attention.sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            next_token, next_prob = next_token.clone(), next_prob.clone()
            input_pos += 1
            new_tokens.append(next_token)
            generation_done = callback(new_tokens)
            new_probs.append(next_prob)
            cur_token = next_token.view(1, -1)
            if generation_done is True:
                break
    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Aria,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_mask: Optional[torch.Tensor] = None,
    callback=lambda x: x,
    cache_size: Optional[int] = None,
    linear_causal_mask: bool = False,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    device = input_ids.device
    original_prompt_token_count = input_ids.numel()

    if pixel_values is not None:
        input_embeds = model.prepare_embeddings(input_ids, pixel_values, pixel_mask)
        prompt_token_count_after_inserting_image_tokens = input_embeds.shape[1]
    else:
        input_embeds = None
        prompt_token_count_after_inserting_image_tokens = input_ids.numel()

    # calculate how many tokens to generate based on max_new_tokens and model's upper bound (block_size)
    max_seq_length = min(
        prompt_token_count_after_inserting_image_tokens + max_new_tokens,
        model.config.block_size,
    )
    new_tokens = max_seq_length - prompt_token_count_after_inserting_image_tokens

    # full prompt+output will be stored in seq
    seq = torch.empty(max_seq_length, dtype=input_ids.dtype, device=device)
    seq[:original_prompt_token_count] = input_ids.view(-1)

    # setup model caches
    with torch.device(device):
        if cache_size is None:
            cache_size = max_seq_length
        assert (
            cache_size >= max_seq_length
        ), "need cache_size to be greater than max_new_tokens + size-of-prompt"
        model.setup_caches(
            max_batch_size=1,
            max_seq_length=cache_size,
            linear_causal_mask=linear_causal_mask,
            prompt_length=prompt_token_count_after_inserting_image_tokens,
        )

    input_pos = torch.arange(
        0,
        prompt_token_count_after_inserting_image_tokens,
        device=device,
        dtype=torch.int,
    )

    # execute prefill
    next_token = prefill(
        model, input_ids, input_pos, input_embeds, **sampling_kwargs
    ).clone()
    seq[original_prompt_token_count] = next_token
    input_pos = torch.tensor(
        [prompt_token_count_after_inserting_image_tokens],
        device=device,
        dtype=torch.int,
    )
    # execute token generation
    generated_tokens, _ = decode_n_tokens(
        model,
        next_token.view(1, -1),
        input_pos,
        new_tokens - 1,
        callback=callback,
        **sampling_kwargs,
    )

    seq = torch.cat((seq[: original_prompt_token_count + 1], *generated_tokens))

    return seq


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision):
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]

    with torch.device("meta"):
        model = Aria(ModelArgs())
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=precision)

    return model.eval()


def recommended_inductor_config_setter():
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch.set_float32_matmul_precision("high")


def load_model_and_tokenizer(checkpoint_path, device, precision):
    print(f"Using device={device}")
    print("Loading model ...")
    t0 = time.time()

    model = _load_model(checkpoint_path, device, precision)
    device_sync(device=device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer_path = checkpoint_path.parent
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=False, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)

    return model, tokenizer, processor


def prepare_image_inputs(image_path, prompt, processor, precision):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": prompt, "type": "text"},
            ],
        }
    ]

    image = Image.open(
        requests.get(image_path, stream=True).raw
        if image_path.startswith(("http://", "https://"))
        else image_path
    )

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    del inputs["attention_mask"]
    inputs["pixel_values"] = inputs["pixel_values"].to(precision)
    return inputs


def prepare_text_inputs(prompt, tokenizer):
    messages = [{"role": "user", "content": [{"text": prompt, "type": "text"}]}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "input_ids": tokenizer(text, return_tensors="pt").input_ids.to(torch.int32),
        "pixel_values": None,
        "pixel_mask": None,
    }


def setup_model_compilation(
    model, compile, compile_prefill, apply_regional_compilation
):
    if apply_regional_compilation:
        print("Compiling Model")
        for layer in model.llm.layers:
            layer.compile()

    if compile:
        print("Compiling Model")
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


def process_generation(
    model,
    inputs,
    tokenizer,
    i,
    num_samples,
    profile,
    device,
    stop_strings=None,
    **generation_kwargs,
):
    t0 = time.perf_counter()

    # Encode stop strings once at the start
    stop_sequences = None
    if stop_strings:
        stop_sequences = [
            torch.tensor(tokenizer.encode(stop), dtype=torch.int, device=device)
            for stop in stop_strings
        ]

    prof = (
        torch.profiler.profile(with_stack=True)
        if i == num_samples - 1 and profile
        else contextlib.nullcontext()
    )

    with prof:

        def callback(new_tokens):
            if stop_sequences:
                generated = torch.cat(new_tokens)
                return any(
                    generated.size(0) >= stop_seq.size(0)
                    and torch.equal(generated[-stop_seq.size(0) :], stop_seq)
                    for stop_seq in stop_sequences
                )
            return False

        output = generate(model, **inputs, callback=callback, **generation_kwargs)

    if i == -1:
        print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
        return None

    if hasattr(prof, "export_chrome_trace"):
        prof.export_chrome_trace(f"{profile}.json")

    device_sync(device=device)
    generation_time = time.perf_counter() - t0

    print(tokenizer.decode(output))
    return output, generation_time


def print_metrics(tokens_per_sec, model_size):
    print("==========")
    tokpersec = torch.mean(torch.tensor(tokens_per_sec)).item()
    bandwidth = model_size * tokpersec
    mem = torch.cuda.max_memory_reserved() / 1e9
    print(f"Average tokens/sec: {tokpersec:.2f}")
    print(f"Average Bandwidth: {bandwidth:.02f} GB/s")
    print(f"Peak Memory Usage: {mem:.02f} GB")
    print(f"Model Size: {model_size:.02f} GB")


def main(
    checkpoint_path,
    prompt: str = "Hello, my name is",
    image_path: str = None,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    cache_size: Optional[int] = None,
    linear_causal_mask: bool = False,
    compile: bool = True,
    compile_prefill: bool = False,
    apply_regional_compilation: bool = False,
    profile: Optional[Path] = None,
    memory_profile: Optional[Path] = None,
    device=default_device,
    precision=torch.bfloat16,
    stop_strings: Optional[list] = None,
) -> None:
    recommended_inductor_config_setter()
    assert checkpoint_path.is_file(), checkpoint_path

    model, tokenizer, processor = load_model_and_tokenizer(
        checkpoint_path, device, precision
    )

    inputs = (
        prepare_image_inputs(image_path, prompt, processor, precision)
        if image_path
        else prepare_text_inputs(prompt, tokenizer)
    )
    inputs = {k: v.to(device) if v is not None else v for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].size(1)
    torch.manual_seed(1234)
    model_size = get_model_size_in_bytes(model, ignore_embeddings=True) / 1e9

    setup_model_compilation(model, compile, compile_prefill, apply_regional_compilation)

    if memory_profile:
        torch.cuda.memory._record_memory_history(
            True, trace_alloc_max_entries=250000, trace_alloc_record_context=True
        )

    tokens_per_sec = []
    start = -1 if compile or apply_regional_compilation else 0

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "cache_size": cache_size,
        "linear_causal_mask": linear_causal_mask,
        "stop_strings": stop_strings,
    }

    for i in range(start, num_samples):
        if i == 0:
            torch.cuda.reset_peak_memory_stats()
        device_sync(device=device)

        result = process_generation(
            model,
            inputs,
            tokenizer,
            i,
            num_samples,
            profile,
            device,
            **generation_kwargs,
        )
        if result is None:
            continue

        output, generation_time = result
        tokens_generated = output.size(0) - prompt_length
        print(f"Tokens generated: {tokens_generated}")
        tokens_sec = tokens_generated / generation_time
        tokens_per_sec.append(tokens_sec)

        print(
            f"Time for inference {i + 1}: {generation_time:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec:.02f} GB/s")

        if memory_profile and i == 0:
            snapshot = torch.cuda.memory._snapshot()
            with open(f"{memory_profile}.pickle", "wb") as f:
                from pickle import dump

                dump(snapshot, f)
            print(
                f"\nmemory profile {memory_profile}.pickle saved, to convert that to a usable file, use",
                "python pytorch/torch/cuda/_memory_viz.py trace_plot <pickle file> -o <desired output name>.html",
            )
            break

    print_metrics(tokens_per_sec, model_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain what is the meaning of life",
        help="Input prompt.",
    )
    parser.add_argument("--image_path", type=str, default=None, help="Image path.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=None,
        help="Force size of cache to be a certain number of tokens, if not set, will use max_new_tokens+prompt_size",
    )
    parser.add_argument(
        "--linear_causal_mask",
        action="store_true",
        help="Whether to use the memory efficient, but slightly less fast, linear causal mask (important for long context lengths)",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile_prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    parser.add_argument(
        "--apply_regional_compilation",
        action="store_true",
        help="Whether to apply regional compilation to the layers of the model",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--memory_profile", type=Path, default=None, help="filename for memory profile."
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )
    parser.add_argument(
        "--precision",
        type=lambda x: getattr(torch, x.split(".")[-1]),
        default=torch.bfloat16,
        help="dtype precision to use",
    )
    parser.add_argument(
        "--stop_strings",
        type=str,
        nargs="+",
        default=["<|im_end|>"],
        help="List of strings that will stop generation when encountered at the end",
    )

    args = parser.parse_args()
    main(
        args.checkpoint_path,
        args.prompt,
        args.image_path,
        args.num_samples,
        args.max_new_tokens,
        args.top_k,
        args.temperature,
        args.cache_size,
        args.linear_causal_mask,
        args.compile,
        args.compile_prefill,
        args.apply_regional_compilation,
        args.profile,
        args.memory_profile,
        args.device,
        args.precision,
        args.stop_strings,
    )
