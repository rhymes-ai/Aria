import time
from pathlib import Path
from statistics import mean, stdev

import requests
from generate import GenerationConfig, Generator, ModelConfig
from PIL import Image


def run_benchmark(
    generator: Generator, messages: list[dict], image: Image.Image, num_runs: int = 5
):
    """Run multimodal generation benchmark."""
    # Warmup runs
    for _ in range(2):
        generator.generate(messages, image)

    # Benchmark runs
    latencies = []
    token_counts = []

    for i in range(num_runs):
        print(f"Running benchmark {i+1}/{num_runs}")
        start_time = time.perf_counter()
        output = generator.generate(messages, image, detokenize=False)
        end_time = time.perf_counter()

        latencies.append(end_time - start_time)
        token_counts.append(len(output))

    results = {
        "mean_latency": mean(latencies),
        "std_latency": stdev(latencies) if len(latencies) > 1 else 0,
        "mean_tokens": mean(token_counts),
        "std_tokens": stdev(token_counts) if len(token_counts) > 1 else 0,
        "tokens_per_second": mean(token_counts) / mean(latencies),
    }

    print("\nBenchmark Results:")
    print(
        f"Average Latency: {results['mean_latency']:.2f}s (±{results['std_latency']:.2f}s)"
    )
    print(
        f"Average Tokens: {results['mean_tokens']:.1f} (±{results['std_tokens']:.1f})"
    )
    print(f"Tokens per Second: {results['tokens_per_second']:.1f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile", action="store_true", help="Enable model compilation"
    )
    args = parser.parse_args()

    model_config = ModelConfig(
        checkpoint_path=Path("checkpoints/rhymes-ai/Aria/model.pth"),
        compile=args.compile,
    )
    generation_config = GenerationConfig(
        max_new_tokens=200, top_k=200, temperature=0.8, stop_strings=None
    )
    generator = Generator(model_config, generation_config)

    # Load test image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    image = Image.open(requests.get(image_url, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": "Describe this image.", "type": "text"},
            ],
        },
    ]

    run_benchmark(generator, messages, image)
