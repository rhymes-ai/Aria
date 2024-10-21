import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from tqdm import trange

from aria.lora.layers import GroupedGemmLoraLayer
from aria.model import AriaForConditionalGeneration, AriaProcessor, GroupedGEMM


def parse_arguments():
    parser = argparse.ArgumentParser(description="Aria Inference Script on HumanEval")
    parser.add_argument(
        "--base_model_path", required=True, help="Path to the base model"
    )
    parser.add_argument("--peft_model_path", help="Path to the PEFT model (optional)")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--save_root", required=True, help="Path to the save_dir")
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


def prepare_input(prompt, processor: AriaProcessor):

    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt, "type": "text"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=text,
        return_tensors="pt",
    )

    return inputs


def build_aria_instruction(question: str):
    return (
        """
Please complete the python function below. The final complete version of your function must be returned within a code block. Here is the unfinished function:\n ```python\n
{}
""".strip().format(
            question.strip()
        )
        + "\n\n"
    )


def inference(
    question,
    model: AriaForConditionalGeneration,
    processor: AriaProcessor,
):
    prompt = build_aria_instruction(question)
    inputs = prepare_input(prompt, processor)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
            do_sample=False,
        )

    for i in range(inputs["input_ids"].shape[0]):
        prompt_len = len(inputs["input_ids"][i])
        output_text = processor.tokenizer.decode(
            output[i][prompt_len:], skip_special_tokens=True
        ).replace("<|im_end|>", "")

    return output_text


def extract_markdown_content(text):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def generate_problem_file(data, fp):
    ld = len(data)
    with open(fp, "w", encoding="utf-8") as f:
        for idx in trange(ld):
            item = data[idx]
            item["test"] += f"check({item['entry_point']})\n"
            f.write(f"{json.dumps(item)}\n")


def main():
    args = parse_arguments()
    os.makedirs(args.save_root, exist_ok=True)

    humaneval_data = load_dataset("openai/openai_humaneval")["test"]
    generate_problem_file(
        humaneval_data, os.path.join(args.save_root, "problem_file.jsonl")
    )

    # if the tokenizer is not put in the same folder as the model, we need to specify the tokenizer path
    processor = AriaProcessor.from_pretrained(
        args.base_model_path, tokenizer_path=args.tokenizer_path
    )
    model = load_model(args.base_model_path, args.peft_model_path)

    l_test = len(humaneval_data)

    with open(os.path.join(args.save_root, "human_eval_predictions.jsonl"), "w") as fo:
        for idx in trange(l_test):
            item = humaneval_data[idx]
            item["test"] += f"check({item['entry_point']})\n"
            out_item = {**item}

            result = inference(
                item["prompt"],
                model,
                processor,
            )
            extraced_result = extract_markdown_content(result)
            if extraced_result:
                out_item["generation"] = extraced_result[0]
            else:
                out_item["generation"] = result

            fo.write(f"{json.dumps(out_item)}\n")


if __name__ == "__main__":
    main()
