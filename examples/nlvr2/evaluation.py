import argparse
import json
import os

import torch
from peft import PeftConfig, PeftModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aria.lora.layers import GroupedGemmLoraLayer
from aria.model import AriaForConditionalGeneration, AriaProcessor, GroupedGEMM

# Add command-line argument parsing
parser = argparse.ArgumentParser(description="NLVR2 Evaluation")
parser.add_argument(
    "--base_model_path", type=str, required=True, help="Path to the base model"
)
parser.add_argument(
    "--peft_model_path", type=str, default=None, help="Path to the PEFT model"
)
parser.add_argument(
    "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
)
parser.add_argument(
    "--save_root", type=str, required=True, help="The root path of output."
)
parser.add_argument("--image_size", type=int, default=980, help="Maximum image size")
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size for evaluation"
)
parser.add_argument(
    "--num_workers", type=int, default=16, help="Number of workers for data loading"
)

args = parser.parse_args()
os.makedirs(args.save_root, exist_ok=True)


class NLVR2ValDataset(Dataset):
    def __init__(self):
        super().__init__()
        annos = "datasets/nlvr2/val.jsonl"
        vis_root = "datasets/nlvr2"

        self.dataset = []
        lines = open(annos).readlines()
        for line in tqdm(lines):
            anno = json.loads(line.strip())
            anno["images"] = [
                os.path.join(vis_root, im_path) for im_path in anno["images"]
            ]
            self.dataset.append(anno)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def load_model_and_tokenizer(args):
    processor = AriaProcessor.from_pretrained(
        args.base_model_path, tokenizer_path=args.tokenizer_path
    )
    processor.tokenizer.padding_side = "left"
    tokenizer = processor.tokenizer

    model = AriaForConditionalGeneration.from_pretrained(
        args.base_model_path, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()
    model.pad_token_id = tokenizer.pad_token_id

    if args.peft_model_path:
        peft_config = PeftConfig.from_pretrained(args.peft_model_path)
        custom_module_mapping = {GroupedGEMM: GroupedGemmLoraLayer}
        peft_config._register_custom_module(custom_module_mapping)
        model = PeftModel.from_pretrained(
            model,
            args.peft_model_path,
            config=peft_config,
            is_trainable=False,
            autocast_adapter_dtype=False,
        )

    return model, tokenizer, processor


def process_batch(model, tokenizer, inputs, original_batch, prompts):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            stop_strings=["<|im_end|>"],
            tokenizer=tokenizer,
        )

    for i, prompt in enumerate(prompts):
        prompt_len = len(inputs["input_ids"][i])
        output_text = tokenizer.decode(
            output[i][prompt_len:], skip_special_tokens=True
        ).replace("<|im_end|>", "")
        original_batch[i]["pred"] = output_text

    return original_batch


def collate_fn(batch, processor, tokenizer):
    messages = []
    images = []
    for item in batch:
        images.extend(
            [Image.open(im_path).convert("RGB") for im_path in item["images"]]
        )
        messages.append(item["messages"])

    texts = [
        processor.apply_chat_template(msg, add_generation_prompt=True)
        for msg in messages
    ]
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="longest",
        max_image_size=args.image_size,
    )
    return inputs, batch, texts


def main():
    model, tokenizer, processor = load_model_and_tokenizer(args)

    dataset = NLVR2ValDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor, tokenizer),
    )

    results = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs, original_batch, prompts = batch
        results.extend(process_batch(model, tokenizer, inputs, original_batch, prompts))

    with open(f"{args.save_root}/nlvr2-dev_result.json", "w") as fo:
        json.dump(results, fo, indent=4, ensure_ascii=False)
    return results


def parse_pred_ans(pred_ans):
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    elif len(pred_ans) == 1:
        if pred_ans == "y":
            pred_label = "yes"
        elif pred_ans == "n":
            pred_label = "no"
        else:
            pred_label = "other"
    else:
        prefix_pred_ans = pred_ans[:4]
        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def evaluate(result):

    correct = total_cnt = 0
    for output in result:
        pred = output["pred"]
        pred_ans = parse_pred_ans(pred)
        gt = output["gt"]
        gt_ans = gt.lower().strip().replace(".", "")
        score = 1.0 if pred_ans == gt_ans else 0.0
        correct += score

        total_cnt += 1

    acc = correct / total_cnt

    if len(result) == 0:
        return {"acc": 0}

    return {"acc": acc * 100}


def get_score(output):

    print(evaluate(output))


if __name__ == "__main__":
    output = main()
    get_score(output)
