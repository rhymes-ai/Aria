import argparse
import json
import os
import re

import torch
from peft import PeftConfig, PeftModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops.boxes import box_area
from tqdm import tqdm

from aria.lora.layers import GroupedGemmLoraLayer
from aria.model import AriaForConditionalGeneration, AriaProcessor, GroupedGEMM

# Add command-line argument parsing
parser = argparse.ArgumentParser(description="RefCOCO Evaluation")
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


class RefCOCOValDataset(Dataset):
    def __init__(self):
        super().__init__()
        annos = "datasets/refcoco_sub30k/val.jsonl"
        vis_root = "datasets/refcoco_sub30k"

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
            max_new_tokens=500,
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

    dataset = RefCOCOValDataset()
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

    with open(f"{args.save_root}/refcoco-val_result.json", "w") as fo:
        json.dump(results, fo, indent=4, ensure_ascii=False)
    return results


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def evaluate(result):
    PATTERN = re.compile(r"\((.*?)\),\((.*?)\)")

    correct = total_cnt = 0
    for output in result:
        predict_bbox = re.findall(PATTERN, output["pred"])
        try:
            if "," not in predict_bbox[0][0] or "," not in predict_bbox[0][1]:
                predict_bbox = (0.0, 0.0, 0.0, 0.0)
            else:
                x1, y1 = [float(tmp) for tmp in predict_bbox[0][0].split(",")]
                x2, y2 = [float(tmp) for tmp in predict_bbox[0][1].split(",")]
                predict_bbox = (x1, y1, x2, y2)
        except:
            predict_bbox = (0.0, 0.0, 0.0, 0.0)
        target_bbox = torch.tensor(output["bbox"], dtype=torch.float32).view(-1, 4)
        predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4) / 999
        predict_bbox[:, 0::2] *= output["hw"][1]
        predict_bbox[:, 1::2] *= output["hw"][0]
        iou, _ = box_iou(predict_bbox, target_bbox)
        iou = iou.item()
        total_cnt += 1
        if iou >= 0.5:
            correct += 1

    precision_top1 = correct / total_cnt

    if len(result) == 0:
        return {"precision@1": 0}

    return {"precision@1": precision_top1 * 100}


def get_score(output):

    print(evaluate(output))


if __name__ == "__main__":
    output = main()
    get_score(output)
