import argparse
import re

import matplotlib.pyplot as plt
import torch
from peft import PeftConfig, PeftModel
from PIL import Image, ImageDraw

from aria.lora.layers import GroupedGemmLoraLayer
from aria.model import AriaForConditionalGeneration, AriaProcessor, GroupedGEMM


def parse_arguments():
    parser = argparse.ArgumentParser(description="Aria Inference Script on RefCOCO")
    parser.add_argument(
        "--base_model_path", required=True, help="Path to the base model"
    )
    parser.add_argument("--peft_model_path", help="Path to the PEFT model (optional)")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument(
        "--max_image_size",
        type=int,
        help="Maximum size of the image to be processed",
        default=980,
    )
    parser.add_argument(
        "--vis_bbox",
        action="store_true",
        help="Whether to draw the bounding box on the image",
    )
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


def prepare_input(image_path, prompt, processor: AriaProcessor, max_image_size):
    image = Image.open(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": prompt, "type": "text"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        max_image_size=max_image_size,
    )

    return inputs


def inference(
    image_path,
    prompt,
    model: AriaForConditionalGeneration,
    processor: AriaProcessor,
    max_image_size,
):
    inputs = prepare_input(image_path, prompt, processor, max_image_size)
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
        result = processor.batch_decode(output, skip_special_tokens=True)
        prompt_len = len(prompt)
        result = result[0][prompt_len:].replace("<|im_end|>", "")

    return result


def parse_bbox(model_output, img_wh):
    PATTERN = re.compile(r"\((.*?)\),\((.*?)\)")
    predict_bbox = re.findall(PATTERN, model_output)

    try:
        if "," not in predict_bbox[0][0] or "," not in predict_bbox[0][1]:
            predict_bbox = (0.0, 0.0, 0.0, 0.0)
        else:
            x1, y1 = [float(tmp) for tmp in predict_bbox[0][0].split(",")]
            x2, y2 = [float(tmp) for tmp in predict_bbox[0][1].split(",")]
            predict_bbox = (x1, y1, x2, y2)
    except:
        predict_bbox = (0.0, 0.0, 0.0, 0.0)

    img_w, img_h = img_wh
    return (
        int(predict_bbox[0] / 999 * img_w),
        int(predict_bbox[1] / 999 * img_h),
        int(predict_bbox[2] / 999 * img_w),
        int(predict_bbox[3] / 999 * img_h),
    )


def main():
    args = parse_arguments()
    # if the tokenizer is not put in the same folder as the model, we need to specify the tokenizer path
    processor = AriaProcessor.from_pretrained(
        args.base_model_path, tokenizer_path=args.tokenizer_path
    )
    model = load_model(args.base_model_path, args.peft_model_path)

    image_path = "./datasets/refcoco_sub30k/images/COCO_train2014_000000580957.jpg"
    prompt = "Given the image, provide the bounding box coordinate of the region this sentence describes:\n{}"
    reference_object = "white dish in the top right corner"
    result = inference(
        image_path,
        prompt.format(reference_object),
        model,
        processor,
        args.max_image_size,
    )
    print(f"Model Output: {result}")

    image = Image.open(image_path).convert("RGB")
    bbox = parse_bbox(result, image.size)
    print(f"Parsed Bbox: {bbox}")

    if args.vis_bbox:
        predicted_image = image.copy()
        draw = ImageDraw.Draw(predicted_image)

        draw.rectangle(bbox, outline="red", width=3)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("original image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_image)
        plt.title(reference_object)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig("./assets/refcoco_example1.png")
        # plt.show()


if __name__ == "__main__":
    main()
