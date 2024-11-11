import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from aria.model.processing_aria import AriaProcessor
from aria.model.vision_processor import AriaVisionProcessor


@pytest.fixture
def processor():
    tokenizer = AutoTokenizer.from_pretrained("rhymes-ai/Aria")
    image_processor = AriaVisionProcessor(max_image_size=490)
    return AriaProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token="<|img|>",
        chat_template=tokenizer.chat_template,
    )


@pytest.fixture
def sample_image():
    return Image.fromarray(np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8))


@pytest.fixture
def sample_messages():
    return [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": "describe the image", "type": "text"},
            ],
        }
    ]


def test_apply_chat_template(processor, sample_messages):
    text = processor.apply_chat_template(sample_messages, add_generation_prompt=True)

    assert (
        text
        == "<|im_start|>user\n<fim_prefix><|img|><fim_suffix>describe the image<|im_end|>\n<|im_start|>assistant\n"
    )

    text = processor.apply_chat_template(sample_messages, add_generation_prompt=False)
    assert (
        text
        == "<|im_start|>user\n<fim_prefix><|img|><fim_suffix>describe the image<|im_end|>\n"
    )


def test_chat_template_with_multiple_messages(processor):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": "What's in this image?", "type": "text"},
            ],
        },
        {
            "role": "assistant",
            "content": "This is a beautiful landscape.",
        },
        {
            "role": "user",
            "content": [
                {"text": "Can you describe it in more detail?", "type": "text"},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    assert (
        text
        == "<|im_start|>user\n<fim_prefix><|img|><fim_suffix>What's in this image?<|im_end|>\n<|im_start|>assistant\nThis is a beautiful landscape.<|im_end|>\n<|im_start|>user\nCan you describe it in more detail?<|im_end|>\n<|im_start|>assistant\n"
    )


def test_end_to_end_processing_980(processor, sample_messages, sample_image):
    text = processor.apply_chat_template(sample_messages, add_generation_prompt=True)
    inputs, prompts = processor(
        text=text,
        images=[sample_image],
        return_tensors="pt",
        max_image_size=980,
        return_final_prompts=True,
    )

    # Verify the output contains all necessary keys
    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "pixel_values" in inputs

    # Check shapes
    assert len(inputs["input_ids"].shape) == 2
    assert len(inputs["attention_mask"].shape) == 2
    assert len(inputs["pixel_values"].shape) == 4

    # Check device and dtype
    assert inputs["input_ids"].device.type == "cpu"
    assert inputs["pixel_values"].dtype == torch.float32

    expected_prompt = "<|im_start|>user\n<fim_prefix><|img|><fim_suffix>describe the image<|im_end|>\n<|im_start|>assistant\n"
    expected_prompt = expected_prompt.replace("<|img|>", "<|img|>" * 256)

    assert prompts[0] == expected_prompt


def test_end_to_end_processing_490(processor, sample_messages, sample_image):
    text = processor.apply_chat_template(sample_messages, add_generation_prompt=True)
    inputs, prompts = processor(
        text=text,
        images=[sample_image],
        return_tensors="pt",
        max_image_size=490,
        return_final_prompts=True,
    )

    expected_prompt = "<|im_start|>user\n<fim_prefix><|img|><fim_suffix>describe the image<|im_end|>\n<|im_start|>assistant\n"
    expected_prompt = expected_prompt.replace("<|img|>", "<|img|>" * 128)

    assert prompts[0] == expected_prompt


def test_end_to_end_processing_invalid_max_image_size(
    processor, sample_messages, sample_image
):
    text = processor.apply_chat_template(sample_messages, add_generation_prompt=True)
    with pytest.raises(ValueError):
        processor(
            text=text, images=[sample_image], return_tensors="pt", max_image_size=1000
        )


def test_multiple_images_in_conversation(processor, sample_image):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": None, "type": "image"},
                {"text": None, "type": "image"},
                {"text": "Compare the two images.", "type": "text"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs, prompts = processor(
        text=text,
        images=[sample_image, sample_image],  # Two images
        return_tensors="pt",
        max_image_size=980,
        return_final_prompts=True,
    )

    assert "pixel_values" in inputs
    assert inputs["pixel_values"].shape[0] == 2  # Batch size should be 2 for two images

    expected_prompt = "<|im_start|>user\n<fim_prefix><|img|><fim_suffix><fim_prefix><|img|><fim_suffix>Compare the two images.<|im_end|>\n<|im_start|>assistant\n"
    expected_prompt = expected_prompt.replace("<|img|>", "<|img|>" * 256)

    assert prompts[0] == expected_prompt


def test_split_image(processor, sample_messages, sample_image):
    text = processor.apply_chat_template(sample_messages, add_generation_prompt=True)
    inputs, prompts = processor(
        text=text,
        images=[sample_image],
        return_tensors="pt",
        max_image_size=490,
        split_image=True,
        return_final_prompts=True,
    )

    assert inputs["pixel_values"].shape == (5, 3, 490, 490)
    assert inputs["pixel_mask"].shape == (5, 490, 490)

    expected_prompt = "<|im_start|>user\n<fim_prefix><|img|><|img|><|img|><|img|><|img|><fim_suffix>describe the image<|im_end|>\n<|im_start|>assistant\n"
    expected_prompt = expected_prompt.replace("<|img|>", "<|img|>" * 128)

    assert prompts[0] == expected_prompt
