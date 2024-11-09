import os
import warnings
from typing import Dict, Iterable, List

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from datasets.features import Features, Sequence, Value


def apply_chat_template_and_tokenize(
    messages_batch: List[List[Dict]], tokenizer, num_image_crop: Iterable[torch.Tensor] = iter([]), max_length: int = 1024
) -> Dict[str, torch.Tensor]:
    IGNORE_TOKEN_ID = -100
    im_start_tokens = tokenizer("<|im_start|>").input_ids
    user_tokens = tokenizer("user").input_ids
    assistant_tokens = tokenizer("assistant").input_ids
    im_end_tokens = tokenizer("<|im_end|>").input_ids
    nl_tokens = tokenizer("\n").input_ids

    def process_content(content) -> str:
        if content["type"] == "text":
            return content["text"]
        elif content["type"] == "image":
            return "<fim_prefix>" + "<|img|>" * next(num_image_crop) + "<fim_suffix>"
        else:
            raise ValueError(f"Unknown content type {content['type']} in message")

    def tokenize_message(role: str, text: str) -> List[int]:
        tokens = (
            im_start_tokens
            + (user_tokens if role == "user" else assistant_tokens)
            + nl_tokens
            + tokenizer(text).input_ids
            + im_end_tokens
            + nl_tokens
        )
        return tokens

    def create_target(role: str, input_id: List[int]) -> List[int]:
        if role == "user":
            return [IGNORE_TOKEN_ID] * len(input_id)
        elif role == "assistant":
            prefix_length = len(im_start_tokens + assistant_tokens + nl_tokens)
            return [IGNORE_TOKEN_ID] * prefix_length + input_id[prefix_length:]
        else:
            raise ValueError(f"Unknown role: {role}")

    input_ids, targets = [], []
    for messages in messages_batch:
        input_id, target = [], []
        for message in messages:
            role = message["role"]
            text = "".join(process_content(content) for content in message["content"])

            _input_id = tokenize_message(role, text)
            input_id.extend(_input_id)
            target.extend(create_target(role, _input_id))

        assert len(input_id) == len(
            target
        ), f"input_ids and target should have the same length, {len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)

    # Find the maximum length in the batch
    max_batch_len = min(max(len(ids) for ids in input_ids), max_length)

    # Pad or truncate to max_batch_len
    for i in range(len(input_ids)):
        pad_length = max_batch_len - len(input_ids[i])
        if pad_length > 0:
            input_ids[i] += [tokenizer.pad_token_id] * pad_length
            targets[i] += [IGNORE_TOKEN_ID] * pad_length
        else:
            input_ids[i] = input_ids[i][:max_batch_len]
            targets[i] = targets[i][:max_batch_len]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(targets, dtype=torch.long),
        "attention_mask": torch.tensor(input_ids, dtype=torch.long).ne(
            tokenizer.pad_token_id
        ),
    }


def apply_chat_template(messages: List[Dict], add_generation_prompt: bool = False) -> str:
    """
    Format chat messages between the user and the assistant for tokenization.
    """
    res = ""
    for message in messages:
        role_marker = "<|im_start|>user\n" if message["role"] == "user" else "<|im_start|>assistant\n"
        res += role_marker
        for content in message["content"]:
            if content["type"] == "text":
                res += content["text"]
            elif content["type"] == "image":
                res += "<fim_prefix><|img|><fim_suffix>"
            else:
                raise ValueError(f"Unknown content type {content['type']} in message")

        res += "<|im_end|>\n"

    if add_generation_prompt:
        res += "<|im_start|>assistant\n"
    return res


def load_local_dataset(path: str, num_proc: int = 8) -> DatasetDict:
    """
    Load a local dataset from the specified path.
    """
    train_file = os.path.join(path, "train.jsonl")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"train.jsonl not found in {path}")

    def convert_to_absolute_path(item):
        if item["images"] and item["video"]:
            raise ValueError("Simultaneous input of images and video is not supported.")
        if item["images"] is not None:
            item["images"] = [os.path.join(path, image) for image in item["images"]]
        if item["video"] is not None:
            if item["video"].get("num_frames", 0) <= 0:
                warnings.warn("`num_frames` is set to 8 by default due to invalid or missing value.")
                item["video"]["num_frames"] = 8
            item["video"]["path"] = os.path.join(path, item["video"]["path"])
        return item

    features = {
        "messages": [
            {
                "content": [{"text": Value(dtype="string"), "type": Value(dtype="string")}],
                "role": Value(dtype="string"),
            }
        ],
        "images": Sequence(feature=Value(dtype="string")),
        "video": {
            "path": Value(dtype="string"),
            "num_frames": Value(dtype="int64"),
        },
    }

    ds = DatasetDict()
    ds["train"] = load_dataset("json", data_files=train_file, features=Features(features), split="train")
    test_file = os.path.join(path, "test.jsonl")
    if os.path.exists(test_file):
        ds["test"] = load_dataset("json", data_files=test_file, features=Features(features), split="test")

    ds = ds.map(convert_to_absolute_path, num_proc=num_proc)
    return ds


def mix_datasets(dataset_config: Dict[str, float], columns_to_keep: List[str] = ["images", "messages", "video"]) -> DatasetDict:
    """
    Mix datasets based on configuration with sampling fraction.
    """
    raw_train_datasets, raw_test_datasets = [], []
    for dataset_path, frac in dataset_config.items():
        ds = load_local_dataset(dataset_path)
        frac = float(frac)

        if "train" in ds:
            train_ds = ds["train"].select(range(int(frac * len(ds["train"]))))
            train_ds = train_ds.remove_columns([col for col in ds["train"].column_names if col not in columns_to_keep])
            raw_train_datasets.append(train_ds)

        if "test" in ds:
            test_ds = ds["test"].remove_columns([col for col in ds["test"].column_names if col not in columns_to_keep])
            raw_test_datasets.append(test_ds)

    return DatasetDict(
        train=concatenate_datasets(raw_train_datasets).shuffle(seed=42),
        test=concatenate_datasets(raw_test_datasets) if raw_test_datasets else None,
    )
