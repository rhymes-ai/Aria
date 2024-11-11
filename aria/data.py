# Copyright 2024 Rhymes AI. All rights reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import warnings
from typing import Dict, Iterable, List

import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset
from datasets.features import Features, Sequence, Value


def apply_chat_template_and_tokenize(
    messages_batch: List[List[Dict]],
    tokenizer,
    num_image_crop: Iterable[torch.Tensor] = iter([]),
    max_length: int = 1024,
    max_image_size: int = 980,
):
    IGNORE_TOKEN_ID = -100
    im_start_tokens = tokenizer("<|im_start|>").input_ids
    user_tokens = tokenizer("user").input_ids
    assistant_tokens = tokenizer("assistant").input_ids
    im_end_tokens = tokenizer("<|im_end|>").input_ids
    nl_tokens = tokenizer("\n").input_ids

    def process_content(content):
        if content["type"] == "text":
            return content["text"]
        elif content["type"] == "image":
            return "<fim_prefix>" + "<|img|>" * next(num_image_crop) + "<fim_suffix>"
        else:
            raise ValueError(f"Unknown content type {content['type']} in message")

    def tokenize_message(role, text):
        return (
            im_start_tokens
            + (user_tokens if role == "user" else assistant_tokens)
            + nl_tokens
            + tokenizer(text).input_ids
            + im_end_tokens
            + nl_tokens
        )

    def create_target(role, input_id):
        if role == "user":
            return [IGNORE_TOKEN_ID] * len(input_id)
        elif role == "assistant":
            role_token_length = len(assistant_tokens)
            im_start_length = len(im_start_tokens)
            nl_length = len(nl_tokens)
            prefix_length = im_start_length + role_token_length + nl_length
            return [IGNORE_TOKEN_ID] * prefix_length + input_id[prefix_length:]
        else:
            raise ValueError(f"Unknown role: {role}")

    input_ids, targets = [], []
    for messages in messages_batch:
        input_id, target = [], []
        for message in messages:
            role = message["role"]
            text = "".join(process_content(content) for content in message["content"])

            if max_image_size == 490:
                num_image_tokens = 128
            elif max_image_size == 980:
                num_image_tokens = 256
            else:
                raise ValueError(
                    f"max_image_size must be either 490 or 980, got {max_image_size}"
                )
            text = text.replace("<|img|>", "<|img|>" * num_image_tokens)

            _input_id = tokenize_message(role, text)
            input_id.extend(_input_id)
            target.extend(create_target(role, _input_id))

        assert len(input_id) == len(
            target
        ), f"input_ids should have the same length as the target, {len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)

    # Find the maximum length in the batch
    max_batch_len = min(max(len(ids) for ids in input_ids), max_length)

    # Pad or truncate to max_batch_len
    for i in range(len(input_ids)):
        pad_length = max_batch_len - len(input_ids[i])
        if pad_length > 0:
            input_ids[i] = input_ids[i] + [tokenizer.pad_token_id] * pad_length
            targets[i] = targets[i] + [IGNORE_TOKEN_ID] * pad_length
        else:  # truncate
            input_ids[i] = input_ids[i][:max_batch_len]
            targets[i] = targets[i][:max_batch_len]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(targets, dtype=torch.long),
        "attention_mask": torch.tensor(input_ids, dtype=torch.long).ne(
            tokenizer.pad_token_id
        ),
    }


def apply_chat_template(messages: List[Dict], add_generation_prompt: bool = False):
    """
    Args:
        messages: List of messages, each message is a dictionary with the following keys:
            - role: str, either "user" or "assistant"
            - content: List of content items, each item is a dictionary with the following keys:
                - type: str, either "text" or "image"
                - text: str, the text content if type is "text"
    Returns:
        str: A formatted string representing the chat messages between the user and the assistant

    Example:
    >>> messages = [
            {
                "content": [
                    {"text": "Who wrote this book?\n", "type": "text"},
                    {"text": None, "type": "image"},
                ],
                "role": "user",
            },
            {
                "content": [{"text": "Sylvie Covey", "type": "text"}],
                "role": "assistant",
            },
        ]
    >>> apply_chat_template(messages)
    """
    res = ""
    for message in messages:
        if message["role"] == "user":
            res += "<|im_start|>user\n"
            for content in message["content"]:
                if content["type"] == "text":
                    res += content["text"]
                elif content["type"] == "image":
                    res += "<fim_prefix><|img|><fim_suffix>"
                else:
                    raise ValueError(
                        f"Unknown content type {content['type']} in user message"
                    )
            res += "<|im_end|>\n"
        elif message["role"] == "assistant":
            res += "<|im_start|>assistant\n"
            for content in message["content"]:
                if content["type"] == "text":
                    res += content["text"]
                else:
                    raise ValueError(
                        f"Unknown content type {content['type']} in assistant message"
                    )
            res += "<|im_end|>\n"
    if add_generation_prompt:
        res += "<|im_start|>assistant\n"
    return res


def load_local_dataset(path, num_proc=8):
    """
    Load a local dataset from the specified path and return it as a DatasetDict.

    Expected directory structure:
    - train.jsonl
    - test.jsonl
    - image_folder (folder containing image files)

    Structure of train.jsonl and test.jsonl files:
    Each item is a dictionary with the following format:
    - messages: List of message dictionaries, each with:
        - role: str, either "user" or "assistant"
        - content: List of content items, each with:
            - type: str, either "text" or "image"
            - text: str, the text content if type is "text"
    - images: List of image file paths relative to the respective JSONL file
    """
    if not os.path.exists(f"{path}/train.jsonl"):
        raise FileNotFoundError(f"train.jsonl not found in {path}")

    def convert_to_absolute_path(item):
        if item["images"] and item["video"]:
            assert False, "Simultaneous input of images and video is not supported."
        if item["images"] is not None:
            item["images"] = [f"{path}/{image}" for image in item["images"]]
        if item["video"] is not None:
            if (item["video"]["num_frames"] is None) or item["video"][
                "num_frames"
            ] <= 0:
                warnings.warn(
                    "`num_frames` is set to 8 by defauble because of a negative value or `None`."
                )
                item["video"]["num_frames"] = 8

            item["video"]["path"] = f"{path}/{item['video']['path']}"
        return item

    features = {
        "messages": [
            {
                "content": [
                    {
                        "text": Value(dtype="string", id=None),
                        "type": Value(dtype="string", id=None),
                    }
                ],
                "role": Value(dtype="string", id=None),
            }
        ],
        "images": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        "video": {
            "path": Value(dtype="string", id=None),
            "num_frames": Value(dtype="int64", id=None),
        },
    }

    ds = DatasetDict()
    train_ds = load_dataset(
        "json",
        data_files=f"{path}/train.jsonl",
        features=Features(features),
        split="train",
    )
    ds["train"] = train_ds
    if os.path.exists(f"{path}/test.jsonl"):
        test_ds = load_dataset(
            "json",
            data_files=f"{path}/test.jsonl",
            features=Features(features),
            split="train",
        )
        ds["test"] = test_ds

    ds = ds.map(convert_to_absolute_path, num_proc=num_proc)
    return ds


def mix_datasets(
    dataset_config: Dict,
    columns_to_keep: List[str] = ["images", "messages", "video"],
):
    raw_train_datasets = []
    raw_test_datasets = []
    for dataset_path, frac in dataset_config.items():
        frac = float(frac)
        print(dataset_path)
        ds = load_local_dataset(dataset_path)

        if "train" in ds:
            train_ds = ds["train"].remove_columns(
                [col for col in ds["train"].column_names if col not in columns_to_keep]
            )
            if frac <= 1:
                to_be_selected = range(int(frac * len(train_ds)))
            elif frac > 1:
                to_be_selected = list(range(len(train_ds))) * int(frac)
            train_ds = train_ds.select(to_be_selected)
            raw_train_datasets.append(train_ds)
        if "test" in ds:
            test_ds = ds["test"].remove_columns(
                [col for col in ds["test"].column_names if col not in columns_to_keep]
            )
            raw_test_datasets.append(test_ds)
    raw_dataset = DatasetDict()
    raw_dataset["train"] = concatenate_datasets(raw_train_datasets).shuffle(seed=42)
    if raw_test_datasets:
        raw_dataset["test"] = concatenate_datasets(raw_test_datasets)
    else:
        raw_dataset["test"] = None
    return raw_dataset
