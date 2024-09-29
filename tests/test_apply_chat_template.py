import pytest
from transformers import AutoTokenizer

from aria.data import apply_chat_template_and_tokenize


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "rhymes-ai/Aria",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def test_apply_chat_template_single_user_message(tokenizer):
    messages = [
        {
            "content": [
                {"text": "Who wrote this book?\n", "type": "text"},
                {"text": None, "type": "image"},
            ],
            "role": "user",
        }
    ]
    expected_output = "<|im_start|>user\nWho wrote this book?\n<fim_prefix><|img|><fim_suffix><|im_end|>\n"
    res = apply_chat_template_and_tokenize([messages], tokenizer=tokenizer)
    input_ids = res["input_ids"]
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    assert input_str == expected_output

    labels = res["labels"]
    assert (labels == -100).sum() == input_ids.numel()


def test_apply_chat_template_single_assistant_message(tokenizer):
    messages = [
        {
            "content": [{"text": "Sylvie Covey", "type": "text"}],
            "role": "assistant",
        }
    ]
    expected_output = "<|im_start|>assistant\nSylvie Covey<|im_end|>\n"
    res = apply_chat_template_and_tokenize([messages], tokenizer=tokenizer)
    input_ids = res["input_ids"]
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    assert input_str == expected_output


def test_apply_chat_template_multiple_messages(tokenizer):
    messages = [
        {
            "content": [
                {"text": "Who wrote this book?\n", "type": "text"},
                {"text": None, "type": "image"},
            ],
            "role": "user",
        },
        {
            "content": [{"index": None, "text": "Sylvie Covey", "type": "text"}],
            "role": "assistant",
        },
    ]
    expected_output = "<|im_start|>user\nWho wrote this book?\n<fim_prefix><|img|><fim_suffix><|im_end|>\n<|im_start|>assistant\nSylvie Covey<|im_end|>\n"
    res = apply_chat_template_and_tokenize([messages], tokenizer=tokenizer)
    input_ids = res["input_ids"]
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    assert input_str == expected_output


def test_apply_chat_template_invalid_content_type(tokenizer):
    messages = [
        {
            "content": [
                {"text": "Who wrote this book?\n", "type": "text"},
                {"text": None, "type": "invalid"},
            ],
            "role": "user",
        }
    ]
    with pytest.raises(ValueError) as excinfo:
        apply_chat_template_and_tokenize([messages], tokenizer=tokenizer)
    assert "Unknown content type invalid in message" in str(excinfo.value)


def test_apply_chat_template_multi_round_messages(tokenizer):
    messages = [
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
        {
            "content": [
                {
                    "text": "What is the title of this book?",
                    "type": "text",
                }
            ],
            "role": "user",
        },
        {
            "content": [
                {
                    "text": "Modern Printmaking: A Guide to Traditional and Digital Techniques",
                    "type": "text",
                }
            ],
            "role": "assistant",
        },
    ]
    expected_output = "<|im_start|>user\nWho wrote this book?\n<fim_prefix><|img|><fim_suffix><|im_end|>\n<|im_start|>assistant\nSylvie Covey<|im_end|>\n<|im_start|>user\nWhat is the title of this book?<|im_end|>\n<|im_start|>assistant\nModern Printmaking: A Guide to Traditional and Digital Techniques<|im_end|>\n"
    res = apply_chat_template_and_tokenize([messages], tokenizer=tokenizer)
    input_ids = res["input_ids"]
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    assert input_str == expected_output


def test_apply_chat_template_batch_messages(tokenizer):
    messages_batch = [
        [
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
        ],
        [
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
            {
                "content": [
                    {
                        "text": "What is the title of this book?",
                        "type": "text",
                    }
                ],
                "role": "user",
            },
            {
                "content": [
                    {
                        "text": "Modern Printmaking: A Guide to Traditional and Digital Techniques",
                        "type": "text",
                    }
                ],
                "role": "assistant",
            },
        ],
    ]

    res = apply_chat_template_and_tokenize(messages_batch, tokenizer=tokenizer)
    input_ids = res["input_ids"]

    expected_output = [
        "<|im_start|>user\nWho wrote this book?\n<fim_prefix><|img|><fim_suffix><|im_end|>\n<|im_start|>assistant\nSylvie Covey<|im_end|>\n",
        "<|im_start|>user\nWho wrote this book?\n<fim_prefix><|img|><fim_suffix><|im_end|>\n<|im_start|>assistant\nSylvie Covey<|im_end|>\n<|im_start|>user\nWhat is the title of this book?<|im_end|>\n<|im_start|>assistant\nModern Printmaking: A Guide to Traditional and Digital Techniques<|im_end|>\n",
    ]
    assert (
        tokenizer.batch_decode(input_ids, skip_special_tokens=True) == expected_output
    )
