import json
import os

from datasets import load_dataset
from tqdm import trange

save_dir = "./datasets/code_sft"
os.makedirs(save_dir, exist_ok=True)

dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")["train"]
l_ds = len(dataset)

with open(os.path.join(save_dir, "train.jsonl"), "w") as fo:
    for i in trange(l_ds):
        instruction, response = dataset[i]["instruction"], dataset[i]["response"]
        item = {
            "messages": [
                {
                    "content": [
                        {"text": instruction, "type": "text"},
                    ],
                    "role": "user",
                },
                {
                    "content": [
                        {"text": response, "type": "text"},
                    ],
                    "role": "assistant",
                },
            ],
        }
        fo.write(f"{json.dumps(item, ensure_ascii=False)}\n")
