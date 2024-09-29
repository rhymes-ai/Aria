import argparse
import os
import time

import huggingface_hub

parser = argparse.ArgumentParser(description="Huggingface Dataset Download")
parser.add_argument("--hf_root", required=True, type=str)
parser.add_argument("--save_root", required=True, type=str)
args = parser.parse_args()
os.makedirs(args.save_root, exist_ok=True)


def download():
    try:
        huggingface_hub.snapshot_download(
            args.hf_root,
            local_dir=args.save_root,
            resume_download=True,
            max_workers=8,
            repo_type="dataset",
        )
        return True
    except:
        print("Caught an exception! Retrying...")
        return False


while True:
    result = download()
    if result:
        print("success")
        break  # Exit the loop if the function ran successfully
    time.sleep(1)  # Wait for 1 second before retrying
