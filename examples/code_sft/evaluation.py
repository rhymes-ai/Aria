from human_eval.evaluation import evaluate_functional_correctness
import json
import argparse

import os

parser = argparse.ArgumentParser()
parser.add_argument("--save_root", required=True, type=str, help="the result directory of humaneval")
args = parser.parse_args()

tmp_dir = os.path.join(args.save_root, "tmp")
os.makedirs(tmp_dir, exist_ok=True)

result = evaluate_functional_correctness(
    input_file=os.path.join(args.save_root, "human_eval_predictions.jsonl"),
    tmp_dir=tmp_dir,
    n_workers=8,
    timeout=3.0,
    problem_file=os.path.join(args.save_root, "problem_file.jsonl"),
    language="python",
)

print(result)