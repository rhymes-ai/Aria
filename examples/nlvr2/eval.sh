#!/bin/bash

BASE_MODEL=/root/workspace/models/aria
TOKENIZER=/root/workspace/models/aria
LORA=""
IMG_SIZE=980
SAVE_ROOT=./eval/nlvr2__$(basename $BASE_MODEL)__$(basename $LORA)__$IMG_SIZE

CMD="python examples/nlvr2/evaluation.py \
    --base_model_path $BASE_MODEL \
    --tokenizer_path $TOKENIZER \
    --save_root $SAVE_ROOT \
    --image_size $IMG_SIZE"

if [ -n "$LORA" ]; then
    CMD="$CMD --peft_model_path $LORA"
fi

time $CMD
