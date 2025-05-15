#!/usr/bin/env bash
GPUS=$1

python -W ignore \
    -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    train.py \
    ${@:2}
