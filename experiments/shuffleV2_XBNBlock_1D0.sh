#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=0 python3 imagenet.py \
-a=shuffleV2_XBNBlock_1D0x \
--arch-cfg=dropout=0 \
--batch-size=256 \
--epochs=100 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=1e-4 \
--lr=0.1 \
--lr-method=step \
--lr-step=30 \
--lr-gamma=0.1 \
--dataset-root=./data/imageNet/ \
--dataset=folder \
--norm=GN \
--norm-cfg=num_groups=29,num_channels=0 \
--seed=1 \
$@
