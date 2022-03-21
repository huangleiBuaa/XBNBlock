#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 imagenet.py \
-a=resnext_forGN101_32x8d \
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
--norm-cfg=num_groups=32,num_channels=0 \
--seed=1 \
$@
#--log-suffix=BN \
