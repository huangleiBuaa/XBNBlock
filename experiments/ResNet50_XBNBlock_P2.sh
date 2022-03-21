#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=0,1 python3 imagenet.py \
-a=resnet_XBNBlock_P2_50 \
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
--norm-cfg=num_groups=64 \
--seed=1 \
$@
#--log-suffix=BN \
