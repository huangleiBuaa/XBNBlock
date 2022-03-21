#!/usr/bin/env bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=0 python3 imagenet.py \
-a=mobilenet_v2 \
--arch-cfg=dropout=0 \
--batch-size=256 \
--epochs=150 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=4e-5 \
--lr=0.1 \
--lr-method=cos \
--lr-step=150 \
--lr-gamma=0.00001 \
--dataset-root=./data/imageNet/ \
--dataset=folder \
--norm=BN \
--seed=1 \
$@
#--log-suffix=BN \
