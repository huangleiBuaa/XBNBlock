export NGPUS=8
#!/usr/bin/env bash
cd "$(dirname $0)/.."
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file "configs/e2e_faster_rcnn_R_50_0selfStep_1FPN_2fc_1x.yaml" MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000
