#!/bin/bash
pip install -r requirements.txt

N_SUP=(744 372 100 -1)

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

for ((i=0;i<${#N_SUP[@]};i++))
do python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT train.py\
 -c config/config-contrastive.json\
 --num_servers 1 --ngpus_per_node 4\
 -ddp\
 --cloud\
 --lr 0.00012\
 --trainer cccseg\
 --batch_size 4\
 --n_sup ${N_SUP[i]}\
 --jitter 1.0 --gray 0.0 --blur 0\
 --name 'c3seg-s1'\
 --cutmix\
 --ema\
 --ema_alpha 0.99\
 --rampup 16000\
 --consist_weight 50\
 --unlabel_start_epoch 10\
 --epochs 100\
 --autoaug\
 --cb_threshold\
 --end_cb 0.8\
 --contrastive_start_epoch 1\
 --contrastive_loss_weight 0.1\
 --contrastive\
 --p_head 'mlp'\
 --temperature 0.15\
 --contrastive_rampup 1000\
 --embed 256\
 --hard_neg_num 20\
 --contrastive_cross_set\
 --mask_contrast\
 --split_seed 12345
done