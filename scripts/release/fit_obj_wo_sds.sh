#!/bin/bash

conda activate bigs

seq_name=$1

python main.py exp_name=test+obj dataset.seq=$seq_name dataset.name=arctic components="['object']" \
    train.num_steps=30000 human.lr.lr_max_steps=30000 \
    human.sds_use_xl=false \
    human.loss.l2_scale=100.0 human.loss.l2_offset=1.0