#!/bin/bash

conda activate bigs

seq_name=$1
component="'right', 'left'"

python main.py exp_name=release+hand dataset.seq=$seq_name components="[$component]" \
    train.num_steps=15000 human.lr.lr_max_steps=15000