#! /bin/bash

export CUDA_VISIBLE_DEVICES=1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

run_cmd="$environs python sample_video_edit.py --base configs/cogvideox_2b_lora_edit.yaml configs/inference_2b_edit.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"
