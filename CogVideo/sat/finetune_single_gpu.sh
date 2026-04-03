#! /bin/bash

echo "RUN on `hostname`, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

# run_cmd="$environs python train_video_edit_oppo.py --base configs/cogvideox_2b_lora_edit_oppo.yaml configs/sft_edit.yaml --seed $RANDOM"
run_cmd="$environs python train_video_edit.py --base configs/cogvideox_2b_lora_edit.yaml configs/sft_edit.yaml --seed $RANDOM"
# run_cmd="$environs python train_video.py --base configs/cogvideox_2b_lora.yaml configs/sft.yaml --seed $RANDOM"


echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"