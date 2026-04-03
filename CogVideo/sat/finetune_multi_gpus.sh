#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# run_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 train_video.py --base configs/test_cogvideox_5b_i2v_lora.yaml configs/test_sft.yaml --seed $RANDOM"
run_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 train_video_edit.py --base configs/cogvideox_2b_lora_edit.yaml configs/sft_edit_resume.yaml --seed $RANDOM > /opt/dlami/nvme/wyh_data/ckpts_2b_lora/log_testresume_1.log 2>&1"
# run_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 train_video_edit.py --base configs/cogvideox_2b_lora_edit.yaml configs/sft_edit_resume.yaml --seed $RANDOM"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"

# echo "MASTER_ADDR=$MASTER_ADDR"
# n_node=$SLURM_JOB_NUM_NODES
# echo "number of nodes:" $n_node
# echo "node rank:" $SLURM_PROCID

# python -m torch.distributed.run \
#     --standalone \
#     --nproc_per_node 8 \
#     train_video_edit.py \
#     --base configs/cogvideox_2b_lora_edit.yaml \
#     configs/sft_edit_resume.yaml \
#     --seed $RANDOM > /opt/dlami/nvme/wyh_data/ckpts_2b_lora/log_testresume_1.log 2>&1 \
    
    # --nnodes 1 --node_rank 0 \
    # --master_addr $(hostname -f) --master_port 25031 \
    # --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    # --master_addr $MASTER_ADDR --master_port 25031 \
