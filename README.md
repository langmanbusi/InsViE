## InsViE-1M: Effective Instruction-based Video Editing with Elaborate Dataset Construction (ICCV25)

[![arXiv](https://img.shields.io/badge/arXiv-InsViE-b31b1b.svg)](https://arxiv.org/abs/2503.20287) 
![Pytorch](https://img.shields.io/badge/PyTorch->=2.4.0-Red?logo=pytorch)  
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/wyh6666/InsViE)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/wyh6666/InsViE)

Yuhui Wu<sup>1,2</sup>, Liyi Chen<sup>1</sup>, Ruibin Li<sup>1,2</sup>, Shihao Wang<sup>1</sup>, Chenxi Xie<sup>1,2</sup>, Lei Zhang<sup>1,2</sup>*

(*Corresponding Author)

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute

[<img src="https://img.shields.io/badge/YouTube-Video-red?logo=youtube&logoColor=white&style=for-the-badge" height="40">](https://www.youtube.com/watch?v=z4t3RkqZ4no)

▶️ Watch our demo video on Youtube. We provide a smoother video and add more editing results. It is suggested to alter the resolution for better visual quality.

https://www.youtube.com/watch?v=z4t3RkqZ4no

https://github.com/user-attachments/assets/846f1fc3-3200-4e26-b4a5-2124cedee571



### Abstract

<details><summary>Click for the full abstract</summary>
Instruction-based video editing allows effective and interactive editing of videos using only instructions without extra inputs such as masks or attributes. However, collecting high-quality training triplets (source video, edited video, instruction) is a challenging task. Existing datasets mostly consist of low-resolution, short duration, and limited amount of source videos with unsatisfactory editing quality, limiting the performance of trained editing models. In this work, we present a high-quality Instruction-based Video Editing dataset with 1M triplets, namely InsViE-1M. We first curate high-resolution and high-quality source videos and images, then design an effective editing-filtering pipeline to construct high-quality editing triplets for model training. For a source video, we generate multiple edited samples of its first frame with different intensities of classifier-free guidance, which are automatically filtered by GPT-4o with carefully crafted guidelines. The edited first frame is propagated to subsequent frames to produce the edited video, followed by another round of filtering for frame quality and motion evaluation. We also generate and filter a variety of video editing triplets from high-quality images. With the InsViE-1M dataset, we propose a multi-stage learning strategy to train our InsViE model, progressively enhancing its instruction following and editing ability. Extensive experiments demonstrate the advantages of our InsViE-1M dataset and the trained model over state-of-the-art works.
</details>

### Updates
- [3/26/2025] Paper is available on ArXiv.


### TODO 
- [x] Release the pretrained model.
- [x] Update the code for inference.
- [x] Release the InsViE-1M dataset.
- [x] Update the code for training.


## Usage

### Installation

Clone the repo and install dependent packages

```bash
https://github.com/langmanbusi/InsViE.git
cd InsViE

# follow the instruction of original CogVideoX repo
cd CogVideo
pip install -r requirements.txt
cd sat
pip install -r requirements.txt
# use the given environment.yml
conda env create -f environment.yml

```

### Inference 

First download the weights of T5 and VAE models follow the instruction of [CogVideoX](https://github.com/THUDM/CogVideo/blob/main/sat/README.md).

Then download the weight of our [InsViE](https://huggingface.co/wyh6666/InsViE). The floder structure is the same with original CogVideo:

```
.
├── train_edit
    ├── 1000 (or 1)
    │   └── mp_rank_00_model_states.pt
    └── latest 
```

You should also modify the configs in `InsViE/CogVideo/sat/config`, such as the path to the pretrained models, refer to [link](https://github.com/THUDM/CogVideo/blob/main/sat/README_zh.md#3-%E4%BF%AE%E6%94%B9configscogvideox_yaml%E4%B8%AD%E7%9A%84%E6%96%87%E4%BB%B6).

```yaml
args:
  image2video: False # True for image2video, False for text2video
  latent_channels: 16
  mode: inference
  load: "/xxx/ckpts_2b_lora/train_edit" # This is for Full model without lora adapter
  batch_size: 1
  input_type: txt # You can choose txt for pure text input, or change to cli for command line input 
  input_file: /xxx/mytest.csv # prepare a test csv, which stores the video file names and instructions in each row
  test_folder: mytest # the folder contains the videos corresponding to the input_file (mytest.csv)
  sampling_image_size: [480, 720] # [480, 720]
  sampling_num_frames: 13  # Must be 13, 11 or 9
  sampling_fps: 7
  fp16: True # For CogVideoX-2B
  # bf16: True # For CogVideoX-5B and CoGVideoX-5B-I2V
  output_dir: /xxx/ # set the folder of the outputs
  force_inference: True
```

Then run the script.

```bash
cd InsViE/CogVideo/sat
bash inference.sh
```

### Training

This project provides scripts and configs for edit fine-tuning (`train_video_edit.py` + `configs/cogvideox_2b_lora_edit.yaml` + `configs/sft_edit_resume_nv.yaml`).

1. Prepare pretrained weights:
   - `CogVideoX-2b-sat/transformer` (or LoRA checkpoint directory)
   - `CogVideoX-2b-sat/vae/3d-vae.pt`
   - `t5-v1_1-xxl` (T5 encoder model)

2. Update configuration paths:
   - In `configs/cogvideox_2b_lora_edit.yaml`:
     - set `conditioner_config.params.emb_models[0].params.model_dir` to your T5 path
     - set `first_stage_config.params.ckpt_path` to your 3D VAE weight path
   - In `configs/sft_edit_resume_nv.yaml`:
     - set `args.load` to transformer weights
     - set `args.save` to output checkpoint directory
     - set `data.params.train_video_root` / `data.params.test_video_root` and `data.path` to dataset paths

3. Single-GPU training example:

```bash
cd InsViE/CogVideo/sat
bash finetune_single_gpu.sh
```

4. Multi-GPU training example:

```bash
cd InsViE/CogVideo/sat
bash finetune_multi_gpus.sh
# or NV environment
bash finetune_multi_gpus_nv.sh
```

5. Alternatively run directly:

```bash
cd InsViE/CogVideo/sat
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 train_video_edit.py --base configs/cogvideox_2b_lora_edit.yaml configs/sft_edit_resume_nv.yaml --seed $RANDOM
```

6. Verify training results:
   - `args.save` directory contains `train_edit_*` checkpoint folders
   - `training_config.yaml` is saved during first iteration for reproducibility

### InsViE-1M Dataset

[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/wyh6666/InsViE)


## Citation

If you find this work helpful, please consider citing:

```
@article{wu2025insvie,
  title={InsViE-1M: Effective Instruction-based Video Editing with Elaborate Dataset Construction},
  author={Wu, Yuhui and Chen, Liyi and Li, Ruibin and Wang, Shihao and Xie, Chenxi and Zhang, Lei},
  journal={arXiv preprint arXiv:2503.20287},
  year={2025}
}
