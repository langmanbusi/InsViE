## InsViE: Effective Instruction-based Video Editing with Elaborate Dataset Construction

[![arXiv](https://img.shields.io/badge/arXiv-InsViE-b31b1b.svg)](https://arxiv.org) ![Pytorch](https://img.shields.io/badge/PyTorch->=2.4.0-Red?logo=pytorch)

### Abstract

<details><summary>Click for the full abstract</summary>
Instruction-based video editing allows effective and interactive editing of videos using only instructions without extra inputs such as masks or attributes. However, collecting high-quality training triplets (source video, edited video, instruction) is a challenging task. Existing datasets mostly consist of low-resolution, short duration, and limited amount of source videos with unsatisfactory editing quality, limiting the performance of trained editing models. In this work, we present a high-quality Instruction-based Video Editing dataset with 1M triplets, namely InsViE-1M. We first curate high-resolution and high-quality source videos and images, then design an effective editing-filtering pipeline to construct high-quality editing triplets for model training. For a source video, we generate multiple edited samples of its first frame with different intensities of classifier-free guidance, which are automatically filtered by GPT-4o with carefully crafted guidelines. The edited first frame is propagated to subsequent frames to produce the edited video, followed by another round of filtering for frame quality and motion evaluation. We also generate and filter a variety of video editing triplets from high-quality images. With the InsViE-1M dataset, we propose a multi-stage learning strategy to train our InsViE model, progressively enhancing its instruction following and editing ability. Extensive experiments demonstrate the advantages of our InsViE-1M dataset and the trained model over state-of-the-art works.
</details>

<br>

### Updates
- [3/xx/2025] Paper is available on ArXiv.


### TODO 
- [ ] Release the pretrained model.
- [ ] Update the code for inference.
- [ ] Release the InsViE-1M dataset.
- [ ] Update the code for training.

<!-- ### Environment

### Inference

### Training

#### -Data Construction

#### -Model Training

### Citation
```


```  -->
