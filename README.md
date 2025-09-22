<h2 align="center"> 
  BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting (CVPR, 2025)
</h2>

<h4 align="center">
  <a href="https://on-jungwoan.github.io/">Jeongwan On</a>, <a href="https://khgwak.github.io/about/">Kyeonghwan Gwak</a>, Gunyoung Kang, <a href="https://junukcha.github.io/about/">Junuk Cha</a>,

  <a href="https://github.com/coding-Hwang">Soohyun Hwang</a>, Hyein Hwang , <a href="https://sites.google.com/site/bsrvision00/">Seungryul Baek</a>
</h4>

<h5 align="center">
  [<a href="https://openaccess.thecvf.com/content/CVPR2025/papers/On_BIGS_Bimanual_Category-agnostic_Interaction_Reconstruction_from_Monocular_Videos_via_3D_CVPR_2025_paper.pdf">Paper</a>]
  [<a href="https://github.com/On-JungWoan/BIGS">Project Page</a>]
</h5>

<br>

<p float="center">
  <img src="assets/bigs_teaser.png" width="100%" />
</p>

## Abstract

> Reconstructing 3Ds of hand-object interaction (HOI) is a fundamental problem that can find numerous applications. Despite recent advances, there is no comprehensive pipeline yet for bimanual class-agnostic interaction reconstruction from a monocular RGB video, where two hands and an unknown object are interacting with each other. Previous works tackled the limited hand-object interaction case, where object templates are pre-known or only one hand is involved in the interaction. The bimanual interaction reconstruction exhibits severe occlusions introduced by complex interactions between two hands and an object. To solve this, we  first introduce BIGS (Bimanual Interaction 3D Gaussian Splatting), a method that reconstructs 3D Gaussians of hands and an unknown object from a monocular video. To robustly obtain object Gaussians avoiding severe occlusions, we leverage prior knowledge of pre-trained diffusion model with score distillation sampling (SDS) loss, to reconstruct unseen object parts. For hand Gaussians, we exploit the 3D priors of hand model (i.e., MANO) and share a single Gaussian for two hands to effectively accumulate hand 3D information, given limited views. To further consider the 3D alignment between hands and objects, we include the interacting-subjects optimization step during Gaussian optimization. Our method achieves the state-of-the-art accuracy on two challenging datasets, in terms of 3D hand pose estimation (MPJPE), 3D object reconstruction (CDh, CDo, F10), and rendering quality (PSNR, SSIM, LPIPS), respectively.

<br>

## News

- 2025.02.27: 🎉 Paper accepted by CVPR 2025.
- 2024.09.30: 🏆 We gave an oral presentation at the HANDS Workshop, ECCV 2024.
- 2024.09.30: 🥇 We won the 1st place in Bimanual category-agnostic interaction reconstruction challenge in conjunction with ECCV 2024. [[Technical Report](https://hands-workshop.org/files/2024/UVHANDS.pdf)]

<br>

## Installation

```
bash scripts/setup.sh
```

<br>

## Datasets

Our test sequences can be found below:

- **ARCTIC & HO3Dv3**
  - We just followed the ECCV 2024 HANDS Workshop Challenge evaluation settings. Our experiments use the officially provided test splits. Additional details are in Section 4.1 (Datasets) of the [BIGS paper](https://openaccess.thecvf.com/content/CVPR2025/papers/On_BIGS_Bimanual_Category-agnostic_Interaction_Reconstruction_from_Monocular_Videos_via_3D_CVPR_2025_paper.pdf). And you can also refer to this link: <https://github.com/zc-alexfan/hold/blob/master/docs/data_doc.md#checkpoints>
- **HO3Dv3 (Limited view)**
  - Please refer to this link: <https://drive.google.com/drive/folders/1Tk2Y4lN0vtRaUlHDg7zuWThQZtWWGHVy?usp=sharing>

<br>

## Demo

```
bash scripts/demo.sh
```

<br>

## Training

### 1. Single-subject optimization step (hand)

```
bash -i scripts/release/fit_hand.sh $seq_name
```


### 2. Single-subject optimization step (object)

```
# SDS loss-based fitting may take a long time to converge.
# For debugging purposes, you may run the fitting without SDS loss for faster results.

# w/ sds loss
bash -i scripts/release/fit_obj.sh $seq_name

# w/o sds loss
bash -i scripts/release/fit_obj_wo_sds.sh $seq_name
```

### 3. Interacting-subjects optimization step

```
bash -i scripts/release/joint_train.sh $seq_name
```

<br>

## Citation

If you find this work useful, please consider citing our paper.

```bibtex
@InProceedings{On_2025_CVPR,
    author    = {On, Jeongwan and Gwak, Kyeonghwan and Kang, Gunyoung and Cha, Junuk and Hwang, Soohyun and Hwang, Hyein and Baek, Seungryul},
    title     = {BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17437-17447}
}
```
