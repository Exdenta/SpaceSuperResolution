# Model Zoo and Baselines

:arrow_double_down: Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing)
:arrow_double_down: 百度网盘: [预训练模型](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g) **|** [复现实验](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ)

---

We provide:

1. Official models converted directly from official released models
1. Reproduced models with `BasicSR`. Pre-trained models and log examples are provided

You can put the downloaded models in the `experiments/pretrained_models` folder.

**[Download official pre-trained models]** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g))

You can use the scrip to download pre-trained models from Google Drive.

```python
python scripts/download_pretrained_models.py ESRGAN
# method can be ESRGAN, EDVR, StyleGAN, EDSR, DUF, DFDNet, dlib
```

**[Download reproduced models and logs]** ([Google Drive](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing), [百度网盘](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ))

In addition, we upload the training process and curves in [wandb](https://www.wandb.com/).

**[Training curves in wandb](https://app.wandb.ai/xintao/basicsr)**

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="../assets/wandb.jpg" height="350">
</a></p>

#### Contents

1. [Image Super-Resolution](#Image-Super-Resolution)
    1. [Image SR Official Models](#Image-SR-Official-Models)
    1. [Image SR Reproduced Models](#Image-SR-Reproduced-Models)

## Image Super-Resolution

When evaluation:

- We crop `scale` border pixels in each border
- Evaluated on RGB channels

### Image SR Official Models

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| EDSR_Mx2_f64b16_DIV2K_official-3ba7b086 | 35.7768 / 0.9442 | 31.4966 / 0.8939 | 34.6291 / 0.9373 |
| EDSR_Mx3_f64b16_DIV2K_official-6908f88a | 32.3597 / 0.903 | 28.3932 / 0.8096 | 30.9438 / 0.8737 |
| EDSR_Mx4_f64b16_DIV2K_official-0c287733 | 30.1821 / 0.8641 | 26.7528 / 0.7432 | 28.9679 / 0.8183 |
| EDSR_Lx2_f256b32_DIV2K_official-be38e77d | 35.9979 / 0.9454 | 31.8583 / 0.8971 | 35.0495 / 0.9407 |
| EDSR_Lx3_f256b32_DIV2K_official-3660f70d | 32.643 / 0.906 | 28.644 / 0.8152 | 31.28 / 0.8798 |
| EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f | 30.5499 / 0.8701 | 27.0011 / 0.7509 | 29.277 / 0.8266 |

### Image SR Reproduced Models

Experiment name conventions are in [Config.md](Config.md).

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb | 30.2468 / 0.8651 | 26.7817 / 0.7451 | 28.9967 / 0.8195 |
| 002_MSRResNet_x2_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 35.7483 / 0.9442 | 31.5403 / 0.8937 |34.6699 / 0.9377|
| 003_MSRResNet_x3_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 32.4038 / 0.9032| 28.4418 / 0.8106|30.9726 / 0.8743 |
| 004_MSRGAN_x4_f64b16_DIV2K_400k_B16G1_wandb | 28.0158 / 0.8087|24.7474 / 0.6623 | 26.6504 / 0.7462|
| | | | |
| 201_EDSR_Mx2_f64b16_DIV2K_300k_B16G1_wandb | 35.7395 / 0.944|31.4348 / 0.8934 |34.5798 / 0.937 |
| 202_EDSR_Mx3_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|32.315 / 0.9026 |28.3866 / 0.8088 |30.9095 / 0.8731|
| 203_EDSR_Mx4_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|30.1726 / 0.8641 |26.721 / 0.743 |28.9506 / 0.818|
| 204_EDSR_Lx2_f256b32_DIV2K_300k_B16G1_wandb | 35.9792 / 0.9453 | 31.7284 / 0.8959 | 34.9544 / 0.9399 |
| 205_EDSR_Lx3_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 32.6467 / 0.9057 | 28.6859 / 0.8152 | 31.2664 / 0.8793 |
| 206_EDSR_Lx4_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 30.4718 / 0.8695 | 26.9616 / 0.7502 | 29.2621 / 0.8265 |
