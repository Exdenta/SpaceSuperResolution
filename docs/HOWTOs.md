# HOWTOs

## How to train StyleGAN2

1. Prepare training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset). More details are in [DatasetPreparation.md](DatasetPreparation.md#StyleGAN2)
    1. Download FFHQ dataset. Recommend to download the tfrecords files from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
    1. Extract tfrecords to images or LMDBs (TensorFlow is required to read tfrecords):

        > python scripts/data_preparation/extract_images_from_tfrecords.py

1. Modify the config file in `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ_800k.yml --launcher pytorch

## How to inference StyleGAN2

1. Download pre-trained models from **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) to the `experiments/pretrained_models` folder.
1. Test.

    > python inference/inference_stylegan2.py

1. The results are in the `samples` folder.
