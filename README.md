# SpaceSuperResolution

Udapted for satellite and aerial images. Based on [BasicSR](https://github.com/xinntao/BasicSR) project

Super Resolution for aerial and satellite imagery

1. Download images: [docs/Datasets.md](./docs/Datasets.md)
2. Split large images into smaller ones:
```
python ./core/scripts/data_preparation/extract_subimages.py
```
or reshape images in already splitted dataset 
```
python ./core/scripts/data_preparation/prepare_scaled_dataset.py
```
3. Modify config for tranining i.e.: [train_EDSR_Argis_30sm_x4.yml](./options/train/EDSR/train_EDSR_Argis_30sm_x4.yml)
4. Train your model:
```
python ./core/train.py ./options/train/EDSR/train_EDSR_Argis_30sm_x4.yml
```
