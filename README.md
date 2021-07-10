# SpaceSuperResolution

Based on [BasicSR](link) project

Super Resolution for aerial and satellite imagery

1. Download images: [docs/Datasets.md](./docs/Datasets.md)
2. Modify config for tranining i.e.: [train_EDSR_Argis_30sm_x4.yml](./options/train/EDSR/train_EDSR_Argis_30sm_x4.yml)
3. Train your model:
```
python ./core/train.py ./options/train/EDSR/train_EDSR_Argis_30sm_x4.yml
```
