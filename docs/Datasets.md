
## 1. Argis dataset

~ 5 sm. aerial imagery
Download Argis dataset from https://geosaurus.maps.arcgis.com/home/item.html?id=abc0812aa82c4fe681662e5ba495b6b8

## 2. SpaceNet6

In September 2020, the SpaceNet partners released an expanded version of the SpaceNet 6 dataset. The dataset is untiled and distributed in its maximum extent to enable research using combinations of SAR and optical imagery. All of the SAR data comes from Capella Space’s X-band quad-pol sensor mounted on an aircraft. We distribute 202 SAR image strips in two formats: one with minimal pre-processing (Single Look Complex) as well as a second set of new six-band georeferenced products that include 4 channels of intensity and 2 channels derived from a Pauli decomposition. The decomposition channels show different types of scattering behavior. Many of these strips overlap to create a dense stack of SAR data with multiple revisits spanning a three-day time period in August 2019. The extent of these image strips covers a large portion of Rotterdam and 120 km² of total area, with each strip spanning approximately 0.7 km by 10 km.

Complimentary to our SAR data, we also release our untiled Maxar WorldView 2 image spanning ~92 km² at 0.5m spatial resolution. We distribute 4 image products including the panchromatic band, pan-sharpened RGB and RGBNIR data (0.5m) and RGBNIR data (2.0m). As in the challenge, we hold back the optical data over the final testing area but distribute these optical data for validation and training.

```
### Download SN6 dataset
aws s3 cp s3://spacenet-dataset/SN6_buildings/ SN6_buildings --recursive

### Download expanded
aws s3 cp s3://spacenet-dataset/AOIs/AOI_11_Rotterdam/ SN6_buildings/expanded --recursive
```

## 3. Openaerialmap

To download images with resolution from 0.01 to 0.02 sm. you can use python script.

```
python download_openaerialmap.py
```



