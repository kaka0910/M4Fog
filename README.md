# M4Fog
This is the code and data for M4Fog: A Global Multi-Regional, Multi-Modal, and Multi-Stage Dataset for Marine Fog Detection and Forecasting to Bridge Ocean and Atmosphere (https://arxiv.org/html/2406.13317v1)

## M4Fog dataset

### Overall
We have collected nearly a decade's worth of multi-modal data related to continuous marine fog stages from four series of geostationary meteorological satellites, along with meteorological observations and numerical analysis, covering 15 marine regions globally where maritime fog frequently occurs. Through pixel-level manual annotation by meteorological experts, we present the most comprehensive marine fog detection and forecasting dataset to date, named M4Fog, to bridge ocean and atmosphere. The dataset comprises 68,000 "super data cubes" along four dimensions: elements, latitude, longitude and time, with a temporal resolution of half an hour and a spatial resolution of 1 kilometer. Considering practical applications, we have defined and explored three meaningful tracks with multi-metric evaluation systems: static or dynamic marine fog detection, and spatio-temporal forecasting for cloud images. 

![arch](https://github.com/kaka0910/M4Fog/blob/main/Imgs/Fig_construction_0605.png)

Figure 1: The overall construction flowchart of the proposed super data cubes in M4Fog is as follows. A variety of meteorological data related to marine fog detection and forecasting is obtained from geostationary satellites, numerical analysis, and observation databases. The super data cube is then constructed along four dimensions: elements, latitude, longitude, and time. According to visibility (VV) and present weather (WW) from observatories, the presence and absence of marine fog are determined. Subsequently, M4Fog can support multiple meaningful tracks, including static/dynamic marine fog detection and spatio-temporal forecasting.

### M4Fog data-processing

In this section, we present several types of meteorological data used in M4Fog, including stationary meteorological satellite data, numerical analysis data, and observation data. We primarily outline the data processing workflow from the RAW files to multi-dimensional Arrays.

**The data-processing of geostationary satellites data**

In the M4Fog dataset, most of the stationary meteorological satellite data can be obtained from the Level 1 (L1) data of their imagers, such as the FY4A/4B and GOES series. However, the Meteo satellite data is sourced from the original Level 1.5 data. After processes such as reading, projection, and format conversion, we can acquire a three-dimensional array (C, H, W). Additionally, we provide synthesis functions for generating true or natural-color images for all kinds of satellites. [Refer to ‘Data_process/Satellites/’]

**The data-processing of numerical analysis data**

Building on the multi-channel stationary meteorological satellite data, we further supplemented the dataset with land-sea masks and sea surface temperature (SST) numerical analysis data. Currently, the publicly available sea surface temperature data is daily, which means that for different times on the same day, the same SST file is provided. We also performed raw data reading, processing, and format conversion on the sea temperature data, ultimately resulting in a two-dimensional array (H, W). Additionally, we offer visualization methods for the sea temperature and other related data. [Refer to ‘Data_process/Numerical/’]

**The data-processing of observation databases data**

Observation data is primarily obtained from offshore buoys, vessels, ocean observation platforms, and coastal observation stations, recording weather phenomena and related meteorological variables such as temperature, humidity, visibility, and weather codes. The original observational data formats are typically .000/.dat or compressed files. We filter the observational records corresponding to the image data based on time and geographic coordinates to create a new text file. It is important to note that for data that do not directly record fog-related meteorological conditions, secondary assessments will be conducted using meteorological principles. [Refer to ‘Data_process/Observations/’]

![arch](https://github.com/kaka0910/M4Fog/blob/main/Imgs/Fig_samples_0518.png)

### M4Fog Dataset Downloading

In order to facilitate data storage and download, we have separated and stored the elements in M4Fog Dataset. Dense data such as geostationary satellite images, sea surface temperature, and constant data are stored in arrays, while sparse data, like observations from monitoring stations, are stored in text format. We have published the data links and brief descriptions in the order of Tracks A to C.

**Track A sub-dataset**

We have released multiple sub-datasets in Track-A, which are divided based on regions and satellites. Each data cube is a labeled three-dimensional array with dimensions (C, H, W), based on multi-channel geostationary satellite data and supplemented with other data such as sea surface temperature. Dense data and sparse data are strictly aligned with the fog events in terms of time and space.

- Himawari-8/9 data in Yellow and Bohai Sea:
  - 1,802 data cubes (CxHxW = (16+1+1)x1024x1024), including Himawari 8/9, SST, Constant Dense Data. 
  - [Baidu Netdisk](https://pan.baidu.com/s/1aOw3TqzkS7E0ymLG0k2BhA?pwd=6vm7) (password: 6vm7)
- Fengyun-4A Yellow and Bohai Sea:
  - 1,724 data cubes (CxHxW = (14+1+1)x1024x1024), including Fengyun 4A, SST, Constant Dense Data.
  - [Baidu Netdisk](https://pan.baidu.com/s/1FSNdZoZCZ4mdu9_ehMRLIA?pwd=wp9k) (password: wp9k)
- GOES-16 Multi-areas:
  -  data cubes (CxHxW = (16+1+1)x1024x1024), including GOES-16, SST, Constant Dense Data.
  - [Baidu Netdisk](https://pan.baidu.com/s/16RCgHSbdbNDgV8HMwKojPw?pwd=jv7g) (password: jv7g)

**Track B sub-dataset**

Building on Track A, we constructed a time series dataset with a length of N+1 and dimensions of ((N+1),C,H,W) to enable dynamic marine fog monitoring. Additionally, we provided the corresponding satellite dataset's Motion data, derived using visible bands and optical flow algorithms.

- Himawari-8/9 data in Yellow and Bohai Sea:
  - 1,696 data cubes ((N+1)xCxHxW = (1+1)x(16+1+1)x1024x1024), including Himawari 8/9, SST, Constant Dense Data and motion data. 
  - [Baidu Netdisk](https://pan.baidu.com/s/1q0s9eqAlydmyX6lu0znd3w?pwd=8q75) (password: 8q75)

**Track C sub-dataset**

We construct several sub-datasets using M4Fog data cubes, with each satellite having a single-region sub-dataset and a multi-region sub-dataset. The datasets are split by year to prevent test set leakage. Considering the timeliness required for practical forecasting applications, we have currently released prediction data at a lower resolution.

- Himawari-8/9 data:
  - The length of data cube sequence is 8, T = T' = 4. The total number of multi-areas: 4,944.
  - Area1 (Yellow and Bohai Sea): 2,512 data cubes (CxHxW = 3x256x256), visible bands (0.47μm, 0.51μm, 0.64μm).
  - Area2 (East China Sea): 2,432 data cubes (CxHxW = 3x256x256), visible bands (0.47μm, 0.51μm, 0.64μm).
  - [Area 1 Baidu Netdisk](https://pan.baidu.com/s/1OsN_nkJuwJDNun8N_O4iMA?pwd=mfcx) (password: mfcx); [Multi-areas Baidu Netdisk](https://pan.baidu.com/s/1it_gK_v_RDLXRoxMX-X4Ow?pwd=99ci) (password: 99ci)
- Fengyun-4A data:
  - The length of data cube sequence is 8, T = T' = 4. The total number of multi-areas: 3,931.
  - 3,931 data cubes (CxHxW = 3x256x256), visible bands (0.47μm, 0.65μm) + Near-Infrared band (0.825μm).
  - [Baidu Netdisk](https://pan.baidu.com/s/1S4oQAw1klyFedaKl1gR7mA?pwd=ugsr) (password: ugsr)
- Meteo-multiareas data:
  - The length of data cube sequence is 8, T = T' = 4. The total number of multi-areas: 2,832.
  - 2,832 data cubes (CxHxW = 3x256x256), natural-color images.
  - [Baidu Netdisk](https://pan.baidu.com/s/1UO1uuihnczQOCeEnbWSsXA?pwd=vv3p) (password: vv3p)
- Meteo-single-area data:
  - The length of data cube sequence is 8, T = T' = 4. The total number of all areas: 464*6 = 2,784.
  - Each area: training data (2020,2021) vs. test data (2022): 328 vs. 136 data cubes, natural-color images.
  - [Baidu Netdisk](https://pan.baidu.com/s/1n9tj4l1qjTxmGApzxI3lgQ?pwd=1gng) (password: 1gng)

**Other raleted-marine fog sub-dataset**

We also provide additional processed data related to marine fog monitoring and forecasting as part of the M4Fog construction. This includes, for example, nearly a decade's worth of ICOADS data used to analyze areas where marine fog occurs.

- ICOADS data spanning 2015-2024 :
  - almost 9.5 million records spanning 2015-2024, which clearly observe or infer the presence or absence of marine fog globally.
  - [Baidu Netdisk](https://pan.baidu.com/s/1_8hTqhCS5ZyrkCqvTrT7KQ?pwd=yfat) (password: yfat)
- Marine weather reviews (in Chinese) 2017-2022: 
  - [Baidu Netdisk](https://pan.baidu.com/s/1QE5QicyA1tFalcguP_fY_Q?pwd=amwc) (password: amwc)
-  SaeMAE 03/04/14 bands images :
  - [Baidu Netdisk](https://pan.baidu.com/s/1zKpGfhruSNLInJB4Bc8HhQ?pwd=1n5v) (password: 1n5v)

## M4Fog benchmarks

### Track-A benchmark

Static marine fog detection refers to identifying and delineating marine fog areas at a specific moment based on super data cubes from that moment. It can support the input of either multi-channel geostationary satellite data alone or in combination with sea surface temperature and constant data. For more details on Track A, please refer to the Track A README. Here is an example of single GPU non-distributed training UNetpp on Himawari-8/9 dataset.

```shell
# Training on H8/9
CUDA_VISIBLE_DEVICES=1 python code/train_h8.py --dataset h8 --batch-size 2 --workers 1 \
           --model aspp_unet --checkname aspp_unet \
           --lr 0.0001 \
           --epochs 100 \
           --weight-decay 0.05 \
           --model_savefolder /nas/cloud/H8/ \
```

### Track-B benchmark

Track B dynamic marine fog detection refers to identifying and delineating marine fog areas at a specific moment by utilizing continuous super data cubes. We also provide the code for calculate the motion feature between adjacent time data. For more details on Track B, please refer to the Track B README. Here is an example of single GPU non-distributed training UNetpp on Himawari-8/9 dataset with motion data.

```shell
# Training on H8/9 with motion data
CUDA_VISIBLE_DEVICES=1 python code/train_h8_motion.py \
            --dataset h8_motion_v2 --batch-size 16 --workers 1 \
            --model unet_motion_v2 --checkname unet_motion_v2 \
            --model_optimizer SGD \
            --lr 0.0001 \
            --epochs 100 \
            --weight-decay 0.05 \
            --model_savefolder /motion/01unet_concatdist_v2/ \
```

### Track-C benchmark

For the experiments, the OpenSTL framework is selected as the foundation benchmark. Based on this codebase, we supply the process M4Fog dataset and generate corresponding data. We also provide MSE, MAE, SSIM and PSNR as evaluation metrics to measure model performance. For more details on Track C, please refer to the Track C README. Here is an example of single GPU non-distributed training PredRNN on Meteo-single-area dataset.

```shell
cd OpenSTL-OpenSTL-Lightning

# Training on meteo
CUDA_VISIBLE_DEVICES=0 python tools/train.py -d meteo -c configs/meteo/PredRNN.py --ex_name meteo_supply_predrnn
# Test on meteo
CUDA_VISIBLE_DEVICES=0 python tools/test.py -d meteo -c configs/meteo/PredRNN.py --ex_name meteo_supply_predrnn --test
# Finetune on meteo
CUDA_VISIBLE_DEVICES=1 python tools/train.py -d meteo -c configs/meteo/PredRNN.py --ex_name meteo_supply_finetue --ckpt_path meteo_supply_predrnn
```

## Citation

If you have any quesetions about the data download, please contact xumengqiu@bupt.edu.cn or wuming@bupt.edu.cn.

If you are using this dataset or code for M4Fog please cite:
> Xu M, Wu M, Chen K, et al. M4Fog: A Global Multi-Regional, Multi-Modal, and Multi-Stage Dataset for Marine Fog Detection and Forecasting to Bridge Ocean and Atmosphere[J]. arXiv preprint arXiv:2406.13317, 2024.
