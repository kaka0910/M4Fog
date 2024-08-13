# M4Fog
This is the code and data for M4Fog: A Global Multi-Regional, Multi-Modal, and Multi-Stage Dataset for Marine Fog Detection and Forecasting to Bridge Ocean and Atmosphere (https://arxiv.org/html/2406.13317v1)

## M4Fog dataset

### Overall
We have collected nearly a decade's worth of multi-modal data related to continuous marine fog stages from four series of geostationary meteorological satellites, along with meteorological observations and numerical analysis, covering 15 marine regions globally where maritime fog frequently occurs. Through pixel-level manual annotation by meteorological experts, we present the most comprehensive marine fog detection and forecasting dataset to date, named M4Fog, to bridge ocean and atmosphere. The dataset comprises 68,000 "super data cubes" along four dimensions: elements, latitude, longitude and time, with a temporal resolution of half an hour and a spatial resolution of 1 kilometer. Considering practical applications, we have defined and explored three meaningful tracks with multi-metric evaluation systems: static or dynamic marine fog detection, and spatio-temporal forecasting for cloud images. 

![arch](https://github.com/kaka0910/M4Fog/blob/main/Imgs/Intro-M4fog.jpg)

Figure 1: The overall construction flowchart of the proposed super data cubes in M4Fog is as follows. A variety of meteorological data related to marine fog detection and forecasting is obtained from geostationary satellites, numerical analysis, and observation databases. The super data cube is then constructed along four dimensions: elements, latitude, longitude, and time. According to visibility (VV) and present weather (WW) from observatories, the presence and absence of marine fog are determined. Subsequently, M4Fog can support multiple meaningful tracks, including static/dynamic marine fog detection and spatio-temporal forecasting.

### M4Fog data-processing

In this section, we present several types of meteorological data used in M4Fog, including stationary meteorological satellite data, numerical analysis data, and observation data. We primarily outline the data processing workflow from the RAW files to multi-dimensional Arrays.

**The data-processing of geostationary satellites data**

[Format Arrangement: HDF—>NPY]] In the M4Fog dataset, most of the stationary meteorological satellite data can be obtained from the Level 1 (L1) data of their imagers, such as the FY4A/4B and GOES series. However, the Meteo satellite data is sourced from the original Level 1.5 data. After processes such as reading, projection, and format conversion, we can acquire a three-dimensional array (C, H, W). Additionally, we provide synthesis functions for generating true or natural-color images for all kinds of satellites.

-----

![arch](https://github.com/kaka0910/M4Fog/blob/main/Imgs/Abb-subdataset.jpg)

The sub-dataset for M4Fog is hosted in the upper links with the following directory structure (taking H8/9 in YB Area as an example):

```
.
|-- M4Fog-H8/9-YB
|   |-- train
|       |-- npy
|           |-- 20180115_0000.npy
|           |-- ...
|           |-- 20200805_0800.npy
|       |-- sst
|           |-- sst_20180115.npy
|           |-- ...
|           |-- sst_20200805.npy
|       |-- label
|           |-- conn_20180115_0000_300_sp.png
|           |-- ...
|           |-- conn_20200805_0800_300_sp.png
|       |-- icoads.txt
|   `-- val
|       |-- npy
|           |-- 20210110_0000.npy
|           |-- ...
|           |-- 20200810_0800.npy
|       |-- sst
|           |-- sst_20210110.npy
|           |-- ...
|           |-- sst_20200810.npy
|       |-- label
|           |-- conn_20210110_300_sp.png
|           |-- ...
|           |-- conn_20200810_300_sp.png
|       |-- icoads.txt
`--
```

You can choose the corresponding M4Fog subdataset according to the required region and satellite to train and test.

- H8/9-YB Sub-dataset : [Baidu Netdisk]() (password: ), [Google Drive]()
- FY4A-YB Sub-dataset : [Baidu Netdisk]() (password: ), [Google Drive]()

## Citation

If you have any quesetions about the data download, please contact xumengqiu@bupt.edu.cn or wuming@bupt.edu.cn.

If you are using this dataset or code for M4Fog please cite 
> Xu M, Wu M, Chen K, et al. M4Fog: A Global Multi-Regional, Multi-Modal, and Multi-Stage Dataset for Marine Fog Detection and Forecasting to Bridge Ocean and Atmosphere[J]. arXiv preprint arXiv:2406.13317, 2024.
