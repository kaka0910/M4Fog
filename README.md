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

In the M4Fog dataset, most of the stationary meteorological satellite data can be obtained from the Level 1 (L1) data of their imagers, such as the FY4A/4B and GOES series. However, the Meteo satellite data is sourced from the original Level 1.5 data. After processes such as reading, projection, and format conversion, we can acquire a three-dimensional array (C, H, W). Additionally, we provide synthesis functions for generating true or natural-color images for all kinds of satellites. [Refer to ‘Data_process/Satellites/’]

**The data-processing of numerical analysis data**

Building on the multi-channel stationary meteorological satellite data, we further supplemented the dataset with land-sea masks and sea surface temperature (SST) numerical analysis data. Currently, the publicly available sea surface temperature data is daily, which means that for different times on the same day, the same SST file is provided. We also performed raw data reading, processing, and format conversion on the sea temperature data, ultimately resulting in a two-dimensional array (H, W). Additionally, we offer visualization methods for the sea temperature and other related data. [Refer to ‘Data_process/Numerical/’]

**The data-processing of observation databases data**

Observation data is primarily obtained from offshore buoys, vessels, ocean observation platforms, and coastal observation stations, recording weather phenomena and related meteorological variables such as temperature, humidity, visibility, and weather codes. The original observational data formats are typically .000/.dat or compressed files. We filter the observational records corresponding to the image data based on time and geographic coordinates to create a new text file. It is important to note that for data that do not directly record fog-related meteorological conditions, secondary assessments will be conducted using meteorological principles. [Refer to ‘Data_process/Observations/’]


### Track-A Sub-dataset

- H8/9-YB Sub-dataset : [Baidu Netdisk]() (password: ), [Google Drive]()
- FY4A-YB Sub-dataset : [Baidu Netdisk]() (password: ), [Google Drive]()

## M4Fog benchmarks

### Track-A benchmark


## Citation

If you have any quesetions about the data download, please contact xumengqiu@bupt.edu.cn or wuming@bupt.edu.cn.

If you are using this dataset or code for M4Fog please cite 
> Xu M, Wu M, Chen K, et al. M4Fog: A Global Multi-Regional, Multi-Modal, and Multi-Stage Dataset for Marine Fog Detection and Forecasting to Bridge Ocean and Atmosphere[J]. arXiv preprint arXiv:2406.13317, 2024.
