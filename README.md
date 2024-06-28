# M4Fog
#### M4Fog: A Global Multi-Regional, Multi-Modal, and Multi-Stage Dataset for Marine Fog Detection and Forecasting to Bridge Ocean and Atmosphere

### Abstract
Marine fog poses a significant hazard to global shipping, necessitating effective detection and forecasting to reduce economic losses. In recent years, several machine learning (ML) methods have demonstrated superior detection accuracy compared to traditional meteorological methods. However, most of these works are developed on proprietary datasets, and the few publicly accessible datasets are often limited to simplistic toy scenarios for research purposes. To advance the field, we have collected nearly a decade's worth of multi-modal data related to continuous marine fog stages from four series of geostationary meteorological satellites, along with meteorological observations and numerical analysis, covering 15 marine regions globally where maritime fog frequently occurs. Through pixel-level manual annotation by meteorological experts, we present the most comprehensive marine fog detection and forecasting dataset to date, named **M4Fog**, to bridge ocean and atmosphere. The dataset comprises 68,000 "super data cubes" along four dimensions: elements, latitude, longitude and time, with a temporal resolution of half an hour and a spatial resolution of 1 kilometer. Considering practical applications, we have defined and explored three meaningful tracks with multi-metric evaluation systems: static or dynamic marine fog detection, and spatio-temporal forecasting for cloud images. Extensive benchmarking and experiments demonstrate the rationality and effectiveness of the construction concept for proposed M4Fog. The data and codes are available to whole researchers through cloud platforms to develop ML-driven marine fog solutions and mitigate adverse impacts on human activities.

----

<img width="680" alt="image" src="https://github.com/kaka0910/M4Fog/assets/23305257/59f058b1-2264-4d40-ac01-b9a8e2522436">

Figure 1: The overall construction flowchart of the proposed super data cubes in M4Fog is as follows. A variety of meteorological data related to marine fog detection and forecasting is obtained from geostationary satellites, numerical analysis, and observation databases. The super data cube is then constructed along four dimensions: elements, latitude, longitude, and time. According to visibility (VV) and present weather (WW) from observatories, the presence and absence of marine fog are determined. Subsequently, M4Fog can support multiple meaningful tracks, including static/dynamic marine fog detection and spatio-temporal forecasting.

Table 1: The comparisons with other marine fog benchmarks or datasets, focusing on tracks (i.e., numbers, w/o pixel-level), regionality (i.e., specific vs. multi-regions, abbreviated as (Sea area, Continent), temporality (e.g., seasonal, annual), variety of training/test (i.e.,satellites vs. observations vs. others), continuity and w/o open.

<img width="680" alt="image" src="https://github.com/kaka0910/M4Fog/assets/23305257/cf648c14-c2a7-4d31-adbf-9c7c386c31bc">

----
