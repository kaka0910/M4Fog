## M4Fog data-processing

In this section, we present several types of meteorological data used in M4Fog, including stationary meteorological satellite data, numerical analysis data, and observation data. We primarily outline the data processing workflow from the RAW files to multi-dimensional Arrays.

### The data-processing of geostationary satellites data

In the M4Fog dataset, most of the stationary meteorological satellite data can be obtained from the Level 1 (L1) data of their imagers, such as the FY4A/4B and GOES series. However, the Meteo satellite data is sourced from the original Level 1.5 data. After processes such as reading, projection, and format conversion, we can acquire a three-dimensional array (C, H, W). Additionally, we provide synthesis functions for generating true or natural-color images for all kinds of satellites.

### The data-processing of numerical analysis data

Building on the multi-channel stationary meteorological satellite data, we further supplemented the dataset with land-sea masks and sea surface temperature (SST) numerical analysis data. Currently, the publicly available sea surface temperature data is daily, which means that for different times on the same day, the same SST file is provided. We also performed raw data reading, processing, and format conversion on the sea temperature data, ultimately resulting in a two-dimensional array (H, W). Additionally, we offer visualization methods for the sea temperature and other related data.

### The data-processing of observation databases data

Observation data is primarily obtained from offshore buoys, vessels, ocean observation platforms, and coastal observation stations, recording weather phenomena and related meteorological variables such as temperature, humidity, visibility, and weather codes. The original observational data formats are typically .000/.dat or compressed files. We filter the observational records corresponding to the image data based on time and geographic coordinates to create a new text file. It is important to note that for data that do not directly record fog-related meteorological conditions, secondary assessments will be conducted using meteorological principles. 
