## M4Fog data-processing

In this section, we present several types of meteorological data used in M4Fog, including stationary meteorological satellite data, numerical analysis data, and observation data. We primarily outline the data processing workflow from the RAW files to multi-dimensional Arrays.

![arch](https://github.com/kaka0910/M4Fog/blob/main/Data_process/dataprocess.png)

### The data-processing of geostationary satellites data

In the M4Fog dataset, most of the stationary meteorological satellite data can be obtained from the Level 1 (L1) data of their imagers, such as the FY4A/4B and GOES series. However, the Meteo satellite data is sourced from the original Level 1.5 data. After processes such as reading, projection, and format conversion, we can acquire a three-dimensional array (C, H, W). Additionally, we provide synthesis functions for generating true or natural-color images for all kinds of satellites.

![arch](https://github.com/kaka0910/M4Fog/blob/main/Data_process/dataprocess-sats.png)

### The data-processing of numerical analysis data

Building on the multi-channel stationary meteorological satellite data, we further supplemented the dataset with land-sea masks and sea surface temperature (SST) numerical analysis data. Currently, the publicly available sea surface temperature data is daily, which means that for different times on the same day, the same SST file is provided. We also performed raw data reading, processing, and format conversion on the sea temperature data, ultimately resulting in a two-dimensional array (H, W). Additionally, we offer visualization methods for the sea temperature and other related data.

![arch](https://github.com/kaka0910/M4Fog/blob/main/Data_process/dataprocess-sst.png)

### The data-processing of observation databases data

Observation data is primarily obtained from offshore buoys, vessels, ocean observation platforms, and coastal observation stations, recording weather phenomena and related meteorological variables such as temperature, humidity, visibility, and weather codes. The original observational data formats are typically .000/.dat or compressed files. We filter the observational records corresponding to the image data based on time and geographic coordinates to create a new text file. It is important to note that for data that do not directly record fog-related meteorological conditions, secondary assessments will be conducted using meteorological principles. 

| Index | Year | Month | Day | Hour | Minute | Longitude | Latitude | Visibility | Present Weather | Label   |
|-------|------|-------|-----|------|--------|-----------|----------|------------|-----------------|---------|
| 1     | 2021 | 1     | 24  | 8    | 0      | 120.1     | 35.4     | 94.0       | 45.0            | Fog     |
| 2     | 2021 | 2     | 20  | 7    | 0      | 121.4     | 40.1     | 96.0       | 5.0             | Unfog   |
| 3     | 2021 | 3     | 4   | 6    | 0      | 122.5     | 30.4     | 96.0       | 5.0             | Unfog   |
| 4     | 2021 | 4     | 28  | 0    | 0      | 122.4     | 31.3     | 91.0       | 47.0            | Fog     |
| 5     | 2021 | 5     | 20  | 3    | 0      | 122.8     | 29.8     | 94.0       | 63.0            | Unfog   |
| 6     | 2021 | 6     | 13  | 8    | 0      | 122.0     | 38.8     | 94.0       | 45.0            | Fog     |
| 7     | 2021 | 7     | 11  | 4    | 0      | 126.2     | 32.9     | 96.0       | 2.0             | Unfog   |
| 8     | 2021 | 8     | 10  | 6    | 0      | 125.5     | 32.0     | 98.0       | 2.0             | Unfog   |
