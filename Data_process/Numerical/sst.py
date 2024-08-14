import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import os
import netCDF4 as nc
input_dir = './' # 请修改存储sst-nc的文件地址
output_dir = './'# 请修改存储图片的文件地址
paths = [x for x in os.listdir(input_dir) if x.split('.')[-1]=='nc']
for path in paths:
        pathdir = input_dir + path
        data = xr.open_dataset(pathdir)

        # 提取特定经纬度范围的数据
        lat_range = slice(28.7, 41.5)  # 纬度范围
        lon_range = slice(116.2, 129.0)  # 经度范围

        subset = data.sel(lat=lat_range, lon=lon_range)
        numpy_array = subset['sst'].values  
        output_filename = path.replace('.nc', '')
        np.save(output_dir + output_filename + '.npy', numpy_array)

        data = xr.open_dataset(pathdir)
        day = '2018-01-01' #  请修改可视化的日期
        first_day_sst = data['sst'].sel(time=day).values
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines()
        plt.contourf(data['lon'], data['lat'], first_day_sst, transform=ccrs.PlateCarree(), cmap='coolwarm')
        plt.colorbar(label='Sea Surface Temperature (SST)', orientation='horizontal', shrink=0.6, pad=0.1)
        plt.title('Sea Surface Temperature')
        plt.savefig(output_dir + 'sst.day.mean.' + day +'.png', dpi=300, bbox_inches='tight')
        