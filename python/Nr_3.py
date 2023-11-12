#!/usr/bin/env python
# coding: utf-8

# In[206]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime
import rasterio as rio
import matplotlib.colors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import math


# Open rasters and masks
with rio.open('Glaciology2/data_raw/lake_dem_2004.asc') as src1:
    lake_2004 = src1.read(1)

with rio.open('Glaciology2/data_raw/lake_dem_2015.asc') as src2:
    lake_2015 = src2.read(1)

with rio.open('Glaciology2/data_raw/lake_dem_2023.asc') as src3:
    lake_2023 = src3.read(1)

# Spillway data
spillways = pd.DataFrame(data = np.array([[2004, 2100, 10975, 50], [2015, 2150, 10950, 58], [2023, 2150, 10950, 64]]),
                        columns = ["year", "easting (m)", "northing (m)", "peak discharge"])

# Compute resolution
transform = src1.transform
bounds = src2.bounds

# Get the resolution in the X and Y directions
res_x = transform.a
res_y = transform.e # value is negative

# coordinates of upper left corner and spillways
upper_left = transform * (0,0)
sp_coord_2004 = (spillways.iloc[0,1], spillways.iloc[0,2])
sp_coord_2015 = (spillways.iloc[1,1], spillways.iloc[1,2])
sp_coord_2023 = (spillways.iloc[2,1], spillways.iloc[2,2])

# get spillway index
def get_index (xy):
    '''Returns tuple with index from tuple of coordinates.'''    
    distance = (xy[0] - upper_left[0], xy[1] - upper_left[1]) # distance in meters
    # use ceil to round up (distance in cells of 1.2 means you're in gridcell 2)
    index = (math.ceil(distance[0]/res_x), math.ceil(distance[1]/res_y)) # position index (distance in cells)
    return index

# add index and elevation to dataframe
spillways["index"] = (get_index(sp_coord_2004), get_index(sp_coord_2015), get_index(sp_coord_2023))
spillways["elevation"] = (lake_2004[spillways["index"][0]], lake_2015[spillways["index"][1]], lake_2023[spillways["index"][2]])

# calculate dems with elevation difference
dem_elev_diff_2004 = lake_2004 - spillways["elevation"][0]
dem_elev_diff_2015 = lake_2015 - spillways["elevation"][1]
dem_elev_diff_2023 = lake_2023 - spillways["elevation"][2]

# create mask (only areas below 0) 
mask_lake_2004 = (np.where(dem_elev_diff_2004 <= 0, 1, 0))
mask_lake_2015 = (np.where(dem_elev_diff_2015 <= 0, 1, 0))
mask_lake_2023 = (np.where(dem_elev_diff_2023 <= 0, 1, 0))

# create dems with lake depth - negative
dem_lake_depth_2004 = dem_elev_diff_2004 * mask_lake_2004
dem_lake_depth_2015 = dem_elev_diff_2015 * mask_lake_2015
dem_lake_depth_2023 = dem_elev_diff_2023 * mask_lake_2023

# compute lake volume - gets positive due to negative res_y
v_lake_2004 = round(dem_lake_depth_2004.sum() * res_x * res_y / (10**6), ndigits=4)
v_lake_2015 = round(dem_lake_depth_2015.sum() * res_x * res_y / (10**6), ndigits=4)
v_lake_2023 = round(dem_lake_depth_2023.sum() * res_x * res_y / (10**6), ndigits=4)

# compute glacier mask
dem_thinning = lake_2004 - lake_2023
# thinning treshhold of 3 meters accounts for height uncertainties and delivers better visual result than 0
mask_glacier = (np.where(dem_thinning >= 3, 1, 0)) 

# for plotting: remove cells outside of lake
dem_lake_depth_2004[mask_lake_2004 == 0] = np.nan
dem_lake_depth_2015[mask_lake_2015 == 0] = np.nan
dem_lake_depth_2023[mask_lake_2023 == 0] = np.nan

# plot lake area and depth
fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(12, 6))
cmap_terrain = matplotlib.colors.LinearSegmentedColormap.from_list("", ["greenyellow", "darkgreen", "khaki", "sienna", "saddlebrown"])
im1 = axs[0].imshow(lake_2004, cmap=cmap_terrain, alpha=0.8, vmin= 2500, vmax = 2850)
im2 = axs[1].imshow(lake_2015, cmap=cmap_terrain, alpha=0.8, vmin= 2500, vmax = 2850)
im3 = axs[2].imshow(lake_2023, cmap=cmap_terrain, alpha=0.8, vmin= 2500, vmax = 2850)
im4 = axs[0].imshow(dem_lake_depth_2004, cmap="Blues_r", vmin= -65, vmax = 0) # adjust all grids to use the same colorbar
im5 = axs[1].imshow(dem_lake_depth_2015, cmap="Blues_r", vmin= -65, vmax = 0)
im6 = axs[2].imshow(dem_lake_depth_2023, cmap="Blues_r", vmin= -65, vmax = 0)
axs[0].contour(mask_lake_2004, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[1].contour(mask_lake_2015, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[2].contour(mask_lake_2023, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[0].contour(mask_glacier, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[1].contour(mask_glacier, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
axs[2].contour(mask_glacier, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
# insert here: code to hatch the rock area
# axs[0].contourf(mask_glacier, levels=..., colors="None", hatches=["xx", None])
axs[0].set_title("Lake Area 2004")
axs[1].set_title("Lake Area 2015")
axs[2].set_title("Lake Area 2023")
axs[0].set_ylabel("# Cells")
axs[0].set_xlabel("# Cells")
axs[1].set_xlabel("# Cells")
axs[2].set_xlabel("# Cells")
fig.colorbar(im6, ax=axs, orientation='vertical', label="Lake depth [m]")
fig.colorbar(im3, ax=axs, orientation='vertical', label="Elevation around lake area [m]")


print("-"*8, "RESULTS", "-"*8)
print()
print("Lake Volume 2004: ", v_lake_2004, "*10^6 m3")
print("Lake Volume 2015: ", v_lake_2015, "*10^6 m3")
print("Lake Volume 2023: ", v_lake_2023, "*10^6 m3")
print("-"*25)


# In[ ]:




