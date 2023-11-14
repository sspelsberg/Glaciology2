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


# Questions to be answered: 
#
# 1) nr 3.1 – which threshhold should we use to distinguish glacier and rock? 
#  --> 3 m thinning works quite well, > 0 m thinning is too patchy probably due to snow cover
# 
# 2) nr 3.3 – Interpolation of the lake: which lake surface elevation should we assume for the predictions of lake volume?
#  --> possible: spillway location remains the same --> surface elevation = future spillway elevation?
# 
# 3) nr 3.3 – should we somehow account for glacier retreat? 
# 
# nr 3.3 not finished
# nr 3.1: shrink color bars



# Open rasters and masks
with rio.open('data_raw/lake_dem_2004.asc') as src1:
    lake_2004 = src1.read(1)

with rio.open('data_raw/lake_dem_2015.asc') as src2:
    lake_2015 = src2.read(1)

with rio.open('data_raw/lake_dem_2023.asc') as src3:
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



# LAKES --------------------------------

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
    # switch coordinates (index goes first row, then column)
    index = (math.ceil(distance[1]/res_y), math.ceil(distance[0]/res_x)) # distance in cells
    return index

# add index and elevation to dataframe
spillways["index"] = (get_index(sp_coord_2004), get_index(sp_coord_2015), get_index(sp_coord_2023))
spillways["elevation"] = (lake_2004[spillways["index"][0]], lake_2015[spillways["index"][1]], lake_2023[spillways["index"][2]])

# calculate dems with elevation difference
dem_elev_diff_2004 = lake_2004 - spillways["elevation"][0]
dem_elev_diff_2015 = lake_2015 - spillways["elevation"][1]
dem_elev_diff_2023 = lake_2023 - spillways["elevation"][2]

# create lake mask (only areas below 0) 
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




# GLACIER -------------------------------

# compute thinning
dem_thinning_2004 = lake_2023 - lake_2004 
dem_thinning_2015 = lake_2023 - lake_2015
dem_thinning_rate = dem_thinning_2015 / (2023-2015) # m/yr

# compute glacier mask
# thinning treshhold of 3 meters accounts for height uncertainties/snow and delivers better visual result than 0
mask_glacier_2004 = (np.where(dem_thinning_2004 <= -3, 1, 0))
mask_glacier_2015 = (np.where(dem_thinning_2015 <= -3, 1, 0))

# compute seperate glacier masks with and without lake
mask_glacier_no_lake = (np.where((mask_glacier_2015 == 1) & (mask_lake_2023 == 0), 1, 0))
mask_glacier_lake = (np.where((mask_glacier_2015 == 1) & (mask_lake_2023 == 1), 1, 0))
mask_rock = (np.where(mask_glacier_2015 == 0, 1, 0))

# compute thinning rates for glacier below lake and glacier
thinning_rate_lake = (dem_thinning_rate * mask_glacier_lake).sum() / mask_glacier_lake.sum() # - m/yr
thinning_rate_glacier = (dem_thinning_rate * mask_glacier_no_lake).sum() / mask_glacier_no_lake.sum() # - m/yr



# PREDICT FUTURE LAKE VOLUME -------------------

# compute future dems with thinning in glacier areas 
# all based on lake area 2023. should we use future lake areas for the mask of the lake?
def get_future_dem(year):
    n_years = year - 2023
    dem_rock = lake_2023 * mask_rock
    dem_glacier = (lake_2023 + thinning_rate_glacier * n_years) * mask_glacier_no_lake
    dem_lake = (lake_2023 + thinning_rate_lake * n_years) * mask_glacier_lake
    return dem_rock + dem_glacier + dem_lake

lake_2024 = get_future_dem(2024)
lake_2027 = get_future_dem(2027)
lake_2032 = get_future_dem(2032)

# add new spillway data
new_years = spillways.copy()
new_years["year"] = [2024, 2027, 2032]
new_years["easting (m)"] = 2150
new_years["northing (m)"] = 10950
new_years["peak discharge"] = np.nan 
new_years["index"] = ((15, 7), (15, 7), (15, 7))
new_years["elevation"] = (lake_2024[new_years["index"][0]], lake_2027[new_years["index"][1]], lake_2032[new_years["index"][2]]) 

spillways = pd.concat([spillways, new_years], ignore_index = True)

# calculate dems with elevation difference
dem_elev_diff_2024 = lake_2024 - spillways["elevation"][3]
dem_elev_diff_2027 = lake_2027 - spillways["elevation"][4]
dem_elev_diff_2032 = lake_2032 - spillways["elevation"][5]

# create lake mask (only areas below 0) 
mask_lake_2024 = (np.where(dem_elev_diff_2024 <= 0, 1, 0))
mask_lake_2027 = (np.where(dem_elev_diff_2027 <= 0, 1, 0))
mask_lake_2032 = (np.where(dem_elev_diff_2032 <= 0, 1, 0))

# create dems with lake depth - negative
dem_lake_depth_2024 = dem_elev_diff_2024 * mask_lake_2024
dem_lake_depth_2027 = dem_elev_diff_2027 * mask_lake_2027
dem_lake_depth_2032 = dem_elev_diff_2032 * mask_lake_2032

# compute lake volume - gets positive due to negative res_y
v_lake_2024 = round(dem_lake_depth_2024.sum() * res_x * res_y / (10**6), ndigits=4)
v_lake_2027 = round(dem_lake_depth_2027.sum() * res_x * res_y / (10**6), ndigits=4)
v_lake_2032 = round(dem_lake_depth_2032.sum() * res_x * res_y / (10**6), ndigits=4)



# PLOTS & RESULTS ----------------------

# for plotting: remove cells outside of lake
dem_lake_depth_2004[mask_lake_2004 == 0] = np.nan
dem_lake_depth_2015[mask_lake_2015 == 0] = np.nan
dem_lake_depth_2023[mask_lake_2023 == 0] = np.nan
dem_lake_depth_2024[mask_lake_2024 == 0] = np.nan
dem_lake_depth_2027[mask_lake_2027 == 0] = np.nan
dem_lake_depth_2032[mask_lake_2032 == 0] = np.nan

# for plotting: remove glacierized cells
dem_elev_diff_2004[mask_glacier_2004 == 1] = np.nan

# plot lake area and depth
fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(12, 6))
cmap_terrain = matplotlib.colors.LinearSegmentedColormap.from_list("", ["greenyellow", "darkgreen", "khaki", "sienna", "saddlebrown"])
im1 = axs[0].imshow(lake_2004, cmap=cmap_terrain, alpha=0.8, vmin= 2450, vmax = 2850)
im2 = axs[1].imshow(lake_2015, cmap=cmap_terrain, alpha=0.8, vmin= 2450, vmax = 2850)
im3 = axs[2].imshow(lake_2023, cmap=cmap_terrain, alpha=0.8, vmin= 2450, vmax = 2850)
axs[0].contourf(dem_elev_diff_2004, colors="white", hatches=["..."], alpha=0) # hatching of rock area
axs[1].contourf(dem_elev_diff_2004, colors="white", hatches=["..."], alpha=0)
axs[2].contourf(dem_elev_diff_2004, colors="white", hatches=["..."], alpha=0)
im4 = axs[0].imshow(dem_lake_depth_2004, cmap="Blues_r", vmin= -65, vmax = 0) # adjust all grids to use the same colorbar
im5 = axs[1].imshow(dem_lake_depth_2015, cmap="Blues_r", vmin= -65, vmax = 0)
im6 = axs[2].imshow(dem_lake_depth_2023, cmap="Blues_r", vmin= -65, vmax = 0)
axs[0].contour(mask_lake_2004, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[1].contour(mask_lake_2015, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[2].contour(mask_lake_2023, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[0].contour(mask_glacier, levels=[0.5], colors='black', linestyles='solid', linewidths=1, alpha=0.7)
axs[1].contour(mask_glacier, levels=[0.5], colors='black', linestyles='solid', linewidths=1, alpha=0.7)
axs[2].contour(mask_glacier, levels=[0.5], colors='black', linestyles='solid', linewidths=1, alpha=0.7)
axs[0].set_title("Lake Area 2004")
axs[1].set_title("Lake Area 2015")
axs[2].set_title("Lake Area 2023")
axs[0].set_ylabel("# Cells")
axs[0].set_xlabel("# Cells")
axs[1].set_xlabel("# Cells")
axs[2].set_xlabel("# Cells")
fig.colorbar(im6, ax=axs, orientation='vertical', label="Lake depth [m]")
fig.colorbar(im3, ax=axs, orientation='vertical', label="Elevation around lake area [m]")
plt.savefig("Glaciology2/plots/lake_depth.png", dpi=300)
plt.show()


# for plotting: remove cells that should not be plotted
dem_thinning_rate[mask_glacier_2015 == 0] = np.nan
lake_2004[mask_glacier_2015 == 1] = np.nan

# Plot Glacial thinning 
plt.imshow(dem_thinning_rate, cmap="Blues_r")
plt.contour(mask_lake_2023, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
plt.contour(mask_glacier_2015, levels=[0.5], colors='black', linestyles='solid', linewidths=1)
plt.contourf(lake_2004, colors="white", hatches=["..."], alpha=0)
plt.title("Glacier elevation change 2015-2023")
plt.ylabel("# Cells")
plt.xlabel("# Cells")
plt.colorbar(fig2, orientation='vertical', label="Average thinning rate [m/yr]")
plt.savefig("Glaciology2/plots/thinning_rate.png", dpi=300)
plt.show()


# Plot future predicted lakes (area and depth)
fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(12, 6))
cmap_terrain = matplotlib.colors.LinearSegmentedColormap.from_list("", ["greenyellow", "darkgreen", "khaki", "sienna", "saddlebrown"])
im1 = axs[0].imshow(lake_2024, cmap=cmap_terrain, alpha=0.8, vmin= 2400, vmax = 2850)
im2 = axs[1].imshow(lake_2027, cmap=cmap_terrain, alpha=0.8, vmin= 2400, vmax = 2850)
im3 = axs[2].imshow(lake_2032, cmap=cmap_terrain, alpha=0.8, vmin= 2400, vmax = 2850)
axs[0].contourf(lake_2004, colors="white", hatches=["..."], alpha=0) # hatching of rock area
axs[1].contourf(lake_2004, colors="white", hatches=["..."], alpha=0)
axs[2].contourf(lake_2004, colors="white", hatches=["..."], alpha=0)
im4 = axs[0].imshow(dem_lake_depth_2024, cmap="Blues_r", vmin= -65, vmax = 0) # adjust all grids to use the same colorbar
im5 = axs[1].imshow(dem_lake_depth_2027, cmap="Blues_r", vmin= -65, vmax = 0)
im6 = axs[2].imshow(dem_lake_depth_2032, cmap="Blues_r", vmin= -65, vmax = 0)
axs[0].contour(mask_lake_2024, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[1].contour(mask_lake_2027, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[2].contour(mask_lake_2032, levels=[0.5], colors='blue', linestyles='solid', linewidths=1)
axs[0].contour(mask_glacier_2015, levels=[0.5], colors='black', linestyles='solid', linewidths=1, alpha=0.7)
axs[1].contour(mask_glacier_2015, levels=[0.5], colors='black', linestyles='solid', linewidths=1, alpha=0.7)
axs[2].contour(mask_glacier_2015, levels=[0.5], colors='black', linestyles='solid', linewidths=1, alpha=0.7)
axs[0].set_title("Lake Area 2024")
axs[1].set_title("Lake Area 2027")
axs[2].set_title("Lake Area 2032")
axs[0].set_ylabel("# Cells")
axs[0].set_xlabel("# Cells")
axs[1].set_xlabel("# Cells")
axs[2].set_xlabel("# Cells")
fig.colorbar(im6, ax=axs, orientation='vertical', label="Lake depth [m]")
fig.colorbar(im3, ax=axs, orientation='vertical', label="Elevation around lake area [m]")
plt.savefig("Glaciology2/plots/predicted_lakes.png", dpi=300)
plt.show()


print("-"*8, "RESULTS", "-"*8)
print()
print("Lake Volume 2004: ", v_lake_2004, "*10^6 m3")
print("Lake Volume 2015: ", v_lake_2015, "*10^6 m3")
print("Lake Volume 2023: ", v_lake_2023, "*10^6 m3")
print() 
print("Average thinning rate underneath lake: ", round(thinning_rate_lake, 3), "m/yr")
print("Average thinning rate of glacier without lake: ", round(thinning_rate_glacier, 3), "m/yr")
print()
print("Predicted Lake Volume 2024: ", v_lake_2024, "*10^6 m3")
print("Predicted Lake Volume 2027: ", v_lake_2027, "*10^6 m3")
print("Predicted Lake Volume 2032: ", v_lake_2032, "*10^6 m3")
print("-"*25)

print()
print("SPILLWAY TABLE")
print(spillways)

