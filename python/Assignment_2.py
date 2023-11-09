


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime
import rasterio as rio
import matplotlib.colors
from matplotlib.patches import Patch
import re


# Open rasters and masks
with rio.open('data_raw/lake_dem_2004.asc') as src1:
    lake_2004 = src1.read(1)

with rio.open('data_raw/lake_dem_2015.asc') as src2:
    lake_2015 = src2.read(1)

with rio.open('data_raw/lake_dem_2023.asc') as src3:
    lake_2023 = src3.read(1)


# Compute resolution
transform = src1.transform

# Get the resolution in the X and Y directions
resolution_x = transform.a
resolution_y = transform.e * (-1) # otherwise negative


# Spillway data
spillways = pd.DataFrame(data = np.array([[2004, 2100, 10975, 50], [2015, 2150, 10950, 58], [2023, 2150, 10950, 64]]),
                        columns = ["year", "easting (m)", "northing (m)", "peak discharge"])

