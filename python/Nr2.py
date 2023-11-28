import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from statistics import mean


# Glacier flow-speed data 
# time  = time since beginning of the measurements (d)
# speed = measured ice flow speed (m/d)

# read in monitoring data
data = pd.read_csv("data_raw/maybeunstable_velocity_data.txt", sep='\s+')

# calculate start speed (first 10% of data points)
v0 = mean(data["speed"][:20])

# define function of speed with parameters to be fitted
def glacier_velocity(t, a, m, tc):
    vt = v0 + a / (tc - t) ** m
    return vt

# x and y variable
t = np.array(data["time"])  # predictor
vt = np.array(data["speed"]) # target

# set bounds (a > 0, m > 0, tc >= last day of measurement)
param_bounds = ([0.001,0.001,5636],[np.inf,np.inf,np.inf])

# fit parameters
popt, pcov = curve_fit(glacier_velocity, t, vt, bounds=param_bounds)

# Compute values for plotting the fitted curve
x_fit = np.linspace(min(t), max(t), 50)
y_fit = glacier_velocity(x_fit, popt[0], popt[1], popt[2])

# round values for output
popt = np.round(popt, 3)

# plot measured and modelled data
plt.scatter(data["time"], data["speed"], label="Measurements")
plt.plot(x_fit, y_fit, "r", label="Modelled relationship")
plt.title("Glacier instability velocity data")
plt.xlabel("Time since beginning of the measurements [d]")
plt.ylabel("Ice flow speed [m/d]")
plt.legend()
plt.savefig("plots/glacier_instability.png", dpi=300)
plt.show()

print("-"*8, "RESULTS", "-"*8)
print()
print("fitted a: ", popt[0])
print("fitted m: ", popt[1])
print("day of collapse (tc): ", popt[2])
print("-"*25)

