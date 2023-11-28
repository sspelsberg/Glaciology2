
import math

x1 = 69577.37
x2 = 69488.03
y1 = 16321.42
y2 = 16119.59
z1 = 3850.29
z2 = 3840.78

dhdx = abs(z2-z1) / (math.sqrt((x2-x1)**2 + (y2-y1)**2))

rho = 850
g = 9.81
H = 536

tau = rho * g * H * dhdx

A = 9.3 * 10**(-25)
n = 3
seconds_year = 60 * 60 * 24 * 365

vds = (2*A/(n+1)) * H * tau**n * seconds_year

print()