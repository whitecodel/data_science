import numpy as np
import matplotlib.pyplot as plt

# Known data points
days = [1, 3, 5]
temperatures = [25, 30, 35]

# Interpolate the value for day 2 using linear interpolation
day_2_temp = np.interp(2, days, temperatures)

# Plot the known points
plt.scatter(days, temperatures, color='blue', label='Known Temperatures')

# Plot the interpolated value
plt.scatter(2, day_2_temp, color='red', label='Interpolated Value for Day 2')
plt.plot(days, temperatures, linestyle='--', color='green', label='Trend Line')

plt.title('Interpolation Example')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
