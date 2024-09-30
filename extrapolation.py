import numpy as np
import matplotlib.pyplot as plt

# Known data points
days = [1, 3, 5]
temperatures = [25, 30, 35]

# Extrapolate the value for day 6 using the trend
m, c = np.polyfit(days, temperatures, 1)  # Fit a linear model
day_6_temp = m * 6 + c  # Extrapolate for day 6

# Plot the known points
plt.scatter(days, temperatures, color='blue', label='Known Temperatures')

# Plot the extrapolated value
plt.scatter(6, day_6_temp, color='red', label='Extrapolated Value for Day 6')
plt.plot([1, 6], [25, day_6_temp], linestyle='--', color='green', label='Trend Line')

plt.title('Extrapolation Example')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
