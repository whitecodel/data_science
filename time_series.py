import matplotlib.pyplot as plt

# Sample temperature data for 30 days
days = list(range(1, 31))
temperature = [28, 30, 31, 29, 32, 34, 33, 35, 36, 37, 38, 36, 35, 34, 33, 
               32, 31, 30, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41]

# Plotting the time series
# plt.plot(days, temperature, marker='o')
# plt.title('Temperature over 30 Days')
# plt.xlabel('Days')
# plt.ylabel('Temperature (°C)')
# plt.grid(True)
# plt.show()

import pandas as pd

# # Convert to pandas series
temp_series = pd.Series(temperature)

# # Difference the data to remove trend
# temp_diff = temp_series.diff().dropna()

# # Plotting differenced data
# plt.plot(temp_diff, marker='o')
# plt.title('Differenced Temperature Data')
# plt.xlabel('Days')
# plt.ylabel('Temperature Difference (°C)')
# plt.grid(True)
# plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model
model = ARIMA(temp_series, order=(1, 1, 1))  # (AR, I, MA) orders
model_fit = model.fit()

# Forecast the next 5 days
forecast = model_fit.forecast(steps=5)
print(forecast)


plt.plot(days, temperature, label='Actual')
plt.plot(range(31, 36), forecast, label='Forecast', marker='o')
plt.title('Temperature Forecast')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
