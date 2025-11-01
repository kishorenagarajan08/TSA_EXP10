# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 18/10/2025

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load TSLA dataset
data = pd.read_csv('/content/TSLA.csv')


# Convert Date to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot Close price
plt.plot(data.index, data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('TSLA Close Price Time Series')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity of Close price
check_stationarity(data['Close'])

# Plot ACF and PACF
plot_acf(data['Close'])
plt.show()

plot_pacf(data['Close'])
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Model Predictions on TSLA Close Price')
plt.legend()
plt.show()
```
### OUTPUT:

<img width="507" height="453" alt="image" src="https://github.com/user-attachments/assets/a85219b7-d62e-4281-89a1-aedea57407a7" />
<img width="496" height="442" alt="image" src="https://github.com/user-attachments/assets/de65c43d-e117-4e19-be72-d1d8850067eb" />
<img width="578" height="438" alt="image" src="https://github.com/user-attachments/assets/d4978521-ddd5-49d4-9b31-128e6dbafc8b" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
