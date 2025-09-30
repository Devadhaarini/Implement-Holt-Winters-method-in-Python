# Ex.No: 06                                       HOLT WINTERS METHOD
### Date: 



### AIM:
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and
Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv(
    "co2-annmean-gl.csv",
    parse_dates=['Year'],
    index_col='Year'
)

# Use only the 'Mean' column
data_mean = data[['Mean']]

# Scale
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_mean.values.reshape(-1, 1)).flatten(),
    index=data_mean.index
)

# Decompose (using additive because annual global CO₂ rise is mainly trend-driven)
decomposition = seasonal_decompose(data_mean, model="additive", period=10)  
decomposition.plot()
plt.show()

# Train-test split
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Holt-Winters Model
model = ExponentialSmoothing(
    train_data, trend='add', seasonal=None, seasonal_periods=None
).fit()

# Forecast
test_predictions = model.forecast(steps=len(test_data))

# Plot train, test, predictions
ax = train_data.plot(label="Train")
test_data.plot(ax=ax, label="Test")
test_predictions.plot(ax=ax, label="Predictions")
ax.legend()
ax.set_title("Visual Evaluation of Holt-Winters Model")
plt.show()

# RMSE
print("RMSE:", np.sqrt(mean_squared_error(test_data, test_predictions)))

# Final Model on full dataset
final_model = ExponentialSmoothing(
    scaled_data, trend='add', seasonal=None, seasonal_periods=None
).fit()
final_predictions = final_model.forecast(steps=20)  # Forecast next 20 years

# Inverse scale back to CO₂ ppm
final_predictions_rescaled = scaler.inverse_transform(
    final_predictions.values.reshape(-1, 1)
)

# Plot final forecast
ax = data_mean.plot(label="Historical")
pd.Series(final_predictions_rescaled.flatten(),
          index=pd.date_range(start=data_mean.index[-1] + pd.offsets.YearBegin(),
                              periods=20, freq='YS')).plot(ax=ax, label="Forecast")
ax.set_xlabel("Year")
ax.set_ylabel("CO₂ concentration (ppm)")
ax.set_title("Global Annual Mean CO₂ Forecast (Holt-Winters)")
ax.legend()
plt.show()
```
### OUTPUT:

<img width="660" height="471" alt="image" src="https://github.com/user-attachments/assets/6fe89711-5035-48b0-b238-1286c86a0a7d" />
<img width="601" height="483" alt="image" src="https://github.com/user-attachments/assets/a3572632-c5cf-400d-b0f6-f2c4e3ffff97" />
<img width="588" height="455" alt="image" src="https://github.com/user-attachments/assets/a8d716cb-24e9-471f-a600-86162de81a83" />

### RESULT:
Thus the implementation of Holt Winters method model using python is done successfully.
