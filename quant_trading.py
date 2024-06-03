import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
dates = pd.date_range('2020-01-01', '2021-01-01')
prices = np.cumsum(np.random.randn(len(dates))) + 100

# Create a DataFrame
df = pd.DataFrame(data={'Price': prices}, index=dates)

# Define the moving averages
short_window = 40
long_window = 100

# Calculate the moving averages
df['Short_MA'] = df['Price'].rolling(window=short_window, min_periods=1).mean()
df['Long_MA'] = df['Price'].rolling(window=long_window, min_periods=1).mean()

# Generate signals
df['Signal'] = 0
df['Signal'][short_window:] = np.where(df['Short_MA'][short_window:] > df['Long_MA'][short_window:], 1, 0)
df['Position'] = df['Signal'].diff()

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df['Price'], label='Price', color='k')
plt.plot(df['Short_MA'], label='40-day MA', color='b')
plt.plot(df['Long_MA'], label='100-day MA', color='r')

# Plot buy signals
plt.plot(df[df['Position'] == 1].index, 
         df['Short_MA'][df['Position'] == 1], 
         '^', markersize=10, color='g', lw=0, label='Buy Signal')

# Plot sell signals
plt.plot(df[df['Position'] == -1].index, 
         df['Short_MA'][df['Position'] == -1], 
         'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('Moving Average Crossover Strategy')
plt.legend()
plt.show()
