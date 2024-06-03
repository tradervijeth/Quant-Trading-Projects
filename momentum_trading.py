import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
dates = pd.date_range('2020-01-01', '2021-01-01')
prices = np.cumsum(np.random.randn(len(dates))) + 100

# Create a DataFrame
df = pd.DataFrame(data={'Price': prices}, index=dates)

# Define the momentum window
momentum_window = 20

# Calculate the momentum
df['Momentum'] = df['Price'].diff(momentum_window)

# Generate signals
df['Signal'] = 0
df['Signal'][momentum_window:] = np.where(df['Momentum'][momentum_window:] > 0, 1, -1)

# Shift signals to align with trades
df['Signal'] = df['Signal'].shift(1)

# Calculate daily returns
df['Daily_Return'] = df['Price'].pct_change()

# Calculate strategy returns
df['Strategy_Return'] = df['Signal'] * df['Daily_Return']

# Calculate cumulative returns
df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df['Cumulative_Return'], label='Strategy Return', color='g')
plt.plot((1 + df['Daily_Return']).cumprod(), label='Buy and Hold Return', color='b')
plt.title('Momentum Trading Strategy')
plt.legend()
plt.show()
