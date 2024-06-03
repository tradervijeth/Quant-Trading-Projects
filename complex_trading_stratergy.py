import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch historical data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Adj Close']]
    data.rename(columns={'Adj Close': 'Price'}, inplace=True)
    return data

# Calculate MACD
def MACD(df, short_window=12, long_window=26, signal_window=9):
    df['Short_EMA'] = df['Price'].ewm(span=short_window, adjust=False).mean()
    df['Long_EMA'] = df['Price'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['Short_EMA'] - df['Long_EMA']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Signal'] = np.where(df['MACD'] > df['Signal_Line'], 1, -1)
    return df

# Calculate RSI
def RSI(df, window=14):
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    return df

# Calculate Bollinger Bands
def Bollinger_Bands(df, window=20):
    df['Middle_Band'] = df['Price'].rolling(window).mean()
    df['Std_Dev'] = df['Price'].rolling(window).std()
    df['Upper_Band'] = df['Middle_Band'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Std_Dev'] * 2)
    df['Bollinger_Signal'] = np.where(df['Price'] < df['Lower_Band'], 1, np.where(df['Price'] > df['Upper_Band'], -1, 0))
    return df

# Combine signals
def combine_signals(df):
    df['Combined_Signal'] = df['MACD_Signal'] + df['RSI_Signal'] + df['Bollinger_Signal']
    df['Position'] = np.where(df['Combined_Signal'] > 1, 1, np.where(df['Combined_Signal'] < -1, -1, 0))
    return df

# Risk management - position sizing
def position_sizing(df, risk_per_trade=0.01, initial_balance=100000):
    df['Cash'] = initial_balance
    df['Position_Size'] = 0
    df['Equity'] = initial_balance

    for i in range(1, len(df)):
        if df['Position'].iloc[i] == 1:
            df.loc[df.index[i], 'Position_Size'] = df['Cash'].iloc[i-1] * risk_per_trade / df['Price'].iloc[i]
        elif df['Position'].iloc[i] == -1:
            df.loc[df.index[i], 'Position_Size'] = -df['Cash'].iloc[i-1] * risk_per_trade / df['Price'].iloc[i]
        df.loc[df.index[i], 'Cash'] = df['Cash'].iloc[i-1] - (df['Position_Size'].iloc[i] * df['Price'].iloc[i])
        df.loc[df.index[i], 'Equity'] = df['Cash'].iloc[i] + (df['Position_Size'].iloc[i] * df['Price'].iloc[i])

    return df

# Performance evaluation
def performance_evaluation(df):
    df['Strategy_Return'] = df['Equity'].pct_change()
    sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)
    drawdown = df['Equity'].div(df['Equity'].cummax()) - 1
    max_drawdown = drawdown.min()
    return sharpe_ratio, max_drawdown

# Main function to run the strategy
def run_strategy(ticker, start_date, end_date):
    df = fetch_data(ticker, start_date, end_date)
    df = MACD(df)
    df = RSI(df)
    df = Bollinger_Bands(df)
    df = combine_signals(df)
    df = position_sizing(df)

    # Performance metrics
    sharpe_ratio, max_drawdown = performance_evaluation(df)
    print(f'Sharpe Ratio: {sharpe_ratio}')
    print(f'Maximum Drawdown: {max_drawdown}')

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(df['Equity'], label='Equity Curve', color='g')
    plt.title('Complex Trading Strategy Performance')
    plt.legend()
    plt.show()

# Run the strategy
run_strategy('AAPL', '2020-01-01', '2021-01-01')
