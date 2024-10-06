import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

st.title("Stock Price Predictor App") 

# User input for stock ID
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set date range for data retrieval
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download historical stock data
google_data = yf.download(stock, start, end)

# Load pre-trained model
model = load_model("Latest_stock_price_models.keras")

# Display the stock data in Streamlit
st.subheader("Stock Data")
st.write(google_data)

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None, title="Stock Price Graph", xlabel="Date", ylabel="Price (USD)"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(values, color='Orange', label='Calculated Values')
    ax.plot(full_data.Close, color='b', label='Actual Stock Data')
    if extra_data:
        ax.plot(extra_dataset, label='Additional Data')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig

# Plot Moving Averages for 100, 200, and 250 days
ma_days = [100, 200, 250]
for ma in ma_days:
    st.subheader(f'Original Close Price and MA for {ma} days')
    google_data[f'MA_for_{ma}_days'] = google_data.Close.rolling(ma).mean()
    st.pyplot(plot_graph((15, 6), google_data[f'MA_for_{ma}_days'], google_data, 0, title=f'Moving Average {ma} Days'))

# Plot comparison of MA for 100 days and 250 days
st.subheader('Original Close Price and MA for 100 and 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days'], title="MA 100 vs MA 250"))

# Data Preprocessing and Prediction Preparation
scaler = MinMaxScaler(feature_range=(0, 1))
splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predictions using the loaded model
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Create a DataFrame for plotting
ploting_data = pd.DataFrame({
    'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len+100:])

# Display original vs predicted values
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot original vs predicted values
st.subheader('Original Close Price vs Predicted Close Price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

# Future Predictions for 15 Days
future_days = 15  # Number of days to predict into the future

# Fit the scaler on the entire 'Adj Close' column
scaler_full = MinMaxScaler(feature_range=(0, 1))
scaled_data_full = scaler_full.fit_transform(google_data['Adj Close'].values.reshape(-1, 1))

# Use the last 60 days for prediction
last_60_days_scaled = scaled_data_full[-60:]

# Empty list for storing future predictions
future_predictions = []

# Predict future prices day by day
for _ in range(future_days):
    last_60_days_reshaped = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0], 1))
    predicted_price = model.predict(last_60_days_reshaped)
    future_predictions.append(predicted_price[0, 0])
    last_60_days_scaled = np.append(last_60_days_scaled, predicted_price)[1:]

# Convert the predictions back to the original scale
future_predictions = scaler_full.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create a pandas DataFrame for the predicted data
dates_future = pd.date_range(start=google_data.index[-1] + pd.Timedelta(days=1), periods=future_days)
predicted_df = pd.DataFrame(future_predictions, columns=["Predicted Closing Price (USD)"], index=dates_future)

# Plot only the predicted prices
st.subheader('Predicted Prices for the Next 15 Days')
fig = plt.figure(figsize=(15, 5))
plt.plot(predicted_df, label='Predicted Prices for Next 15 Days', color='green')
plt.title(f'{stock} Stock Price Prediction for the Next 15 Days')
plt.xlabel('Date')
plt.ylabel('Predicted Closing Price (USD)')
plt.legend()
st.pyplot(fig)

# Plot the historical data and predicted prices
st.subheader('Historical Data and Predicted Prices')
fig = plt.figure(figsize=(15, 5))
plt.plot(google_data['Adj Close'], label='Historical Adjusted Close Price', color='blue')
plt.plot(predicted_df, label='Predicted Prices for Next 15 Days', color='red')
plt.title(f'Stock Price Prediction for the Next {future_days} Days')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Display the predicted prices
st.subheader('Predicted Prices Table for the Next 15 Days')
st.write(predicted_df)


# Compare yesterday's actual and predicted prices
st.subheader("Yesterday's Adj Close vs Predicted Price")
yesterday_adj_close = google_data['Adj Close'].iloc[-1]  # Last actual price
yesterday_prediction = future_predictions[0][0]  # First predicted price

# Display yesterday's Adj Close price and predicted price
st.write(f"Yesterday's Adj Close Price: ${yesterday_adj_close:.2f}")
st.write(f"Yesterday's Predicted Stock Price: ${yesterday_prediction:.2f}")

# Plot the results for comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the Adj Close price
ax.plot(google_data.index[-100:], google_data['Adj Close'][-100:], label='Adj Close Price', color='blue')

# Highlight yesterday's actual Adj Close price
ax.scatter(google_data.index[-1], yesterday_adj_close, color='blue', s=100, label="Yesterday's Adj Close")

# Plot the predicted price
ax.plot(predicted_df.index, predicted_df['Predicted Closing Price (USD)'], label='Predictions', color='red', linestyle='--')

# Highlight yesterday's predicted price
ax.scatter(predicted_df.index[0], yesterday_prediction, color='red', s=100, label="Yesterday's Prediction")

# Add title and labels
ax.set_title('Adj Close vs Predicted Stock Price')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Comparison of Last Actual and Predicted Prices in Test Set
st.subheader("Test Set: Last Actual vs Predicted Price")
last_actual_test_price = inv_y_test[-1][0]
last_predicted_test_price = inv_pre[-1][0]

st.write(f"Last Actual Test Price: ${last_actual_test_price:.2f}")
st.write(f"Last Predicted Test Price: ${last_predicted_test_price:.2f}")

# Plot the comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the actual test prices
ax.plot(google_data.index[splitting_len+100:], inv_y_test.reshape(-1), label='Actual Test Prices', color='blue')

# Plot the predicted test prices
ax.plot(google_data.index[splitting_len+100:], inv_pre.reshape(-1), label='Predicted Test Prices', color='red', linestyle='--')

# Highlight last actual test price
ax.scatter(google_data.index[-1], last_actual_test_price, color='blue', s=100, label="Last Actual Test Price")

# Highlight last predicted test price
ax.scatter(google_data.index[-1], last_predicted_test_price, color='red', s=100, label="Last Predicted Test Price")

ax.set_title('Test Set: Actual vs Predicted Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Show today's stock price
today_price = google_data['Adj Close'].iloc[-1]
st.subheader(f"Today's Stock Price: ${today_price:.2f}")

# Plot today's stock price
today_df = pd.DataFrame({
    'Date': [google_data.index[-1]],
    'Price (USD)': [today_price]
})

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(today_df['Date'], today_df['Price (USD)'], color='blue', s=100, label=f"Today's Stock Price: ${today_price:.2f}")
ax.set_title(f"Today's {stock} Stock Price")
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True)
st.pyplot(fig)
st.write("THANK YOU...")
