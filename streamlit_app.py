import math
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests


st.title ("stock market")

def get_ticker(name):
	company = yf.Ticker(name)
	return company
def plot_raw_data():
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data.Date, y=data['Open'], name="stock_open",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(
        x=df.Date, y=df['Close'], name="stock_close", line_color='deepskyblue'))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig

c1 = get_ticker("AAPL")
c2 = get_ticker("MSFT")
c3 = get_ticker("TSLA")

apple = yf.download("AAPL", start="11-11-2011",end="11-11-2022")
microsoft = yf.download("MSFT", start="11-11-2011",end="11-11-2022")
tesla= yf.download("TSLA", start="11-11-2011",end="11-11-2022")

data1 = c1. history(period="3mo")
data2 = c2. history(period="3mo")
data3 = c3. history(period="3mo")

st.write(""" ### Apple """)
st.write(c1.info['longBusinessSummary'])
st.write(apple)
st.line_chart(data1.values)

st.write(""" ### MSFT """)
st.write(c2.info['longBusinessSummary'])
st.write(microsoft)
st.line_chart(data2.values)

st.write(""" ### TSLA """)
st.write(c3.info['longBusinessSummary'])
st.write(tesla)
st.line_chart(data3.values)

#new
ticker = st.text_input('Ticker',"NFLX").upper()
buttonClicked = st.button('Set')

if buttonClicked:
	requestString=f"""https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile%2Cprice"""
	request=requests.get(f"{requestString}",headers={"USER-AGENT":"Mozilla/5.0"})
	json = request.json()
	data=json["quoteSummary"]["result"][0]

	st.header("Profile")

	st.metric("sector",data["assetProfile"]["sector"])
	st.metric("industry",data["assetProfile"]["industry"])
	st.metric("website",data["assetProfile"]["website"])
	st.metric("marketCap",data["price"]["marketCap"]["fmt"])

	with st.expander("About Company"):
		st.write(data["assetProfile"]["longBusinessSummary"])

	











#new

st.title('Stock prediction ')

stocks = ("RELIANCE.NS", "BHARTIARTL.NS", "ICICIBANK.NS", "TATASTEEL.NS")
selected_stock = st.selectbox("Select Stocks for prediction", stocks)



def load_data(ticker):
    data = yf.download(ticker)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
df = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Last Five Days')
st.write(df.tail())


def plot_raw_data():
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=data.Date, y=data['Open'], name="stock_open",line_color='deepskyblue'))
    fig.add_trace(go.Scatter(
        x=df.Date, y=df['Close'], name="stock_close", line_color='deepskyblue'))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig

plot_raw_data()

# Model

# imports

data_load_state = st.text('Loading please wait...')

data = df.filter(['Close'])
current_data = np.array(data).reshape(-1, 1).tolist()

df = np.array(data).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(np.array(df).reshape(-1, 1))
train_data = scaled_df[0:, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaled_df[-60:, :].tolist()
x_test = []
y_test = []
for i in range(60, 70):
    x_test = (test_data[i-60:i])
    x_test = np.asarray(x_test)
    pred_data = model.predict(x_test.reshape(1, x_test.shape[0], 1).tolist())

    y_test.append(pred_data[0][0])
    test_data.append(pred_data)

pred_next_10 = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1))

data_load_state.text('Loading Model... done!')




st.subheader("Next 5 Days")
st.write(pred_next_10)

pred_next_10 = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1))

data_load_state.text('Loading Model... done!')


st.subheader("Next 10 Days")
st.write(pred_next_10)



st.title('Charts Dashboard')

tickers = (
  "AACG",
  "AACI",
  "AACIU",
  "AACIW",
  "AADI",
  "AAL",
 
)

dropdown = st.multiselect('Select Assets you want', tickers)

start = st.date_input('Start', value = pd.to_datetime('2021-01-01'))
end = st.date_input('End', value = pd.to_datetime('today'))

def relativeret(df):
  rel = df.pct_change()
  cumret = (1+rel).cumprod() - 1
  cumret = cumret.fillna(0)
  return cumret

if len(dropdown) > 0:
  #df = yf.download(dropdown, start, end)['Adj Close']
  df = relativeret(yf.download(dropdown, start, end)['Adj Close'])
  st.header(f'Stock Analysis of {dropdown}')
  
  st.line_chart(df)


#new 

