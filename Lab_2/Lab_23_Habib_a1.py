import yfinance as yf
import streamlit as st


# Définir le symbole de l'action, la date de début et la date de fin
st.header("Stock price prediction")
stock_symbol = st.sidebar.selectbox('Select stock symbol',['AAPL','GOOGL','TSLA'])
start_date = st.sidebar.date_input('train start time')
end_date = st.sidebar.date_input('train end time')
dataset_train=yf.download(symbol,stat)


# Télécharger les données depuis Yahoo Finance
dataset_train = yf.download(stock_symbol, start=start_date, end=end_date)
print(dataset_train)