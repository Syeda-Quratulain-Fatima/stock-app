import pickle

import pandas as pd
import streamlit as st

MODEL_FILE = "bac_price_model.pkl"
FEATURES = ["Open", "High", "Low", "Close", "Volume",
            "Return_1d", "SMA_5", "SMA_20", "Volatility_10"]

st.set_page_config(page_title="BAC Stock Price Predictor", page_icon="📈")


@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("📈 Bank of America (BAC) Next-Day Price Predictor")
st.write(
    "Enter today's trading data for BAC and the model predicts the **closing price of the next "
    "trading day**. The model is the best of 5 algorithms compared with 10-fold cross validation "
    "(Ridge Regression, RMSE about $0.46). Default values are the last day in the dataset (12 Sep 2025)."
)

col1, col2 = st.columns(2)
with col1:
    open_p = st.number_input("Open ($)", min_value=0.01, value=50.70, step=0.01)
    high_p = st.number_input("High ($)", min_value=0.01, value=50.93, step=0.01)
    low_p = st.number_input("Low ($)", min_value=0.01, value=50.46, step=0.01)
    close_p = st.number_input("Close ($)", min_value=0.01, value=50.58, step=0.01)
    volume = st.number_input("Volume (shares)", min_value=0.0, value=27849900.0, step=100000.0)
with col2:
    ret_pct = st.number_input("Daily return (%) vs previous close", value=-0.33, step=0.01)
    sma5 = st.number_input("5-day average close ($)", min_value=0.01, value=50.24, step=0.01)
    sma20 = st.number_input("20-day average close ($)", min_value=0.01, value=49.62, step=0.01)
    vol10_pct = st.number_input("10-day volatility (% std of daily returns)", min_value=0.0, value=1.06, step=0.01)

if st.button("Predict next-day close"):
    if not (low_p <= open_p <= high_p and low_p <= close_p <= high_p):
        st.warning("Open and Close should lie between Low and High. Please check your inputs.")
    else:
        row = pd.DataFrame([[open_p, high_p, low_p, close_p, volume,
                             ret_pct / 100, sma5, sma20, vol10_pct / 100]], columns=FEATURES)
        prediction = float(model.predict(row)[0])
        st.success(f"Predicted closing price for the next trading day: **${prediction:,.2f}**")
        st.caption(f"Change vs today's close: {prediction - close_p:+.2f} USD. "
                   "Typical model error is about $0.46 per share.")

st.info("This is a student machine-learning project. It is not financial advice.")
