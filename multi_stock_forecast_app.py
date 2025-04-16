
import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("📈 Multi-Stock Forecast (Alpha Vantage - Free Tier)")

API_KEY = st.text_input("🔑 Enter your Alpha Vantage API Key", type="password")
symbols_input = st.text_input("📊 Enter comma-separated stock symbols (e.g., AAPL, FSLR, TSLA)", "AAPL, FSLR")
alert_threshold = st.number_input("🔔 Alert if price increases more than ($):", value=2.0, step=0.5)

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

if API_KEY and symbols:
    for symbol in symbols:
        st.header(f"📊 {symbol} Forecast (10-Day)")
        url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
            f"&symbol={symbol}&outputsize=compact&apikey={API_KEY}"
        )
        response = requests.get(url)
        data = response.json()

        if "Time Series (Daily)" not in data:
            st.warning(f"❌ Could not fetch data for {symbol}.")
            continue

        ts_data = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts_data, orient="index")
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.astype(float)

        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["Momentum"] = df["Close"] - df["Close"].shift(10)
        df["Volatility"] = df["Close"].pct_change().rolling(window=10).std()
        df["Future_10d"] = df["Close"].shift(-10)
        df.dropna(inplace=True)

        if len(df) < 30:
            st.warning(f"⚠️ Not enough data to train model for {symbol}.")
            continue

        X = df[["SMA_10", "Momentum", "Volatility"]]
        y = df["Future_10d"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        current_price = df["Close"].iloc[-1]
        if hasattr(current_price, "item"):
            current_price = current_price.item()
        latest_features = X.iloc[[-1]]
        predicted_price = model.predict(latest_features)[0]
        price_diff = predicted_price - current_price
        percent_change = (price_diff / current_price) * 100

        st.write(f"📍 Current Price: **${current_price:.2f}**")
        st.write(f"🔮 Predicted Price in 10 Days: **${predicted_price:.2f}**")
        st.write(f"📊 Change: **${price_diff:.2f} ({percent_change:.2f}%)**")

        st.subheader("🧠 Why it may be a good buy")
        reasons = []
        if df["Momentum"].iloc[-1] > 0:
            reasons.append("📈 Positive momentum — price is trending upward.")
        if df["Volatility"].iloc[-1] < df["Volatility"].mean():
            reasons.append("🛡️ Low volatility — less risky than usual.")
        if df["SMA_10"].iloc[-1] < predicted_price:
            reasons.append("📊 Forecasted price is above SMA — possible bullish signal.")
        if reasons:
            st.info("\n".join(reasons))
        else:
            st.info("ℹ️ Model suggests a buy but without strong supporting indicators.")

        if price_diff > alert_threshold:
            st.success("🚨 STRONG BUY SIGNAL!")

        st.subheader("📉 Current vs Forecasted Price")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(["Today", "In 10 Days"], [current_price, predicted_price], color=["blue", "green"])
        ax.set_ylabel("Price ($)")
        ax.set_title(f"{symbol.upper()} Price Forecast")
        st.pyplot(fig)

        forecast_df = pd.DataFrame({
            "Ticker": [symbol],
            "Current Price": [round(current_price, 2)],
            "Predicted Price (10d)": [round(predicted_price, 2)],
            "Change ($)": [round(price_diff, 2)],
            "Change (%)": [round(percent_change, 2)]
        })
        st.download_button(f"⬇️ Download {symbol} Forecast CSV", forecast_df.to_csv(index=False), file_name=f"{symbol}_forecast.csv")
