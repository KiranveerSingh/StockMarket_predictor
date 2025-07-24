import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# Load saved artifacts once
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(r"Models/XGBoost.pkl")  # Adjust filename if needed
    scaler = joblib.load(r"Models/scaler.pkl")
    categorical_columns = joblib.load(r"Models/X_train_cat_enc_columns.pkl")
    return model, scaler, categorical_columns


model, scaler, categorical_columns = load_model_artifacts()

# Feature engineering with all requested features
def add_features(df):
    # Moving averages
    for window in [5, 10, 20]:
        df[f"SMA_{window}"] = df.groupby("Symbol")["Close"].transform(lambda x: x.rolling(window).mean())

    # Daily returns
    df['Return_1D'] = df.groupby('Symbol')['Close'].pct_change()

    # Rolling volatility
    df['Volatility_10'] = df.groupby('Symbol')['Return_1D'].transform(lambda x: x.rolling(10).std())

    # Price ratios
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']

    # Volume features
    df['Volume_SMA_10'] = df.groupby('Symbol')['Volume'].transform(lambda x: x.rolling(10).mean())
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']

    # Lagged features (Close and Volume)
    for lag in [1, 2, 3]:
        df[f'Close_Lag_{lag}'] = df.groupby('Symbol')['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df.groupby('Symbol')['Volume'].shift(lag)

    # %Deliverble features
    if '%Deliverble' in df.columns:
        df['Pct_Deliverable'] = pd.to_numeric(df['%Deliverble'], errors='coerce') / 100.0
        df['Pct_Deliverable_SMA_5'] = df.groupby('Symbol')['Pct_Deliverable'].transform(lambda x: x.rolling(5).mean())
    else:
        df['Pct_Deliverable'] = np.nan
        df['Pct_Deliverable_SMA_5'] = np.nan

    # Deliverable Volume with rolling stats
    if 'Deliverable Volume' in df.columns:
        df['Deliverable_Volume'] = pd.to_numeric(df['Deliverable Volume'], errors='coerce')
        df['Deliverable_Volume_SMA_5'] = df.groupby('Symbol')['Deliverable_Volume'].transform(lambda x: x.rolling(5).mean())
        df['Deliverable_Volume_Ratio'] = df['Deliverable_Volume'] / (df['Deliverable_Volume_SMA_5'] + 1e-5)
    else:
        df['Deliverable_Volume'] = np.nan
        df['Deliverable_Volume_SMA_5'] = np.nan
        df['Deliverable_Volume_Ratio'] = np.nan

    # Trades with rolling stats
    if 'Trades' in df.columns:
        df['Trades'] = pd.to_numeric(df['Trades'], errors='coerce')
        df['Trades_SMA_5'] = df.groupby('Symbol')['Trades'].transform(lambda x: x.rolling(5).mean())
        df['Trades_Ratio'] = df['Trades'] / (df['Trades_SMA_5'] + 1e-5)
    else:
        df['Trades'] = np.nan
        df['Trades_SMA_5'] = np.nan
        df['Trades_Ratio'] = np.nan

    # Turnover with rolling stats
    if 'Turnover' in df.columns:
        df['Turnover'] = pd.to_numeric(df['Turnover'], errors='coerce')
        df['Turnover_SMA_5'] = df.groupby('Symbol')['Turnover'].transform(lambda x: x.rolling(5).mean())
        df['Turnover_Ratio'] = df['Turnover'] / (df['Turnover_SMA_5'] + 1e-5)
    else:
        df['Turnover'] = np.nan
        df['Turnover_SMA_5'] = np.nan
        df['Turnover_Ratio'] = np.nan

    # VWAP with rolling stats
    if 'VWAP' in df.columns:
        df['VWAP'] = pd.to_numeric(df['VWAP'], errors='coerce')
        df['VWAP_SMA_5'] = df.groupby('Symbol')['VWAP'].transform(lambda x: x.rolling(5).mean())
        df['VWAP_Ratio'] = df['VWAP'] / (df['VWAP_SMA_5'] + 1e-5)
    else:
        df['VWAP'] = np.nan
        df['VWAP_SMA_5'] = np.nan
        df['VWAP_Ratio'] = np.nan

    # RSI Calculation
    def calc_rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=period - 1, adjust=False).mean()
        ema_down = down.ewm(com=period - 1, adjust=False).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI_14'] = df.groupby("Symbol")["Close"].transform(lambda x: calc_rsi(x, 14))

    return df


def preprocess_input(input_df):
    df = input_df.copy()
    df = add_features(df)

    exclude_cols = ['Date', 'Symbol', 'Will_Grow']
    all_features = [col for col in df.columns if col not in exclude_cols]

    numeric_features = df[all_features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = list(set(all_features) - set(numeric_features))

    X_num = df[numeric_features]
    X_cat = df[categorical_features]

    X_cat_enc = pd.get_dummies(X_cat, drop_first=True)
    X_cat_enc = X_cat_enc.reindex(columns=categorical_columns, fill_value=0)
    X_processed = pd.concat([X_num.reset_index(drop=True), X_cat_enc.reset_index(drop=True)], axis=1)

    # Reorder columns to exact order scaler expects
    expected_order = scaler.feature_names_in_
    X_processed = X_processed.reindex(columns=expected_order, fill_value=0)

    # Apply scaling
    numeric_in_order = [col for col in expected_order if col in numeric_features]
    X_processed[numeric_in_order] = scaler.transform(X_processed[numeric_in_order])

    return X_processed


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Introduction", "Prediction", "Know More Companies", "About Us"])

if page == "Introduction":
    st.header("Introduction")
    st.write("""
    The stock market is a centralized platform where shares of publicly traded companies are bought and sold. It plays a crucial role in the global financial system, serving as a barometer of economic health and providing a structured environment for companies to raise capital and for investors to grow wealth.

    In simple terms, the stock market allows investors to purchase a small ownership stake—called a share—in businesses they believe will succeed. These shares are traded on stock exchanges such as the Bombay Stock Exchange (BSE) and National Stock Exchange (NSE) in India.

    The prices of stocks fluctuate based on a wide range of factors, including company performance, economic indicators, investor sentiment, and global events. The stock market also includes other instruments such as bonds, mutual funds, ETFs, and derivatives.

    Key participants in the stock market include:
    - Retail Investors: Individuals who buy and sell stocks through brokers.
    - Institutional Investors: Banks, insurance companies, mutual funds, and pension funds.
    - Regulators: Authorities like SEBI (Securities and Exchange Board of India) ensure market fairness, transparency, and investor protection.

    Investing in the stock market can offer high returns, but it also comes with risks. Hence, understanding market dynamics and informed decision-making are critical for success.
    """)

elif page == "Prediction":
    st.header("Stock Growth Prediction")

    with st.form("prediction_form"):
        company_name = st.text_input("Company Name (e.g., Reliance Industries)")
        symbol = st.text_input("Symbol (e.g., RELIANCE)").upper().strip()
        series = st.text_input("Series (e.g., EQ)").upper().strip()
        prev_close = st.number_input("Prev Close", min_value=0.0, step=0.01, format="%.2f")
        open_price = st.number_input("Open", min_value=0.0, step=0.01, format="%.2f")
        high_price = st.number_input("High", min_value=0.0, step=0.01, format="%.2f")
        low_price = st.number_input("Low", min_value=0.0, step=0.01, format="%.2f")
        last_price = st.number_input("Last", min_value=0.0, step=0.01, format="%.2f")
        close_price = st.number_input("Close", min_value=0.0, step=0.01, format="%.2f")
        vwap = st.number_input("VWAP", min_value=0.0, step=0.01, format="%.2f")
        volume = st.number_input("Volume", min_value=0, step=1)
        turnover = st.number_input("Turnover", min_value=0.0, step=0.01, format="%.2f")
        deliverable_volume = st.number_input("Deliverable Volume", min_value=0, step=1)
        trades = st.number_input("Trades", min_value=0, step=1)
        pct_deliverable = st.number_input("%Deliverable", min_value=0.0, max_value=100.0, step=0.01, format="%.2f")

        submitted = st.form_submit_button("Predict")

    if submitted:
        if not all([symbol, series, close_price > 0]):
            st.error("Please fill in all required fields correctly.")
        else:
            input_dict = {
                "Date": pd.to_datetime("today").normalize(),
                "Symbol": symbol,
                "Series": series,
                "Prev Close": prev_close,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Last": last_price,
                "Close": close_price,
                "VWAP": vwap,
                "Volume": volume,
                "Turnover": turnover,
                "Deliverable Volume": deliverable_volume,
                "Trades": trades,
                "%Deliverble": pct_deliverable,
            }

            input_df = pd.DataFrame([input_dict])

            try:
                X_processed = preprocess_input(input_df)
                prediction = model.predict(X_processed)[0]
                direction = "Grow 📈" if prediction == 1 else "Fall 📉"

                prob = None
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_processed)[0][1]

                st.success(f"Prediction: **{direction}**")
                if prob is not None:
                    st.info(f"Confidence: {prob:.2%}")

                st.warning(
                    "Disclaimer: This prediction is not guaranteed and should not be solely relied upon for investment decisions."
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")

elif page == "Know More Companies":
    st.header("Top 20 Stocks for Investment Recommendation")

    companies_data = [
        {"Rank": 1, "Company": "Reliance Industries", "Market Cap (USD billion)": 237.7,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 2, "Company": "HDFC Bank", "Market Cap (USD billion)": 201.3,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 3, "Company": "Tata Consultancy Services (TCS)", "Market Cap (USD billion)": 141.4,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 4, "Company": "Bharti Airtel", "Market Cap (USD billion)": 140.5,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 5, "Company": "ICICI Bank", "Market Cap (USD billion)": 123.0,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 6, "Company": "State Bank of India (SBI)", "Market Cap (USD billion)": 87.1,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 7, "Company": "Infosys", "Market Cap (USD billion)": 73.8,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 8, "Company": "Bajaj Finance", "Market Cap (USD billion)": 68.9,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 9, "Company": "Life Insurance Corporation (LIC)", "Market Cap (USD billion)": 67.1,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 10, "Company": "Hindustan Unilever (HUL)", "Market Cap (USD billion)": 66.3,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 11, "Company": "ITC", "Market Cap (USD billion)": 59.5,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 12, "Company": "Larsen & Toubro (L&T)", "Market Cap (USD billion)": 56.9,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 13, "Company": "HCL Technologies", "Market Cap (USD billion)": 52.8,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 14, "Company": "Kotak Mahindra Bank", "Market Cap (USD billion)": 49.3,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 15, "Company": "Sun Pharma", "Market Cap (USD billion)": 47.0,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 16, "Company": "Maruti Suzuki", "Market Cap (USD billion)": 45.7,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 17, "Company": "Mahindra & Mahindra (M&M)", "Market Cap (USD billion)": 45.3,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
        {"Rank": 18, "Company": "UltraTech Cement", "Market Cap (USD billion)": 41.9,
         "Source": "https://www.moneyworks4me.com/screener/top-25-stocks-in-india?utm_source=chatgpt.com"},
        {"Rank": 19, "Company": "Axis Bank", "Market Cap (USD billion)": 41.8,
         "Source": "https://www.moneyworks4me.com/screener/top-25-stocks-in-india?utm_source=chatgpt.com"},
        {"Rank": 20, "Company": "Hindustan Aeronautics Limited (HAL)", "Market Cap (USD billion)": 38.5,
         "Source": "https://companiesmarketcap.com/india/largest-companies-in-india-by-market-cap/?utm_source=chatgpt.com"},
    ]

    df_comp = pd.DataFrame(companies_data)
    for idx, row in df_comp.iterrows():
        st.markdown(
            f"**{row['Rank']}. [{row['Company']}]({row['Source']})** - Market Cap: ${row['Market Cap (USD billion)']} Billion"
        )

elif page == "About Us":
    st.header("About Us")
    st.write("""
    A Stock Market Prediction Interface is an intelligent, user-friendly tool designed to forecast future stock prices or market trends based on historical and real-time financial data. These interfaces combine data science, machine learning, and financial modeling to assist investors, traders, and analysts in making informed investment decisions.

    The primary goal of such a system is to analyze various factors that influence stock prices—such as past price trends, trading volume, technical indicators, economic news, and market sentiment—and use predictive algorithms to estimate future performance.
    """)
