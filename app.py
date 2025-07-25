import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts once
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(r"./Models/XGBoost.pkl")  # Adjust filename if needed
    scaler = joblib.load(r"./Models/scaler.pkl")
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
st.sidebar.title("Home")
page = st.sidebar.radio("Go to:", ["Introduction", "Prediction", "Model Used Information", "Know About More Companies", "About Us"])




if page == "Introduction":
    image_path = r"C:\My stuff\Coding\ML project\KiranveerSingh_projectfinal\Images\OIP.jpeg"
    st.image(image_path, use_container_width=True)

    st.header("Introduction")
    st.write("""
    🔹 What is the Stock Market?
             
        The stock market is a public platform that enables individuals and institutions to buy, sell, and trade shares (also known as stocks or equities) of publicly listed companies. It is a key component of the financial system, facilitating capital formation, investment, and economic growth.

    🔹 Key Concepts
             
        1. Stock:
            A stock represents a fractional ownership in a company. For example, if you own 100 shares of a company that has issued 1 million shares, you own 0.01% of that company.

        2. Shareholder:
            Anyone who owns at least one share of a company is called a shareholder, and they are entitled to a share of the company's profits (dividends) and voting rights.

        3. Stock Exchange:
            A stock exchange is a centralized marketplace where stocks are traded. In India, the two main exchanges are:
                BSE (Bombay Stock Exchange) - Oldest in Asia, established in 1875
                NSE (National Stock Exchange) - Founded in 1992, introduced electronic trading

    🔹 How the Stock Market Works
             
        Company Listing:        A company goes public by issuing shares via an Initial Public Offering (IPO).

        Trading Begins:         Once listed, investors can buy or sell these shares through stock exchanges.

        Price Fluctuation:      Stock prices are determined by supply and demand, influenced by:
                                    Company performance
                                    Industry growth

        Economic indicators:    GDP, Inflation, Interest rates

        Market sentiment:       News, Rumors, Global trends

        Settlement:             After a trade, ownership is transferred within T+1 day in India.

    🔹 Types of Markets
             
        Type of Market-             Description-
             
        Primary Market:             Where securities are created (IPOs)
             
        Secondary Market:           Where securities are traded after the IPO
             
        Cash Market:                Full payment must be made for buying shares
             
        Derivatives Market:         Futures and options based on stock prices
             
        Commodities & Currency:     Traded via exchanges like MCX & NSE FX

    🔹 Market Participants
             
        Participant Type:           Role
             
        Retail Investors:           Individual investors,
                                    Institutional Investors,
                                    Mutual funds,
                                    Insurance companies,
                                    Banks
             
        SEBI:                       Regulator ensuring transparency and fairness
             
        Brokers	Intermediaries:     Execute trades
             
        Market Makers:              Ensure liquidity by placing buy/sell orders

    🔹 Stock Market Indices:\n An index tracks the performance of a group of stocks to represent market trends.


        Index-               Description-
        Nifty 50 :           Top 50 companies on NSE
        Sensex :             Top 30 companies on BSE
        Bank Nifty  :        Major banks listed on NSE
        Midcap, Smallcap  :  Medium and small-sized companies

    🔹 Benefits of Stock Market
             
        💰 Capital for Companies:       Helps companies raise funds for growth

        📈 Wealth Creation:             Long-term investment can generate significant returns

        🔄 Liquidity:                   Easy to buy/sell shares

        👥 Democratization:             Anyone can own a piece of a major company

    🔹 Risks Involved
             
        📉 Market Volatility:           Prices fluctuate daily

        🧠 Emotional Investing:         Fear and greed can cause irrational decisions

        💼 Company Risk:                A company's poor performance may impact share prices

        🌍 Global Events:               Geopolitical and economic events can impact all markets

    🔹 Tools & Platforms
             
        Brokerage apps:                 Zerodha, Groww, Upstox, Angel One

        Market data providers:          NSE India, Moneycontrol, TradingView

        Stock screeners:                Screener.in, TickerTape, Equitymaster

    📚 Real-World Examples
             
        Reliance Industries:            raised billions through the stock market to fund Jio and retail expansion.

        Zomato, Paytm, Nykaa:           IPOs brought tech startups into the public market.

        COVID-19 crash:                 showed both risk and opportunity in equity investing.

    🏦 Regulatory Body: SEBI
             
        1.Securities and Exchange Board of India (SEBI) is the primary regulator.

        2.Its role is to protect investors, develop the market, and regulate intermediaries.

        3.SEBI also introduces rules for IPOs, mutual funds, insider trading, etc.

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

elif page == "Model Used Information":
    st.header("Model Used:")
    st.write("""
    You are using an **XGBoost Classifier** as your main prediction model in the stock prediction pipeline.
    """)

    st.header("What is **XGBoost**?")
    st.write("""
    XGBoost (Extreme Gradient Boosting) is a popular and highly efficient machine learning algorithm based on the concept of gradient boosting.

    It builds an ensemble of decision trees, where each new tree corrects errors made by the previous one, leading to a strong overall prediction.

    XGBoost is known for its speed, accuracy, and handling of missing data and is often used in machine learning competitions and production environments.
    """)

    st.header("Why is XGBoost Used in Stock Prediction?")
    st.write("""
    Handles complex patterns and non-linear relationships often found in financial time series data.

    Robust to different types of input features (both numerical and categorical when properly encoded).

    Supports regularization, minimizing the risk of overfitting common in financial prediction.

    Provides feature importance scores, helping explain which factors drive predictions.
    """)

    st.header("Features of Your Model Setup")
    st.write("""
    Type: _Classification Model_ (predicts categorical outcome — e.g., whether a stock will grow or fall).

    Uses: Various engineered features from stock data, such as rolling averages, volatility, technical indicators (RSI), volume statistics, and more.

    Advantages:

        High accuracy on tabular data.

        Fast training and prediction.

        Good with large numbers of features.

    Outputs: Probability of "grow" (or "fall") and a predicted class for each sample.
    """)

    image_path = r"C:\My stuff\Coding\ML project\KiranveerSingh_projectfinal\Images\79337a1e-f158-4168-9ea3-3f81b4a11d23.png"
    st.image(image_path,caption="XGBoost Classifer", use_container_width=True)

elif page == "Know About More Companies":
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
    st.header("Overview")
    st.write("""
    A Stock Market Prediction Interface is an intelligent, user-friendly tool designed to forecast future stock prices or market trends based on historical and real-time financial data. These interfaces combine data science, machine learning, and financial modeling to assist investors, traders, and analysts in making informed investment decisions.

    The primary goal of such a system is to analyze various factors that influence stock prices—such as past price trends, trading volume, technical indicators, economic news, and market sentiment—and use predictive algorithms to estimate future performance.
    """)

    st.header("Project Disclaimer⚠️")
    st.write("""
    
    This stock market prediction project and its associated user interface are for informational and educational purposes only.

    🚫 No Financial Advice: The predictions and outputs generated by this system are not intended as financial advice or recommendations to buy, sell, or hold any security or investment.

    ❗ No Guarantees: While the model uses advanced machine learning techniques and historical data, all predictions are subject to substantial uncertainty. Stock prices are influenced by numerous unpredictable factors, including market sentiment, economic events, and unforeseen company developments.

    🔄 Past Performance Is Not Indicative of Future Results: Model accuracy on past data does not guarantee similar results in the future. Stock markets can experience volatility outside the scope of any model.

    👤 User Responsibility: Users should always conduct their own research, consult with professional advisors, and consider their own risk tolerance before making any investment decisions.

    ⚖️ No Liability: The creators and deployers of this project accept no liability for any losses or damages arising from the use or interpretation of the predictions, outputs, or recommendations provided by this tool.

    By using this application, you acknowledge and accept these terms and agree not to hold the developers responsible for any investment outcomes resulting from its use.
    """)

    st.header("Step Performed")
    st.write("""
    
    1. Problem Definition & Objectives:
             
            -Clearly define the prediction goal: “Will a given stock grow or fall on the next trading day?”.
             
            -Specify the end-user target, business value, and main deliverables: an interactive prediction app, model performance report, and investment insights.
    
    2. Data Acquisition:
             
            -Collect historical stock data (OHLCV: Open, High, Low, Close, Volume) for companies from trusted sources (e.g., NSE/BSE, Yahoo Finance).

            -Include additional columns if available: Prev Close, Deliverable Volume, %Deliverble, Trades, Turnover, VWAP, etc.
             
    3. Data Cleaning:
             
            -Standardize and rename columns for consistency.

            -Convert all string-based and mixed columns to appropriate numeric types (using error coercion).

            -Remove duplicates, invalid entries, and fill or drop missing values.

            -Parse and sort date columns chronologically for each symbol.
             
    4. Feature Engineering:
             
            -Generate key technical indicators: moving averages, RSI, daily returns, rolling volatility, volume ratios, and price ratios.

            -Compute and add lagged features (prior days’ data) to capture temporal trends.

            -Engineer rolling statistics and ratios for new features such as Deliverable Volume, Trades, Turnover, VWAP, and %Deliverble.

            -Create the binary prediction target: Will_Grow = 1 if next day’s close > today’s, else 0.

            -Save the feature-enriched dataset for modeling.
             
    5. Data Preprocessing:
             
            -Split the data into training and testing sets chronologically to avoid data leakage.

            -Separate numeric and categorical columns for each sample.

            -One-hot encode categorical features and align columns for train/test consistency.

            -Apply feature scaling (e.g., StandardScaler) to numeric features.

            -Save the fitted preprocessing objects (scaler, categorical column list) for use in live prediction.
             
    6. Model Development:
             
            -Train and compare multiple classification models: Random Forest, XGBoost, LightGBM, Decision Tree, Logistic Regression, etc.

            -Optionally compare regression models, thresholding their outputs for classification use.

            -Collect and compare performance metrics: accuracy, F1-score, precision, recall, ROC-AUC, and confusion matrix.

            -Select and export the best-performing model and log all model artifacts.
             
    7. Model Robustness Check:
             
            -Test the selected model on a new, unseen dataset processed with the identical pipeline.

            -Analyze results for consistency, overfitting, and generalization.

            -Adjust features or retrain if required based on performance drift.
             
    8. Unified Prediction Pipeline:
             
            -Integrate all steps into a single processing and prediction function.

            -Ensure future user or batch data flows through the same preprocessing and feature engineering as the training pipeline.

    9. User Interface (UI) Development:
             
            -Build an intuitive interactive application (Streamlit UI).
    """)
    st.header("Submitted By:")
    st.write("""
        **Kiranveer Singh Randhawa**
             
        🎓B.Tech Electronics and Communication Engineering (5th Semester)

        🏫Guru Nanak Dev University, Amritsar, Punjab
             
        
    """)