
# Cell 3: Write the full, CORRECTED Streamlit application to a file named app.py

import streamlit as st
import pandas as pd
import re
import plotly.express as px
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# --- IMPORTANT: FOR GOOGLE COLAB, PASTE YOUR API KEY HERE ---
IPO_ALERTS_API_KEY = "40de9583c60112ccc067cc09094b7593569e2f02d8b43af714a3f612e2ff9bae"

# --- Page Configuration ---
st.set_page_config(page_title="IPO Analytics & Tracker", page_icon="📈", layout="wide")

# --- Helper function to parse issue size ---
def parse_issue_size(size_str):
    if not isinstance(size_str, str): return None
    size_str = size_str.lower().replace('cr', '').strip()
    try: return float(size_str)
    except (ValueError, TypeError): return None

# --- API Data Fetching Functions ---
@st.cache_data(ttl=3600)
def fetch_ipos_by_status(status):
    if not IPO_ALERTS_API_KEY or IPO_ALERTS_API_KEY == "your_actual_api_key_goes_here": return None
    
    # THIS IS THE FIX: The base URL has no parameters
    API_ENDPOINT = "https://api.ipoalerts.in/ipos" 
    headers = {"Authorization": f"Bearer {IPO_ALERTS_API_KEY}"}
    params = {'status': status, 'limit': 200} 
    
    try:
        response = requests.get(API_ENDPOINT, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get('ipos', []))
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed for status '{status}': {e}")
        return None

# --- Data Loading and Preparation for Model ---
@st.cache_data(ttl=3600)
def load_and_prepare_model_data():
    raw_df = fetch_ipos_by_status('closed')
    if raw_df is None or raw_df.empty: return None, None

    df = raw_df.copy()
    df['price_band_high'] = df['priceRange'].apply(lambda x: float(x.split('-')[-1]) if isinstance(x, str) else None)
    # The API doesn't seem to provide listingPrice for 'closed' status, so we can't build the model as is.
    # We will simulate it for now. In a real scenario, you'd need an endpoint with this data.
    if 'listingPrice' not in df.columns:
        # Simulate listing price for demonstration purposes
        np.random.seed(42)
        df['listing_price'] = df['price_band_high'] * (1 + np.random.uniform(-0.1, 0.5, size=len(df)))
    else:
        df['listing_price'] = pd.to_numeric(df.get('listingPrice'), errors='coerce')

    df['issue_size_cr'] = df['issueSize'].apply(parse_issue_size)
    df['ipo_type'] = df['type']
    df['num_strengths'] = df['strengths'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['num_risks'] = df['risks'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['issue_price'] = df['price_band_high']

    df.dropna(subset=['issue_price', 'listing_price'], inplace=True)
    df = df[df['issue_price'] > 0]
    df['FirstDayReturn_percent'] = ((df['listing_price'] - df['issue_price']) / df['issue_price']) * 100
    
    return raw_df, df

# --- Main App ---
# (The rest of the app code remains the same as the last full version)
st.sidebar.title("IPO Analytics & Tracker")
st.sidebar.markdown("Navigate through the project sections.")
page = st.sidebar.radio("Select Section", 
    ["Live IPO Tracker",
     "1. Problem Definition & Objectives", 
     "2 & 3. Data Sourcing & Preparation", 
     "4 & 5. Exploration & Visualization", 
     "6. Insights & Interpretation"])
st.sidebar.info(f"Report Date: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%d %B %Y')}")

# ==============================================================================
# LIVE IPO TRACKER PAGE
# ==============================================================================
if page == "Live IPO Tracker":
    st.title("📈 Live Indian IPO Market Tracker")
    st.markdown(f"Live status as of **{pd.Timestamp.now(tz='Asia/Kolkata').strftime('%I:%M %p IST')}**.")
    
    open_ipos = fetch_ipos_by_status('open')
    upcoming_ipos = fetch_ipos_by_status('upcoming')

    st.header("Open IPOs")
    if open_ipos is not None and not open_ipos.empty:
        for index, ipo in open_ipos.iterrows():
            with st.expander(f"**{ipo['name']}** ({ipo['symbol']}) - Closing on {ipo['endDate']}"):
                st.markdown(f"**Price Band:** ₹{ipo['priceRange']} | **Issue Size:** {ipo['issueSize']} | **Min Investment:** ₹{ipo.get('minAmount', 'N/A')}")
                st.write(ipo.get('about', 'No description available.'))
    else:
        st.info("No IPOs are currently open for subscription.")
        
    st.header("Upcoming IPOs")
    if upcoming_ipos is not None and not upcoming_ipos.empty:
        st.dataframe(upcoming_ipos[['name', 'startDate', 'endDate', 'priceRange', 'issueSize']])
    else:
        st.info("No upcoming IPOs have been announced with dates.")

# ==============================================================================
# OTHER PAGES (Your Original Rubric Structure)
# ==============================================================================
else:
    raw_data, model_df = load_and_prepare_model_data()
    
    if page == "1. Problem Definition & Objectives":
        st.title("1. Problem Definition & Objectives")
        st.header("Project: Predictive Modeling for Indian IPO Performance")
        st.subheader("Problem Definition")
        st.markdown("Investing in IPOs in the Indian market involves significant financial risk. This project aims to mitigate this risk by developing a data-driven model to predict the listing day performance of an IPO.")
        st.subheader("Key Objectives")
        st.markdown("1.  **Data Collection:** Source historical IPO data via the `ipoalerts.in` API.\n2.  **Feature Engineering:** Create powerful predictive features from the available data.\n3.  **Model Development:** Build a machine learning model to predict `FirstDayReturn_percent`.\n4.  **Insight Generation:** Interpret the model to provide actionable insights for investors.")

    elif page == "2 & 3. Data Sourcing & Preparation":
        st.title("2. Data Collection & 3. Data Cleaning")
        st.header("2. Data Collection & Sources")
        st.markdown("Historical data is sourced from the `https://api.ipoalerts.in/ipos?status=closed` endpoint.")
        if raw_data is not None:
            st.dataframe(raw_data.head())
        
        st.header("3. Data Cleaning & Preparation")
        st.markdown("The raw data is cleaned and new features are engineered:")
        if model_df is not None:
            st.dataframe(model_df.head())
    
    elif page == "4 & 5. Exploration & Visualization":
        st.title("4. Data Exploration & 5. Data Visualization")
        if model_df is not None:
            st.header("4. Data Exploration & Summarization")
            st.write(model_df[['FirstDayReturn_percent', 'issue_size_cr', 'num_strengths', 'num_risks', 'price_band_high']].describe())

            st.header("5. Data Visualization")
            st.subheader("IPO Performance by Type (Mainboard vs. SME)")
            fig_box = px.box(model_df, x='ipo_type', y='FirstDayReturn_percent', title='First Day Return by IPO Type')
            st.plotly_chart(fig_box)
        else:
            st.warning("Data for analysis not available.")

    elif page == "6. Insights & Interpretation":
        st.title("6. Insights & Interpretation through Predictive Modeling")
        st.markdown("NOTE: The `closed` endpoint from the API does not provide the actual `listingPrice`. For this demonstration, the listing price has been **simulated** to allow the model to be built. The model's accuracy reflects this simulated data.")
        if model_df is not None:
            features = ['issue_size_cr', 'num_strengths', 'num_risks', 'price_band_high', 'ipo_type']
            target = 'FirstDayReturn_percent'
            df_model = model_df[features + [target]].dropna()
            X = df_model[features]
            y = df_model[target]

            categorical_features = ['ipo_type']
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]), columns=encoder.get_feature_names_out(categorical_features))
            X_numeric = X.drop(columns=categorical_features).reset_index(drop=True)
            X_processed = pd.concat([X_numeric, X_encoded], axis=1)

            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.header("Model Performance")
            r2, mae = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
            col1, col2 = st.columns(2)
            col1.metric(label="R-squared (R²)", value=f"{r2:.3f}")
            col2.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}%")
        else:
            st.warning("Data for modeling not available.")