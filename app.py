# ===================================================================
# IPO Analytics App - Robust Web Scraping Version
# Data Source: chittorgarh.com
# ===================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# --- Page Configuration ---
st.set_page_config(page_title="IPO Analytics Project", page_icon="📊", layout="wide")

# --- Data Caching and Scraping (Robust Version) ---
@st.cache_data(ttl=3600) # Cache the data for 1 hour
def fetch_and_clean_ipo_data():
    """
    Scrapes IPO data from chittorgarh.com, cleans it, and prepares it for analysis.
    This version dynamically finds tables instead of using fixed indices.
    """
    try:
        url = "https://www.chittorgarh.com/ipo/ipo_dashboard.asp"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        all_tables = pd.read_html(response.text)
        
        # --- Dynamically find the correct tables ---
        historical_dfs = []
        current_df = pd.DataFrame()

        # Loop through all scraped tables to find the ones we need
        for table in all_tables:
            # Clean column names for easier checking
            clean_columns = [str(col).strip() for col in table.columns]
            table.columns = clean_columns

            # Identify historical tables by looking for unique columns
            if 'Listing Price' in clean_columns and 'Listing Gain %' in clean_columns:
                historical_dfs.append(table)
            
            # Identify the current IPO table
            if 'Close Date' in clean_columns and 'QIB' in clean_columns:
                current_df = table

        if not historical_dfs:
            st.warning("Could not dynamically find any historical IPO data tables on the page.")
            return None, None
            
        historical_df = pd.concat(historical_dfs, ignore_index=True)
        
        # --- Data Cleaning ---
        historical_df.columns = ['company_name', 'exchange', 'issue_price', 'current_price', 'listing_price', 
                                 'listing_gain_pct', 'all_time_high', 'all_time_low', '3_month_return', '6_month_return']
        
        def clean_numeric_column(series):
            return pd.to_numeric(series.astype(str).str.replace(r'[₹,Cr%]', '', regex=True), errors='coerce')

        for col in ['issue_price', 'current_price', 'listing_price', 'listing_gain_pct', 'all_time_high', 'all_time_low']:
            historical_df[col] = clean_numeric_column(historical_df[col])
            
        historical_df.dropna(subset=['issue_price', 'listing_price', 'listing_gain_pct'], inplace=True)
        
        return historical_df, current_df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data. Network error: {e}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during scraping or cleaning: {e}")
        return None, None

# --- Main App Structure ---
st.sidebar.title("Data Analytics Rubric")
st.sidebar.markdown("Navigate through the project sections.")
page = st.sidebar.radio("Select Section", 
    ["1. Problem Definition & Objectives", 
     "2. Data Collection & Sources",
     "3. Data Cleaning & Preparation", 
     "4. Data Exploration & Summarization", 
     "5. Data Visualization",
     "6. Insights & Interpretation",
     "7. Report & Presentation",
     "Live IPO Tracker"
     ])
st.sidebar.info(f"Report Date: 29 August 2025")

# Fetch the data
model_df, current_df = fetch_and_clean_ipo_data()

# ==============================================================================
# RUBRIC SECTIONS AS APP PAGES
# ==============================================================================
if page == "1. Problem Definition & Objectives":
    st.title("1. Problem Definition & Objectives")
    st.header("Project: Predictive Modeling for Indian IPO Performance")
    st.subheader("Problem Definition")
    st.markdown("Investing in IPOs involves significant risk. This project aims to build a data-driven model to predict the listing day performance of an IPO, using publicly available historical data to provide investors with a quantitative tool.")
    st.subheader("Key Objectives")
    st.markdown("1. **Data Collection:** Scrape a comprehensive dataset of historical IPOs from a public financial website.\n2. **Data Cleaning:** Process and clean the raw scraped data.\n3. **Feature Identification:** Perform EDA to identify key factors that correlate with high listing gains.\n4. **Model Development:** Build a machine learning model to predict `listing_gain_pct`.\n5. **Insight Generation:** Interpret the model to provide actionable insights.")

elif page == "2. Data Collection & Sources":
    st.title("2. Data Collection & Sources")
    st.markdown("""
    The data for this project is sourced via **web scraping** from the [Chittorgarh IPO Dashboard](https://www.chittorgarh.com/ipo/ipo_dashboard.asp).
    - **Methodology:** A Python script using the `requests` library fetches the page's HTML. The `pandas.read_html` function parses all HTML tables. The script then dynamically identifies the correct tables by searching for key column headers (e.g., 'Listing Price').
    - **Data Collected:** Tables for past Mainboard and SME IPOs are extracted to create a historical dataset.
    """)
    if model_df is not None:
        st.subheader("Sample of Scraped and Combined Historical Data")
        st.dataframe(model_df.head())

elif page == "3. Data Cleaning & Preparation":
    st.title("3. Data Cleaning & Preparation")
    st.markdown("""
    The raw scraped data requires significant cleaning:
    1.  **Dynamic Table Finding:** The script intelligently locates the correct data tables on the page, making it resilient to website layout changes.
    2.  **Column Renaming:** Columns are given simple, Python-friendly names (e.g., `listing_gain_pct`).
    3.  **Data Type Conversion:** Text columns containing currency ('₹', 'Cr') and percentage symbols ('%') are cleaned and converted to numeric types.
    4.  **Handling Missing Values:** Rows missing essential data for the model (like `listing_price`) are removed.
    """)
    if model_df is not None and not model_df.empty:
        st.subheader("Cleaned & Prepared Data for Modeling (First 5 Rows)")
        st.dataframe(model_df.head())

elif page == "4. Data Exploration & Summarization":
    st.title("4. Data Exploration & Summarization")
    if model_df is not None and not model_df.empty:
        st.markdown("Descriptive statistics of key numerical features from the historical IPO data:")
        st.write(model_df[['issue_price', 'listing_price', 'listing_gain_pct']].describe())
    
elif page == "5. Data Visualization":
    st.title("5. Data Visualization")
    if model_df is not None and not model_df.empty:
        st.subheader("Distribution of IPO Listing Gains")
        fig_hist = px.histogram(model_df, x='listing_gain_pct', nbins=50, title='Distribution of Listing Day Gains (%)', marginal='box')
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Listing Gains by Exchange (NSE vs. BSE vs. SME)")
        fig_box = px.box(model_df, x='exchange', y='listing_gain_pct', title='Listing Gains by Exchange')
        st.plotly_chart(fig_box, use_container_width=True)

elif page == "6. Insights & Interpretation":
    st.title("6. Insights & Interpretation through Predictive Modeling")
    if model_df is not None and not model_df.empty:
        features = ['issue_price']
        target = 'listing_gain_pct'
        
        df_model = model_df[features + [target]].dropna()
        if not df_model.empty:
            X = df_model[features]
            y = df_model[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            st.header("Model Performance")
            r2 = r2_score(y_test, model.predict(X_test))
            mae = mean_absolute_error(y_test, model.predict(X_test))
            col1, col2 = st.columns(2)
            col1.metric(label="R-squared (R²)", value=f"{r2:.3f}")
            col2.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}%")

            st.header("Insights & Interpretation")
            st.markdown("""
            - A baseline model was built using `issue_price` to predict `listing_gain_pct`.
            - **Key Insight:** The `R-squared` value indicates how much of the variance in listing gains is explained by the model. A low value suggests that `issue_price` alone is not a strong predictor.
            - **To build a more powerful model**, the next step would be to scrape subscription data (QIB, NII, RII) for each IPO, as these are known to be highly influential factors.
            """)
        else:
             st.warning("Not enough clean data to build a model.")

elif page == "7. Report & Presentation":
    st.title("7. Report & Presentation")
    st.markdown("""
    This Streamlit application serves as the interactive report for the project.
    - **Summary:** The project successfully implemented a robust web scraping pipeline to collect historical IPO data. The data was cleaned, analyzed, and used to build a baseline predictive model.
    - **Conclusion:** Web scraping is a powerful method for data collection when APIs are limited. The analysis shows that while a simple model can be built, richer features like subscription data (available on individual IPO pages) are necessary to create a highly accurate predictive tool.
    """)

elif page == "Live IPO Tracker":
    st.title("📈 Live IPO Tracker")
    st.markdown("Data on current and upcoming IPOs, scraped from Chittorgarh.")
    if current_df is not None and not current_df.empty:
        st.subheader("Current & Upcoming IPOs")
        st.dataframe(current_df)
    else:
        st.info("Could not find a table for current or upcoming IPOs on the page.")
