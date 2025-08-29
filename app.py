# ===================================================================
# FINAL APP.PY - Simplified for the IPO Alerts Free API Plan
# ===================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# ===================================================================
# 🔽🔽🔽 PASTE YOUR VALID IPO ALERTS API KEY HERE 🔽🔽🔽
# ===================================================================
IPO_ALERTS_API_KEY = "6d34f66ba89aac521b895f43bf389256900e20ac4b62d1ee7d1bd099ff07164e"
# ===================================================================

# --- Page Configuration ---
st.set_page_config(page_title="IPO Data Analysis", page_icon="📊", layout="wide")

# --- API Data Fetching ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_ipos_from_free_plan():
    """
    Fetches the generic list of IPOs available to free plan users.
    This call does NOT use the 'status' parameter.
    """
    if not IPO_ALERTS_API_KEY or IPO_ALERTS_API_KEY == "YOUR_API_KEY_GOES_HERE":
        st.error("API key is missing. Please edit this file and add your key.")
        return None
    
    API_ENDPOINT = "https://api.ipoalerts.in/ipos"
    headers = {"x-api-key": IPO_ALERTS_API_KEY}
    
    try:
        response = requests.get(API_ENDPOINT, headers=headers)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get('ipos', []))
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        st.error("This may be due to an invalid API key or a network issue.")
        return None

# --- Main App ---
st.sidebar.title("Data Analytics Rubric")
st.sidebar.markdown("""
This report is structured to align with the Data Analytics Marking Rubric. 

Due to API plan limitations, this version focuses on the data available from the free tier.
""")
st.sidebar.info(f"Report Date: 29 August 2025")

# --- 1. Problem Definition & Objectives ---
st.title("1. Problem Definition & Objectives")
st.header("Project: Analysis of Available IPO Data")
st.markdown("""
**Problem Definition:** Accessing and analyzing IPO data requires a structured approach. This project demonstrates the process of connecting to a financial data API, processing the retrieved data, and presenting key insights, while working within the constraints of a free-tier API plan.

**Key Objectives:**
* **Data Collection:** Source a list of IPOs via the IPO Alerts API.
* **Data Cleaning:** Process and display the raw JSON data in a structured table.
* **Data Exploration & Visualization:** Summarize and visualize basic characteristics of the available IPO data.
* **Insights & Reporting:** Present the findings in an interactive web application.
""")
st.markdown("---")


# --- 2. Data Collection & Sources ---
st.header("2. Data Collection & Sources")
st.markdown("""
The data is sourced live from the `https://api.ipoalerts.in/ipos` endpoint using a Python script. Authentication is performed using an `x-api-key` header. 

**Constraint:** The API key is for a free plan, which does not support filtering by parameters like `status`. Therefore, the application fetches the default list of IPOs provided by the API.
""")
ipo_data = fetch_ipos_from_free_plan()


if ipo_data is not None and not ipo_data.empty:
    # --- 3. Data Cleaning & Preparation ---
    st.header("3. Data Cleaning & Preparation")
    st.markdown("The raw JSON response from the API is loaded directly into a Pandas DataFrame. For this analysis, we select key columns for clarity and ensure proper data types.")
    
    # Create a cleaned-up version for display and analysis
    columns_to_display = ['name', 'symbol', 'status', 'type', 'startDate', 'endDate', 'priceRange', 'issueSize']
    # Filter out columns that don't exist in the DataFrame to prevent errors
    existing_columns = [col for col in columns_to_display if col in ipo_data.columns]
    cleaned_df = ipo_data[existing_columns].copy()
    
    # --- 4. Data Exploration & Summarization ---
    st.header("4. Data Exploration & Summarization")
    st.markdown("Here is the structured table of the IPO data retrieved from the API:")
    st.dataframe(cleaned_df)
    
    # --- 5. Data Visualization ---
    st.header("5. Data Visualization")
    
    # Check if 'status' column exists for visualization
    if 'status' in cleaned_df.columns:
        st.subheader("Count of IPOs by Status")
        status_counts = cleaned_df['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        fig = px.bar(status_counts, x='status', y='count', title="Number of IPOs by Current Status", text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'status' column was available in the data for visualization.")

    # --- 6. Insights & Interpretation / 7. Report & Presentation ---
    st.header("6. Insights & Interpretation")
    st.markdown("""
    **Insight:** The API's free tier provides a limited, mixed list of IPOs. The visualization above shows the distribution of these IPOs across different statuses (e.g., 'open', 'upcoming'). This demonstrates the ability to extract and summarize key categorical data from the source.

    **Limitation:** Without access to filtered historical data (specifically for `closed` IPOs with their `listingPrice`), building a *predictive model* for performance is not feasible. [cite_start]The "Upcoming Features" section of the API documentation indicates that full historical data access is planned for the future[cite: 210].
    """)

    st.header("7. Report & Presentation")
    st.markdown("This interactive Streamlit dashboard serves as the final report. It successfully demonstrates the core data analytics workflow: connecting to an API, processing data, and generating visualizations and insights, all while clearly documenting the constraints of the data source.")
    
else:
    st.warning("Could not retrieve any data from the API. Please check your API key and network connection.")

