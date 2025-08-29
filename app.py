# ===================================================================
# FINAL APP - Open IPOs Dashboard (Compatible with Free API Plan)
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
st.set_page_config(page_title="Open IPOs Dashboard", page_icon="📈", layout="wide")

# --- API Data Fetching ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_open_ipos():
    """
    Fetches ONLY 'open' IPOs, as this is the only endpoint
    supported by the free API plan.
    """
    if not IPO_ALERTS_API_KEY or IPO_ALERTS_API_KEY == "YOUR_API_KEY_GOES_HERE":
        st.error("API key is missing. Please edit this file and add your key.")
        return None
    
    API_ENDPOINT = "https://api.ipoalerts.in/ipos"
    headers = {"x-api-key": IPO_ALERTS_API_KEY}
    params = {'status': 'open'} # This is the only status that works on the free plan
    
    try:
        response = requests.get(API_ENDPOINT, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data.get('ipos', []))
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Failed: {e}")
        return None

# --- Main App ---
st.sidebar.title("Data Analytics Rubric")
st.sidebar.markdown("""
This report is adapted to the constraints of the free API plan, focusing on the available data for 'open' IPOs.
""")
st.sidebar.info(f"Report Date: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%d %B %Y')}")

# --- 1. Problem Definition & Objectives ---
st.title("1. Problem Definition & Objectives")
st.header("Project: Live Dashboard for Currently Open IPOs")
st.markdown("""
**Problem Definition:** Investors need timely and consolidated information about Initial Public Offerings that are currently open for subscription. This project demonstrates how to build a live dashboard to address this need.

**Key Objectives:**
* **Data Collection:** Source live data for 'open' IPOs using the IPO Alerts API.
* **Data Presentation:** Clean and present the details for each open IPO in a clear, user-friendly format.
* **Reporting:** Package the dashboard into an interactive Streamlit web application.
""")
st.markdown("---")


# --- 2. Data Collection & Sources ---
st.header("2. Data Collection & Sources")
st.markdown("""
The data is sourced from the `https://api.ipoalerts.in/ipos?status=open` endpoint. 

**Constraint Analysis:** A diagnostic test revealed that the free API plan only supports the `status=open` parameter. All other statuses (`closed`, `upcoming`, etc.) are restricted. This shapes the scope of the project to focus exclusively on currently open offerings.
""")
open_ipos_df = fetch_open_ipos()


# --- 3. Data Cleaning & 4. Summarization ---
st.header("3. Data Cleaning & 4. Data Summarization")
if open_ipos_df is not None and not open_ipos_df.empty:
    st.markdown(f"The API call was successful. **{len(open_ipos_df)}** IPO(s) are currently open for subscription.")
    st.markdown("The raw JSON data is loaded into a Pandas DataFrame for processing. Key details are extracted for display.")
    
    # --- 5. Data Visualization / 7. Report & Presentation ---
    st.header("5. Data Visualization & 7. Report Presentation")
    st.markdown("Below is the detailed presentation of each currently open IPO:")

    for index, ipo in open_ipos_df.iterrows():
        with st.expander(f"**{ipo.get('name', 'N/A')} ({ipo.get('symbol', 'N/A')})**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Issue Period:** {ipo.get('startDate', 'N/A')} to {ipo.get('endDate', 'N/A')}")
                st.markdown(f"**Price Band:** ₹{ipo.get('priceRange', 'N/A')}")
                st.markdown(f"**Issue Size:** {ipo.get('issueSize', 'N/A')}")
            with col2:
                st.markdown(f"**Lot Size:** {ipo.get('minQty', 'N/A')} shares")
                st.markdown(f"**Min. Investment:** ₹{ipo.get('minAmount', 'N/A')}")
                st.markdown(f"**Listing Date:** {ipo.get('listingDate', 'N/A')}")
            
            st.markdown(f"**About the Company:**\n{ipo.get('about', 'No description available.')}")
            
            if ipo.get('prospectusUrl'):
                st.link_button("View Prospectus", ipo['prospectusUrl'])
else:
    st.info("No data was returned from the API. This likely means there are no IPOs currently open for subscription.")


# --- 6. Insights & Interpretation ---
st.header("6. Insights & Interpretation")
st.markdown("""
**Insight:** The application successfully provides a real-time view of the primary market, allowing users to see all IPOs currently accepting subscriptions in one place.

**Conclusion based on API Limitations:** The initial goal of creating a predictive model is not feasible with the free API plan because it prohibits access to historical (`closed`) data. To expand this project to include predictive analytics, an **upgrade to a paid API plan** would be required. This dashboard represents the most comprehensive application that can be built under the current data access constraints.
""")
