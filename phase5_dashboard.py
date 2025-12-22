import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text
from datetime import timedelta

# Import your database module
from db_mysql import get_engine

# === CONFIGURATION ===
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide")

# === DATA LOADING ===
@st.cache_data(ttl=60)  # Cache data for 1 minute
def load_data():
    engine = get_engine()
    
    # FIXED QUERY: Uses `Close` as price_close to match your DB schema
    query = """
    SELECT 
        id, ticker, datetime, headline, url,
        sentiment_combined, sentiment_category, ml_confidence,
        `Close` as price_close, pct_change_eod,
        rsi_14, vix_close
    FROM articles
    ORDER BY datetime DESC
    LIMIT 2000
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    return df

# === SIDEBAR FILTERS ===
st.sidebar.header("🔍 Filters")

try:
    df = load_data()
except Exception as e:
    st.error(f"Database Connection Error: {e}")
    st.stop()

# 1. Ticker Filter
all_tickers = ["All"] + sorted(df['ticker'].dropna().unique().tolist())
selected_ticker = st.sidebar.selectbox("Select Ticker", all_tickers)

# 2. Gatekeeper Filter
if 'sentiment_category' in df.columns:
    categories = ["All"] + sorted(df['sentiment_category'].dropna().unique().tolist())
else:
    categories = ["All"]
selected_category = st.sidebar.selectbox("Sentiment Category", categories)

# 3. Confidence Threshold
min_conf = st.sidebar.slider("Minimum AI Confidence", 0.0, 1.0, 0.5)

# === APPLY FILTERS ===
filtered_df = df.copy()

if selected_ticker != "All":
    filtered_df = filtered_df[filtered_df['ticker'] == selected_ticker]

if selected_category != "All" and 'sentiment_category' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['sentiment_category'] == selected_category]

if 'ml_confidence' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['ml_confidence'] >= min_conf]

# === MAIN DASHBOARD ===
st.title("🚀 AI Financial Sentiment Dashboard")

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Articles", len(filtered_df))
col2.metric("High Confidence Signals", len(filtered_df[filtered_df['sentiment_category'] == 'Signal']) if 'sentiment_category' in filtered_df.columns else 0)
col3.metric("Avg Sentiment", f"{filtered_df['sentiment_combined'].mean():.2f}")
col4.metric("Latest Market VIX", f"{filtered_df['vix_close'].iloc[0] if not filtered_df.empty and 'vix_close' in filtered_df.columns else 'N/A'}")

# === SECTION 1: TOMORROW'S WATCHLIST ===
st.subheader("🔮 Top Picks for Next Market Open")
st.caption("Articles from the weekend (or post-market) with High AI Confidence")

# Filter for rows with NO price change yet (Inference rows)
# We check if pct_change_eod is NULL (NaN)
if 'pct_change_eod' in filtered_df.columns:
    watchlist = filtered_df[filtered_df['pct_change_eod'].isna()]
    
    if 'ml_confidence' in watchlist.columns:
        watchlist = watchlist.sort_values(by='ml_confidence', ascending=False)

    if not watchlist.empty:
        display_cols = ['ticker', 'datetime', 'headline', 'sentiment_combined', 'ml_confidence', 'sentiment_category']
        # Filter for cols that actually exist in the dataframe
        display_cols = [c for c in display_cols if c in watchlist.columns]
        
        st.dataframe(
            watchlist[display_cols].head(10),
            use_container_width=True,
            column_config={
                "ml_confidence": st.column_config.ProgressColumn("AI Confidence", format="%.2f", min_value=0, max_value=1),
                "sentiment_combined": st.column_config.NumberColumn("Sentiment", format="%.2f"),
            }
        )
    else:
        st.info("No active watchlist items found (all loaded articles have past price data).")

# === SECTION 2: VISUAL ANALYSIS ===
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
    if not filtered_df.empty:
        fig_hist = px.histogram(filtered_df, x="sentiment_combined", nbins=20, title="Sentiment Score Spread", color_discrete_sequence=['#3366cc'])
        st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.subheader("AI Confidence vs. Sentiment")
    if not filtered_df.empty and 'ml_confidence' in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df, 
            x="sentiment_combined", 
            y="ml_confidence", 
            color="sentiment_category" if 'sentiment_category' in filtered_df.columns else None,
            hover_data=["ticker", "headline"],
            title="Does the AI trust the Sentiment?",
            color_discrete_map={"Signal": "green", "Noise": "gray"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# === SECTION 3: RAW DATA EXPLORER ===
st.subheader("📝 Data Explorer")
explore_cols = ['id', 'ticker', 'datetime', 'headline', 'sentiment_combined', 'rsi_14', 'vix_close', 'price_close']
explore_cols = [c for c in explore_cols if c in filtered_df.columns]

st.dataframe(
    filtered_df[explore_cols], 
    use_container_width=True, 
    height=400
)

# === SIDEBAR REFRESH BUTTON ===
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()