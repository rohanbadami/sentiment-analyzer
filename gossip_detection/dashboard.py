"""
dashboard.py - Streamlit dashboard for the gossip detection system.

Real-time visualization of:
- Active alerts (high gossip scores)
- Mention velocity trends
- Ticker heatmaps
- Top gossip scores
- Raw events
- CSV export

Follows patterns from phase5_dashboard.py (caching, sidebar filters, Plotly charts).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional

from gossip_detection.db_gossip import get_latest_gossip_scores, get_recent_mentions
from gossip_detection.redis_client import GossipRedisClient
from gossip_detection.rolling_tracker import RollingTracker
from gossip_detection.config import ALERT_GOSSIP_SCORE_THRESHOLD

logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(page_title="Gossip Detector", layout="wide")

# ============================================================================
# Password Gate
# ============================================================================

def check_password():
    """Prompt for password if DASHBOARD_PASSWORD is configured."""
    try:
        expected = st.secrets["DASHBOARD_PASSWORD"]
    except (KeyError, FileNotFoundError):
        expected = os.environ.get("DASHBOARD_PASSWORD")

    if not expected:
        return  # No password configured — allow access (local dev)

    if st.session_state.get("authenticated"):
        return

    pwd = st.text_input("Enter dashboard password", type="password")
    if pwd == expected:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd:
        st.error("Incorrect password")
    st.stop()

check_password()

# ============================================================================
# Cached Data Loaders (TTL = 60 seconds)
# ============================================================================

@st.cache_data(ttl=60)
def load_latest_gossip_scores(limit: int = 100) -> pd.DataFrame:
    """Load latest gossip scores for all tickers."""
    try:
        df = get_latest_gossip_scores(limit=limit)
        return df
    except Exception as e:
        logger.error(f"Error loading gossip scores: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_active_alerts() -> pd.DataFrame:
    """Load recent alerts from Redis."""
    try:
        redis_client = GossipRedisClient()
        alerts = redis_client.get_recent_alerts(count=50)

        if not alerts:
            return pd.DataFrame()

        df = pd.DataFrame(alerts)
        return df
    except Exception as e:
        logger.error(f"Error loading alerts: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_ticker_mentions(ticker: str, window_minutes: int = 60) -> pd.DataFrame:
    """Load recent mentions for a ticker."""
    try:
        df = get_recent_mentions(ticker, minutes=window_minutes)
        return df
    except Exception as e:
        logger.error(f"Error loading mentions for {ticker}: {e}")
        return pd.DataFrame()


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.header("⚙️ Filters & Controls")

# Sidebar: Ticker selection
all_scores_df = load_latest_gossip_scores()
available_tickers = sorted(all_scores_df["ticker"].unique().tolist()) if not all_scores_df.empty else []

selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=available_tickers,
    default=available_tickers[:5] if len(available_tickers) > 5 else available_tickers,
    help="Leave empty to show all tickers"
)

# Sidebar: Rolling window selector
window_minutes = st.sidebar.slider(
    "Rolling Window (minutes)",
    min_value=5,
    max_value=60,
    value=15,
    step=5,
)

# Sidebar: Gossip score threshold
gossip_threshold = st.sidebar.slider(
    "Gossip Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=ALERT_GOSSIP_SCORE_THRESHOLD,
    step=0.05,
)

# Sidebar: Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)

# Sidebar: Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

# Sidebar: CSV export button
all_scores_export = load_latest_gossip_scores(limit=500)
csv_data = all_scores_export.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Export to CSV",
    data=csv_data,
    file_name=f"gossip_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🔊 Retail Market Gossip Detector")
st.markdown("Real-time Reddit sentiment monitoring for trading signals")

# ========================================================================
# Section 1: Active Alerts
# ========================================================================

st.header("🚨 Active Alerts")

alerts_df = load_active_alerts()

if not alerts_df.empty:
    alert_col1, alert_col2, alert_col3 = st.columns(3)

    with alert_col1:
        st.metric("Total Alerts (24h)", len(alerts_df))

    with alert_col2:
        high_score_alerts = alerts_df[alerts_df.get("gossip_score", 0) > gossip_threshold]
        st.metric("High-Confidence Alerts", len(high_score_alerts))

    with alert_col3:
        unique_tickers_alert = len(alerts_df["ticker"].unique()) if "ticker" in alerts_df.columns else 0
        st.metric("Unique Tickers", unique_tickers_alert)

    # Alert details table
    alert_display_df = alerts_df[[
        "ticker", "gossip_score", "mention_velocity", "timestamp"
    ]].head(20) if not alerts_df.empty else pd.DataFrame()

    if not alert_display_df.empty:
        st.dataframe(
            alert_display_df,
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("No alerts in the last 24 hours")

# ========================================================================
# Section 2: Gossip Scores Table
# ========================================================================

st.header("📊 Top Gossip Scores")

scores_df = load_latest_gossip_scores(limit=50)

if not scores_df.empty:
    # Filter by selected tickers if any
    if selected_tickers:
        scores_df = scores_df[scores_df["ticker"].isin(selected_tickers)]

    # Filter by gossip score threshold
    scores_df = scores_df[scores_df["gossip_score"] >= gossip_threshold]

    # Color-code the display
    def color_score(val):
        if val >= 0.7:
            return "background-color: #ff4444"  # Red for high
        elif val >= 0.4:
            return "background-color: #ffaa00"  # Yellow for medium
        else:
            return "background-color: #cccccc"  # Grey for low

    styled_df = scores_df.style.applymap(
        color_score,
        subset=["gossip_score"],
        na_action='ignore'
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
    )

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Tickers Scored", len(scores_df))

    with col2:
        avg_score = scores_df["gossip_score"].mean() if not scores_df.empty else 0
        st.metric("Avg Gossip Score", f"{avg_score:.2f}")

    with col3:
        max_velocity = scores_df["mention_velocity"].max() if not scores_df.empty else 0
        st.metric("Max Velocity (mentions/min)", f"{max_velocity:.2f}")

    with col4:
        max_authors = scores_df["unique_authors"].max() if not scores_df.empty else 0
        st.metric("Max Unique Authors", int(max_authors))

else:
    st.info("No gossip scores available yet")

# ========================================================================
# Section 3: Mention Velocity Time Series
# ========================================================================

st.header("📈 Mention Velocity Trends")

if selected_tickers and not scores_df.empty:
    # Create time series chart
    velocity_data = []

    for ticker in selected_tickers[:5]:  # Limit to 5 tickers for readability
        try:
            mentions_df = load_ticker_mentions(ticker, window_minutes=120)

            if not mentions_df.empty:
                mentions_df["mentioned_at"] = pd.to_datetime(mentions_df["mentioned_at"])
                mentions_df = mentions_df.sort_values("mentioned_at")

                # Group by 5-minute buckets
                mentions_df["bucket"] = mentions_df["mentioned_at"].dt.floor("5T")
                velocity_by_bucket = mentions_df.groupby("bucket").size() / 5.0  # mentions per minute

                for bucket, velocity in velocity_by_bucket.items():
                    velocity_data.append({
                        "Time": bucket,
                        "Ticker": ticker,
                        "Velocity": velocity,
                    })

        except Exception as e:
            logger.debug(f"Error loading velocity data for {ticker}: {e}")

    if velocity_data:
        velocity_df = pd.DataFrame(velocity_data)

        fig = px.line(
            velocity_df,
            x="Time",
            y="Velocity",
            color="Ticker",
            title="Mention Velocity Over Time (5m buckets)",
            labels={"Velocity": "Mentions per Minute"},
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No velocity data available for selected tickers")
else:
    st.info("Select 1-5 tickers to view velocity trends")

# ========================================================================
# Section 4: Ticker Heatmap
# ========================================================================

st.header("🔥 Ticker Heatmap (Gossip Score Intensity)")

if not scores_df.empty:
    # Prepare heatmap data
    heatmap_df = scores_df[["ticker", "gossip_score"]].head(20)

    if not heatmap_df.empty:
        heatmap_df["Time"] = datetime.now().strftime("%H:%M:%S")
        heatmap_pivot = heatmap_df.pivot_table(
            index="ticker",
            columns="Time",
            values="gossip_score",
            aggfunc="first"
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns,
                y=heatmap_pivot.index,
                colorscale="Reds",
            )
        )

        fig.update_layout(
            title="Gossip Score Heatmap (Top 20 Tickers)",
            xaxis_title="Time",
            yaxis_title="Ticker",
        )

        st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# Section 5: Raw Events
# ========================================================================

st.header("📝 Raw Recent Events")

with st.expander("View raw Reddit events"):
    if selected_tickers:
        ticker = selected_tickers[0] if selected_tickers else None

        if ticker:
            mentions_df = load_ticker_mentions(ticker, window_minutes=60)

            if not mentions_df.empty:
                # Truncate long text
                mentions_df_display = mentions_df.copy()

                if "text" in mentions_df_display.columns:
                    mentions_df_display["text"] = mentions_df_display["text"].str[:100] + "..."

                st.dataframe(
                    mentions_df_display,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(f"No recent events for {ticker}")
    else:
        st.info("Select a ticker to view raw events")

# ========================================================================
# Auto-refresh
# ========================================================================

if auto_refresh:
    import time
    time.sleep(300)  # 5 minutes
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "🤖 *Generated with [Claude Code](https://claude.com/claude-code)*"
)
