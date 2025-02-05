import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from src.models.volume_analyzer import VolumeAnalyzer
from src.data.alpha_vantage_client import AlphaVantageClient
from src import RAGVolumeAnalyzer
from src.models.ml_volume_analyzer import MLVolumeAnalyzer
from src.utils.signals import SignalGenerator
from src.utils.alerts import AlertSystem
from src.services.trading_signal_service import TradingSignalService
from src.services.volume_analysis_service import VolumeAnalysisService
import numpy as np
import logging

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Oracle of Delphi - Financial Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #666;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

try:
    # Initialize services
    alpha_vantage = AlphaVantageClient()
    volume_analyzer = VolumeAnalyzer()

    # Sidebar Navigation
    st.sidebar.title("Oracle of Delphi 🏛️")

    # AI Assessments Section
    st.sidebar.markdown("## 🤖 AI Assessments")
    ai_section = st.sidebar.radio(
        "Select Analysis Type",
        ["Earnings Analysis", "Sentiment Trends", "Market Driver Insights"],
        help="""
        Earnings Analysis: Identify key earnings drivers
        Sentiment Trends: Analyze market sentiment patterns
        Market Driver Insights: Discover key market influencers
        """
    )

    # Volume Indicators Section
    st.sidebar.markdown("## 📈 Volume Indicators")
    volume_section = st.sidebar.radio(
        "Select Volume Analysis",
        ["Anomaly Detection", "Momentum Forecasting", "Volume Divergence"],
        help="""
        Anomaly Detection: Identify unusual volume patterns
        Momentum Forecasting: Predict volume trends
        Volume Divergence: Analyze price-volume relationships
        """
    )

    # Asset Selection - Always visible
    st.sidebar.markdown("## 🎯 Asset Selection")
    selected_category = st.sidebar.selectbox(
        "Asset Category",
        ["US Stocks", "International", "Crypto"]
    )

    # Symbol Selection
    symbols = {
        "US Stocks": {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "NVDA": "NVIDIA Corporation",
            "META": "Meta Platforms Inc.",
            "TSLA": "Tesla Inc.",
            "JPM": "JPMorgan Chase & Co.",
            "BAC": "Bank of America Corp",
            "WMT": "Walmart Inc.",
        },
        "International": {
            "BVI.PA": "Bureau Veritas SA",
            "PRX.AS": "Prosus NV",
            "SAP.DE": "SAP SE",
            "SONY": "Sony Group Corporation",
            "TCEHY": "Tencent Holdings",
        },
        "Crypto": {
            "BTC-USD": "Bitcoin USD",
            "ETH-USD": "Ethereum USD",
            "HUT": "Hut 8 Mining Corp",
            "RIOT": "Riot Platforms Inc.",
            "COIN": "Coinbase Global Inc.",
        }
    }

    symbol_options = symbols[selected_category]
    selected_symbol = st.sidebar.selectbox(
        "Select Asset",
        options=list(symbol_options.keys()),
        format_func=lambda x: f"{x} - {symbol_options[x]}"
    )

    # Main Content Area
    if ai_section in ["Earnings Analysis", "Sentiment Trends", "Market Driver Insights"]:
        st.title(f"AI Assessment - {ai_section}")
        st.info("🚧 This feature is coming soon! Stay tuned for AI-powered market insights.")

    elif volume_section == "Anomaly Detection":
        # Fetch and process data
        data = alpha_vantage.fetch_daily_adjusted(selected_symbol)

        if data is not None and not data.empty:
            st.title(f"Volume Analysis - {selected_symbol}")

            # Calculate Z-score for anomaly detection
            data['volume_z_score'] = (data['Volume'] - data['Volume'].rolling(window=20).mean()) / \
                                   data['Volume'].rolling(window=20).std()
            data['is_anomaly'] = data['volume_z_score'].abs() > 2

            # Calculate momentum score
            data['momentum_score'] = data['Volume'].pct_change(periods=3).fillna(0) * 100

            # Main metrics with tooltips
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="tooltip">Volume Z-Score
                        <span class="tooltiptext">Statistical measure of volume deviation from the 20-day average</span>
                    </div>
                    <h3>{data['volume_z_score'].iloc[-1]:.2f}</h3>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="tooltip">Momentum Score
                        <span class="tooltiptext">3-day volume change percentage</span>
                    </div>
                    <h3>{data['momentum_score'].iloc[-1]:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                anomaly_status = "ANOMALY DETECTED" if data['is_anomaly'].iloc[-1] else "NORMAL"
                anomaly_color = "red" if data['is_anomaly'].iloc[-1] else "green"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="tooltip">Volume Status
                        <span class="tooltiptext">Current volume pattern classification</span>
                    </div>
                    <h3 style="color: {anomaly_color}">{anomaly_status}</h3>
                </div>
                """, unsafe_allow_html=True)

            # Volume Analysis Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.03,
                              row_heights=[0.7, 0.3])

            # Price chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Price"
                ),
                row=1, col=1
            )

            # Volume bars with anomaly highlighting
            colors = ['red' if is_anomaly else 'blue' for is_anomaly in data['is_anomaly']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name="Volume",
                    marker_color=colors
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                template="plotly_dark",
                showlegend=True,
                title=f"Price and Volume Analysis - {selected_symbol}"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Anomaly Table
            st.subheader("Recent Volume Anomalies")
            anomalies = data[data['is_anomaly']].tail(5)
            if not anomalies.empty:
                anomaly_data = []
                for idx, row in anomalies.iterrows():
                    anomaly_data.append({
                        "Date": idx.strftime("%Y-%m-%d"),
                        "Volume": f"{row['Volume']:,.0f}",
                        "Z-Score": f"{row['volume_z_score']:.2f}",
                        "Momentum": f"{row['momentum_score']:.1f}%"
                    })
                st.table(pd.DataFrame(anomaly_data))
            else:
                st.info("No recent volume anomalies detected")

        else:
            st.error("Unable to fetch market data. Please try again later.")

    elif volume_section in ["Momentum Forecasting", "Volume Divergence"]:
        st.title(f"Volume Analysis - {volume_section}")
        st.info("🚧 This feature is coming soon! Stay tuned for advanced volume analytics.")

    # Add feedback form at the bottom
    st.markdown("---")
    st.subheader("📝 Feedback")
    feedback = st.text_area("Help us improve! Share your thoughts:", placeholder="Enter your suggestions here...")

    if feedback:
        if st.button("Submit Feedback"):
            try:
                logging.info(f"User Feedback: {feedback}")
                st.success("Thank you for your feedback! It helps us improve.")
            except Exception as e:
                st.error("Unable to submit feedback. Please try again.")
                logging.error(f"Error saving feedback: {str(e)}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please refresh the page to try again.")