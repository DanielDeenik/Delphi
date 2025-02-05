import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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
import plotly.express as px
from datetime import datetime, timedelta

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
    sentiment_service = SentimentAnalysisService()

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

    # Main Content Area for AI Assessments
    if ai_section == "Earnings Analysis":
        st.title("Earnings Analysis")

        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=90)
            )
        with col2:
            end_date = st.date_input("End Date", datetime.now())

        # Fetch historical data and sentiment data
        data = alpha_vantage.fetch_daily_adjusted(selected_symbol)
        if data is not None and not data.empty:
            # Earnings Metrics
            st.subheader("Earnings Performance")
            metrics_cols = st.columns(3)

            # Mock earnings metrics (replace with real data integration)
            with metrics_cols[0]:
                st.metric(
                    "Latest EPS Surprise",
                    "+$0.15",
                    "+12%",
                    help="Difference between actual and estimated EPS"
                )
            with metrics_cols[1]:
                st.metric(
                    "Revenue Growth",
                    "8.5%",
                    "+2.3%",
                    help="Year-over-year revenue growth"
                )
            with metrics_cols[2]:
                st.metric(
                    "Analyst Consensus",
                    "Buy",
                    "↑ 2 upgrades",
                    help="Overall analyst recommendation"
                )

            # Earnings Trend Chart
            st.subheader("Historical Earnings Trends")
            earnings_fig = go.Figure()
            # Mock earnings data (replace with real data)
            quarters = ['Q1 2024', 'Q4 2023', 'Q3 2023', 'Q2 2023']
            actual = [2.15, 1.95, 1.88, 1.75]
            estimated = [2.00, 1.90, 1.85, 1.70]

            earnings_fig.add_trace(go.Bar(
                x=quarters,
                y=actual,
                name="Actual EPS",
                marker_color='rgb(26, 118, 255)'
            ))
            earnings_fig.add_trace(go.Bar(
                x=quarters,
                y=estimated,
                name="Estimated EPS",
                marker_color='rgba(26, 118, 255, 0.5)'
            ))

            earnings_fig.update_layout(
                title="Quarterly EPS Comparison",
                barmode='group',
                template="plotly_dark"
            )
            st.plotly_chart(earnings_fig, use_container_width=True)

    elif ai_section == "Sentiment Trends":
        st.title("Sentiment Analysis")

        if data is not None and not data.empty:
            # Get sentiment data
            sentiment_data = sentiment_service.get_sentiment_analysis(selected_symbol)

            # Sentiment Overview
            st.subheader("Sentiment Overview")
            sent_cols = st.columns(3)

            with sent_cols[0]:
                sentiment_score = sentiment_data['sentiment_metrics']['overall_score']
                st.metric(
                    "Overall Sentiment",
                    f"{sentiment_score:.2f}",
                    f"{sentiment_data['sentiment_metrics']['sentiment_change_24h']:.1%}",
                    help="Aggregated sentiment score from multiple sources"
                )

            with sent_cols[1]:
                st.metric(
                    "Social Media Buzz",
                    "High",
                    "↑ 15%",
                    help="Social media mention volume"
                )

            with sent_cols[2]:
                st.metric(
                    "Viral Coefficient",
                    f"{sentiment_data['sentiment_metrics']['viral_coefficient']:.2f}",
                    help="Measure of content virality"
                )

            # STEPPS Framework Analysis
            st.subheader("STEPPS Framework Analysis")
            stepps_cols = st.columns(2)

            with stepps_cols[0]:
                stepps_data = sentiment_data['stepps_analysis']
                stepps_fig = go.Figure(data=[
                    go.Scatterpolar(
                        r=[
                            stepps_data['social_currency'],
                            stepps_data['triggers'],
                            stepps_data['emotion'],
                            stepps_data['public'],
                            stepps_data['practical_value'],
                            stepps_data['stories']
                        ],
                        theta=['Social Currency', 'Triggers', 'Emotion',
                               'Public', 'Practical Value', 'Stories'],
                        fill='toself'
                    )
                ])
                stepps_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    template="plotly_dark"
                )
                st.plotly_chart(stepps_fig, use_container_width=True)

            with stepps_cols[1]:
                st.subheader("Trend Classification")
                st.write(sentiment_data['trend_classification']['classification'])
                st.progress(sentiment_data['trend_classification']['confidence'])

    elif ai_section == "Market Driver Insights":
        st.title("Market Driver Analysis")

        # Market Impact Factors
        st.subheader("Key Market Drivers")
        impact_cols = st.columns(3)

        with impact_cols[0]:
            st.metric(
                "Interest Rate Impact",
                "Moderate",
                "-2.3%",
                help="Effect of interest rate changes"
            )
        with impact_cols[1]:
            st.metric(
                "Sector Momentum",
                "Strong",
                "+5.1%",
                help="Industry sector performance"
            )
        with impact_cols[2]:
            st.metric(
                "Market Sentiment",
                "Bullish",
                "↑ trending",
                help="Overall market sentiment"
            )

        # Market Correlation Analysis
        st.subheader("Market Correlations")
        correlations = {
            'S&P 500': 0.85,
            'Sector Index': 0.92,
            'VIX': -0.45,
            'US Dollar': -0.25,
            '10Y Treasury': -0.35
        }

        corr_fig = go.Figure(data=[
            go.Bar(
                x=list(correlations.keys()),
                y=list(correlations.values()),
                marker_color=['blue' if v >= 0 else 'red' for v in correlations.values()]
            )
        ])
        corr_fig.update_layout(
            title="Asset Correlations",
            template="plotly_dark",
            showlegend=False
        )
        st.plotly_chart(corr_fig, use_container_width=True)

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