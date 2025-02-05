import os
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
from src.services.sentiment_analysis_service import SentimentAnalysisService
from src.services.timeseries_storage_service import TimeSeriesStorageService
import numpy as np
import logging
import plotly.express as px
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    storage_service = TimeSeriesStorageService()

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

        try:
            # Fetch historical data first
            data = alpha_vantage.fetch_daily_adjusted(selected_symbol)

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
            else:
                st.error("Unable to fetch market data for sentiment analysis.")
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")

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
        st.title(f"Volume Analysis - {selected_symbol}")

        try:
            # Fetch data with cloud storage integration
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # Last 90 days
            data = storage_service.get_market_data(
                symbol=selected_symbol,
                start_date=start_date,
                end_date=end_date
            )

            if data.empty:
                # Fallback to Alpha Vantage API
                data = alpha_vantage.fetch_daily_adjusted(selected_symbol)

            if data is not None and not data.empty:
                # Initialize columns for metrics
                col1, col2, col3 = st.columns(3)

                # Calculate anomaly scores using MLVolumeAnalyzer
                ml_analyzer = MLVolumeAnalyzer()
                features, _ = ml_analyzer._extract_advanced_features(data)
                anomaly_scores = ml_analyzer.anomaly_detector.score_samples(features)
                data['anomaly_score'] = anomaly_scores

                # Display metrics
                with col1:
                    latest_anomaly = anomaly_scores[-1]
                    anomaly_status = "ANOMALY" if latest_anomaly < -0.5 else "NORMAL"
                    st.metric(
                        "Volume Status",
                        anomaly_status,
                        delta=f"{latest_anomaly:.2f}",
                        delta_color="inverse"
                    )

                with col2:
                    volume_change = data['Volume'].pct_change().iloc[-1]
                    st.metric(
                        "Volume Change",
                        f"{volume_change:.2%}",
                        delta=f"{volume_change:.1%}",
                        delta_color="normal"
                    )

                with col3:
                    relative_vol = (data['Volume'] / data['Volume'].rolling(20).mean()).iloc[-1]
                    st.metric(
                        "Relative Volume",
                        f"{relative_vol:.2f}x",
                        delta=f"{(relative_vol-1):.1%}",
                        delta_color="normal"
                    )

                # Create anomaly detection chart
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=('Price', 'Volume Anomaly Score'),
                    row_heights=[0.7, 0.3]
                )

                # Add candlestick chart
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

                # Add volume bars colored by anomaly score
                colors = ['red' if score < -0.5 else 'blue' for score in anomaly_scores]
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        marker_color=colors,
                        name="Volume"
                    ),
                    row=2, col=1
                )

                # Update layout
                fig.update_layout(
                    height=800,
                    template="plotly_dark",
                    showlegend=True,
                    title=f"Volume Anomaly Detection - {selected_symbol}"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Simplified real-time updates section
                st.subheader("Manual Refresh")
                if st.button("Refresh Data"):
                    try:
                        new_data = alpha_vantage.fetch_daily_adjusted(selected_symbol)
                        if new_data is not None and not new_data.empty:
                            st.metric(
                                "Latest Volume",
                                f"{new_data['Volume'].iloc[-1]:,.0f}",
                                delta=f"{new_data['Volume'].pct_change().iloc[-1]:.1%}"
                            )
                            st.success("Data refreshed successfully!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error updating data: {str(e)}")

        except Exception as e:
            st.error(f"Error fetching data for Anomaly Detection: {e}")

    elif volume_section == "Momentum Forecasting":
        st.title(f"Volume Momentum Analysis - {selected_symbol}")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            data = storage_service.get_market_data(
                symbol=selected_symbol,
                start_date=start_date,
                end_date=end_date
            )
            if data.empty:
                data = alpha_vantage.fetch_daily_adjusted(selected_symbol)

            if data is not None and not data.empty:
                # Calculate momentum indicators
                data['volume_ma5'] = data['Volume'].rolling(window=5).mean()
                data['volume_ma20'] = data['Volume'].rolling(window=20).mean()
                data['momentum_score'] = (data['Volume'] - data['volume_ma20']) / data['volume_ma20']

                # Display momentum metrics
                col1, col2 = st.columns(2)

                with col1:
                    momentum = data['momentum_score'].iloc[-1]
                    st.metric(
                        "Momentum Score",
                        f"{momentum:.2f}",
                        delta=f"{(momentum - data['momentum_score'].iloc[-2]):.2f}"
                    )

                with col2:
                    trend = "Bullish" if momentum > 0 else "Bearish"
                    st.metric("Volume Trend", trend)

                # Create momentum chart
                fig = go.Figure()

                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name="Volume",
                        opacity=0.3
                    )
                )

                # Add moving averages
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['volume_ma5'],
                        name="5-day MA",
                        line=dict(color='orange')
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['volume_ma20'],
                        name="20-day MA",
                        line=dict(color='blue')
                    )
                )

                # Update layout
                fig.update_layout(
                    height=600,
                    template="plotly_dark",
                    title=f"Volume Momentum - {selected_symbol}"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Add momentum distribution
                st.subheader("Momentum Distribution")
                fig_dist = go.Figure()
                fig_dist.add_trace(
                    go.Histogram(
                        x=data['momentum_score'].dropna(),
                        nbinsx=50,
                        name="Momentum Distribution"
                    )
                )
                fig_dist.update_layout(
                    height=400,
                    template="plotly_dark",
                    title="Momentum Score Distribution"
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching data for Momentum Forecasting: {e}")


    elif volume_section == "Volume Divergence":
        st.title(f"Volume Divergence Analysis - {selected_symbol}")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            data = storage_service.get_market_data(
                symbol=selected_symbol,
                start_date=start_date,
                end_date=end_date
            )
            if data.empty:
                data = alpha_vantage.fetch_daily_adjusted(selected_symbol)

            if data is not None and not data.empty:
                # Calculate divergence metrics
                data['price_change'] = data['Close'].pct_change()
                data['volume_change'] = data['Volume'].pct_change()
                data['divergence_score'] = (
                    data['price_change'].rolling(5).corr(data['volume_change'].rolling(5))
                )

                # Display divergence metrics
                col1, col2 = st.columns(2)

                with col1:
                    div_score = data['divergence_score'].iloc[-1]
                    st.metric(
                        "Divergence Score",
                        f"{div_score:.2f}",
                        delta=f"{(div_score - data['divergence_score'].iloc[-2]):.2f}"
                    )

                with col2:
                    signal = "Bullish" if div_score > 0.5 else "Bearish" if div_score < -0.5 else "Neutral"
                    st.metric("Divergence Signal", signal)

                # Create divergence chart
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=('Price', 'Volume', 'Divergence Score'),
                    row_heights=[0.4, 0.3, 0.3]
                )

                # Add price line
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        name="Price",
                        line=dict(color='white')
                    ),
                    row=1, col=1
                )

                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name="Volume",
                        marker_color='blue'
                    ),
                    row=2, col=1
                )

                # Add divergence score
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['divergence_score'],
                        name="Divergence",
                        line=dict(color='yellow')
                    ),
                    row=3, col=1
                )

                # Update layout
                fig.update_layout(
                    height=800,
                    template="plotly_dark",
                    showlegend=True,
                    title=f"Volume Divergence Analysis - {selected_symbol}"
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching data for Volume Divergence: {e}")


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