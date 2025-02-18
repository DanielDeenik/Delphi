import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from functools import lru_cache
from src.models.volume_analyzer import VolumeAnalyzer
from src.data.alpha_vantage_client import AlphaVantageClient
from src import RAGVolumeAnalyzer
from src.models.ml_volume_analyzer import MLVolumeAnalyzer
from src.models.mosaic_agent import MosaicTheoryAgent
import numpy as np
import logging
import plotly.express as px
from datetime import datetime, timedelta
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for settings if not exists
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'theme': 'dark',
        'default_symbol': 'AAPL',
        'update_interval': 5,
        'enable_ai_features': True,
        'watchlist': ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    }

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Oracle of Delphi - Financial Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/project',
        'Report a bug': "https://github.com/yourusername/project/issues",
        'About': "# Oracle of Delphi 🏛️\nAdvanced Financial Intelligence Platform"
    }
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2e3440;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3b4252;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #2e3440;
    }
    .sidebar-text {
        color: white !important;
    }
    h1, h2, h3 {
        color: #2e3440;
        font-weight: 600;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .watchlist-card {
        background-color: #2e3440;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .settings-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data fetching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_market_data(symbol: str, start_date: datetime, end_date: datetime):
    try:
        data = storage_service.get_market_data(symbol=symbol, start_date=start_date, end_date=end_date)
        if data.empty:
            data = alpha_vantage.fetch_daily_adjusted(symbol)
        return data
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_sentiment_analysis(symbol: str):
    try:
        return sentiment_service.get_sentiment_analysis(symbol)
    except Exception as e:
        logger.error(f"Error getting sentiment analysis: {e}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def fetch_watchlist_data(symbols):
    """Fetch data for watchlist symbols"""
    data = {}
    for symbol in symbols:
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            df = fetch_market_data(symbol, start_date, end_date)
            if df is not None and not df.empty:
                latest_price = df['Close'].iloc[-1]
                price_change = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
                data[symbol] = {
                    'price': latest_price,
                    'change': price_change
                }
        except Exception as e:
            logger.error(f"Error fetching watchlist data for {symbol}: {e}")
    return data

try:
    # Initialize services
    alpha_vantage = AlphaVantageClient()
    volume_analyzer = VolumeAnalyzer()
    sentiment_service = SentimentAnalysisService()
    storage_service = TimeSeriesStorageService()
    mosaic_agent = MosaicTheoryAgent()

    # Create tabs
    tab_main, tab_settings, tab_watchlist, tab_performance = st.tabs([
        "🏛️ Main Dashboard",
        "⚙️ Settings",
        "📈 Watchlist",
        "⚡ Performance"
    ])

    with tab_main:
        # Modern sidebar navigation with icons
        st.sidebar.markdown('<h1 class="sidebar-text">🏛️ Oracle of Delphi</h1>', unsafe_allow_html=True)

        # AI Assessments Section with modern styling
        st.sidebar.markdown('<h2 class="sidebar-text">🤖 AI Assessments</h2>', unsafe_allow_html=True)
        ai_section = st.sidebar.radio(
            "",  # Empty label for cleaner look
            ["📊 Earnings Analysis", "📈 Sentiment Trends", "🔍 Market Driver Insights"],
            key="ai_section"
        )

        # Volume Indicators Section
        st.sidebar.markdown('<h2 class="sidebar-text">📈 Volume Indicators</h2>', unsafe_allow_html=True)
        volume_section = st.sidebar.radio(
            "",
            ["🔍 Anomaly Detection", "📈 Momentum Forecasting", "↔️ Volume Divergence"],
            key="volume_section"
        )

        # Mosaic Theory Section
        st.sidebar.markdown('<h2 class="sidebar-text">🧠 Mosaic Theory AI</h2>', unsafe_allow_html=True)
        mosaic_section = st.sidebar.radio(
            "",
            ["🎯 Market Intelligence", "💹 Investment Flows", "🕸️ Knowledge Graph"],
            key="mosaic_section"
        )

        # Asset Selection with modern dropdown
        st.sidebar.markdown('<h2 class="sidebar-text">🎯 Asset Selection</h2>', unsafe_allow_html=True)
        selected_category = st.sidebar.selectbox(
            "",
            ["US Stocks", "International", "Crypto"],
            key="asset_category"
        )

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
            "",
            options=list(symbol_options.keys()),
            format_func=lambda x: f"{x} - {symbol_options[x]}"
        )


        # Main Content Area for AI Assessments
        if ai_section == "📊 Earnings Analysis":
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
                    with st.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Latest EPS Surprise",
                            "+$0.15",
                            "+12%",
                            help="Difference between actual and estimated EPS"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                with metrics_cols[1]:
                    with st.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Revenue Growth",
                            "8.5%",
                            "+2.3%",
                            help="Year-over-year revenue growth"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                with metrics_cols[2]:
                    with st.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Analyst Consensus",
                            "Buy",
                            "↑ 2 upgrades",
                            help="Overall analyst recommendation"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

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

        elif ai_section == "📈 Sentiment Trends":
            st.title("Sentiment Analysis")

            try:
                # Fetch historical data first
                data = alpha_vantage.fetch_daily_adjusted(selected_symbol)

                if data is not None and not data.empty:
                    # Get sentiment data
                    sentiment_data = get_sentiment_analysis(selected_symbol)

                    # Sentiment Overview
                    st.subheader("Sentiment Overview")
                    sent_cols = st.columns(3)

                    with sent_cols[0]:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            sentiment_score = sentiment_data['sentiment_metrics']['overall_score']
                            st.metric(
                                "Overall Sentiment",
                                f"{sentiment_score:.2f}",
                                f"{sentiment_data['sentiment_metrics']['sentiment_change_24h']:.1%}",
                                help="Aggregated sentiment score from multiple sources"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with sent_cols[1]:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                "Social Media Buzz",
                                "High",
                                "↑ 15%",
                                help="Social media mention volume"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with sent_cols[2]:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                "Viral Coefficient",
                                f"{sentiment_data['sentiment_metrics']['viral_coefficient']:.2f}",
                                help="Measure of content virality"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

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

        elif ai_section == "🔍 Market Driver Insights":
            st.title("Market Driver Analysis")

            # Market Impact Factors
            st.subheader("Key Market Drivers")
            impact_cols = st.columns(3)

            with impact_cols[0]:
                with st.container():
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Interest Rate Impact",
                        "Moderate",
                        "-2.3%",
                        help="Effect of interest rate changes"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            with impact_cols[1]:
                with st.container():
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Sector Momentum",
                        "Strong",
                        "+5.1%",
                        help="Industry sector performance"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            with impact_cols[2]:
                with st.container():
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Market Sentiment",
                        "Bullish",
                        "↑ trending",
                        help="Overall market sentiment"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

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

        elif volume_section == "🔍 Anomaly Detection":
            st.title(f"Volume Analysis - {selected_symbol}")

            try:
                # Fetch data with cloud storage integration
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)  # Last 90 days
                data = fetch_market_data(selected_symbol, start_date, end_date)

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
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            latest_anomaly = anomaly_scores[-1]
                            anomaly_status = "ANOMALY" if latest_anomaly < -0.5 else "NORMAL"
                            st.metric(
                                "Volume Status",
                                anomaly_status,
                                delta=f"{latest_anomaly:.2f}",
                                delta_color="inverse"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            volume_change = data['Volume'].pct_change().iloc[-1]
                            st.metric(
                                "Volume Change",
                                f"{volume_change:.2%}",
                                delta=f"{volume_change:.1%}",
                                delta_color="normal"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with col3:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            relative_vol = (data['Volume'] / data['Volume'].rolling(20).mean()).iloc[-1]
                            st.metric(
                                "Relative Volume",
                                f"{relative_vol:.2f}x",
                                delta=f"{(relative_vol-1):.1%}",
                                delta_color="normal"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

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

        elif volume_section == "📈 Momentum Forecasting":
            st.title(f"Volume Momentum Analysis - {selected_symbol}")

            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                data = fetch_market_data(selected_symbol, start_date, end_date)

                if data is not None and not data.empty:
                    # Calculate momentum indicators
                    data['volume_ma5'] = data['Volume'].rolling(window=5).mean()
                    data['volume_ma20'] = data['Volume'].rolling(window=20).mean()
                    data['momentum_score'] = (data['Volume'] - data['volume_ma20']) / data['volume_ma20']

                    # Display momentum metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            momentum = data['momentum_score'].iloc[-1]
                            st.metric(
                                "Momentum Score",
                                f"{momentum:.2f}",
                                delta=f"{(momentum - data['momentum_score'].iloc[-2]):.2f}"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            trend = "Bullish" if momentum > 0 else "Bearish"
                            st.metric("Volume Trend", trend)
                            st.markdown('</div>', unsafe_allow_html=True)

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


        elif volume_section == "↔️ Volume Divergence":
            st.title(f"Volume Divergence Analysis - {selected_symbol}")

            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                data = fetch_market_data(selected_symbol, start_date, end_date)

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
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            div_score = data['divergence_score'].iloc[-1]
                            st.metric(
                                "Divergence Score",
                                f"{div_score:.2f}",
                                delta=f"{(div_score - data['divergence_score'].iloc[-2]):.2f}"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            signal = "Bullish" if div_score > 0.5 else "Bearish" if div_score < -0.5 else "Neutral"
                            st.metric("Divergence Signal", signal)
                            st.markdown('</div>', unsafe_allow_html=True)

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

        # Added Mosaic Theory sections to main content
        if mosaic_section == "🎯 Market Intelligence":
            st.title("AI Market Intelligence")

            # Market Sentiment Analysis
            st.subheader("Market Sentiment & Trends")
            sentiment_cols = st.columns(3)

            # Mock news data for sentiment analysis
            news_data = [
                f"Earnings report for {selected_symbol} shows strong growth",
                f"Market analysts upgrade {selected_symbol} rating",
                f"Industry outlook positive for {selected_symbol} sector"
            ]

            sentiment_results = mosaic_agent.analyze_market_sentiment(news_data)

            if sentiment_results:
                with sentiment_cols[0]:
                    with st.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "AI Sentiment Score",
                            f"{sentiment_results['aggregate_score']:.2f}",
                            "↑ 0.05",
                            help="AI-generated market sentiment score"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                with sentiment_cols[1]:
                    with st.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Confidence Level",
                            "High",
                            "↑ trending",
                            help="AI model confidence in analysis"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                with sentiment_cols[2]:
                    with st.container():
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "Risk Assessment",
                            "Moderate",
                            "↓ 2%",
                            help="AI-evaluated risk level"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

            # Market Trends Forecast
            st.subheader("AI-Driven Market Forecast")
            if data is not None and not data.empty:
                forecast = mosaic_agent.forecast_trends(data['Close'])
                if forecast:
                    forecast_cols = st.columns(2)

                    with forecast_cols[0]:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                "Predicted Price",
                                f"${forecast['predicted_price']:.2f}",
                                f"{((forecast['predicted_price'] / data['Close'].iloc[-1]) - 1) * 100:.1f}%",
                                help="AI-predicted price target"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                    with forecast_cols[1]:
                        with st.container():
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                "Prediction Confidence",
                                f"{forecast['confidence_score']:.1%}",
                                help="AI model confidence score"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

        elif mosaic_section == "💹 Investment Flows":
            st.title("Investment Flow Analysis")

            # Sample market data structure
            market_data = {
                "Technology": {
                    "Company A": {"market_cap": 100},
                    "Company B": {"market_cap": 80}
                },
                "Finance": {
                    "Company C": {"market_cap": 90},
                    "Company D": {"market_cap": 70}
                }
            }

            # Update knowledge graph with market data
            mosaic_agent.update_knowledge_graph(market_data)

            # Get investment flows for Sankey diagram
            flows_df = mosaic_agent.get_investment_flows()

            if not flows_df.empty:
                # Create Sankey diagram
                fig = go.Figure(go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=list(set(flows_df["source"].tolist() + flows_df["target"].tolist())),
                    ),
                    link=dict(
                        source=[list(set(flows_df["source"].tolist() + flows_df["target"].tolist())).index(s) for s in flows_df["source"]],
                        target=[list(set(flows_df["source"].tolist()+ flows_df["target"].tolist())).index(t) for t in flows_df["target"]],
                        value=flows_df["value"]
                    )
                ))

                fig.update_layout(
                    title_text="Investment Flow Analysis",
                    font_size=12,
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

        elif mosaic_section == "🕸️ Knowledge Graph":
            st.title("Market Knowledge Graph")
            import networkx as nx

            graph_data = mosaic_agent.knowledge_graph
            if graph_data:
                # Create network visualization using NetworkX and Plotly
                pos = nx.spring_layout(graph_data)

                # Create edges trace
                edge_x = []
                edge_y = []
                for edge in graph_data.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edges_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines')

                # Create nodes trace
                node_x = []
                node_y = []
                for node in graph_data.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                nodes_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=[node for node in graph_data.nodes()],
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        size=10,
                        colorbar=dict(
                            thickness=15,
                            title='Node Connections',
                            xanchor='left',
                            titleside='right'                    )
                    )
                )

                # Create the figure
                fig = go.Figure(data=[edges_trace, nodes_trace],
                                 layout=go.Layout(
                                     title='Market Knowledge Graph',
                                     showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20,l=5,r=5,t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                 ))

                st.plotly_chart(fig, use_container_width=True)

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

    with tab_settings:
        st.title("⚙️ Settings")

        with st.form("settings_form"):
            st.markdown('<div class="settings-section">', unsafe_allow_html=True)

            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["dark", "light"],
                index=0 if st.session_state.settings['theme'] == 'dark' else 1
            )

            # Default symbol
            default_symbol = st.text_input(
                "Default Symbol",
                value=st.session_state.settings['default_symbol']
            )

            # Update interval
            update_interval = st.slider(
                "Update Interval (minutes)",
                min_value=1,
                max_value=60,
                value=st.session_state.settings['update_interval']
            )

            # AI features toggle
            enable_ai = st.checkbox(
                "Enable AI Features",
                value=st.session_state.settings['enable_ai_features']
            )

            # Watchlist management
            watchlist = st.text_input(
                "Watchlist (comma-separated symbols)",
                value=",".join(st.session_state.settings['watchlist'])
            )

            if st.form_submit_button("Save Settings"):
                st.session_state.settings.update({
                    'theme': theme,
                    'default_symbol': default_symbol,
                    'update_interval': update_interval,
                    'enable_ai_features': enable_ai,
                    'watchlist': [s.strip() for s in watchlist.split(",")]
                })
                st.success("Settings saved successfully!")

            st.markdown('</div>', unsafe_allow_html=True)

    with tab_watchlist:
        st.title("📈 Watchlist")

        # Fetch watchlist data
        watchlist_data = fetch_watchlist_data(st.session_state.settings['watchlist'])

        # Display watchlist in grid layout
        cols = st.columns(4)
        for idx, (symbol, data) in enumerate(watchlist_data.items()):
            with cols[idx % 4]:
                st.markdown(f'<div class="watchlist-card">', unsafe_allow_html=True)
                st.metric(
                    symbol,
                    f"${data['price']:.2f}",
                    f"{data['change']:.2%}",
                    delta_color="normal" if data['change'] >= 0 else "inverse"
                )
                st.markdown('</div>', unsafe_allow_html=True)

        # Display watchlist table
        st.subheader("Watchlist Overview")
        watchlist_df = pd.DataFrame([
            {
                'Symbol': symbol,
                'Price': data['price'],
                'Change': f"{data['change']:.2%}"
            }
            for symbol, data in watchlist_data.items()
        ])
        st.dataframe(watchlist_df, use_container_width=True)

    with tab_performance:
        st.title("⚡ Caching & Performance")

        st.markdown("""
        ### Caching Implementation
        This application uses Streamlit's caching mechanisms to optimize performance:

        1. **Market Data Cache** (5 minutes TTL):
           - Reduces API calls to external data providers
           - Improves dashboard responsiveness

        2. **Sentiment Analysis Cache** (10 minutes TTL):
           - Optimizes expensive NLP computations
           - Maintains analysis consistency

        3. **Watchlist Cache** (1 minute TTL):
           - Efficient watchlist updates
           - Reduces API load

        ### Cache Statistics
        """)

        # Display cache statistics
        cache_stats = {
            'Market Data Hits': fetch_market_data.cache_info().hits,
            'Market Data Misses': fetch_market_data.cache_info().misses,
            'Sentiment Analysis Hits': get_sentiment_analysis.cache_info().hits,
            'Sentiment Analysis Misses': get_sentiment_analysis.cache_info().misses,
            'Watchlist Hits': fetch_watchlist_data.cache_info().hits,
            'Watchlist Misses': fetch_watchlist_data.cache_info().misses
        }

        # Display cache statistics in columns
        cols = st.columns(3)
        for idx, (metric, value) in enumerate(cache_stats.items()):
            with cols[idx % 3]:
                st.metric(metric, value)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    logger.error(f"Application error: {str(e)}")