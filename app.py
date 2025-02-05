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
from src.models.hmm_regime_classifier import MarketRegimeClassifier
from src.models.lstm_price_predictor import LSTMPricePredictor
from src.services.trading_signal_service import TradingSignalService
from src.services.volume_analysis_service import VolumeAnalysisService
import time
from src.models.volume_shift_analyzer import VolumeShiftAnalyzer

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Oracle of Delphi - Market Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
try:
    alpha_vantage = AlphaVantageClient()
    trading_service = TradingSignalService()
    volume_analysis_service = VolumeAnalysisService()
    volume_analyzer = VolumeAnalyzer()
    rag_analyzer = RAGVolumeAnalyzer()
    ml_analyzer = MLVolumeAnalyzer()
    signal_generator = SignalGenerator()
    alert_system = AlertSystem()
    volume_shift_analyzer = VolumeShiftAnalyzer()

    # Initialize session state for symbols if not present
    if 'available_symbols' not in st.session_state:
        st.session_state.available_symbols = {
            "US Stocks": {
                "AAPL": {"name": "Apple Inc.", "currency": "USD"},
                "MSFT": {"name": "Microsoft Corporation", "currency": "USD"},
                "GOOGL": {"name": "Alphabet Inc.", "currency": "USD"},
                "AMZN": {"name": "Amazon.com Inc.", "currency": "USD"},
                "NVDA": {"name": "NVIDIA Corporation", "currency": "USD"},
                "META": {"name": "Meta Platforms Inc.", "currency": "USD"},
                "TSLA": {"name": "Tesla Inc.", "currency": "USD"},
                "JPM": {"name": "JPMorgan Chase & Co.", "currency": "USD"},
                "BAC": {"name": "Bank of America Corp", "currency": "USD"},
                "WMT": {"name": "Walmart Inc.", "currency": "USD"},
            },
            "International": {
                "BVI.PA": {"name": "Bureau Veritas SA", "currency": "EUR"},
                "PRX.AS": {"name": "Prosus NV", "currency": "EUR"},
                "SAP.DE": {"name": "SAP SE", "currency": "EUR"},
                "SONY": {"name": "Sony Group Corporation", "currency": "JPY"},
                "TCEHY": {"name": "Tencent Holdings", "currency": "HKD"},
            },
            "Crypto": {
                "BTC-USD": {"name": "Bitcoin USD", "currency": "USD"},
                "ETH-USD": {"name": "Ethereum USD", "currency": "USD"},
                "HUT": {"name": "Hut 8 Mining Corp", "currency": "USD"},
                "RIOT": {"name": "Riot Platforms Inc.", "currency": "USD"},
                "COIN": {"name": "Coinbase Global Inc.", "currency": "USD"},
            }
        }

    # Sidebar Navigation
    st.sidebar.title("Oracle of Delphi 🏛️")

    # Volume Indicators Section
    st.sidebar.markdown("## 📈 Volume Analysis")
    selected_view = st.sidebar.radio(
        "",
        ["Volume Overview", "Anomaly Detection", "Momentum Analysis"]
    )

    # Asset Selection
    st.title("Volume Analysis Dashboard")
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_category = st.selectbox(
            "Asset Category",
            options=list(st.session_state.available_symbols.keys())
        )

    with col2:
        available_assets = st.session_state.available_symbols[selected_category]
        selected_asset = st.selectbox(
            "Select Asset",
            options=[f"{symbol} - {details['name']}"
                     for symbol, details in available_assets.items()]
        )

    # Extract symbol from selection
    selected_symbol = selected_asset.split(" - ")[0]

    # Fetch market data
    data = alpha_vantage.fetch_daily_adjusted(selected_symbol)

    if data is not None and not data.empty:
        # Process volume analysis
        volume_metrics = volume_analyzer.calculate_vwap(data)
        volume_patterns = volume_analyzer.detect_volume_patterns(data)

        if selected_view == "Volume Overview":
            # Volume Overview Section
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Current Volume",
                    f"{data['Volume'].iloc[-1]:,.0f}",
                    f"{((data['Volume'].iloc[-1] / data['Volume'].iloc[-2]) - 1):.1%}"
                )

            with col2:
                st.metric(
                    "VWAP",
                    f"${data['VWAP'].iloc[-1]:.2f}",
                    f"{((data['VWAP'].iloc[-1] / data['VWAP'].iloc[-2]) - 1):.1%}"
                )

            with col3:
                volume_trend = "BULLISH" if data['Volume'].iloc[-1] > data['Volume'].mean() else "BEARISH"
                st.metric("Volume Trend", volume_trend)

            # Volume Trend Graph
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03,
                                row_heights=[0.7, 0.3])

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

            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name="Volume",
                    marker_color='blue'
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                template="plotly_dark",
                showlegend=True,
                title="Price and Volume Analysis"
            )

            st.plotly_chart(fig, use_container_width=True)

        elif selected_view == "Anomaly Detection":
            # Anomaly Detection Section
            st.subheader("Volume Anomalies")

            # Get anomalies from ML analyzer
            anomalies = ml_analyzer.detect_volume_patterns(data)

            # Display anomaly metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Anomaly Score",
                    f"{anomalies['anomaly_score']:.2f}",
                    f"{anomalies['anomaly_change']:.1%}"
                )
            with col2:
                st.metric(
                    "Pattern Type",
                    anomalies['volume_profile']
                )

            # Anomaly Timeline
            if len(anomalies['anomalies']) > 0:
                anomaly_dates = data.index[
                    [i for i, x in enumerate(anomalies['anomalies']) if x == -1]
                ]

                if len(anomaly_dates) > 0:
                    st.markdown("### Recent Anomalies")
                    anomaly_df = pd.DataFrame({
                        'Date': anomaly_dates,
                        'Volume': data.loc[anomaly_dates, 'Volume'],
                        'Price Change': data.loc[anomaly_dates, 'Close'].pct_change()
                    })

                    st.dataframe(
                        anomaly_df.style.format({
                            'Volume': '{:,.0f}',
                            'Price Change': '{:.1%}'
                        }),
                        hide_index=True
                    )

        else:  # Momentum Analysis
            # Momentum Analysis Section
            st.subheader("Volume Momentum")

            # Calculate momentum metrics
            data['momentum'] = data['Close'].pct_change().rolling(window=20).mean()
            data['volume_momentum'] = data['Volume'].pct_change().rolling(window=20).mean()

            # Display momentum metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Price Momentum",
                    f"{data['momentum'].iloc[-1]:.1%}",
                    f"{data['momentum'].iloc[-1] - data['momentum'].iloc[-2]:.1%}"
                )
            with col2:
                st.metric(
                    "Volume Momentum",
                    f"{data['volume_momentum'].iloc[-1]:.1%}",
                    f"{data['volume_momentum'].iloc[-1] - data['volume_momentum'].iloc[-2]:.1%}"
                )

            # Momentum Chart
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['momentum'],
                    name="Price Momentum",
                    line=dict(color='blue')
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['volume_momentum'],
                    name="Volume Momentum",
                    line=dict(color='orange')
                )
            )

            fig.update_layout(
                height=400,
                template="plotly_dark",
                title="Momentum Analysis"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Unable to fetch market data. Please try again later.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please refresh the page to try again.")

#CURRENCY_SYMBOLS = { ... }  Removed as not used in the edited code.
#format_price function removed as it's not used in the edited code.
#Rest of the original code removed as it's not relevant to the simplified UI.