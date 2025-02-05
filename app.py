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

# Initialize Alpha Vantage client for fetching symbols
alpha_vantage = AlphaVantageClient()

# Initialize session state
if 'available_symbols' not in st.session_state:
    try:
        # Fetch available symbols from Alpha Vantage
        # This is a mock list since Alpha Vantage doesn't provide a direct endpoint for all symbols
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
    except Exception as e:
        st.error(f"Error fetching symbols: {str(e)}")
        st.session_state.available_symbols = {}

# Initialize watchlist if not present
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = {
        "Bank of America Corp": {"symbol": "BAC", "currency": "USD"},
        "Bureau Veritas SA": {"symbol": "BVI.PA", "currency": "EUR"},
        "Hut 8 Mining Corp": {"symbol": "HUT", "currency": "USD"},
        "Prosus NV": {"symbol": "PRX.AS", "currency": "EUR"}
    }

# Update the currency symbols mapping
CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "HKD": "HK$",
    "CNY": "¥",
    "AUD": "A$",
    "CAD": "C$"
}

# Helper function to format price with currency
def format_price(price, currency):
    if isinstance(price, str) and price == "N/A":
        return "N/A"
    currency_symbol = CURRENCY_SYMBOLS.get(currency, currency)
    return f"{currency_symbol}{price:.2f}"

try:
    # Initialize services
    trading_service = TradingSignalService()
    volume_analysis_service = VolumeAnalysisService()
    volume_analyzer = VolumeAnalyzer()
    rag_analyzer = RAGVolumeAnalyzer()
    ml_analyzer = MLVolumeAnalyzer()
    signal_generator = SignalGenerator()
    alert_system = AlertSystem()
    volume_shift_analyzer = VolumeShiftAnalyzer()

    # Sidebar configuration
    with st.sidebar:
        st.title("Oracle of Delphi 🏛️")
        st.write("Welcome to the Market Intelligence Dashboard")

        # Add new asset to watchlist
        st.subheader("Add New Asset")

        # Asset category selection
        category = st.selectbox(
            "Asset Category",
            options=list(st.session_state.available_symbols.keys())
        )

        # Create a dictionary of all symbols in selected category
        category_symbols = st.session_state.available_symbols[category]
        symbol_names = [f"{symbol} - {name['name']}" for symbol, name in category_symbols.items()]

        # Symbol selection with search
        selected_symbol = st.selectbox(
            "Select Symbol",
            options=symbol_names,
            key="symbol_selector"
        )

        if st.button("Add to Watchlist"):
            if selected_symbol:
                symbol = selected_symbol.split(" - ")[0]
                name = category_symbols[symbol]['name']
                if name not in st.session_state.watchlist:
                    st.session_state.watchlist[name] = {"symbol": symbol, "currency": category_symbols[symbol]['currency']}
                    st.success(f"Added {name} ({symbol}) to watchlist")
                else:
                    st.warning(f"{name} is already in your watchlist")

        # Option to remove from watchlist
        st.subheader("Remove from Watchlist")
        remove_symbol = st.selectbox(
            "Select Asset to Remove",
            options=list(st.session_state.watchlist.keys())
        )
        if st.button("Remove"):
            if remove_symbol in st.session_state.watchlist:
                del st.session_state.watchlist[remove_symbol]
                st.success(f"Removed {remove_symbol} from watchlist")

    # Main content
    st.title("Oracle of Delphi 🏛️")

    # Create tabs for different views
    watchlist_tab, analysis_tab = st.tabs(["📈 Watchlist", "🔍 Detailed Analysis"])

    with watchlist_tab:
        # Watchlist controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("🔍 Search Assets", key="asset_search")
        with col2:
            category = st.selectbox(
                "Category",
                options=list(st.session_state.available_symbols.keys())
            )
        with col3:
            st.write("")  # Spacing
            show_add_asset = st.button("➕ Add Asset")

        # Asset addition modal (shown when button is clicked)
        if show_add_asset:
            st.subheader("Add New Asset")
            # Filter assets based on search query
            category_symbols = st.session_state.available_symbols[category]
            filtered_symbols = {
                symbol: name for symbol, name in category_symbols.items()
                if (search_query.lower() in symbol.lower() or
                    search_query.lower() in name['name'].lower())
            }
            symbol_names = [f"{symbol} - {name['name']}" for symbol, name in filtered_symbols.items()]

            selected_symbol = st.selectbox(
                "Select Asset",
                options=symbol_names,
                key="filtered_symbol_selector"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Add"):
                    if selected_symbol:
                        symbol = selected_symbol.split(" - ")[0]
                        name = filtered_symbols[symbol]['name']
                        if name not in st.session_state.watchlist:
                            st.session_state.watchlist[name] = {"symbol": symbol, "currency": filtered_symbols[symbol]['currency']}
                            st.success(f"Added {name} ({symbol}) to watchlist")
                        else:
                            st.warning(f"{name} is already in your watchlist")

        # Watchlist management
        if st.session_state.watchlist:
            st.subheader("Current Watchlist")

            # Initialize watchlist data
            watchlist_data = []

            # Process each asset in watchlist
            for company_name, details in st.session_state.watchlist.items():
                try:
                    symbol = details["symbol"]
                    currency = details["currency"]

                    # Fetch market data
                    data = alpha_vantage.fetch_daily_adjusted(symbol)
                    if data is not None:
                        # Calculate metrics
                        latest_data = data.iloc[-1]
                        current_price = latest_data['Close']

                        # Train models if needed
                        trading_service.train_models(data)
                        volume_analysis_service.train_models(data)

                        # Get analysis after training
                        volume_analysis = volume_analysis_service.analyze_volume_patterns(data)
                        trading_signals = trading_service.generate_trading_signals(data)
                        entry_signals = trading_service.get_entry_signals(data)

                        # Get market regime info
                        regime_info = trading_signals.get('market_regime', {
                            'regime_type': 'NEUTRAL',
                            'regime_probability': 0.5,
                            'regime_strength': 0.5
                        })

                        # Get price forecast
                        price_forecast = trading_signals.get('price_forecast', {
                            'predicted_price': latest_data['Close'],
                            'confidence_score': 0.5,
                            'price_change': 0.0
                        })

                        # Calculate price change
                        price_change = ((price_forecast['predicted_price'] - current_price) / current_price) * 100

                        # Get stop recommendation
                        stop_info = trading_signals.get('stop_recommendation', {
                            'suggested_stop_price': current_price * 0.95,
                            'optimal_distance_percent': 0.05,
                            'confidence_level': 0.5
                        })

                        # Compile asset data with ML signals
                        asset_data = {
                            "Asset": company_name,
                            "Symbol": symbol,
                            "Price": format_price(current_price, currency),
                            "Volume": f"{latest_data['Volume']:,.0f}",
                            "Volume Trend": volume_analysis.get('recent_pattern', {}).get('type', 'NEUTRAL'),
                            "Market Regime": regime_info['regime_type'],
                            "Regime Confidence": f"{regime_info['regime_probability']:.1%}",
                            "Predicted Price": format_price(price_forecast['predicted_price'], currency),
                            "Price Change": f"{price_change:+.1f}%",
                            "Forecast Confidence": f"{price_forecast['confidence_score']:.1%}",
                            "Stop Loss": format_price(stop_info['suggested_stop_price'], currency),
                            "Stop Distance": f"{stop_info['optimal_distance_percent']:.1%}",
                            "Trading Signal": entry_signals['signal'] if entry_signals and 'signal' in entry_signals else 'NEUTRAL',
                            "Signal Confidence": f"{entry_signals.get('confidence', 0.5):.1%}" if entry_signals else "0.0%",
                            "Alerts": len(volume_analysis.get('alerts', [])),
                        }
                        watchlist_data.append(asset_data)
                except Exception as e:
                    st.error(f"Error processing {symbol}: {str(e)}")
                    watchlist_data.append({
                        "Asset": company_name,
                        "Symbol": symbol,
                        "Price": "N/A",
                        "Volume": "N/A",
                        "Volume Trend": "ERROR",
                        "Market Regime": "ERROR",
                        "Regime Confidence": "0.0%",
                        "Predicted Price": "N/A",
                        "Price Change": "0.0%",
                        "Forecast Confidence": "0.0%",
                        "Stop Loss": "N/A",
                        "Stop Distance": "0.0%",
                        "Trading Signal": "ERROR",
                        "Signal Confidence": "0.0%",
                        "Alerts": 0,
                    })

            # Convert to DataFrame for display
            if watchlist_data:
                watchlist_df = pd.DataFrame(watchlist_data)

                # Add sorting and filtering
                col1, col2 = st.columns([2, 2])
                with col1:
                    sort_column = st.selectbox("Sort by", watchlist_df.columns)
                with col2:
                    sort_order = st.radio("Sort order", ["Ascending", "Descending"], horizontal=True)

                # Apply sorting
                watchlist_df = watchlist_df.sort_values(
                    by=sort_column,
                    ascending=(sort_order == "Ascending")
                )

                # Display the watchlist with enhanced styling
                st.dataframe(
                    watchlist_df,
                    use_container_width=True,
                    column_config={
                        "Asset": st.column_config.TextColumn("Asset", width="medium"),
                        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                        "Volume": st.column_config.TextColumn("Volume", width="medium"),
                        "Volume Trend": st.column_config.TextColumn("Volume Trend", width="medium"),
                        "Market Regime": st.column_config.TextColumn("Market Regime", width="medium"),
                        "Regime Confidence": st.column_config.TextColumn("Regime Conf.", width="small"),
                        "Predicted Price": st.column_config.TextColumn("Pred. Price", width="small"),
                        "Price Change": st.column_config.TextColumn("Pred. Change", width="small"),
                        "Forecast Confidence": st.column_config.TextColumn("Forecast Conf.", width="small"),
                        "Stop Loss": st.column_config.TextColumn("Stop Loss", width="small"),
                        "Stop Distance": st.column_config.TextColumn("Stop Dist.", width="small"),
                        "Trading Signal": st.column_config.TextColumn("Signal", width="small"),
                        "Signal Confidence": st.column_config.TextColumn("Signal Conf.", width="small"),
                        "Alerts": st.column_config.NumberColumn("Alerts", width="small"),
                    },
                    hide_index=True,
                )

                # Add remove button for each asset
                to_remove = st.multiselect(
                    "Select assets to remove",
                    options=list(st.session_state.watchlist.keys())
                )
                if to_remove and st.button("Remove Selected"):
                    for asset in to_remove:
                        del st.session_state.watchlist[asset]
                    st.success(f"Removed {len(to_remove)} assets from watchlist")
                    st.rerun()

    with analysis_tab:
        # Selected Asset Analysis
        selected_stock = st.selectbox(
            "Select Asset for Detailed Analysis",
            list(st.session_state.watchlist.keys()),
            format_func=lambda x: f"{x} ({st.session_state.watchlist[x]['symbol']})"
        )

        symbol = st.session_state.watchlist[selected_stock]['symbol']
        currency = st.session_state.watchlist[selected_stock]['currency']

        timeframe = st.selectbox(
            "Select Timeframe",
            ["1d", "5d", "1mo", "3mo"]
        )

        # Main content
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Show loading message while fetching data
        status_text.text('Initializing...')
        progress_bar.progress(10)

        with st.spinner('Loading market data...'):
            status_text.text('Fetching market data...')
            progress_bar.progress(20)
            data = alpha_vantage.fetch_daily_adjusted(symbol)
            progress_bar.progress(30)

        if data is not None and not data.empty:
            # Train models if needed
            with st.spinner('Analyzing market patterns...'):
                status_text.text('Training machine learning models...')
                progress_bar.progress(40)
                trading_service.train_models(data)
                progress_bar.progress(60)

                status_text.text('Analyzing volume patterns...')
                volume_analysis_service.train_models(data)
                progress_bar.progress(70)

                # Get trading signals
                status_text.text('Generating trading signals...')
                trading_signals = trading_service.generate_trading_signals(data)
                progress_bar.progress(80)

                status_text.text('Calculating entry points...')
                entry_signals = trading_service.get_entry_signals(data)
                progress_bar.progress(90)

            # Clear progress indicators once loading is complete
            progress_bar.progress(100)
            status_text.text('Analysis complete!')
            time.sleep(1)  # Brief pause to show completion
            progress_bar.empty()
            status_text.empty()

            # Add new section for ML Trading Signals
            st.subheader("🎯 ML Trading Signals")

            # Display market regime information
            st.markdown("""
                #### Market Regime Analysis
                """)

            # Ensure trading_signals has market_regime with defaults
            if 'error' in trading_signals:
                st.error(f"Error generating trading signals: {trading_signals['error']}")
                regime_info = {
                    'regime_type': 'NEUTRAL',
                    'regime_probability': 0.5,
                    'regime_strength': 0.5
                }
            else:
                regime_info = trading_signals.get('market_regime', {
                    'regime_type': 'NEUTRAL',
                    'regime_probability': 0.5,
                    'regime_strength': 0.5
                })

            # Display regime metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Current Regime",
                    regime_info.get('regime_type', 'NEUTRAL'),
                    f"{regime_info.get('regime_probability', 0.5):.1%} confidence"
                )
            with col2:
                st.metric(
                    "Regime Strength",
                    f"{regime_info.get('regime_strength', 0.5):.1%}",
                    "Market conviction"
                )
            # Price Predictions
            st.markdown("""
                #### Price Forecast
                """)
            price_forecast = trading_signals.get('price_forecast', {
                'predicted_price': data['Close'].iloc[-1],
                'confidence_score': 0.5,
                'volume_impact': {'trend': 'NEUTRAL'}
            })
            current_price = data['Close'].iloc[-1]
            price_change = (price_forecast['predicted_price'] - current_price) / current_price

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Predicted Price",
                    format_price(price_forecast['predicted_price'], currency),
                    f"{price_change:.1%}",
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "Forecast Confidence",
                    f"{price_forecast['confidence_score']:.1%}",
                    f"Volume Impact: {price_forecast['volume_impact'].get('trend', 'NEUTRAL')}"
                )

            # Stop Recommendations
            st.markdown("""
                #### Stop Loss Recommendations
                """)
            stop_info = trading_signals['stop_recommendation']

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Suggested Stop Price",
                    format_price(stop_info['suggested_stop_price'], currency),
                    f"{stop_info['optimal_distance_percent']:.1%} from current price"
                )
            with col2:
                st.metric(
                    "Stop Confidence",
                    f"{stop_info['confidence_level']:.1%}",
                    f"Based on {trading_signals['market_regime']['regime_type']}"
                )

            # Entry Signals
            if entry_signals:
                signal_color = "green" if entry_signals['signal'] == 'LONG' else "red"
                st.markdown(f"""
                    <div style='padding: 10px; border-left: 5px solid {signal_color}; background-color: rgba(0,0,0,0.1);'>
                        <h4>Entry Signal: {entry_signals['signal']}</h4>
                        <b>Confidence:</b> {entry_signals['confidence']:.1%}<br>
                        <b>Expected Return:</b> {entry_signals['expected_return']:.1%}<br>
                        <b>Suggested Stop:</b> {format_price(entry_signals['suggested_stop'], currency)}<br>
                        <b>Reasoning:</b><br>
                        {'<br>'.join(f'• {reason}' for reason in entry_signals['reasoning'])}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No clear entry signals at this time", icon="ℹ️")

            # Process volume data
            data = volume_analyzer.calculate_vwap(data)
            patterns = volume_analyzer.detect_volume_patterns(data)

            # Get RAG and ML insights
            rag_insights = rag_analyzer.analyze_volume_pattern(data)
            ml_insights = ml_analyzer.detect_volume_patterns(data)

            # Analyze volume shifts
            volume_shift_analysis = volume_shift_analyzer.analyze_volume_shift(data)
            explanatory_insights = volume_shift_analyzer.get_explanatory_insights(data)

            # Display volume shift insights
            st.markdown("### 🔄 Volume Shift Analysis")

            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Relative Volume",
                    f"{volume_shift_analysis['latest_metrics']['rvol']:.1f}x",
                    f"{volume_shift_analysis['latest_metrics']['volume_trend']:.1%}"
                )
            with col2:
                st.metric(
                    "Volume Pattern",
                    volume_shift_analysis['classification']
                )
            with col3:
                st.metric(
                    "Price-Volume Correlation",
                    f"{volume_shift_analysis['latest_metrics']['price_volume_correlation']:.2f}"
                )

            # Display key insights
            st.markdown("#### Key Insights")
            for insight in explanatory_insights:
                st.markdown(f"• {insight}")

            # Display feature importance
            st.markdown("#### Volume Drivers")
            importance_data = volume_shift_analysis['feature_importance']

            fig = go.Figure(data=[
                go.Bar(
                    y=list(importance_data.keys()),
                    x=list(importance_data.values()),
                    orientation='h',
                    marker_color='#00A68C'
                )
            ])

            fig.update_layout(
                title="Driver Attribution",
                xaxis_title="Impact Score",
                template="plotly_dark",
                height=200
            )

            st.plotly_chart(fig, use_container_width=True)


            # Create main chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03,
                                row_heights=[0.7, 0.3])

            # Candlestick chart
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

            # VWAP line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['VWAP'],
                    name="VWAP",
                    line=dict(color='purple')
                ),
                row=1, col=1
            )

            # Volume bars with color based on ML anomaly detection
            colors = ['red' if a == -1 else 'green' for a in ml_insights['anomalies']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name="Volume",
                    marker_color=colors
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                height=800,
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ML Insights Section
            st.subheader("🤖 Machine Learning Insights")

            # Display volume patterns
            if ml_insights['patterns']:
                st.markdown("#### Recent Volume Patterns")
                for pattern in ml_insights['patterns']:
                    pattern_color = {
                        'BREAKOUT': 'green',
                        'FAKE_BREAKOUT': 'red',
                        'CLIMAX': 'orange',
                        'ACCUMULATION': 'blue',
                        'DISTRIBUTION': 'purple',
                        'NEUTRAL': 'gray'
                    }.get(pattern['pattern'], 'gray')

                    st.markdown(f"""
                        <div style='padding: 10px; border-left: 5px solid {pattern_color}; background-color: rgba(0,0,0,0.1);'>
                            <h4>{pattern['pattern']}</h4>
                            <b>Confidence:</b> {pattern['confidence']:.1%}<br>
                            <b>Volume Ratio:</b> {pattern['metrics']['volume_ratio']:.1f}x average<br>
                            <b>Price-Volume Correlation:</b> {pattern['metrics']['price_volume_corr']:.2f}<br>
                            <b>Trend Strength:</b> {pattern['metrics']['trend_strength']:.2f}
                        </div>
                    """, unsafe_allow_html=True)

            # Recent Pattern Analysis
            recent_pattern = ml_insights['recent_pattern']
            pattern_color = {
                'BULLISH_MOMENTUM': 'green',
                'BEARISH_MOMENTUM': 'red',
                'VOLUME_CLIMAX': 'yellow',
                'CONSOLIDATION': 'blue',
                'NEUTRAL': 'gray'
            }.get(recent_pattern['type'], 'gray')

            st.markdown("""
                #### Overall Volume Analysis
            """)
            st.markdown(f"""
                <div style='padding: 10px; border-left: 5px solid {pattern_color}; background-color: rgba(0,0,0,0.1);'>
                    <h3>Volume Pattern Analysis</h3>
                    <b>Pattern Type:</b> {recent_pattern['type']}<br>
                    <b>Strength:</b> {recent_pattern['strength']:.2f}<br>
                    <b>Confidence:</b> {recent_pattern['confidence']:.2%}<br>
                    <b>Volume Profile:</b> {ml_insights['volume_profile']}<br>
                </div>
            """, unsafe_allow_html=True)

            # Show volume anomalies on a separate chart if needed
            if len(ml_insights['anomalies']) > 0:
                st.markdown("#### Volume Anomalies")
                anomaly_fig = go.Figure()
                anomaly_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Volume'],
                    mode='lines',
                    name='Volume',
                    line=dict(color='blue', width=1)
                ))

                # Highlight anomalies
                anomaly_indices = [i for i, x in enumerate(ml_insights['anomalies']) if x == -1]
                if anomaly_indices:
                    anomaly_fig.add_trace(go.Scatter(
                        x=data.index[anomaly_indices],
                        y=data['Volume'].iloc[anomaly_indices],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=10)
                    ))

                anomaly_fig.update_layout(
                    height=300,
                    title="Volume Anomalies Detection",
                    template="plotly_dark"
                )
                st.plotly_chart(anomaly_fig, use_container_width=True)

            # AI Insights Section
            st.subheader("🤖 AI-Powered Volume Insights")
            if rag_insights['insights']:
                for insight in rag_insights['insights']:
                    with st.expander(f"📊 {insight['pattern'].replace('_', ' ').title()}", expanded=True):
                        st.markdown(f"""
                            **Pattern Description:**  
                            {insight['description']}

                            **Signal:**  
                            {insight['signal']}

                            **Metrics:**
                            - Volume Ratio: {insight['metrics']['volume_ratio']:.2f}x average
                            - Price Change: {insight['metrics']['price_change']:.2%}
                        """)
            else:
                st.info("No significant volume patterns detected by AI", icon="ℹ️")

            # Volume Patterns Analysis
            st.subheader("📈 Volume Patterns")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                accumulation_count = patterns['accumulation'].sum()
                st.metric("Accumulation Patterns", accumulation_count)

            with col2:
                distribution_count = patterns['distribution'].sum()
                st.metric("Distribution Patterns", distribution_count)

            with col3:
                breakout_count = patterns['breakout'].sum()
                st.metric("Breakout Patterns", breakout_count)

            with col4:
                exhaustion_count = patterns['exhaustion'].sum()
                st.metric("Exhaustion Patterns", exhaustion_count)

            # Pattern Search
            st.subheader("🔍 Pattern Search")
            search_query = st.text_input("Search for similar volume patterns",
                                        placeholder="e.g., 'high volume breakout with positive price action'")
            if search_query:
                similar_patterns = rag_analyzer.find_similar_patterns(search_query)
                for pattern in similar_patterns:
                    st.markdown(f"""
                        <div style='padding: 10px; border-left: 5px solid purple; background-color: rgba(0,0,0,0.1);'>
                            <b>{pattern['pattern'].replace('_', ' ').title()}</b> (Score: {pattern['similarity_score']:.2f})<br>
                            {pattern['description']}<br>
                            <i>{pattern['signal']}</i>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Trading Signals
            st.subheader("🎯 Trading Signals")
            signals = signal_generator.generate_volume_signals(data)
            if signals:
                for signal in signals[-5:]:  # Show last 5 signals
                    severity_color = {
                        'HIGH': 'red',
                        'MEDIUM': 'orange',
                        'LOW': 'blue'
                    }.get(signal['strength'], 'blue')

                    st.markdown(
                        f"""
                        <div style='padding: 10px; border-left: 5px solid {severity_color}; background-color: rgba(0,0,0,0.1);'>
                            🎯 <b>{signal['type']}</b><br>
                            📊 Strength: <span style='color: {severity_color}'>{signal['strength']}</span><br>
                            💡 {signal['message']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No active trading signals", icon="ℹ️")

            # Get volume analysis
            volume_analysis = volume_analysis_service.analyze_volume_patterns(data)

            # Display volume analysis results
            st.subheader("🔍 Volume Analysis")

            if 'error' not in volume_analysis:
                # Display predicted spikes
                if volume_analysis['predicted_spikes']:
                    st.markdown("### 📈 Predicted Volume Spikes")
                    for spike in volume_analysis['predicted_spikes']:
                        st.markdown(f"""
                            <div style='padding: 10px; border-left: 5px solid orange; background-color: rgba(0,0,0,0.1);'>
                                <h4>Predicted {spike['volume_ratio']:.1f}x Volume Spike</h4>
                                <b>When:</b> {spike['predicted_period']}<br>
                                <b>Confidence:</b> {spike['confidence_score']:.1%}<br>
                                <b>Current Avg Volume:</b> {spike['current_avg_volume']:,.0f}
                            </div>
                        """, unsafe_allow_html=True)

                # Display anomalies
                if volume_analysis['volume_anomalies']:
                    st.markdown("### ⚠️ Volume Anomalies")
                    for anomaly in volume_analysis['volume_anomalies']:
                        severity_color = 'red' if anomaly['anomaly_score'] > 3.0 else 'orange'
                        st.markdown(f"""
                            <div style='padding: 10px; border-left: 5px solid {severity_color}; background-color: rgba(0,0,0,0.1);'>
                                <h4>Volume Anomaly Detected</h4>
                                <b>Score:</b> {anomaly['anomaly_score']:.1f}<br>
                                <b>Confidence:</b> {anomaly['confidence']:.1%}<br>
                                <b>Pattern:</b> {anomaly['pattern_start']} to {anomaly['pattern_end']}
                            </div>
                        """, unsafe_allow_html=True)

                # Display alerts
                if volume_analysis['alerts']:
                    st.markdown("### 🚨 Active Alerts")
                    for alert in volume_analysis['alerts']:
                        alert_color = 'red' if alert['priority'] == 'HIGH' else 'orange'
                        st.markdown(f"""
                            <div style='padding: 10px; border-left: 5px solid {alert_color}; background-color: rgba(0,0,0,0.1);'>
                                <h4>{alert['type'].replace('_', ' ').title()}</h4>
                                <b>Priority:</b> {alert['priority']}<br>
                                <b>Confidence:</b> {alert['confidence']:.1%}<br>
                                <b>Message:</b> {alert['message']}
                            </div>
                        """, unsafe_allow_html=True)

                # Get external insights
                insights = volume_analysis_service.get_external_insights(symbol, volume_analysis['alerts'])

                # Display external insights
                st.subheader("🌐 Market Intelligence")

                # Add Sentiment Analysis Section
                st.markdown("### 📊 Social Sentiment Analysis")

                # Create metrics for sentiment
                col1, col2, col3 = st.columns(3)

                with col1:
                    sentiment_score = insights['social_sentiment']['overall_score']
                    sentiment_change = insights['social_sentiment']['sentiment_change_24h']
                    st.metric(
                        "Overall Sentiment",
                        f"{sentiment_score:.1%}",
                        f"{sentiment_change:+.1%}",
                        delta_color="normal"
                    )

                with col2:
                    volume_change = insights['social_sentiment']['volume_change']
                    st.metric(
                        "Social Volume Change",
                        f"{volume_change:+.1f}x"
                    )

                with col3:
                    confidence = insights['social_sentiment']['confidence']
                    st.metric(
                        "Analysis Confidence",
                        f"{confidence:.1%}"
                    )

                # Display sentiment signals
                if 'sentiment_signals' in insights:
                    signal = insights['sentiment_signals']
                    signal_color = 'green' if signal['signal'] == 'BULLISH' else 'red'

                    st.markdown(f"""
                        <div style='padding: 10px; border-left: 5px solid {signal_color}; background-color: rgba(0,0,0,0.1);'>
                            <h4>Sentiment Signal: {signal['signal']}</h4>
                            <b>Strength:</b> {signal['strength']:.1%}<br>
                            <b>Confidence:</b> {signal['confidence']:.1%}<br>
                            <b>Risk Level:</b> {signal['risk_level']}<br>
                            <b>Suggested Action:</b> {signal['suggested_action']}
                        </div>
                    """, unsafe_allow_html=True)

                # Source Breakdown
                st.markdown("#### Source Analysis")
                source_data = insights['social_sentiment']['source_breakdown']
                source_df = pd.DataFrame({
                    'Source': list(source_data.keys()),
                    'Sentiment': list(source_data.values())
                })

                fig = go.Figure(data=[
                                        go.Bar(
                        x=source_df['Source'],
                        y=source_df['Sentiment'],
                        marker_color=['#1DA1F2', '#FF4500', '#00A68C']  # Twitter, Reddit, StockTwits colors
                    )
                ])

                fig.update_layout(
                    title="Sentiment by Platform",
                    yaxis_title="Sentiment Score",
                    template="plotly_dark",
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display trending topics
                if 'trending_topics' in insights['social_sentiment']:
                    st.markdown("#### 🔥 Trending Topics")
                    for topic in insights['social_sentiment']['trending_topics']:
                        st.markdown(f"• {topic.title()}")
            else:
                st.error("Error analyzing volume patterns")

        else:
            st.error("Unable to fetch market data. Please try again later or select a different stock.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please refresh the page to try again.")