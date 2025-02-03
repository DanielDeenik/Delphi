import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
from src.data.alpha_vantage_client import AlphaVantageClient
from src.services.volume_analysis_service import VolumeAnalysisService
from src.services.trading_signal_service import TradingSignalService
from src.services.data_processing_service import DataProcessingService

logger = logging.getLogger(__name__)

class SchedulerService:
    """Service for scheduling and running nightly data processing tasks"""

    def __init__(self):
        logger.info("Initializing SchedulerService")

        # Initialize all required services
        self.data_processing = DataProcessingService()
        self.volume_analysis = VolumeAnalysisService()
        self.trading_service = TradingSignalService()

        # Available stocks for processing
        self.available_stocks = {
            "Bank of America Corp": "BAC",
            "Bureau Veritas SA": "BVI.PA",
            "Hut 8 Mining Corp": "HUT",
            "Prosus NV": "PRX.AS"
        }
        logger.info(f"Initialized with {len(self.available_stocks)} stocks to monitor")

    async def run_nightly_processing(self) -> Dict:
        """Run the nightly processing routine with parallel processing"""
        logger.info("Starting nightly processing routine")
        processing_results = {
            'timestamp': datetime.now().isoformat(),
            'processed_symbols': [],
            'failed_symbols': [],
            'signals_generated': 0,
            'alerts_generated': 0
        }

        try:
            # Process all symbols in parallel using DataProcessingService
            symbols = list(self.available_stocks.values())
            data_results = await self.data_processing.process_market_data(symbols)

            # Process results for each symbol
            for symbol, data in data_results['data'].items():
                try:
                    company_name = [k for k, v in self.available_stocks.items() if v == symbol][0]

                    # Train models if needed
                    logger.info(f"Training models for {symbol}")
                    self.volume_analysis.train_models(data)
                    self.trading_service.train_models(data)

                    # Generate analysis and signals
                    volume_analysis = self.volume_analysis.analyze_volume_patterns(data)
                    trading_signals = self.trading_service.generate_trading_signals(data)

                    # Process alerts and signals
                    alert_count = len(volume_analysis.get('alerts', []))
                    signal_count = 1 if trading_signals.get('market_regime') else 0

                    logger.info(f"Generated {alert_count} alerts and {signal_count} signals for {symbol}")

                    processing_results['processed_symbols'].append({
                        'symbol': symbol,
                        'company_name': company_name,
                        'alerts': alert_count,
                        'signals': signal_count,
                        'timestamp': datetime.now().isoformat()
                    })

                    processing_results['signals_generated'] += signal_count
                    processing_results['alerts_generated'] += alert_count

                except Exception as e:
                    error_msg = f"Error processing {symbol}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    processing_results['failed_symbols'].append({
                        'symbol': symbol,
                        'company_name': company_name,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })

        except Exception as e:
            logger.error(f"Error in nightly processing: {str(e)}", exc_info=True)
            raise

        logger.info(f"Completed nightly processing. "
                   f"Processed: {len(processing_results['processed_symbols'])} symbols, "
                   f"Failed: {len(processing_results['failed_symbols'])} symbols")
        return processing_results

    def get_processing_summary(self, results: Dict) -> str:
        """Generate a human-readable summary of processing results"""
        summary = [
            f"Nightly Processing Summary ({results['timestamp']})",
            f"Successfully processed: {len(results['processed_symbols'])} symbols",
            f"Failed processing: {len(results['failed_symbols'])} symbols",
            f"Total signals generated: {results['signals_generated']}",
            f"Total alerts generated: {results['alerts_generated']}",
            "\nProcessed Symbols:"
        ]

        for symbol in results['processed_symbols']:
            summary.append(
                f"- {symbol['company_name']} ({symbol['symbol']}): "
                f"{symbol['alerts']} alerts, {symbol['signals']} signals"
            )

        if results['failed_symbols']:
            summary.append("\nFailed Symbols:")
            for symbol in results['failed_symbols']:
                summary.append(
                    f"- {symbol['company_name']} ({symbol['symbol']}): {symbol['error']}"
                )

        return "\n".join(summary)