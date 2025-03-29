"""
Integrated Bot Module

This is the main orchestration module for the Option Hunter system.
It coordinates the operation of all other components and implements
the core trading logic.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import os
import traceback

from src.tradier_api import TradierAPI
from src.option_pricing import OptionPricer, OptionType
from src.option_scanner import OptionScanner
from src.paper_trading.trade_entry import TradeEntry
from src.paper_trading.trade_monitor import TradeMonitor
from src.market_regime.market_state_tracker import MarketStateTracker
from src.sentiment_analyzer import SentimentAnalyzer
from src.position_sizing import PositionSizer
from src.order_flow_analyzer import OrderFlowAnalyzer
from src.time_sales_analyzer import TimeSalesAnalyzer
from src.volatility_regime import VolatilityRegimeDetector

class IntegratedBot:
    """
    Main bot class that orchestrates all components and implements the trading strategy.
    """
    
    def __init__(self, config, parameter_hub, drive_manager, notification_system, market_hours, paper_trading=True, debug_mode=False, focus_ticker=None):
        """
        Initialize the IntegratedBot.
        
        Args:
            config (dict): Main configuration
            parameter_hub: MasterParameterHub instance
            drive_manager: GoogleDriveManager instance
            notification_system: NotificationSystem instance
            market_hours: MarketHoursAwareness instance
            paper_trading (bool): If True, run in paper trading mode
            debug_mode (bool): If True, enable additional logging
            focus_ticker (str, optional): Focus on a specific ticker for testing
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.parameter_hub = parameter_hub
        self.drive_manager = drive_manager
        self.notification_system = notification_system
        self.market_hours = market_hours
        self.paper_trading = paper_trading
        self.debug_mode = debug_mode
        self.focus_ticker = focus_ticker
        
        # State variables
        self.running = False
        self.paused = False
        self.threads = []
        self.opportunity_queue = queue.Queue()
        self.active_trades = {}
        self.trade_history = []
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }
        
        # Initialize API
        self.tradier_api = TradierAPI(
            config["api_credentials"]["tradier"]["api_key"],
            config["api_credentials"]["tradier"]["account_id"],
            config["api_credentials"]["tradier"]["sandbox_mode"]
        )
        
        # Initialize components
        self._initialize_components()
        
        # Create directories
        os.makedirs("logs/trades", exist_ok=True)
        os.makedirs("data/collected_options_data", exist_ok=True)
        
        self.logger.info("IntegratedBot initialized")
    
    def _initialize_components(self):
        """Initialize all trading system components."""
        try:
            # Initialize option pricing
            self.logger.info("Initializing option pricer")
            self.option_pricer = OptionPricer(self.config)
            
            # Initialize option scanner
            self.logger.info("Initializing option scanner")
            self.option_scanner = OptionScanner(self.config, self.tradier_api, self.option_pricer)
            
            # Initialize trade entry and monitoring
            self.logger.info("Initializing trade entry and monitoring")
            self.trade_entry = TradeEntry(
                self.config,
                self.tradier_api,
                self.parameter_hub,
                self.notification_system
            )
            
            self.trade_monitor = TradeMonitor(
                self.config,
                self.tradier_api,
                self.notification_system,
                self.parameter_hub
            )
            
            # Initialize market state tracking
            self.logger.info("Initializing market state tracking")
            self.market_state_tracker = MarketStateTracker(self.config, self.tradier_api, self.drive_manager)
            
            # Initialize sentiment analysis
            self.logger.info("Initializing sentiment analyzer")
            self.sentiment_analyzer = SentimentAnalyzer(self.config["api_credentials"]["twitter"])
            
            # Initialize position sizing
            self.logger.info("Initializing position sizer")
            self.position_sizer = PositionSizer(self.config["trade_parameters"], self.parameter_hub)
            
            # Initialize advanced analyzers
            self.logger.info("Initializing advanced analyzers")
            self.order_flow_analyzer = OrderFlowAnalyzer(self.tradier_api)
            self.time_sales_analyzer = TimeSalesAnalyzer(self.tradier_api)
            self.volatility_detector = VolatilityRegimeDetector(self.tradier_api)
            
            # Get ticker lists
            self.all_tickers = self.config["ticker_list"]
            
            # If focus ticker is specified, only use that
            if self.focus_ticker:
                self.watchlist = [self.focus_ticker]
                self.logger.info(f"Focusing on single ticker: {self.focus_ticker}")
            else:
                # Use configured watchlist
                self.watchlist = []
                for priority in ["high_priority", "medium_priority", "low_priority"]:
                    if priority in self.config["watchlist"]:
                        self.watchlist.extend(self.config["watchlist"][priority])
                self.logger.info(f"Using watchlist with {len(self.watchlist)} tickers")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def run(self):
        """Run the main trading loop."""
        self.running = True
        
        try:
            # Send startup notification
            self.notification_system.send_system_notification(
                "Option Hunter Starting",
                f"Starting in {'paper trading' if self.paper_trading else 'live trading'} mode"
            )
            
            # Load trade history if available
            self._load_trade_history()
            
            # Start background threads
            self._start_background_threads()
            
            # Main loop
            self.logger.info("Starting main loop")
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, shutting down")
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            self.notification_system.send_error_notification(
                "Critical Error",
                f"Main loop error: {str(e)}",
                traceback.format_exc()
            )
        finally:
            # Clean shutdown
            self._shutdown()
    
    def _main_loop(self):
        """Main trading loop."""
        while self.running:
            try:
                # Check if market is open
                is_market_open = self.market_hours.is_market_open()
                
                if not is_market_open and not self.debug_mode:
                    # Market is closed, sleep and continue
                    next_event = self.market_hours.get_next_market_event()
                    seconds_to_next = (next_event["time"] - datetime.now(self.market_hours.timezone)).total_seconds()
                    
                    # Show message every hour when market is closed
                    if seconds_to_next % 3600 < 60:
                        self.logger.info(f"Market is closed. Next event: {next_event['description']} in {seconds_to_next//60:.0f} minutes")
                    
                    # Sleep for 60 seconds between checks
                    time.sleep(min(60, max(1, seconds_to_next)))
                    continue
                
                # Market is open or we're in debug mode
                
                # Process any new opportunities in the queue
                while not self.opportunity_queue.empty() and not self.paused:
                    opportunity = self.opportunity_queue.get()
                    self._evaluate_opportunity(opportunity)
                    self.opportunity_queue.task_done()
                
                # Check active trades
                self._monitor_active_trades()
                
                # Determine if we should scan for new opportunities
                should_scan = (len(self.active_trades) < self.config["trade_parameters"]["max_open_positions"]) and not self.paused
                
                if should_scan:
                    self._scan_for_opportunities()
                
                # Check for end of day
                self._check_end_of_day()
                
                # Sleep for a bit to avoid excessive CPU usage
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in main loop iteration: {str(e)}")
                if self.debug_mode:
                    self.logger.error(traceback.format_exc())
                time.sleep(30)  # Sleep longer if there's an error
    
    def _start_background_threads(self):
        """Start background threads for various tasks."""
        # Thread for market regime detection
        market_regime_thread = threading.Thread(
            target=self._market_regime_thread,
            daemon=True,
            name="MarketRegimeThread"
        )
        self.threads.append(market_regime_thread)
        market_regime_thread.start()
        
        # Thread for sentiment analysis
        sentiment_thread = threading.Thread(
            target=self._sentiment_analysis_thread,
            daemon=True,
            name="SentimentThread"
        )
        self.threads.append(sentiment_thread)
        sentiment_thread.start()
        
        # Thread for volatility regime detection
        volatility_thread = threading.Thread(
            target=self._volatility_regime_thread,
            daemon=True,
            name="VolatilityThread"
        )
        self.threads.append(volatility_thread)
        volatility_thread.start()
        
        self.logger.info(f"Started {len(self.threads)} background threads")
    
    def _market_regime_thread(self):
        """Background thread for market regime detection."""
        self.logger.info("Market regime thread started")
        
        while self.running:
            try:
                # Only run during market hours unless in debug mode
                if self.market_hours.is_market_open() or self.debug_mode:
                    # Update market regime
                    self.logger.debug("Updating market regime")
                    
                    # Get SPY as a proxy for overall market
                    current_regime = self.market_state_tracker.detect_current_regime("SPY")
                    
                    # Log the regime
                    self.logger.info(f"Current market regime: {current_regime}")
                    
                    # Feed into parameter hub for adaptation
                    self.parameter_hub.update_parameter_history(
                        self.parameter_hub.get_all_parameters(),
                        self.daily_stats,
                        current_regime
                    )
                
                # Sleep between updates (check every 30 minutes)
                time.sleep(1800)  
                
            except Exception as e:
                self.logger.error(f"Error in market regime thread: {str(e)}")
                if self.debug_mode:
                    self.logger.error(traceback.format_exc())
                time.sleep(300)  # Sleep for 5 minutes on error
    
    def _sentiment_analysis_thread(self):
        """Background thread for sentiment analysis."""
        self.logger.info("Sentiment analysis thread started")
        
        while self.running:
            try:
                # Only run during market hours unless in debug mode
                if self.market_hours.is_market_open() or self.debug_mode:
                    # Update sentiment for watchlist
                    for symbol in self.watchlist:
                        try:
                            sentiment = self.sentiment_analyzer.analyze_sentiment(symbol)
                            self.logger.debug(f"Sentiment for {symbol}: {sentiment}")
                            
                            # Store sentiment data for use in trading decisions
                            # (implementation depends on how sentiment is used in the strategy)
                        except Exception as e:
                            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
                
                # Sleep between updates (check every hour)
                time.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Error in sentiment analysis thread: {str(e)}")
                if self.debug_mode:
                    self.logger.error(traceback.format_exc())
                time.sleep(300)  # Sleep for 5 minutes on error
    
    def _volatility_regime_thread(self):
        """Background thread for volatility regime detection."""
        self.logger.info("Volatility regime thread started")
        
        while self.running:
            try:
                # Only run during market hours unless in debug mode
                if self.market_hours.is_market_open() or self.debug_mode:
                    # Update volatility regime for SPY and VIX
                    vol_regime = self.volatility_detector.detect_regime("SPY")
                    self.logger.info(f"Current volatility regime: {vol_regime}")
                    
                    # Get VIX level if needed
                    try:
                        vix_quote = self.tradier_api.get_quotes("VIX")
                        if not vix_quote.empty:
                            vix_level = vix_quote.iloc[0]['last']
                            self.logger.info(f"Current VIX level: {vix_level}")
                    except Exception as e:
                        self.logger.error(f"Error getting VIX data: {str(e)}")
                
                # Sleep between updates (check every 15 minutes)
                time.sleep(900)
                
            except Exception as e:
                self.logger.error(f"Error in volatility regime thread: {str(e)}")
                if self.debug_mode:
                    self.logger.error(traceback.format_exc())
                time.sleep(300)  # Sleep for 5 minutes on error
    
    def _scan_for_opportunities(self):
        """Scan watchlist for new trading opportunities."""
        # Skip scanning if we have reached max positions
        if len(self.active_trades) >= self.config["trade_parameters"]["max_open_positions"]:
            return
        
        self.logger.debug("Scanning for new opportunities")
        
        try:
            # Use a smaller batch size to avoid overwhelming the API
            batch_size = min(5, len(self.watchlist))
            ticker_batch = self.watchlist[:batch_size]
            
            # Scan the batch
            opportunities = self.option_scanner.scan_watchlist(ticker_batch)
            
            # Filter and prioritize opportunities
            if opportunities:
                # Apply additional filtering based on current market conditions
                # and trading parameters
                filtered_opportunities = self.option_scanner.filter_opportunities(
                    opportunities,
                    max_results=10,
                    min_volume=self.config["option_scanner"]["min_volume"],
                    min_open_interest=self.config["option_scanner"]["min_open_interest"]
                )
                
                # Add to the queue for evaluation
                for opportunity in filtered_opportunities:
                    self.opportunity_queue.put(opportunity)
                
                self.logger.info(f"Added {len(filtered_opportunities)} opportunities to the queue")
            
            # Rotate the watchlist for next scan
            self.watchlist = self.watchlist[batch_size:] + self.watchlist[:batch_size]
            
        except Exception as e:
            self.logger.error(f"Error scanning for opportunities: {str(e)}")
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
    
    def _evaluate_opportunity(self, opportunity):
        """
        Evaluate a trading opportunity and decide whether to enter a trade.
        
        Args:
            opportunity (dict): Option opportunity data
        """
        self.logger.info(f"Evaluating opportunity: {opportunity['symbol']} {opportunity['option_type']} {opportunity['strike']} {opportunity['expiration']}")
        
        try:
            # Get current market state
            market_regime = self.market_state_tracker.get_current_regime()
            volatility_regime = self.volatility_detector.get_current_regime()
            
            # Get sentiment for the symbol
            sentiment = self.sentiment_analyzer.get_sentiment(opportunity['symbol'])
            
            # Get order flow data
            order_flow_signal = self.order_flow_analyzer.analyze_symbol(opportunity['symbol'])
            
            # Get time & sales data
            time_sales_signal = self.time_sales_analyzer.analyze_symbol(opportunity['symbol'])
            
            # Combine signals for entry decision
            entry_score = self._calculate_entry_score(
                opportunity,
                market_regime,
                volatility_regime,
                sentiment,
                order_flow_signal,
                time_sales_signal
            )
            
            # Set minimum threshold for entry
            min_entry_threshold = 70  # Score from 0-100
            
            if entry_score >= min_entry_threshold:
                # Calculate position size
                position_size = self.position_sizer.calculate_position_size(
                    opportunity['symbol'],
                    opportunity['option_type'],
                    opportunity['mid_price'],
                    market_regime,
                    volatility_regime
                )
                
                # Check if we have enough capital
                account_info = self.tradier_api.get_account_balances()
                available_capital = float(account_info.get('total_cash', 0))
                
                if position_size <= available_capital:
                    # Enter the trade
                    self._enter_trade(opportunity, position_size, entry_score)
                else:
                    self.logger.warning(f"Insufficient capital for trade. Required: ${position_size}, Available: ${available_capital}")
            else:
                self.logger.info(f"Entry score {entry_score} below threshold {min_entry_threshold}, skipping trade")
                
        except Exception as e:
            self.logger.error(f"Error evaluating opportunity: {str(e)}")
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
    
    def _calculate_entry_score(self, opportunity, market_regime, volatility_regime, sentiment, order_flow, time_sales):
        """
        Calculate a composite score to determine if we should enter a trade.
        
        Args:
            opportunity (dict): Option opportunity data
            market_regime (str): Current market regime
            volatility_regime (str): Current volatility regime
            sentiment (str): Sentiment analysis result
            order_flow (dict): Order flow analysis
            time_sales (dict): Time & sales analysis
            
        Returns:
            float: Entry score from 0-100
        """
        # Start with a base score based on price difference
        # More undervalued = higher score
        base_score = min(50, abs(opportunity['diff_percent']) * 2)
        
        # Add points for strong sentiment alignment
        sentiment_score = 0
        if opportunity['option_type'] == 'call' and sentiment == 'very_bullish':
            sentiment_score = 15
        elif opportunity['option_type'] == 'call' and sentiment == 'bullish':
            sentiment_score = 10
        elif opportunity['option_type'] == 'put' and sentiment == 'very_bearish':
            sentiment_score = 15
        elif opportunity['option_type'] == 'put' and sentiment == 'bearish':
            sentiment_score = 10
        
        # Add points for favorable market regime
        regime_score = 0
        if (opportunity['option_type'] == 'call' and market_regime == 'bullish') or \
           (opportunity['option_type'] == 'put' and market_regime == 'bearish'):
            regime_score = 15
        elif market_regime == 'volatile' and volatility_regime == 'high':
            # In volatile markets, both calls and puts can be profitable
            regime_score = 10
        
        # Add points for favorable order flow
        order_flow_score = 0
        if order_flow.get('signal') == 'bullish' and opportunity['option_type'] == 'call':
            order_flow_score = 10
        elif order_flow.get('signal') == 'bearish' and opportunity['option_type'] == 'put':
            order_flow_score = 10
        
        # Add points for favorable time & sales
        time_sales_score = 0
        if time_sales.get('signal') == 'bullish' and opportunity['option_type'] == 'call':
            time_sales_score = 10
        elif time_sales.get('signal') == 'bearish' and opportunity['option_type'] == 'put':
            time_sales_score = 10
        
        # Calculate total score (max 100)
        total_score = min(100, base_score + sentiment_score + regime_score + order_flow_score + time_sales_score)
        
        self.logger.debug(
            f"Entry score breakdown - Base: {base_score}, Sentiment: {sentiment_score}, "
            f"Regime: {regime_score}, OrderFlow: {order_flow_score}, TimeSales: {time_sales_score}, "
            f"Total: {total_score}"
        )
        
        return total_score
    
    def _enter_trade(self, opportunity, position_size, entry_score):
        """
        Enter a new trade based on an opportunity.
        
        Args:
            opportunity (dict): Option opportunity data
            position_size (float): Dollar amount to invest
            entry_score (float): Entry score for this opportunity
            
        Returns:
            bool: True if trade was entered successfully
        """
        self.logger.info(f"Entering trade for {opportunity['option_symbol']} with position size ${position_size}")
        
        try:
            # Calculate number of contracts
            option_price = opportunity['ask']  # Use ask price for entry
            contracts = max(1, int(position_size / (option_price * 100)))
            
            # Create trade entry data
            trade_data = {
                'symbol': opportunity['symbol'],
                'underlying_price': opportunity['underlying_price'],
                'option_symbol': opportunity['option_symbol'],
                'option_type': opportunity['option_type'],
                'strike': opportunity['strike'],
                'expiration': opportunity['expiration'],
                'entry_price': option_price,
                'quantity': contracts,
                'position_size': option_price * contracts * 100,
                'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'entry_score': entry_score,
                'model_price': opportunity['model_price'],
                'market_price': option_price,
                'diff_percent': opportunity['diff_percent'],
                'greeks': {
                    'delta': opportunity['delta'],
                    'gamma': opportunity['gamma'],
                    'theta': opportunity['theta'],
                    'vega': opportunity['vega']
                },
                'trade_id': f"{opportunity['symbol']}_{int(time.time())}",
                'status': 'active',
                'exit_price': None,
                'exit_time': None,
                'pnl': 0.0,
                'pnl_percent': 0.0
            }
            
            # Execute the paper trade
            if self.paper_trading:
                result = self.trade_entry.enter_paper_trade(trade_data)
            else:
                # For live trading, use the Tradier API to place the order
                result = self.trade_entry.enter_live_trade(trade_data)
            
            if result:
                # Add to active trades
                self.active_trades[trade_data['trade_id']] = trade_data
                
                # Track daily stats
                self.daily_stats['total_trades'] += 1
                
                # Send notification
                self.notification_system.send_trade_entry_notification(trade_data)
                
                # Log the trade
                self._log_trade(trade_data)
                
                return True
            else:
                self.logger.error(f"Failed to enter trade for {opportunity['option_symbol']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error entering trade: {str(e)}")
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
            return False
    
    def _monitor_active_trades(self):
        """Monitor and manage active trades."""
        if not self.active_trades:
            return
        
        self.logger.debug(f"Monitoring {len(self.active_trades)} active trades")
        
        for trade_id, trade in list(self.active_trades.items()):
            try:
                # Get current option price
                option_quote = self.tradier_api.get_option_quote(trade['option_symbol'])
                
                if not option_quote:
                    self.logger.warning(f"Could not get quote for {trade['option_symbol']}")
                    continue
                
                # Update current price in trade data
                current_price = (option_quote['bid'] + option_quote['ask']) / 2
                trade['current_price'] = current_price
                
                # Calculate current P&L
                trade['current_pnl'] = (current_price - trade['entry_price']) * trade['quantity'] * 100
                trade['current_pnl_percent'] = (current_price / trade['entry_price'] - 1) * 100
                
                # Check if we should exit the trade
                should_exit, exit_reason = self.trade_monitor.check_exit_conditions(trade)
                
                if should_exit:
                    self._exit_trade(trade_id, exit_reason)
                
            except Exception as e:
                self.logger.error(f"Error monitoring trade {trade_id}: {str(e)}")
                if self.debug_mode:
                    self.logger.error(traceback.format_exc())
    
    def _exit_trade(self, trade_id, exit_reason):
        """
        Exit an active trade.
        
        Args:
            trade_id (str): ID of the trade to exit
            exit_reason (str): Reason for exiting the trade
            
        Returns:
            bool: True if trade was exited successfully
        """
        if trade_id not in self.active_trades:
            self.logger.warning(f"Trade {trade_id} not found in active trades")
            return False
        
        trade = self.active_trades[trade_id]
        self.logger.info(f"Exiting trade {trade_id}: {exit_reason}")
        
        try:
            # Get current option price for exit
            option_quote = self.tradier_api.get_option_quote(trade['option_symbol'])
            
            if not option_quote:
                self.logger.warning(f"Could not get quote for {trade['option_symbol']}, using last known price")
                exit_price = trade.get('current_price', trade['entry_price'])
            else:
                # Use bid price for exit
                exit_price = option_quote['bid']
            
            # Update trade data
            trade['exit_price'] = exit_price
            trade['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trade['pnl'] = (exit_price - trade['entry_price']) * trade['quantity'] * 100
            trade['pnl_percent'] = (exit_price / trade['entry_price'] - 1) * 100
            trade['exit_reason'] = exit_reason
            trade['status'] = 'closed'
            
            # Calculate trade duration
            entry_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
            exit_time = datetime.strptime(trade['exit_time'], '%Y-%m-%d %H:%M:%S')
            duration = exit_time - entry_time
            trade['duration'] = str(duration)
            
            # Execute the exit
            if self.paper_trading:
                result = self.trade_monitor.exit_paper_trade(trade)
            else:
                result = self.trade_monitor.exit_live_trade(trade)
            
            if result:
                # Update statistics
                self.daily_stats['total_pnl'] += trade['pnl']
                
                if trade['pnl'] > 0:
                    self.daily_stats['winning_trades'] += 1
                else:
                    self.daily_stats['losing_trades'] += 1
                
                # Send notification
                self.notification_system.send_trade_exit_notification(trade)
                
                # Move from active to history
                self.trade_history.append(trade)
                del self.active_trades[trade_id]
                
                # Update trade log
                self._log_trade(trade)
                
                # Update parameter hub with trade result
                self._update_parameters_with_trade_result(trade)
                
                return True
            else:
                self.logger.error(f"Failed to exit trade {trade_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exiting trade {trade_id}: {str(e)}")
            if self.debug_mode:
                self.logger.error(traceback.format_exc())
            return False
    
    def _update_parameters_with_trade_result(self, trade):
        """
        Update parameter hub with trade result for reinforcement learning.
        
        Args:
            trade (dict): Completed trade data
        """
        try:
            # Extract features from the trade
            features = {
                'symbol': trade['symbol'],
                'option_type': trade['option_type'],
                'days_to_expiration': (datetime.strptime(trade['expiration'], '%Y-%m-%d') - \
                                      datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')).days,
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl': trade['pnl'],
                'pnl_percent': trade['pnl_percent'],
                'duration': (datetime.strptime(trade['exit_time'], '%Y-%m-%d %H:%M:%S') - \
                            datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600,  # In hours
                'delta': trade['greeks']['delta'],
                'gamma': trade['greeks']['gamma'],
                'theta': trade['greeks']['theta'],
                'vega': trade['greeks']['vega'],
                'entry_score': trade['entry_score']
            }
            
            # Pass to appropriate ML component for learning
            # (This will be implemented in the ML components)
            
        except Exception as e:
            self.logger.error(f"Error updating parameters with trade result: {str(e)}")
    
    def _check_end_of_day(self):
        """Check if it's end of day and perform EOD tasks if needed."""
        # Only check during last 15 minutes of trading day
        if not self.market_hours.is_market_open():
            return
        
        progress = self.market_hours.get_trading_day_progress()
        if progress is not None and progress > 95:  # Last 5% of trading day
            # Check if we've already done EOD tasks
            today = datetime.now().strftime('%Y-%m-%d')
            eod_flag_file = f"logs/eod_completed_{today}.flag"
            
            if not os.path.exists(eod_flag_file):
                self.logger.info("Performing end-of-day tasks")
                
                # Close any remaining trades
                for trade_id in list(self.active_trades.keys()):
                    self._exit_trade(trade_id, "End of day closing")
                
                # Generate daily summary
                self._generate_daily_summary()
                
                # Save trade history
                self._save_trade_history()
                
                # Create flag file to indicate EOD tasks are complete
                with open(eod_flag_file, 'w') as f:
                    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                self.logger.info("End-of-day tasks completed")
    
    def _generate_daily_summary(self):
        """Generate and send daily trading summary."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Filter today's trades
        today_trades = [t for t in self.trade_history if t['entry_time'].startswith(today)]
        
        if not today_trades and len(self.daily_stats['total_trades']) == 0:
            self.logger.info("No trades today, skipping daily summary")
            return
        
        # Compile statistics
        total_trades = self.daily_stats['total_trades']
        winning_trades = self.daily_stats['winning_trades']
        losing_trades = self.daily_stats['losing_trades']
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = self.daily_stats['total_pnl']
        
        # Calculate additional statistics
        if today_trades:
            biggest_win = max([t['pnl'] for t in today_trades] or [0])
            biggest_loss = min([t['pnl'] for t in today_trades] or [0])
            avg_win = sum([t['pnl'] for t in today_trades if t['pnl'] > 0]) / winning_trades if winning_trades > 0 else 0
            avg_loss = sum([t['pnl'] for t in today_trades if t['pnl'] <= 0]) / losing_trades if losing_trades > 0 else 0
            profit_factor = abs(sum([t['pnl'] for t in today_trades if t['pnl'] > 0]) / 
                                sum([t['pnl'] for t in today_trades if t['pnl'] < 0])) if sum([t['pnl'] for t in today_trades if t['pnl'] < 0]) != 0 else 0
        else:
            biggest_win = 0
            biggest_loss = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Calculate top performing symbols
        symbol_pnl = {}
        for trade in today_trades:
            symbol = trade['symbol']
            if symbol not in symbol_pnl:
                symbol_pnl[symbol] = 0
            symbol_pnl[symbol] += trade['pnl']
        
        # Sort symbols by P&L
        top_symbols = dict(sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Create summary data
        summary_data = {
            'date': today,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'biggest_win': biggest_win,
            'biggest_loss': biggest_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'top_symbols': top_symbols,
            'trade_history': today_trades
        }
        
        # Send daily summary notification
        self.notification_system.send_daily_summary(summary_data)
        
        # Reset daily stats
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }
        
        self.logger.info(f"Daily summary generated: {total_trades} trades, {winning_trades} wins, ${total_pnl:.2f} P&L")
    
    def _log_trade(self, trade):
        """
        Log a trade to file and Google Drive.
        
        Args:
            trade (dict): Trade data to log
        """
        try:
            # Create trade log directory if it doesn't exist
            os.makedirs("logs/trades", exist_ok=True)
            
            # Determine log file name based on date
            log_date = datetime.now().strftime('%Y-%m-%d')
            log_file = f"logs/trades/trades_{log_date}.json"
            
            # Load existing trades if file exists
            existing_trades = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing_trades = json.load(f)
            
            # Find if this trade already exists
            updated = False
            for i, existing_trade in enumerate(existing_trades):
                if existing_trade.get('trade_id') == trade.get('trade_id'):
                    existing_trades[i] = trade
                    updated = True
                    break
            
            # Add trade if it doesn't exist
            if not updated:
                existing_trades.append(trade)
            
            # Save to file
            with open(log_file, 'w') as f:
                json.dump(existing_trades, f, indent=2)
            
            # Upload to Google Drive if configured
            if self.drive_manager:
                self.drive_manager.log_trade(trade)
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {str(e)}")
    
    def _load_trade_history(self):
        """Load trade history from files."""
        try:
            # Get all trade log files
            trade_logs = [f for f in os.listdir("logs/trades") if f.startswith("trades_") and f.endswith(".json")]
            
            if not trade_logs:
                self.logger.info("No trade history files found")
                return
            
            # Load each file
            all_trades = []
            for log_file in trade_logs:
                with open(f"logs/trades/{log_file}", 'r') as f:
                    trades = json.load(f)
                    all_trades.extend(trades)
            
            # Filter for closed trades and active trades
            self.trade_history = [t for t in all_trades if t.get('status') == 'closed']
            active_trades = {t['trade_id']: t for t in all_trades if t.get('status') == 'active'}
            
            # Load active trades
            self.active_trades = active_trades
            
            self.logger.info(f"Loaded {len(self.trade_history)} historical trades and {len(self.active_trades)} active trades")
            
        except Exception as e:
            self.logger.error(f"Error loading trade history: {str(e)}")
    
    def _save_trade_history(self):
        """Save complete trade history to file."""
        try:
            # Combine active and completed trades
            all_trades = self.trade_history.copy()
            all_trades.extend(list(self.active_trades.values()))
            
            # Save to single file for backup
            os.makedirs("data", exist_ok=True)
            with open("data/all_trades_history.json", 'w') as f:
                json.dump(all_trades, f, indent=2)
            
            # Upload to Google Drive if configured
            if self.drive_manager:
                with open("data/all_trades_history.json", 'r') as f:
                    content = f.read()
                self.drive_manager.upload_file(
                    "all_trades_history.json",
                    content,
                    "trades",
                    mime_type="application/json"
                )
            
            self.logger.info(f"Saved {len(all_trades)} trades to history file")
            
        except Exception as e:
            self.logger.error(f"Error saving trade history: {str(e)}")
    
    def _shutdown(self):
        """Perform cleanup for graceful shutdown."""
        self.logger.info("Shutting down Option Hunter")
        
        # Set running flag to false to stop threads
        self.running = False
        
        # Close any open trades
        for trade_id in list(self.active_trades.keys()):
            self._exit_trade(trade_id, "System shutdown")
        
        # Save trade history
        self._save_trade_history()
        
        # Generate daily summary
        self._generate_daily_summary()
        
        # Send shutdown notification
        self.notification_system.send_system_notification(
            "Option Hunter Shutdown",
            f"System shutting down at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                self.logger.info(f"Waiting for thread {thread.name} to finish...")
                thread.join(timeout=5)
        
        self.logger.info("Shutdown complete")
    
    def pause(self):
        """Pause trading operations."""
        self.paused = True
        self.logger.info("Trading operations paused")
        self.notification_system.send_system_notification(
            "Trading Paused",
            "Option Hunter trading operations have been paused"
        )
    
    def resume(self):
        """Resume trading operations."""
        self.paused = False
        self.logger.info("Trading operations resumed")
        self.notification_system.send_system_notification(
            "Trading Resumed",
            "Option Hunter trading operations have been resumed"
        )
    
    def run_backtest(self):
        """Run backtesting mode."""
        self.logger.info("Starting backtest mode")
        
        try:
            # Import backtesting components
            from src.backtesting.backtest_engine import BacktestEngine
            
            # Create backtesting engine
            backtest_engine = BacktestEngine(
                self.config,
                self.parameter_hub,
                self.option_pricer,
                self.drive_manager
            )
            
            # Run backtest
            backtest_results = backtest_engine.run_backtest()
            
            # Analyze and report results
            backtest_engine.analyze_results(backtest_results)
            
            # Save results
            backtest_engine.save_results(backtest_results)
            
            self.logger.info("Backtest completed")
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            self.notification_system.send_error_notification(
                "Backtest Error",
                f"Error running backtest: {str(e)}",
                traceback.format_exc()
            )
"""
Integrated Bot Module

This is the main orchestration module for the Option Hunter system.
It coordinates the operation of all other components and implements
the core trading logic.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import os
import traceback

from src.tradier_api import TradierAPI
from src.option_pricing import OptionPricer, OptionType
from src.option_scanner import OptionScanner
from src.paper_trading.trade_entry import TradeEntry
from src.paper_trading.trade_monitor import TradeMonitor
from src.market_regime.market_state_tracker import MarketStateTracker
from src.sentiment_analyzer import SentimentAnalyzer
from src.position_sizing import PositionSizer
from src.order_flow_analyzer import OrderFlowAnalyzer
from src.time_sales_analyzer import TimeSalesAnalyzer
from src.volatility_regime import VolatilityRegimeDetector

class IntegratedBot:
    """
    Main bot class that orchestrates all components and implements the trading strategy.
    """
    
    def __init__(self, config, parameter_hub, drive_manager, notification_system, market_hours, paper_trading=True, debug_mode=False, focus_ticker=None):
        """
        Initialize the IntegratedBot.
        
        Args:
            config (dict): Main configuration
            parameter_hub: MasterParameterHub instance
            drive_manager: GoogleDriveManager instance
            notification_system: NotificationSystem instance
            market_hours: MarketHoursAwareness instance
            paper_trading (bool): If True, run in paper trading mode
            debug_mode (bool): If True, enable additional logging
            focus_ticker (str, optional): Focus on a specific ticker for testing
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.parameter_hub = parameter_hub
        self.drive_manager = drive_manager
        self.notification_system = notification_system
        self.market_hours = market_hours
        self.paper_trading = paper_trading
        self.debug_mode = debug_mode
        self.focus_ticker = focus_ticker
        
        # State variables
        self.running = False
        self.paused = False
        self.threads = []
        self.opportunity_queue = queue.Queue()
        self.active_trades = {}
        self.trade_history = []
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }
        
        # Initialize API
        self.tradier_api = TradierAPI(
            config["api_credentials"]["tradier"]["api_key"],
            config["api_credentials"]["tradier"]["account_id"],
            config["api_credentials"]["tradier"]["sandbox_mode"]
        )
        
        # Initialize components
        self._initialize_components()
        
        # Create directories
        os.makedirs("logs/trades", exist_ok=True)
        os.makedirs("data/collected_options_data", exist_ok=True)
        
        self.logger.info("IntegratedBot initialized")
    
    def _initialize_components(self):
        """Initialize all trading system components."""
        try:
            # Initialize option pricing
            self.logger.info("Initializing option pricer")
            self.option_pricer = OptionPricer(self.config)
            
            # Initialize option scanner
            self.logger.info("Initializing option scanner")
            self.option_scanner = OptionScanner(self.config, self.tradier_api, self.option_pricer)
            
            # Initialize trade entry and monitoring
            self.logger.info("Initializing trade entry and monitoring")
            self.trade_entry = TradeEntry(
                self.config,
                self.tradier_api,
                self.parameter_hub,
                self.notification_system
            )
            
            self.trade_monitor = TradeMonitor(
                self.config,
                self.tradier_api,
                self.notification_system,
                self.parameter_hub
            )
            
            # Initialize market state tracking
            self.logger.info("Initializing market state tracking")
            self.market_state_tracker = MarketStateTracker(self.config, self.tradier_api, self.drive_manager)
            
            # Initialize sentiment analysis
            self.logger.info("Initializing sentiment analyzer")
            self.sentiment_analyzer = SentimentAnalyzer(self.config["api_credentials"]["twitter"])
            
            # Initialize position sizing
            self.logger.info("Initializing position sizer")
            self.position_sizer = PositionSizer(self.config["trade_parameters"], self.parameter_hub)
            
            # Initialize advanced analyzers
            self.logger.info("Initializing advanced analyzers")
            self.order_flow_analyzer = OrderFlowAnalyzer(self.tradier_api)
            self.time_sales_analyzer = TimeSalesAnalyzer(self.tradier_api)
            self.volatility_detector = VolatilityRegimeDetector(self.tradier_api)
            
            # Get ticker lists
            self.all_tickers = self.config["ticker_list"]
            
            # If focus ticker is specified, only use that
            if self.focus_ticker:
                self.watchlist = [self.focus_ticker]
                self.logger.info(f"Focusing on single ticker: {self.focus_ticker}")
            else:
                # Use configured watchlist
                self.watchlist = []
                for priority in ["high_priority", "medium_priority", "low_priority"]:
                    if priority in self.config["watchlist"]:
                        self.watchlist.extend(self.config["watchlist"][priority])
                self.logger.info(f"Using watchlist with {len(self.watchlist)} tickers")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def run(self):
        """Run the main trading loop."""
        self.running = True
        
        try:
            # Send startup notification
            self.notification_system.send_system_notification(
                "Option Hunter Starting",
                f"Starting in {'paper trading' if self.paper_trading else 'live trading'} mode"
            )
            
            # Load trade history if available
            self._load_trade_history()
            
            # Start background threads
            self._start_background_threads()
            
            # Main loop
            self.logger.info("Starting main loop")
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, shutting down")
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            self.notification_system.send_error_notification(
                "Critical Error",
                f"Main loop error: {str(e)}",
                traceback.format_exc()
            )
        finally:
            # Clean shutdown
            self._shutdown()
    
    def _main_loop(self):
        """Main trading loop."""
        while self.running:
            try:
                # Check if market is open
                is_market_open = self.market_hours.is_market_open()
                
                if not is_market_open and not self.debug_mode:
                    # Market is closed, sleep and continue
                    next_event = self.market_hours.get_next_market_event()
                    seconds_to_next = (next_event["time"] - datetime.now(self.market_hours.timezone)).total_seconds()
                    
                    # Show message every hour when market is closed
                    if seconds_to_next % 3600 < 60:
                        self.logger.info(f"Market is closed. Next event: {next_event['description']} in {seconds_to_next//60:.0f} minutes")
                    
                    # Sleep for 60 seconds between checks
                    time.sleep(min(60, max(1, seconds_to_next)))
                    continue
                
                # Market is open or we're in debug mode
                
                # Process any new opportunities in the queue
                while not self.opportunity_queue.empty() and not self.paused:
                    opportunity = self.opportunity_queue.get()
                    self._evaluate_opportunity(opportunity)
                    self.opportunity_queue.task_done()
                
                # Check active trades
                self._monitor_active_trades()
                
                # Determine if we should scan for new opportunities
                should_scan = (len(self.active_trades) < self.config["trade_parameters"]["max_open_positions"]) and not self.paused
                
                if should_scan:
                    self._scan_for_opportunities()
                
                # Check for end of day
                self._check_
