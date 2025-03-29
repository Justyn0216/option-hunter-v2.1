"""
Order Flow Analyzer Module

This module analyzes option order flow to detect unusual activity,
large orders, and other significant market signals.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

class OrderFlowAnalyzer:
    """
    Analyzes option order flow to detect significant market activity.
    
    Features:
    - Detection of unusual options activity
    - Order size analysis
    - Unusual volume detection
    - Price impact analysis
    - Large block trade detection
    """
    
    def __init__(self, tradier_api):
        """
        Initialize the OrderFlowAnalyzer.
        
        Args:
            tradier_api: TradierAPI instance for market data
        """
        self.logger = logging.getLogger(__name__)
        self.tradier_api = tradier_api
        
        # Thresholds for analysis
        self.unusual_volume_threshold = 3.0  # Volume > 3x average
        self.large_order_threshold = 100     # Contracts per order
        self.significant_oi_change = 0.25    # 25% change in open interest
        
        # Cache to store historical data
        self.volume_cache = {}  # {symbol: {date: volume}}
        
        self.logger.info("OrderFlowAnalyzer initialized")
    
    def analyze_symbol(self, symbol, option_symbol=None):
        """
        Analyze order flow for a specific symbol or option.
        
        Args:
            symbol (str): Stock symbol
            option_symbol (str, optional): Specific option symbol to analyze
            
        Returns:
            dict: Order flow analysis results
        """
        self.logger.debug(f"Analyzing order flow for {symbol}")
        
        try:
            # Initialize results
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': 'neutral',
                'confidence': 0.0,
                'unusual_activity': False,
                'notable_strikes': [],
                'call_put_ratio': 1.0,
                'large_orders': []
            }
            
            # If specific option symbol provided, analyze just that option
            if option_symbol:
                option_data = self._analyze_option(option_symbol)
                results.update(option_data)
                return results
            
            # Get option chain for the symbol
            expirations = self.tradier_api.get_option_expirations(symbol)
            
            if not expirations:
                self.logger.warning(f"No option expirations found for {symbol}")
                return results
            
            # Focus on nearest expiration
            expiration = expirations[0]
            
            # Get option chain
            chain = self.tradier_api.get_option_chains(symbol, expiration)
            
            if chain.empty:
                self.logger.warning(f"No option chain data found for {symbol} {expiration}")
                return results
            
            # Calculate call/put volume ratio
            call_volume = chain[chain['option_type'] == 'call']['volume'].sum()
            put_volume = chain[chain['option_type'] == 'put']['volume'].sum()
            
            if put_volume > 0:
                call_put_ratio = call_volume / put_volume
            else:
                call_put_ratio = float('inf') if call_volume > 0 else 1.0
            
            results['call_put_ratio'] = call_put_ratio
            
            # Determine overall sentiment based on call/put ratio
            if call_put_ratio > 2.0:
                results['signal'] = 'bullish'
                results['confidence'] = min(0.8, (call_put_ratio - 2.0) / 8.0 + 0.5)
            elif call_put_ratio < 0.5:
                results['signal'] = 'bearish'
                results['confidence'] = min(0.8, (0.5 - call_put_ratio) / 0.5 + 0.5)
            
            # Find strikes with unusual activity
            unusual_activity = False
            notable_strikes = []
            
            for _, option in chain.iterrows():
                # Check for unusual volume
                strike = option['strike']
                option_type = option['option_type']
                volume = option['volume']
                open_interest = option.get('open_interest', 0)
                
                # Skip options with very low volume
                if volume < 10:
                    continue
                
                # Calculate volume vs open interest ratio
                vol_oi_ratio = volume / open_interest if open_interest > 0 else float('inf')
                
                # Check for unusual activity
                if vol_oi_ratio > 0.5 and volume >= self.large_order_threshold:
                    unusual_activity = True
                    
                    # Determine if buying or selling pressure
                    trade_type = 'buy' if option['ask'] > option.get('ask_prev_close', 0) else 'sell'
                    
                    notable_strikes.append({
                        'strike': strike,
                        'option_type': option_type,
                        'volume': volume,
                        'open_interest': open_interest,
                        'vol_oi_ratio': vol_oi_ratio,
                        'trade_type': trade_type
                    })
            
            # Update results
            results['unusual_activity'] = unusual_activity
            results['notable_strikes'] = notable_strikes
            
            # Adjust signal based on unusual activity
            if unusual_activity:
                # Count bullish vs bearish signals in notable strikes
                bullish_signals = sum(1 for s in notable_strikes if 
                                    (s['option_type'] == 'call' and s['trade_type'] == 'buy') or
                                    (s['option_type'] == 'put' and s['trade_type'] == 'sell'))
                
                bearish_signals = sum(1 for s in notable_strikes if 
                                    (s['option_type'] == 'call' and s['trade_type'] == 'sell') or
                                    (s['option_type'] == 'put' and s['trade_type'] == 'buy'))
                
                # Determine signal based on predominant activity
                if bullish_signals > bearish_signals:
                    results['signal'] = 'bullish'
                    results['confidence'] = min(0.9, 0.5 + (bullish_signals - bearish_signals) / len(notable_strikes))
                elif bearish_signals > bullish_signals:
                    results['signal'] = 'bearish'
                    results['confidence'] = min(0.9, 0.5 + (bearish_signals - bullish_signals) / len(notable_strikes))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _analyze_option(self, option_symbol):
        """
        Analyze order flow for a specific option.
        
        Args:
            option_symbol (str): Option symbol
            
        Returns:
            dict: Order flow analysis results
        """
        try:
            # Get option quote
            option_data = self.tradier_api.get_option_quote(option_symbol)
            
            if not option_data:
                self.logger.warning(f"No data found for option {option_symbol}")
                return {}
            
            volume = option_data.get('volume', 0)
            open_interest = option_data.get('open_interest', 0)
            
            # Calculate volume vs open interest ratio
            vol_oi_ratio = volume / open_interest if open_interest > 0 else float('inf')
            
            # Determine if unusual activity
            unusual_activity = vol_oi_ratio > 0.3 and volume >= self.large_order_threshold
            
            # Determine trade direction (approximate based on price change)
            if 'change' in option_data:
                trade_type = 'buy' if option_data['change'] > 0 else 'sell'
            else:
                trade_type = 'unknown'
            
            return {
                'option_symbol': option_symbol,
                'volume': volume,
                'open_interest': open_interest,
                'vol_oi_ratio': vol_oi_ratio,
                'unusual_activity': unusual_activity,
                'trade_type': trade_type
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing option {option_symbol}: {str(e)}")
            return {}
    
    def detect_large_orders(self, symbol, lookback_minutes=30):
        """
        Detect large orders in recent time & sales data.
        
        Args:
            symbol (str): Stock symbol
            lookback_minutes (int): Minutes to look back
            
        Returns:
            list: List of large orders detected
        """
        try:
            # Get time window
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            # Format times for API
            start_str = start_time.strftime('%Y-%m-%d %H:%M')
            end_str = end_time.strftime('%Y-%m-%d %H:%M')
            
            # Get time & sales data
            time_sales = self.tradier_api.get_time_and_sales(
                symbol, start=start_str, end=end_str, interval='1min'
            )
            
            if time_sales.empty:
                return []
            
            # Aggregate by minute to find large volume spikes
            large_orders = []
            
            # Group by minute and sum volume
            if 'time' in time_sales.columns and 'volume' in time_sales.columns:
                time_sales['minute'] = time_sales['time'].dt.floor('min')
                volume_by_minute = time_sales.groupby('minute')['volume'].sum().reset_index()
                
                # Calculate average volume per minute
                avg_volume = volume_by_minute['volume'].mean()
                
                # Find minutes with unusually high volume
                for _, row in volume_by_minute.iterrows():
                    if row['volume'] > avg_volume * self.unusual_volume_threshold:
                        # Get the trades during this minute
                        minute_trades = time_sales[time_sales['minute'] == row['minute']]
                        
                        # Determine if buying or selling pressure
                        price_change = minute_trades['price'].iloc[-1] - minute_trades['price'].iloc[0]
                        trade_type = 'buy' if price_change > 0 else 'sell'
                        
                        large_orders.append({
                            'time': row['minute'],
                            'volume': row['volume'],
                            'avg_volume_ratio': row['volume'] / avg_volume,
                            'trade_type': trade_type
                        })
            
            return large_orders
            
        except Exception as e:
            self.logger.error(f"Error detecting large orders for {symbol}: {str(e)}")
            return []
    
    def get_historical_volume_trend(self, symbol, days=10):
        """
        Get the trend in option volume over time.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to analyze
            
        Returns:
            dict: Volume trend data
        """
        try:
            # Get option expirations
            expirations = self.tradier_api.get_option_expirations(symbol)
            
            if not expirations:
                return {'trend': 'neutral', 'data': []}
            
            # Focus on nearest expiration
            expiration = expirations[0]
            
            # Check if we have this data cached
            if symbol in self.volume_cache:
                cached_data = self.volume_cache[symbol]
            else:
                # Initialize new cache entry
                self.volume_cache[symbol] = {}
                cached_data = self.volume_cache[symbol]
            
            # Get current date
            today = datetime.now().date()
            
            # Get option chain for the current day
            chain_today = self.tradier_api.get_option_chains(symbol, expiration)
            
            if chain_today.empty:
                return {'trend': 'neutral', 'data': []}
            
            # Calculate current day volumes
            call_volume_today = chain_today[chain_today['option_type'] == 'call']['volume'].sum()
            put_volume_today = chain_today[chain_today['option_type'] == 'put']['volume'].sum()
            total_volume_today = call_volume_today + put_volume_today
            
            # Store in cache
            cached_data[today.strftime('%Y-%m-%d')] = {
                'call_volume': call_volume_today,
                'put_volume': put_volume_today,
                'total_volume': total_volume_today,
                'call_put_ratio': call_volume_today / put_volume_today if put_volume_today > 0 else float('inf')
            }
            
            # Build the historical trend
            dates = []
            call_volumes = []
            put_volumes = []
            call_put_ratios = []
            
            # Sort dates in ascending order
            for date_str, data in sorted(cached_data.items()):
                dates.append(date_str)
                call_volumes.append(data['call_volume'])
                put_volumes.append(data['put_volume'])
                call_put_ratios.append(data['call_put_ratio'])
            
            # Determine the trend
            if len(call_put_ratios) >= 3:
                # Calculate trend in call/put ratio
                if call_put_ratios[-1] > call_put_ratios[-2] > call_put_ratios[-3]:
                    trend = 'increasingly_bullish'
                elif call_put_ratios[-1] < call_put_ratios[-2] < call_put_ratios[-3]:
                    trend = 'increasingly_bearish'
                elif call_put_ratios[-1] > 1.5:
                    trend = 'bullish'
                elif call_put_ratios[-1] < 0.6:
                    trend = 'bearish'
                else:
                    trend = 'neutral'
            else:
                # Not enough data for trend
                trend = 'neutral'
            
            return {
                'trend': trend,
                'data': {
                    'dates': dates,
                    'call_volumes': call_volumes,
                    'put_volumes': put_volumes,
                    'call_put_ratios': call_put_ratios
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume trend for {symbol}: {str(e)}")
            return {'trend': 'neutral', 'data': []}
    
    def analyze_open_interest_change(self, symbol, option_symbol=None):
        """
        Analyze changes in open interest to detect significant activity.
        
        Args:
            symbol (str): Stock symbol
            option_symbol (str, optional): Specific option symbol to analyze
            
        Returns:
            dict: Open interest analysis results
        """
        try:
            # Initialize results
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'significant_changes': []
            }
            
            # Get option chain for the symbol
            expirations = self.tradier_api.get_option_expirations(symbol)
            
            if not expirations:
                return results
            
            # Focus on nearest expiration
            expiration = expirations[0]
            
            # Get option chain
            chain = self.tradier_api.get_option_chains(symbol, expiration)
            
            if chain.empty:
                return results
            
            # Look at current open interest vs. previous day (if available)
            significant_changes = []
            
            for _, option in chain.iterrows():
                current_oi = option.get('open_interest', 0)
                if current_oi < 100:  # Ignore low OI options
                    continue
                
                previous_oi = option.get('open_interest_prev_day', 0)
                
                # If we don't have previous day data, can't calculate change
                if previous_oi == 0:
                    continue
                
                # Calculate percent change
                percent_change = (current_oi - previous_oi) / previous_oi
                
                # Check if change is significant
                if abs(percent_change) >= self.significant_oi_change:
                    change_type = 'increase' if percent_change > 0 else 'decrease'
                    
                    # Determine if bullish or bearish signal
                    signal = 'neutral'
                    if option['option_type'] == 'call' and change_type == 'increase':
                        signal = 'bullish'
                    elif option['option_type'] == 'put' and change_type == 'increase':
                        signal = 'bearish'
                    elif option['option_type'] == 'call' and change_type == 'decrease':
                        signal = 'bearish'
                    elif option['option_type'] == 'put' and change_type == 'decrease':
                        signal = 'bullish'
                    
                    significant_changes.append({
                        'option_symbol': option['symbol'],
                        'option_type': option['option_type'],
                        'strike': option['strike'],
                        'current_oi': current_oi,
                        'previous_oi': previous_oi,
                        'percent_change': percent_change * 100,
                        'change_type': change_type,
                        'signal': signal
                    })
            
            # Sort by absolute percent change (largest first)
            significant_changes.sort(key=lambda x: abs(x['percent_change']), reverse=True)
            
            # Add to results
            results['significant_changes'] = significant_changes
            
            # Determine overall signal based on significant changes
            if significant_changes:
                bullish_signals = sum(1 for s in significant_changes if s['signal'] == 'bullish')
                bearish_signals = sum(1 for s in significant_changes if s['signal'] == 'bearish')
                
                if bullish_signals > bearish_signals:
                    results['signal'] = 'bullish'
                    results['confidence'] = min(0.9, 0.5 + (bullish_signals - bearish_signals) / len(significant_changes))
                elif bearish_signals > bullish_signals:
                    results['signal'] = 'bearish'
                    results['confidence'] = min(0.9, 0.5 + (bearish_signals - bullish_signals) / len(significant_changes))
                else:
                    results['signal'] = 'neutral'
                    results['confidence'] = 0.5
            else:
                results['signal'] = 'neutral'
                results['confidence'] = 0.0
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing open interest change for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
