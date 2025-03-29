"""
Time Sales Analyzer Module

This module analyzes time and sales data to detect patterns and signals
that can inform trading decisions.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

class TimeSalesAnalyzer:
    """
    Analyzes time and sales data (tick by tick trades) to detect patterns
    and provide trading signals.
    
    Features:
    - Trade volume and price analysis
    - Buy/sell pressure detection
    - Momentum detection
    - Block trade identification
    - Pattern recognition
    """
    
    def __init__(self, tradier_api):
        """
        Initialize the TimeSalesAnalyzer.
        
        Args:
            tradier_api: TradierAPI instance for market data
        """
        self.logger = logging.getLogger(__name__)
        self.tradier_api = tradier_api
        
        # Detection thresholds
        self.volume_spike_threshold = 2.0  # Volume > 2x average
        self.large_tick_threshold = 1000   # Large trade (shares)
        self.momentum_lookback = 10        # Number of ticks for momentum calculation
        
        # Data caching
        self.recent_data_cache = {}  # {symbol: DataFrame} of recent time & sales
        
        self.logger.info("TimeSalesAnalyzer initialized")
    
    def analyze_symbol(self, symbol, lookback_minutes=30):
        """
        Analyze recent time and sales data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            lookback_minutes (int): Minutes to look back
            
        Returns:
            dict: Analysis results
        """
        self.logger.debug(f"Analyzing time & sales for {symbol}")
        
        try:
            # Get time & sales data
            data = self._get_time_sales_data(symbol, lookback_minutes)
            
            if data.empty:
                self.logger.warning(f"No time & sales data available for {symbol}")
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'signal': 'neutral',
                    'confidence': 0.0
                }
            
            # Store in cache
            self.recent_data_cache[symbol] = data
            
            # Analyze data
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(data)
            }
            
            # Detect buy/sell pressure
            pressure_results = self._detect_buy_sell_pressure(data)
            results.update(pressure_results)
            
            # Detect momentum
            momentum_results = self._detect_momentum(data)
            results.update(momentum_results)
            
            # Detect volume spikes
            volume_results = self._detect_volume_patterns(data)
            results.update(volume_results)
            
            # Overall signal based on combined factors
            signal_results = self._generate_overall_signal(
                pressure_results, momentum_results, volume_results
            )
            results.update(signal_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing time & sales for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'signal': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _get_time_sales_data(self, symbol, lookback_minutes):
        """
        Get time & sales data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            lookback_minutes (int): Minutes to look back
            
        Returns:
            DataFrame: Time & sales data
        """
        try:
            # Calculate time window
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            # Format for API
            start_str = start_time.strftime('%Y-%m-%d %H:%M')
            end_str = end_time.strftime('%Y-%m-%d %H:%M')
            
            # Get data from API
            data = self.tradier_api.get_time_and_sales(
                symbol, start=start_str, end=end_str, interval='1min'
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # Sort by time
            if 'time' in data.columns:
                data = data.sort_values('time')
            
            # Add derived columns if needed
            if 'price' in data.columns and 'volume' in data.columns:
                # Add price change column
                data['price_change'] = data['price'].diff()
                
                # Infer trade direction (up = buy, down = sell, flat = neutral)
                data['direction'] = np.where(
                    data['price_change'] > 0, 'buy',
                    np.where(data['price_change'] < 0, 'sell', 'neutral')
                )
                
                # Calculate dollar volume
                data['dollar_volume'] = data['price'] * data['volume']
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting time & sales data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _detect_buy_sell_pressure(self, data):
        """
        Detect buying and selling pressure in time & sales data.
        
        Args:
            data (DataFrame): Time & sales data
            
        Returns:
            dict: Buy/sell pressure analysis
        """
        try:
            if 'direction' not in data.columns or 'volume' not in data.columns:
                return {'pressure': 'neutral', 'pressure_ratio': 1.0}
            
            # Calculate buy and sell volume
            buy_volume = data[data['direction'] == 'buy']['volume'].sum()
            sell_volume = data[data['direction'] == 'sell']['volume'].sum()
            neutral_volume = data[data['direction'] == 'neutral']['volume'].sum()
            
            total_volume = buy_volume + sell_volume + neutral_volume
            
            if total_volume == 0:
                return {'pressure': 'neutral', 'pressure_ratio': 1.0}
            
            # Calculate pressure ratio (buy volume / sell volume)
            pressure_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            
            # Determine pressure
            if pressure_ratio > 1.5:
                pressure = 'buying'
                confidence = min(0.9, 0.5 + (pressure_ratio - 1.5) / 5.0)
            elif pressure_ratio < 0.67:  # 1/1.5
                pressure = 'selling'
                confidence = min(0.9, 0.5 + (0.67 - pressure_ratio) / 0.67)
            else:
                pressure = 'neutral'
                confidence = 0.5
            
            # Also look at recent trades (last 25% of data)
            recent_cutoff = int(len(data) * 0.75)
            recent_data = data.iloc[recent_cutoff:]
            
            recent_buy_volume = recent_data[recent_data['direction'] == 'buy']['volume'].sum()
            recent_sell_volume = recent_data[recent_data['direction'] == 'sell']['volume'].sum()
            
            if recent_buy_volume + recent_sell_volume > 0:
                recent_pressure_ratio = recent_buy_volume / recent_sell_volume if recent_sell_volume > 0 else float('inf')
                
                # If recent pressure is significantly different, adjust
                if (pressure == 'buying' and recent_pressure_ratio < 1.0) or \
                   (pressure == 'selling' and recent_pressure_ratio > 1.0):
                    # Recent pressure contradicts overall pressure
                    confidence *= 0.7  # Reduce confidence
                elif (pressure == 'buying' and recent_pressure_ratio > pressure_ratio) or \
                     (pressure == 'selling' and recent_pressure_ratio < pressure_ratio):
                    # Recent pressure strengthens overall pressure
                    confidence = min(0.95, confidence * 1.2)
            
            return {
                'pressure': pressure,
                'pressure_ratio': pressure_ratio,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'pressure_confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting buy/sell pressure: {str(e)}")
            return {'pressure': 'neutral', 'pressure_ratio': 1.0}
    
    def _detect_momentum(self, data):
        """
        Detect price momentum in time & sales data.
        
        Args:
            data (DataFrame): Time & sales data
            
        Returns:
            dict: Momentum analysis
        """
        try:
            if 'price' not in data.columns or len(data) < 5:
                return {'momentum': 'neutral', 'momentum_strength': 0.0}
            
            # Calculate short-term momentum (last 10 ticks or fewer if not enough data)
            lookback = min(self.momentum_lookback, len(data) - 1)
            
            if lookback <= 0:
                return {'momentum': 'neutral', 'momentum_strength': 0.0}
            
            recent_data = data.iloc[-lookback-1:]
            start_price = recent_data['price'].iloc[0]
            end_price = recent_data['price'].iloc[-1]
            
            price_change = end_price - start_price
            percent_change = (price_change / start_price) * 100
            
            # Normalize momentum strength to a 0-1 scale
            # 2% move in lookback period is considered very strong
            norm_strength = min(1.0, abs(percent_change) / 2.0)
            
            # Determine momentum direction
            if percent_change > 0.1:  # Positive momentum
                momentum = 'up'
                momentum_strength = norm_strength
            elif percent_change < -0.1:  # Negative momentum
                momentum = 'down'
                momentum_strength = norm_strength
            else:
                momentum = 'neutral'
                momentum_strength = 0.0
            
            # Check if momentum is accelerating or decelerating
            if len(data) >= 3:
                # Calculate price changes in first and second half of the sample
                mid_point = len(recent_data) // 2
                first_half = recent_data.iloc[:mid_point]
                second_half = recent_data.iloc[mid_point:]
                
                if len(first_half) > 0 and len(second_half) > 0:
                    first_change = first_half['price'].iloc[-1] - first_half['price'].iloc[0]
                    second_change = second_half['price'].iloc[-1] - second_half['price'].iloc[0]
                    
                    if abs(second_change) > abs(first_change):
                        momentum_acceleration = 'accelerating'
                    else:
                        momentum_acceleration = 'decelerating'
                else:
                    momentum_acceleration = 'unknown'
            else:
                momentum_acceleration = 'unknown'
            
            return {
                'momentum': momentum,
                'momentum_strength': momentum_strength,
                'percent_change': percent_change,
                'momentum_acceleration': momentum_acceleration
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum: {str(e)}")
            return {'momentum': 'neutral', 'momentum_strength': 0.0}
    
    def _detect_volume_patterns(self, data):
        """
        Detect volume patterns in time & sales data.
        
        Args:
            data (DataFrame): Time & sales data
            
        Returns:
            dict: Volume pattern analysis
        """
        try:
            if 'volume' not in data.columns or len(data) < 5:
                return {'volume_pattern': 'normal', 'volume_spikes': []}
            
            # Calculate average volume
            avg_volume = data['volume'].mean()
            
            # Detect volume spikes
            volume_spikes = []
            
            for i, row in data.iterrows():
                if row['volume'] > avg_volume * self.volume_spike_threshold:
                    # This is a volume spike
                    spike = {
                        'time': row['time'] if 'time' in row else i,
                        'volume': row['volume'],
                        'price': row['price'] if 'price' in row else None,
                        'ratio': row['volume'] / avg_volume
                    }
                    
                    # Determine if it's a buying or selling spike
                    if 'direction' in row:
                        spike['direction'] = row['direction']
                    
                    volume_spikes.append(spike)
            
            # Determine overall volume pattern
            if not volume_spikes:
                volume_pattern = 'normal'
            else:
                # Look at the predominant direction of spikes
                buy_spikes = sum(1 for s in volume_spikes if s.get('direction') == 'buy')
                sell_spikes = sum(1 for s in volume_spikes if s.get('direction') == 'sell')
                
                if buy_spikes > sell_spikes:
                    volume_pattern = 'buying_spikes'
                elif sell_spikes > buy_spikes:
                    volume_pattern = 'selling_spikes'
                else:
                    volume_pattern = 'mixed_spikes'
            
            # Also check for volume trend
            if len(data) >= 10:
                first_half = data.iloc[:len(data)//2]
                second_half = data.iloc[len(data)//2:]
                
                first_half_avg = first_half['volume'].mean()
                second_half_avg = second_half['volume'].mean()
                
                if second_half_avg > first_half_avg * 1.25:
                    volume_trend = 'increasing'
                elif second_half_avg < first_half_avg * 0.75:
                    volume_trend = 'decreasing'
                else:
                    volume_trend = 'stable'
            else:
                volume_trend = 'insufficient_data'
            
            return {
                'volume_pattern': volume_pattern,
                'volume_spikes': volume_spikes,
                'volume_trend': volume_trend,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting volume patterns: {str(e)}")
            return {'volume_pattern': 'normal', 'volume_spikes': []}
    
    def _generate_overall_signal(self, pressure_results, momentum_results, volume_results):
        """
        Generate an overall trading signal based on all factors.
        
        Args:
            pressure_results (dict): Buy/sell pressure analysis
            momentum_results (dict): Momentum analysis
            volume_results (dict): Volume pattern analysis
            
        Returns:
            dict: Overall signal
        """
        try:
            # Start with neutral
            signal = 'neutral'
            confidence = 0.5
            
            # Combine pressure and momentum
            pressure = pressure_results.get('pressure', 'neutral')
            momentum = momentum_results.get('momentum', 'neutral')
            
            # Bullish signals
            if (pressure == 'buying' and momentum == 'up'):
                signal = 'bullish'
                confidence = 0.7
            elif (pressure == 'buying' and momentum == 'neutral') or \
                 (pressure == 'neutral' and momentum == 'up'):
                signal = 'slightly_bullish'
                confidence = 0.6
            
            # Bearish signals
            elif (pressure == 'selling' and momentum == 'down'):
                signal = 'bearish'
                confidence = 0.7
            elif (pressure == 'selling' and momentum == 'neutral') or \
                 (pressure == 'neutral' and momentum == 'down'):
                signal = 'slightly_bearish'
                confidence = 0.6
                
            # Mixed signals
            elif (pressure == 'buying' and momentum == 'down') or \
                 (pressure == 'selling' and momentum == 'up'):
                # Conflicting signals
                signal = 'mixed'
                confidence = 0.5
            
            # Adjust based on volume pattern
            volume_pattern = volume_results.get('volume_pattern', 'normal')
            
            if volume_pattern == 'buying_spikes' and signal in ['bullish', 'slightly_bullish']:
                confidence = min(0.9, confidence + 0.1)
            elif volume_pattern == 'selling_spikes' and signal in ['bearish', 'slightly_bearish']:
                confidence = min(0.9, confidence + 0.1)
            elif volume_pattern == 'buying_spikes' and signal in ['bearish', 'slightly_bearish']:
                confidence = max(0.3, confidence - 0.1)
            elif volume_pattern == 'selling_spikes' and signal in ['bullish', 'slightly_bullish']:
                confidence = max(0.3, confidence - 0.1)
            
            # Adjust based on momentum strength
            momentum_strength = momentum_results.get('momentum_strength', 0.0)
            if momentum_strength > 0.5:
                # Strong momentum, increase confidence
                confidence = min(0.95, confidence + 0.1)
            
            # Map signal to simplified version for trading strategy
            simplified_signal = 'neutral'
            if signal in ['bullish', 'slightly_bullish']:
                simplified_signal = 'bullish'
            elif signal in ['bearish', 'slightly_bearish']:
                simplified_signal = 'bearish'
            
            return {
                'signal': simplified_signal,
                'detailed_signal': signal,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error generating overall signal: {str(e)}")
            return {'signal': 'neutral', 'confidence': 0.5}
    
    def analyze_large_trades(self, symbol, min_dollar_volume=50000):
        """
        Analyze large trades to detect potential institutional activity.
        
        Args:
            symbol (str): Stock symbol
            min_dollar_volume (float): Minimum dollar volume to be considered large
            
        Returns:
            dict: Large trade analysis
        """
        try:
            # Check if we have recent data cached
            if symbol in self.recent_data_cache:
                data = self.recent_data_cache[symbol]
            else:
                # Get new data
                data = self._get_time_sales_data(symbol, 60)  # 1 hour lookback
            
            if data.empty or 'dollar_volume' not in data.columns:
                return {'large_trades': []}
            
            # Filter for large trades
            large_trades = data[data['dollar_volume'] >= min_dollar_volume].copy()
            
            if large_trades.empty:
                return {'large_trades': []}
            
            # Format results
            large_trades_list = []
            
            for _, trade in large_trades.iterrows():
                formatted_trade = {
                    'time': trade['time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(trade['time'], (datetime, pd.Timestamp)) else trade['time'],
                    'price': trade['price'],
                    'volume': trade['volume'],
                    'dollar_volume': trade['dollar_volume'],
                    'direction': trade.get('direction', 'unknown')
                }
                large_trades_list.append(formatted_trade)
            
            # Calculate aggregate stats
            buy_volume = large_trades[large_trades['direction'] == 'buy']['volume'].sum()
            sell_volume = large_trades[large_trades['direction'] == 'sell']['volume'].sum()
            
            pressure = 'neutral'
            if buy_volume > sell_volume * 1.5:
                pressure = 'buying'
            elif sell_volume > buy_volume * 1.5:
                pressure = 'selling'
            
            return {
                'large_trades': large_trades_list,
                'count': len(large_trades_list),
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'pressure': pressure
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing large trades for {symbol}: {str(e)}")
            return {'large_trades': []}
    
    def detect_block_trades(self, symbol, lookback_hours=4):
        """
        Detect potential block trades that could indicate institutional activity.
        
        Args:
            symbol (str): Stock symbol
            lookback_hours (int): Hours to look back
            
        Returns:
            list: Detected block trades
        """
        try:
            # Get time & sales data with longer lookback
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)
            
            # Format for API
            start_str = start_time.strftime('%Y-%m-%d %H:%M')
            end_str = end_time.strftime('%Y-%m-%d %H:%M')
            
            # Get data
            data = self.tradier_api.get_time_and_sales(
                symbol, start=start_str, end=end_str, interval='15min'
            )
            
            if data.empty or 'volume' not in data.columns:
                return []
            
            # Get quote for current price and average volume
            quote = self.tradier_api.get_quotes(symbol)
            
            if quote.empty:
                avg_daily_volume = data['volume'].sum() * (6.5 / lookback_hours)  # Estimate
                current_price = data['price'].iloc[-1] if 'price' in data.columns else 0
            else:
                avg_daily_volume = quote.iloc[0].get('average_volume', 0)
                current_price = quote.iloc[0].get('last', 0)
            
            # Minimum size for block trade (0.5% of avg daily volume)
            min_block_size = max(10000, int(avg_daily_volume * 0.005))
            
            # Detect block trades (large individual trades)
            block_trades = []
            
            for _, trade in data.iterrows():
                if trade['volume'] >= min_block_size:
                    # Calculate percentage of avg daily volume
                    pct_of_daily = (trade['volume'] / avg_daily_volume) * 100
                    
                    # Calculate dollar value
                    dollar_value = trade['price'] * trade['volume']
                    
                    block_trades.append({
                        'time': trade['time'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(trade['time'], (datetime, pd.Timestamp)) else trade['time'],
                        'price': trade['price'],
                        'volume': trade['volume'],
                        'dollar_value': dollar_value,
                        'percent_of_daily': pct_of_daily,
                        'direction': trade.get('direction', 'unknown')
                    })
            
            # Sort by volume (largest first)
            block_trades.sort(key=lambda x: x['volume'], reverse=True)
            
            return block_trades
            
        except Exception as e:
            self.logger.error(f"Error detecting block trades for {symbol}: {str(e)}")
            return []
