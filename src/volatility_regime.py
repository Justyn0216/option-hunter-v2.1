"""
Volatility Regime Detector Module

This module detects the current volatility regime in the market to
inform trading decisions and risk management.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

class VolatilityRegimeDetector:
    """
    Detects and classifies market volatility regimes.
    
    Features:
    - Historical volatility calculation
    - Volatility regime classification
    - Trend detection in volatility
    - VIX-based market state analysis
    """
    
    # Volatility regime constants
    LOW_VOLATILITY = 'low'
    NORMAL_VOLATILITY = 'normal'
    HIGH_VOLATILITY = 'high'
    EXTREME_VOLATILITY = 'extreme'
    
    def __init__(self, tradier_api):
        """
        Initialize the VolatilityRegimeDetector.
        
        Args:
            tradier_api: TradierAPI instance for market data
        """
        self.logger = logging.getLogger(__name__)
        self.tradier_api = tradier_api
        
        # Thresholds for volatility regimes (annualized)
        self.low_volatility_threshold = 0.10      # 10%
        self.high_volatility_threshold = 0.20     # 20%
        self.extreme_volatility_threshold = 0.30  # 30%
        
        # VIX thresholds
        self.low_vix_threshold = 15.0
        self.high_vix_threshold = 25.0
        self.extreme_vix_threshold = 35.0
        
        # Cache for regime data
        self.current_regime = self.NORMAL_VOLATILITY
        self.volatility_history = {}  # {date: {symbol: volatility}}
        self.vix_history = {}  # {date: vix_value}
        
        # By default, use S&P 500 as the market proxy
        self.market_proxy = 'SPY'
        
        self.logger.info("VolatilityRegimeDetector initialized")
    
    def detect_regime(self, symbol=None, lookback_days=30):
        """
        Detect the current volatility regime for a symbol or the overall market.
        
        Args:
            symbol (str, optional): Stock symbol (uses market proxy if None)
            lookback_days (int): Number of days to look back for volatility calculation
            
        Returns:
            str: Volatility regime (low, normal, high, extreme)
        """
        self.logger.debug(f"Detecting volatility regime for {symbol or self.market_proxy}")
        
        try:
            # Use market proxy if symbol is None
            if symbol is None or symbol.upper() == 'SPX' or symbol.upper() == 'SPY':
                # Try to get VIX data first (best indicator of market volatility)
                vix_regime = self._detect_vix_regime()
                if vix_regime:
                    self.current_regime = vix_regime
                    return vix_regime
                
                # Fall back to historical volatility of SPY
                symbol = self.market_proxy
            
            # Calculate historical volatility
            volatility = self._calculate_historical_volatility(symbol, lookback_days)
            
            if volatility is None:
                self.logger.warning(f"Could not calculate volatility for {symbol}")
                return self.current_regime  # Return last known regime
            
            # Classify regime based on volatility thresholds
            if volatility <= self.low_volatility_threshold:
                regime = self.LOW_VOLATILITY
            elif volatility <= self.high_volatility_threshold:
                regime = self.NORMAL_VOLATILITY
            elif volatility <= self.extreme_volatility_threshold:
                regime = self.HIGH_VOLATILITY
            else:
                regime = self.EXTREME_VOLATILITY
            
            # Update current regime
            self.current_regime = regime
            
            self.logger.info(f"Detected {regime} volatility regime for {symbol}: {volatility:.2f}")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility regime: {str(e)}")
            return self.current_regime  # Return last known regime
    
    def _calculate_historical_volatility(self, symbol, lookback_days):
        """
        Calculate historical volatility for a symbol.
        
        Args:
            symbol (str): Stock symbol
            lookback_days (int): Number of days to look back
            
        Returns:
            float: Annualized historical volatility
        """
        try:
            # Get end date (today) and start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days * 2)  # Extra days for calculation
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get historical data
            historical_data = self.tradier_api.get_historical_data(
                symbol, interval='daily', start_date=start_str, end_date=end_str
            )
            
            if historical_data.empty or len(historical_data) < 5:
                self.logger.warning(f"Insufficient historical data for {symbol}")
                return None
            
            # Calculate log returns
            historical_data = historical_data.sort_values('date')
            historical_data['log_return'] = np.log(historical_data['close'] / historical_data['close'].shift(1))
            
            # Drop NA values
            historical_data = historical_data.dropna()
            
            # Limit to lookback period
            if len(historical_data) > lookback_days:
                historical_data = historical_data.iloc[-lookback_days:]
            
            # Calculate volatility (standard deviation of log returns)
            daily_volatility = historical_data['log_return'].std()
            
            # Annualize (multiply by square root of trading days per year)
            annualized_volatility = daily_volatility * math.sqrt(252)
            
            # Store in history
            today_str = end_date.strftime('%Y-%m-%d')
            if today_str not in self.volatility_history:
                self.volatility_history[today_str] = {}
            
            self.volatility_history[today_str][symbol] = annualized_volatility
            
            return annualized_volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility for {symbol}: {str(e)}")
            return None
    
    def _detect_vix_regime(self):
        """
        Detect volatility regime based on the VIX index.
        
        Returns:
            str: Volatility regime based on VIX
        """
        try:
            # Get VIX quote
            vix_quote = self.tradier_api.get_quotes('VIX')
            
            if vix_quote.empty:
                self.logger.warning("Could not get VIX data")
                return None
            
            # Get current VIX value
            vix_value = vix_quote.iloc[0]['last']
            
            # Store in history
            today_str = datetime.now().strftime('%Y-%m-%d')
            self.vix_history[today_str] = vix_value
            
            # Classify regime based on VIX thresholds
            if vix_value <= self.low_vix_threshold:
                regime = self.LOW_VOLATILITY
            elif vix_value <= self.high_vix_threshold:
                regime = self.NORMAL_VOLATILITY
            elif vix_value <= self.extreme_vix_threshold:
                regime = self.HIGH_VOLATILITY
            else:
                regime = self.EXTREME_VOLATILITY
            
            self.logger.info(f"VIX value: {vix_value:.2f}, regime: {regime}")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting VIX regime: {str(e)}")
            return None
    
    def get_current_regime(self):
        """
        Get the current volatility regime.
        
        Returns:
            str: Current volatility regime
        """
        return self.current_regime
    
    def detect_volatility_trend(self, symbol=None, lookback_days=30, window=5):
        """
        Detect trend in volatility (increasing or decreasing).
        
        Args:
            symbol (str, optional): Stock symbol (uses market proxy if None)
            lookback_days (int): Number of days to look back
            window (int): Window size for trend detection
            
        Returns:
            dict: Volatility trend information
        """
        try:
            # Use market proxy if symbol is None
            if symbol is None:
                symbol = self.market_proxy
            
            # Get end date (today) and start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + window)
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get historical data
            historical_data = self.tradier_api.get_historical_data(
                symbol, interval='daily', start_date=start_str, end_date=end_str
            )
            
            if historical_data.empty or len(historical_data) < window + 5:
                return {'trend': 'unknown', 'volatility_change': 0.0}
            
            # Calculate rolling volatility
            historical_data = historical_data.sort_values('date')
            historical_data['log_return'] = np.log(historical_data['close'] / historical_data['close'].shift(1))
            
            # Create rolling window of volatility
            rolling_vol = historical_data['log_return'].rolling(window=window).std() * math.sqrt(252)
            
            # Drop NA values
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) < 2:
                return {'trend': 'unknown', 'volatility_change': 0.0}
            
            # Calculate recent trend
            start_vol = rolling_vol.iloc[-window] if len(rolling_vol) >= window else rolling_vol.iloc[0]
            end_vol = rolling_vol.iloc[-1]
            
            # Calculate percent change in volatility
            vol_change = (end_vol - start_vol) / start_vol if start_vol > 0 else 0
            
            # Determine trend
            if vol_change > 0.2:  # 20% increase
                trend = 'strongly_increasing'
            elif vol_change > 0.05:  # 5% increase
                trend = 'increasing'
            elif vol_change < -0.2:  # 20% decrease
                trend = 'strongly_decreasing'
            elif vol_change < -0.05:  # 5% decrease
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'volatility_change': vol_change * 100,  # Convert to percentage
                'start_volatility': start_vol,
                'end_volatility': end_vol
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility trend for {symbol}: {str(e)}")
            return {'trend': 'unknown', 'volatility_change': 0.0}
    
    def detect_volatility_spikes(self, symbol=None, lookback_days=30, threshold=2.0):
        """
        Detect recent spikes in volatility.
        
        Args:
            symbol (str, optional): Stock symbol (uses market proxy if None)
            lookback_days (int): Number of days to look back
            threshold (float): Threshold for spike detection (multiple of average)
            
        Returns:
            list: Volatility spikes
        """
        try:
            # Use market proxy if symbol is None
            if symbol is None:
                symbol = self.market_proxy
            
            # Get end date (today) and start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days * 2)  # Extra days for baseline
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get historical data
            historical_data = self.tradier_api.get_historical_data(
                symbol, interval='daily', start_date=start_str, end_date=end_str
            )
            
            if historical_data.empty or len(historical_data) < 10:
                return []
            
            # Calculate absolute returns
            historical_data = historical_data.sort_values('date')
            historical_data['abs_return'] = abs(historical_data['close'] / historical_data['close'].shift(1) - 1)
            
            # Drop NA values
            historical_data = historical_data.dropna()
            
            # Limit to lookback period
            if len(historical_data) > lookback_days:
                baseline_data = historical_data.iloc[:-lookback_days]
                recent_data = historical_data.iloc[-lookback_days:]
            else:
                # Not enough data for proper baseline, use half for baseline
                mid_point = len(historical_data) // 2
                baseline_data = historical_data.iloc[:mid_point]
                recent_data = historical_data.iloc[mid_point:]
            
            # Calculate baseline average absolute return
            baseline_avg = baseline_data['abs_return'].mean()
            
            # Detect spikes
            spikes = []
            
            for _, row in recent_data.iterrows():
                if row['abs_return'] > baseline_avg * threshold:
                    spike = {
                        'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], (datetime, pd.Timestamp)) else row['date'],
                        'abs_return': row['abs_return'] * 100,  # Convert to percentage
                        'close': row['close'],
                        'ratio': row['abs_return'] / baseline_avg
                    }
                    spikes.append(spike)
            
            return spikes
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility spikes for {symbol}: {str(e)}")
            return []
    
    def get_volatility_percentile(self, symbol=None, lookback_days=252):
        """
        Calculate the current volatility's percentile relative to historical data.
        
        Args:
            symbol (str, optional): Stock symbol (uses market proxy if None)
            lookback_days (int): Number of days to look back
            
        Returns:
            float: Percentile (0-100) of current volatility
        """
        try:
            # Use market proxy if symbol is None
            if symbol is None:
                symbol = self.market_proxy
            
            # Get current volatility (30-day)
            current_vol = self._calculate_historical_volatility(symbol, 30)
            
            if current_vol is None:
                return 50.0  # Default to middle percentile
            
            # Get long-term volatility data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days * 2)  # Extra days for calculation
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get historical data
            historical_data = self.tradier_api.get_historical_data(
                symbol, interval='daily', start_date=start_str, end_date=end_str
            )
            
            if historical_data.empty or len(historical_data) < 60:  # Need at least 60 days for meaningful percentile
                return 50.0
            
            # Calculate log returns
            historical_data = historical_data.sort_values('date')
            historical_data['log_return'] = np.log(historical_data['close'] / historical_data['close'].shift(1))
            
            # Drop NA values
            historical_data = historical_data.dropna()
            
            # Calculate rolling 30-day volatility
            rolling_vol = historical_data['log_return'].rolling(window=30).std() * math.sqrt(252)
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) < 10:
                return 50.0
            
            # Calculate percentile of current volatility
            percentile = 100 * (rolling_vol < current_vol).mean()
            
            return percentile
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility percentile for {symbol}: {str(e)}")
            return 50.0
    
    def get_cross_asset_volatility(self):
        """
        Get volatility across different asset classes.
        
        Returns:
            dict: Volatility by asset class
        """
        try:
            # Define representative symbols for each asset class
            assets = {
                'equities': 'SPY',
                'tech': 'QQQ',
                'small_cap': 'IWM',
                'bonds': 'TLT',
                'gold': 'GLD',
                'oil': 'USO',
                'volatility': 'VIX'
            }
            
            # Get volatility for each asset
            results = {}
            
            for asset_class, symbol in assets.items():
                # Skip VIX (need price, not volatility)
                if symbol == 'VIX':
                    vix_quote = self.tradier_api.get_quotes('VIX')
                    if not vix_quote.empty:
                        results[asset_class] = vix_quote.iloc[0]['last']
                    continue
                
                volatility = self._calculate_historical_volatility(symbol, 30)
                if volatility is not None:
                    results[asset_class] = volatility * 100  # Convert to percentage
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting cross-asset volatility: {str(e)}")
            return {}
