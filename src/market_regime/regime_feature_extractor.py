"""
Regime Feature Extractor Module

This module extracts features from market data that are useful for
regime detection and classification.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class RegimeFeatureExtractor:
    """
    Extracts features from market data for regime classification.
    """
    
    def __init__(self, tradier_api):
        """
        Initialize the RegimeFeatureExtractor.
        
        Args:
            tradier_api: Instance of TradierAPI for market data
        """
        self.logger = logging.getLogger(__name__)
        self.tradier_api = tradier_api
        
        self.logger.info("RegimeFeatureExtractor initialized")
    
    def extract_features(self, historical_data):
        """
        Extract regime-relevant features from historical data.
        
        Args:
            historical_data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: Features dataframe
        """
        try:
            if historical_data.empty or len(historical_data) < 5:
                self.logger.warning("Not enough data for feature extraction")
                return pd.DataFrame()
            
            # Create copy to avoid modifying original data
            data = historical_data.copy()
            
            # Make sure data is sorted by date
            if 'date' in data.columns:
                data = data.sort_values('date')
            
            # Calculate returns
            data['daily_return'] = data['close'].pct_change()
            
            # Feature dataframe
            features = pd.DataFrame()
            
            # Extract date
            if 'date' in data.columns:
                features['date'] = data['date']
            
            # Price features
            features['close_price'] = data['close']
            features['price_change_pct'] = data['close'].pct_change(5) * 100  # 5-day price change %
            
            # Volatility features
            features['daily_volatility'] = data['daily_return'].rolling(window=10).std() * np.sqrt(252)
            features['volatility'] = data['daily_return'].rolling(window=20).std() * np.sqrt(252)  # Annualized volatility
            
            # Volume features
            if 'volume' in data.columns:
                features['volume'] = data['volume']
                features['volume_change_pct'] = data['volume'].pct_change(5) * 100
                features['relative_volume'] = data['volume'] / data['volume'].rolling(window=20).mean()
            
            # Trend features
            self._add_trend_features(data, features)
            
            # Momentum features
            self._add_momentum_features(data, features)
            
            # Volatility regime features
            self._add_volatility_regime_features(data, features)
            
            # Add calendar features
            self._add_calendar_features(features)
            
            # Fill NaN values with 0 for initial periods
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return pd.DataFrame()
    
    def _add_trend_features(self, data, features):
        """
        Add trend-related features.
        
        Args:
            data (pd.DataFrame): Historical price data
            features (pd.DataFrame): Features dataframe to update
        """
        # Calculate moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Calculate moving average relationships
        features['price_to_sma20'] = data['close'] / data['sma_20']
        features['sma20_to_sma50'] = data['sma_20'] / data['sma_50']
        
        # ADX (Average Directional Index) - simplified version
        # Calculate +DI and -DI
        data['high_change'] = data['high'].diff()
        data['low_change'] = -data['low'].diff()
        
        # Positive and negative directional movement
        data['+dm'] = np.where((data['high_change'] > data['low_change']) & (data['high_change'] > 0), 
                               data['high_change'], 0)
        data['-dm'] = np.where((data['low_change'] > data['high_change']) & (data['low_change'] > 0), 
                               data['low_change'], 0)
        
        # Smoothed +DM and -DM
        data['+dm_smoothed'] = data['+dm'].rolling(window=14).mean()
        data['-dm_smoothed'] = data['-dm'].rolling(window=14).mean()
        
        # ATR calculation (simplified)
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        data['atr_14'] = data['tr'].rolling(window=14).mean()
        
        # Calculate +DI and -DI
        data['+di_14'] = 100 * data['+dm_smoothed'] / data['atr_14']
        data['-di_14'] = 100 * data['-dm_smoothed'] / data['atr_14']
        
        # Directional movement index
        data['dx'] = 100 * abs(data['+di_14'] - data['-di_14']) / (data['+di_14'] + data['-di_14'])
        
        # ADX is smoothed DX
        data['adx'] = data['dx'].rolling(window=14).mean()
        
        # Add to features
        features['adx'] = data['adx']
        
        # Trend strength (0-1 scale)
        features['trend_strength'] = data['adx'] / 100
        
        # Trend direction (-1 to 1 scale)
        features['trend_direction'] = (data['+di_14'] - data['-di_14']) / (data['+di_14'] + data['-di_14'])
    
    def _add_momentum_features(self, data, features):
        """
        Add momentum-related features.
        
        Args:
            data (pd.DataFrame): Historical price data
            features (pd.DataFrame): Features dataframe to update
        """
        # RSI calculation
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        features['rsi'] = rsi
        
        # MACD (simplified)
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        macd_line = data['ema_12'] - data['ema_26']
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_line - signal_line
        
        # Rate of change
        features['roc_5'] = data['close'].pct_change(5) * 100
        features['roc_10'] = data['close'].pct_change(10) * 100
        features['roc_20'] = data['close'].pct_change(20) * 100
    
    def _add_volatility_regime_features(self, data, features):
        """
        Add volatility regime features.
        
        Args:
            data (pd.DataFrame): Historical price data
            features (pd.DataFrame): Features dataframe to update
        """
        # Bollinger Bands
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['stddev_20'] = data['close'].rolling(window=20).std()
        data['upper_band'] = data['sma_20'] + (data['stddev_20'] * 2)
        data['lower_band'] = data['sma_20'] - (data['stddev_20'] * 2)
        
        # Bollinger Band width (normalized)
        bb_width = (data['upper_band'] - data['lower_band']) / data['sma_20']
        features['bb_width'] = bb_width
        
        # Percent B (position within Bollinger Bands, 0-1)
        percent_b = (data['close'] - data['lower_band']) / (data['upper_band'] - data['lower_band'])
        features['percent_b'] = percent_b
        
        # Historic volatility comparison
        vol_20 = data['daily_return'].rolling(window=20).std() * np.sqrt(252)
        vol_60 = data['daily_return'].rolling(window=60).std() * np.sqrt(252)
        
        # Volatility ratio (current vs longer-term)
        features['volatility_ratio'] = vol_20 / vol_60
        
        # Volatility trend
        features['volatility_trend'] = vol_20.pct_change(5)
        
        # Average True Range (ATR)
        if 'atr_14' not in data.columns:
            data['tr'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )
            data['atr_14'] = data['tr'].rolling(window=14).mean()
        
        # Normalized ATR
        features['normalized_atr'] = data['atr_14'] / data['close']
    
    def _add_calendar_features(self, features):
        """
        Add calendar-based features.
        
        Args:
            features (pd.DataFrame): Features dataframe to update
        """
        if 'date' not in features.columns:
            return
        
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(features['date']):
            features['date'] = pd.to_datetime(features['date'])
        
        # Extract day of week, month, etc.
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        features['quarter'] = features['date'].dt.quarter
        
        # Is month end
        features['is_month_end'] = features['date'].dt.is_month_end.astype(int)
        
        # Days to month end
        features['days_in_month'] = features['date'].dt.days_in_month
        features['day_of_month'] = features['date'].dt.day
        features['days_to_month_end'] = features['days_in_month'] - features['day_of_month']
    
    def get_current_market_features(self, symbol="SPY"):
        """
        Get current market features for regime detection.
        
        Args:
            symbol (str): Symbol to use for market features
            
        Returns:
            pd.DataFrame: Current market features (single row)
        """
        try:
            # Get historical data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            
            historical_data = self.tradier_api.get_historical_data(
                symbol, interval='daily', start_date=start_date, end_date=end_date
            )
            
            if historical_data.empty:
                self.logger.warning(f"Could not get historical data for {symbol}")
                return pd.DataFrame()
            
            # Extract features
            features = self.extract_features(historical_data)
            
            # Return the most recent row
            if not features.empty:
                return features.iloc[-1:].copy()
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting current market features: {str(e)}")
            return pd.DataFrame()
