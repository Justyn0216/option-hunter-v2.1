"""
Data Processor Module

This module processes collected historical data for backtesting.
It transforms raw data into formats suitable for simulation,
handles missing data, and creates synthetic data where needed.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import pickle
from scipy.interpolate import interp1d

class DataProcessor:
    """
    Processes historical data for backtesting.
    
    Features:
    - Data cleaning and normalization
    - Missing data imputation
    - Feature engineering
    - Option chain interpolation and extrapolation
    - Regime labeling
    """
    
    def __init__(self, config):
        """
        Initialize the DataProcessor.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Create necessary directories
        self.data_dir = "data/collected_options_data"
        self.processed_dir = "data/processed_data"
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.logger.info("DataProcessor initialized")
        
    def prepare_backtest_data(self, dataset_path, target_symbols=None):
        """
        Prepare a complete dataset for backtesting.
        
        Args:
            dataset_path (str): Path to the raw dataset
            target_symbols (list, optional): List of symbols to focus on (if None, use all)
            
        Returns:
            dict: Processed dataset ready for backtesting
        """
        self.logger.info(f"Preparing backtest data from {dataset_path}")
        
        # Load the dataset
        raw_dataset = self.load_dataset(dataset_path)
        
        if raw_dataset is None:
            self.logger.error("Failed to load dataset")
            return None
        
        # Extract components
        metadata = raw_dataset.get('metadata', {})
        stock_data = raw_dataset.get('stock_data', {})
        market_data = raw_dataset.get('market_data', {})
        option_chains = raw_dataset.get('option_chains', {})
        
        # Filter symbols if requested
        if target_symbols is not None:
            stock_data = {symbol: data for symbol, data in stock_data.items() if symbol in target_symbols}
            option_chains = {symbol: data for symbol, data in option_chains.items() if symbol in target_symbols}
        
        # Process each component
        processed_stock_data = self.process_stock_data(stock_data)
        
        # Process market data if available
        processed_market_data = None
        if market_data:
            market_df = pd.DataFrame(market_data) if isinstance(market_data, dict) else market_data
            processed_market_data = self.process_market_data(market_df)
        
        # Process option chains if available
        processed_option_chains = None
        if option_chains:
            processed_option_chains = self.process_option_chains(option_chains)
        
        # Create backtest-ready dataset
        processed_dataset = {
            'metadata': metadata,
            'stock_data': processed_stock_data,
            'market_data': processed_market_data,
            'option_chains': processed_option_chains,
            'processing_info': {
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_symbols': len(processed_stock_data),
                'date_range': [
                    min([df.index.min() for df in processed_stock_data.values()]).strftime('%Y-%m-%d'),
                    max([df.index.max() for df in processed_stock_data.values()]).strftime('%Y-%m-%d')
                ] if processed_stock_data else None
            }
        }
        
        # Save the processed dataset
        output_path = f"{self.processed_dir}/processed_backtest_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pickle"
        with open(output_path, 'wb') as f:
            pickle.dump(processed_dataset, f)
        
        self.logger.info(f"Processed backtest data saved to {output_path}")
        
        return processed_dataset
    
    def load_dataset(self, dataset_path):
        """
        Load a raw dataset.
        
        Args:
            dataset_path (str): Path to the dataset file
            
        Returns:
            dict or DataFrame: The loaded dataset
        """
        try:
            if dataset_path.endswith('.csv'):
                return pd.read_csv(dataset_path)
            elif dataset_path.endswith('.pickle'):
                with open(dataset_path, 'rb') as f:
                    return pickle.load(f)
            else:
                self.logger.error(f"Unknown file format for {dataset_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_path}: {str(e)}")
            return None
    
    def process_market_data(self, market_data, label_regimes=True):
        """
        Process market data and identify market regimes.
        
        Args:
            market_data (DataFrame): Market data with various market indicators
            label_regimes (bool): Whether to label market regimes
            
        Returns:
            DataFrame: Processed market data with regime labels
        """
        self.logger.info("Processing market data")
        
        try:
            # Make a copy to avoid modifying the original
            processed_df = market_data.copy()
            
            # Calculate additional market indicators
            if 'SPY_close' in processed_df.columns:
                # Trend indicators
                processed_df['spy_ma50'] = processed_df['SPY_close'].rolling(window=50).mean()
                processed_df['spy_ma200'] = processed_df['SPY_close'].rolling(window=200).mean()
                processed_df['spy_trend'] = processed_df['spy_ma50'] - processed_df['spy_ma200']
                
                # Volatility indicators
                if 'VIX_close' in processed_df.columns:
                    processed_df['vix_ma10'] = processed_df['VIX_close'].rolling(window=10).mean()
                    processed_df['vix_ma50'] = processed_df['VIX_close'].rolling(window=50).mean()
                    processed_df['vix_trend'] = processed_df['vix_ma10'] - processed_df['vix_ma50']
                
                # Label market regimes
                if label_regimes:
                    processed_df = self._label_market_regimes(processed_df)
            
            self.logger.info(f"Processed market data: {len(processed_df)} rows")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            return market_data
    
    def _label_market_regimes(self, market_df):
        """
        Label market regimes based on price action and volatility.
        
        Args:
            market_df (DataFrame): Market data
            
        Returns:
            DataFrame: Market data with regime labels
        """
        # Make sure we have the necessary columns
        required_cols = ['SPY_close', 'SPY_return', 'VIX_close']
        if not all(col in market_df.columns for col in required_cols):
            self.logger.warning("Missing required columns for market regime labeling")
            market_df['market_regime'] = 'unknown'
            return market_df
        
        # Create a new column for market regime
        market_df['market_regime'] = 'neutral'
        
        # Calculate rolling indicators
        market_df['spy_return_20d'] = market_df['SPY_return'].rolling(window=20).sum()
        market_df['spy_vol_20d'] = market_df['SPY_return'].rolling(window=20).std() * np.sqrt(252)
        
        # Check if SPY is above/below moving averages
        market_df['spy_above_ma50'] = market_df['SPY_close'] > market_df['spy_ma50']
        market_df['spy_above_ma200'] = market_df['SPY_close'] > market_df['spy_ma200']
        
        # Label regimes based on rules (simplified approach)
        # Bullish: SPY above MAs, positive returns, low volatility
        bullish_mask = (
            market_df['spy_above_ma50'] & 
            market_df['spy_above_ma200'] & 
            (market_df['spy_return_20d'] > 0.02) &
            (market_df['VIX_close'] < 20)
        )
        market_df.loc[bullish_mask, 'market_regime'] = 'bullish'
        
        # Bearish: SPY below MAs, negative returns
        bearish_mask = (
            ~market_df['spy_above_ma50'] & 
            ~market_df['spy_above_ma200'] & 
            (market_df['spy_return_20d'] < -0.02)
        )
        market_df.loc[bearish_mask, 'market_regime'] = 'bearish'
        
        # Volatile: High VIX, high volatility
        volatile_mask = (
            (market_df['VIX_close'] > 25) |
            (market_df['spy_vol_20d'] > 0.2)
        )
        market_df.loc[volatile_mask, 'market_regime'] = 'volatile'
        
        # Sideways/Ranging: Low returns, low volatility
        sideways_mask = (
            (abs(market_df['spy_return_20d']) < 0.01) &
            (market_df['spy_vol_20d'] < 0.15) &
            (market_df['VIX_close'] < 18)
        )
        market_df.loc[sideways_mask, 'market_regime'] = 'sideways'
        
        # Trending: Consistent direction with moderate volatility
        trending_mask = (
            ((market_df['spy_return_20d'] > 0.03) | (market_df['spy_return_20d'] < -0.03)) &
            (market_df['spy_vol_20d'] < 0.25) &
            (market_df['spy_vol_20d'] > 0.1)
        )
        market_df.loc[trending_mask, 'market_regime'] = 'trending'
        
        # Count regimes
        regime_counts = market_df['market_regime'].value_counts()
        self.logger.info(f"Market regime distribution: {regime_counts.to_dict()}")
        
        return market_df
    
    def process_stock_data(self, stock_data, fill_gaps=True, calculate_features=True):
        """
        Process historical stock data.
        
        Args:
            stock_data (dict): Dictionary of stock DataFrames by symbol
            fill_gaps (bool): Whether to fill gaps in the data
            calculate_features (bool): Whether to calculate additional features
            
        Returns:
            dict: Dictionary of processed stock DataFrames
        """
        self.logger.info(f"Processing stock data for {len(stock_data)} symbols")
        
        processed_data = {}
        
        for symbol, df in stock_data.items():
            try:
                # Make a copy to avoid modifying the original
                processed_df = df.copy()
                
                # Convert date to datetime and set as index if it's not already
                if 'date' in processed_df.columns:
                    processed_df['date'] = pd.to_datetime(processed_df['date'])
                    processed_df.set_index('date', inplace=True)
                
                # Fill gaps in the data
                if fill_gaps:
                    # Create a complete date range
                    date_range = pd.date_range(start=processed_df.index.min(), end=processed_df.index.max(), freq='B')
                    processed_df = processed_df.reindex(date_range)
                    
                    # Forward fill missing values (use previous day's values)
                    processed_df.fillna(method='ffill', inplace=True)
                    
                    # If there are still NaNs (at the beginning), backfill
                    processed_df.fillna(method='bfill', inplace=True)
                
                # Calculate additional features
                if calculate_features:
                    # Returns
                    processed_df['daily_return'] = processed_df['close'].pct_change()
                    processed_df['log_return'] = np.log(processed_df['close'] / processed_df['close'].shift(1))
                    
                    # Moving averages
                    for window in [5, 10, 20, 50, 200]:
                        processed_df[f'ma{window}'] = processed_df['close'].rolling(window=window).mean()
                    
                    # Volatility (20-day rolling standard deviation of returns)
                    processed_df['volatility_20d'] = processed_df['daily_return'].rolling(window=20).std() * np.sqrt(252)
                    
                    # RSI (14-day)
                    delta = processed_df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    processed_df['rsi_14'] = 100 - (100 / (1 + rs))
                    
                    # MACD
                    exp12 = processed_df['close'].ewm(span=12, adjust=False).mean()
                    exp26 = processed_df['close'].ewm(span=26, adjust=False).mean()
                    processed_df['macd'] = exp12 - exp26
                    processed_df['macd_signal'] = processed_df['macd'].ewm(span=9, adjust=False).mean()
                    processed_df['macd_hist'] = processed_df['macd'] - processed_df['macd_signal']
                    
                    # Bollinger Bands (20-day, 2 standard deviations)
                    processed_df['bb_middle'] = processed_df['close'].rolling(window=20).mean()
                    std_dev = processed_df['close'].rolling(window=20).std()
                    processed_df['bb_upper'] = processed_df['bb_middle'] + 2 * std_dev
                    processed_df['bb_lower'] = processed_df['bb_middle'] - 2 * std_dev
                
                processed_data[symbol] = processed_df
                self.logger.debug(f"Processed stock data for {symbol}: {len(processed_df)} rows, {len(processed_df.columns)} columns")
                
            except Exception as e:
                self.logger.error(f"Error processing stock data for {symbol}: {str(e)}")
        
        self.logger.info(f"Completed processing stock data for {len(processed_data)} symbols")
        return processed_data
    
    def process_option_chains(self, option_chains, interpolate_strikes=True, calculate_greeks=True):
        """
        Process historical option chain data.
        
        Args:
            option_chains (dict): Dictionary of option chain data by symbol and expiration
            interpolate_strikes (bool): Whether to interpolate missing strikes
            calculate_greeks (bool): Whether to calculate missing Greeks
            
        Returns:
            dict: Dictionary of processed option chain data
        """
        self.logger.info(f"Processing option chains for {len(option_chains)} symbols")
        
        processed_chains = {}
        
        for symbol, expirations in option_chains.items():
            try:
                processed_chains[symbol] = {}
                
                for expiration, chain_df in expirations.items():
                    # Make a copy to avoid modifying the original
                    processed_df = chain_df.copy()
                    
                    # Split into calls and puts
                    calls = processed_df[processed_df['option_type'] == 'call']
                    puts = processed_df[processed_df['option_type'] == 'put']
                    
                    # Interpolate missing strikes if needed
                    if interpolate_strikes and (len(calls) > 2 or len(puts) > 2):
                        # Process calls
                        if len(calls) > 2:
                            calls = self._interpolate_option_chain(calls)
                        
                        # Process puts
                        if len(puts) > 2:
                            puts = self._interpolate_option_chain(puts)
                        
                        # Recombine
                        processed_df = pd.concat([calls, puts])
                    
                    # Calculate missing Greeks if needed
                    if calculate_greeks:
                        processed_df = self._ensure_greeks(processed_df, symbol, expiration)
                    
                    # Add to processed chains
                    processed_chains[symbol][expiration] = processed_df
                
                self.logger.debug(f"Processed option chains for {symbol}: {len(expirations)} expirations")
                
            except Exception as e:
                self.logger.error(f"Error processing option chains for {symbol}: {str(e)}")
        
        self.logger.info(f"Completed processing option chains for {len(processed_chains)} symbols")
        return processed_chains
    
    def _interpolate_option_chain(self, chain_df):
        """
        Interpolate missing strikes in an option chain.
        
        Args:
            chain_df (DataFrame): Option chain DataFrame (calls or puts)
            
        Returns:
            DataFrame: Interpolated option chain
        """
        # Sort by strike
        chain_df = chain_df.sort_values(by='strike')
        
        # Get sorted strikes
        strikes = chain_df['strike'].values
        
        # If we have less than 2 strikes, we can't interpolate
        if len(strikes) < 2:
            return chain_df
        
        # Calculate ideal strike spacing (min of actual spacing to avoid too many strikes)
        strike_diffs = np.diff(strikes)
        min_diff = max(strike_diffs.min(), 0.5)  # At least 0.5 to avoid too many strikes
        
        # Create a range of strikes with regular spacing
        ideal_strikes = np.arange(strikes.min(), strikes.max() + min_diff, min_diff)
        
        # Columns to interpolate
        cols_to_interpolate = ['bid', 'ask', 'volume', 'open_interest']
        
        # Create interpolation functions for each column
        interpolators = {}
        for col in cols_to_interpolate:
            # Make sure we have values to interpolate
            if chain_df[col].notna().sum() >= 2:
                interpolators[col] = interp1d(
                    strikes, 
                    chain_df[col].values, 
                    kind='linear', 
                    bounds_error=False,
                    fill_value='extrapolate'
                )
        
        # Create new DataFrame with interpolated values
        new_rows = []
        
        for strike in ideal_strikes:
            # Check if strike already exists
            if strike in strikes:
                # Use existing data
                new_rows.append(chain_df[chain_df['strike'] == strike].iloc[0])
            else:
                # Create new row
                new_row = {
                    'strike': strike,
                    'option_type': chain_df['option_type'].iloc[0],  # All should be the same
                    'symbol': f"{chain_df['symbol'].iloc[0]}_interp",  # Mark as interpolated
                    'expiration': chain_df['expiration'].iloc[0] if 'expiration' in chain_df.columns else None
                }
                
                # Interpolate values
                for col, interpolator in interpolators.items():
                    new_row[col] = interpolator(strike)
                
                # Round volume and open interest to integers
                if 'volume' in new_row:
                    new_row['volume'] = max(0, int(round(new_row['volume'])))
                if 'open_interest' in new_row:
                    new_row['open_interest'] = max(0, int(round(new_row['open_interest'])))
                
                # Add to new rows
                new_rows.append(new_row)
        
        # Convert to DataFrame
        return pd.DataFrame(new_rows)
