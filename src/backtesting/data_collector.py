"""
Data Collector Module

This module collects and stores historical option and stock data for backtesting.
It interfaces with Tradier API to fetch historical prices, option chains,
implied volatility data, and other market information needed for simulation.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import concurrent.futures
import pickle

class DataCollector:
    """
    Collects and manages historical data for backtesting.
    
    Features:
    - Historical stock price data collection
    - Historical option chain data collection
    - Market regime classification data
    - Saving and organizing data for backtesting
    """
    
    def __init__(self, config, tradier_api, drive_manager=None):
        """
        Initialize the DataCollector.
        
        Args:
            config (dict): Configuration dictionary
            tradier_api: Tradier API instance for market data
            drive_manager: Google Drive manager for cloud storage (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        self.drive_manager = drive_manager
        
        # Create necessary directories
        self.data_dir = "data/collected_options_data"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/stocks", exist_ok=True)
        os.makedirs(f"{self.data_dir}/options", exist_ok=True)
        os.makedirs(f"{self.data_dir}/market_regimes", exist_ok=True)
        
        self.logger.info("DataCollector initialized")
    
    def collect_stock_data(self, symbols, start_date, end_date=None, interval='daily'):
        """
        Collect historical stock price data for a list of symbols.
        
        Args:
            symbols (list): List of stock symbols
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            interval (str): Data interval ('daily', 'weekly', or 'monthly')
            
        Returns:
            dict: Dictionary mapping symbols to their historical data DataFrames
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.logger.info(f"Collecting {interval} stock data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel data collection
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Create a future for each symbol
            future_to_symbol = {
                executor.submit(self._collect_single_stock, symbol, start_date, end_date, interval): symbol
                for symbol in symbols
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[symbol] = data
                        self.logger.debug(f"Collected {len(data)} data points for {symbol}")
                    else:
                        self.logger.warning(f"No data collected for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error collecting data for {symbol}: {str(e)}")
        
        self.logger.info(f"Collected stock data for {len(results)} symbols successfully")
        return results
    
    def _collect_single_stock(self, symbol, start_date, end_date, interval):
        """
        Collect historical data for a single stock.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date
            interval (str): Data interval
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        # Get data from Tradier API
        df = self.tradier_api.get_historical_data(symbol, interval, start_date, end_date)
        
        if df.empty:
            return df
            
        # Save data to file
        filename = f"{self.data_dir}/stocks/{symbol}_{interval}_{start_date}_to_{end_date.replace('-', '')}.csv"
        df.to_csv(filename, index=False)
        
        # Upload to Google Drive if available
        if self.drive_manager:
            with open(filename, 'r') as f:
                content = f.read()
            self.drive_manager.upload_file(
                f"backtest_data/stocks/{symbol}_{interval}_{start_date}_to_{end_date}.csv",
                content,
                mime_type="text/csv"
            )
            
        return df
    
    def collect_option_chains(self, symbols, lookback_days=30, expiration_range_days=(7, 60)):
        """
        Collect historical option chain data for a list of symbols.
        
        Args:
            symbols (list): List of stock symbols
            lookback_days (int): Number of days to look back for data
            expiration_range_days (tuple): Min and max days to expiration to include
            
        Returns:
            dict: Dictionary mapping symbols to their option chain data
        """
        self.logger.info(f"Collecting option chain data for {len(symbols)} symbols, {lookback_days} days lookback")
        
        results = {}
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in symbols:
            try:
                # Get underlying stock data first
                stock_data = self.tradier_api.get_historical_data(
                    symbol, 
                    'daily', 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d')
                )
                
                if stock_data.empty:
                    self.logger.warning(f"No historical stock data for {symbol}, skipping option collection")
                    continue
                
                # Get option expirations for this symbol
                expirations = self.tradier_api.get_option_expirations(symbol)
                if not expirations:
                    self.logger.warning(f"No option expirations found for {symbol}")
                    continue
                    
                # Filter expirations based on range
                filtered_expirations = []
                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                    days_to_exp = (exp_date - end_date).days
                    
                    if expiration_range_days[0] <= days_to_exp <= expiration_range_days[1]:
                        filtered_expirations.append(exp)
                
                self.logger.debug(f"Found {len(filtered_expirations)} valid expirations for {symbol}")
                
                # Collect option chains for each expiration
                symbol_chains = {}
                for expiration in filtered_expirations:
                    chain = self.tradier_api.get_option_chains(symbol, expiration)
                    if not chain.empty:
                        symbol_chains[expiration] = chain
                        
                        # Save to file
                        exp_date_str = expiration.replace('-', '')
                        filename = f"{self.data_dir}/options/{symbol}_chain_{exp_date_str}.csv"
                        chain.to_csv(filename, index=False)
                
                if symbol_chains:
                    results[symbol] = symbol_chains
                    
                # Sleep to avoid overwhelming the API
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error collecting option chains for {symbol}: {str(e)}")
        
        self.logger.info(f"Collected option chain data for {len(results)} symbols")
        return results
    
    def collect_market_regime_data(self, start_date, end_date=None, symbols=None):
        """
        Collect data for market regime classification.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            symbols (list, optional): List of symbols to use as market indicators
            
        Returns:
            pandas.DataFrame: Market regime data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        if symbols is None:
            # Default market indicator symbols
            symbols = ['SPY', 'VIX', 'QQQ', 'IWM', 'TLT']
            
        self.logger.info(f"Collecting market regime data from {start_date} to {end_date}")
        
        # Collect data for each symbol
        all_data = {}
        for symbol in symbols:
            try:
                df = self.tradier_api.get_historical_data(symbol, 'daily', start_date, end_date)
                if not df.empty:
                    all_data[symbol] = df
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {str(e)}")
        
        if not all_data:
            self.logger.warning("No market regime data collected")
            return pd.DataFrame()
            
        # Combine data into a single DataFrame with date as index
        combined_data = {}
        
        for symbol, df in all_data.items():
            # Make sure date is a datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Add columns with symbol prefix
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    combined_data[f'{symbol}_{col}'] = df[col]
        
        # Create DataFrame
        result_df = pd.DataFrame(combined_data)
        
        # Calculate additional features
        for symbol in symbols:
            if f'{symbol}_close' in result_df.columns:
                # Add returns
                result_df[f'{symbol}_return'] = result_df[f'{symbol}_close'].pct_change()
                
                # Add moving averages
                for window in [5, 10, 20, 50, 200]:
                    result_df[f'{symbol}_ma{window}'] = result_df[f'{symbol}_close'].rolling(window).mean()
                
                # Add volatility
                result_df[f'{symbol}_vol20'] = result_df[f'{symbol}_return'].rolling(20).std() * np.sqrt(252)
        
        # Save to file
        filename = f"{self.data_dir}/market_regimes/market_data_{start_date}_to_{end_date.replace('-', '')}.csv"
        result_df.to_csv(filename)
        
        self.logger.info(f"Collected market regime data with {len(result_df)} rows and {len(result_df.columns)} features")
        return result_df
    
    def collect_option_time_series(self, symbol, option_type, strike_percentage, lookback_days=90, duration_days=30):
        """
        Collect time series data for a specific option strategy (e.g., 30-delta puts).
        
        Args:
            symbol (str): Underlying stock symbol
            option_type (str): 'call' or 'put'
            strike_percentage (float): Strike as percentage of the underlying price (e.g., 0.95 for 5% OTM puts)
            lookback_days (int): Days to look back
            duration_days (int): Duration to track each option
            
        Returns:
            pandas.DataFrame: Option time series data
        """
        self.logger.info(f"Collecting {option_type} option time series for {symbol}, {strike_percentage:.0%} strike")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Get underlying stock data
        stock_data = self.tradier_api.get_historical_data(
            symbol, 
            'daily', 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        if stock_data.empty:
            self.logger.warning(f"No historical stock data for {symbol}")
            return pd.DataFrame()
        
        # Convert date to datetime and set as index
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.set_index('date', inplace=True)
        
        # Build option time series
        option_series = []
        
        for date, row in stock_data.iterrows():
            try:
                # Calculate target strike price
                target_strike = row['close'] * strike_percentage
                
                # Get option expirations for this date
                available_date = date.strftime('%Y-%m-%d')
                expirations = self.tradier_api.get_option_expirations(symbol)
                
                if not expirations:
                    continue
                
                # Find the expiration closest to our desired duration
                target_date = date + timedelta(days=duration_days)
                closest_exp = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d').date() - target_date.date()).days))
                
                # Get strikes for this expiration
                strikes = self.tradier_api.get_option_strikes(symbol, closest_exp)
                
                if not strikes:
                    continue
                
                # Find the strike closest to our target
                closest_strike = min(strikes, key=lambda x: abs(float(x) - target_strike))
                
                # Get option chain for this expiration
                chain = self.tradier_api.get_option_chains(symbol, closest_exp)
                
                if chain.empty:
                    continue
                
                # Filter for our target option
                option_data = chain[(chain['option_type'] == option_type) & (chain['strike'] == closest_strike)]
                
                if option_data.empty:
                    continue
                
                # Get first matching option
                option = option_data.iloc[0]
                
                # Add to our time series
                option_entry = {
                    'date': date,
                    'underlying': symbol,
                    'underlying_price': row['close'],
                    'option_symbol': option['symbol'],
                    'strike': option['strike'],
                    'expiration': closest_exp,
                    'days_to_expiration': (datetime.strptime(closest_exp, '%Y-%m-%d').date() - date.date()).days,
                    'bid': option['bid'],
                    'ask': option['ask'],
                    'mid': (option['bid'] + option['ask']) / 2,
                    'volume': option['volume'],
                    'open_interest': option['open_interest']
                }
                
                # Add greeks if available
                if 'greeks' in option and option['greeks'] is not None:
                    for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                        if greek in option['greeks']:
                            option_entry[greek] = option['greeks'][greek]
                
                option_series.append(option_entry)
                
            except Exception as e:
                self.logger.error(f"Error on {date.strftime('%Y-%m-%d')} for {symbol}: {str(e)}")
        
        if not option_series:
            self.logger.warning(f"No option time series data collected for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        result_df = pd.DataFrame(option_series)
        
        # Save to file
        strike_str = f"{int(strike_percentage * 100)}pct"
        filename = f"{self.data_dir}/options/{symbol}_{option_type}_{strike_str}_{duration_days}d.csv"
        result_df.to_csv(filename, index=False)
        
        self.logger.info(f"Collected {len(result_df)} option time series data points for {symbol}")
        return result_df
    
    def save_dataset(self, data, name, description=None):
        """
        Save a dataset for backtesting.
        
        Args:
            data: Data to save (DataFrame, dict, or other serializable object)
            name (str): Dataset name
            description (str, optional): Dataset description
            
        Returns:
            str: Path to saved dataset
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.data_dir}/{name}_{timestamp}"
        
        # Create metadata
        metadata = {
            'name': name,
            'description': description,
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rows': len(data) if hasattr(data, '__len__') else 'unknown'
        }
        
        # Save metadata
        with open(f"{filename}.meta", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save data based on type
        if isinstance(data, pd.DataFrame):
            data.to_csv(f"{filename}.csv", index=False)
            file_path = f"{filename}.csv"
        else:
            with open(f"{filename}.pickle", 'wb') as f:
                pickle.dump(data, f)
            file_path = f"{filename}.pickle"
        
        self.logger.info(f"Saved dataset {name} to {file_path}")
        
        # Upload to Google Drive if available
        if self.drive_manager:
            if isinstance(data, pd.DataFrame):
                with open(file_path, 'r') as f:
                    content = f.read()
                self.drive_manager.upload_file(
                    f"backtest_data/{name}_{timestamp}.csv",
                    content,
                    mime_type="text/csv"
                )
            else:
                with open(file_path, 'rb') as f:
                    content = f.read()
                self.drive_manager.upload_file(
                    f"backtest_data/{name}_{timestamp}.pickle",
                    content,
                    mime_type="application/octet-stream"
                )
        
        return file_path
    
    def load_dataset(self, file_path):
        """
        Load a saved dataset.
        
        Args:
            file_path (str): Path to the dataset file
            
        Returns:
            The loaded dataset
        """
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.pickle'):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                self.logger.error(f"Unknown file format for {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading dataset {file_path}: {str(e)}")
            return None
    
    def create_backtesting_dataset(self, symbols, start_date, end_date=None, include_options=True):
        """
        Create a complete dataset for backtesting.
        
        Args:
            symbols (list): List of stock symbols to include
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            include_options (bool): Whether to include option data
            
        Returns:
            str: Path to the saved dataset
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.logger.info(f"Creating backtesting dataset for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Collect stock data
        stock_data = self.collect_stock_data(symbols, start_date, end_date)
        
        # Collect market regime data
        market_data = self.collect_market_regime_data(start_date, end_date)
        
        # Initialize dataset
        dataset = {
            'metadata': {
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'include_options': include_options
            },
            'stock_data': stock_data,
            'market_data': market_data.to_dict() if not market_data.empty else {}
        }
        
        # Add option data if requested
        if include_options:
            option_chains = self.collect_option_chains(
                symbols, 
                lookback_days=min(90, (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days)
            )
            dataset['option_chains'] = option_chains
        
        # Save the dataset
        dataset_name = f"backtest_dataset_{len(symbols)}symbols"
        return self.save_dataset(
            dataset, 
            dataset_name, 
            f"Backtesting dataset with {len(symbols)} symbols from {start_date} to {end_date}"
        )
