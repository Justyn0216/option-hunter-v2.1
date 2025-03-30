"""
Backtesting Data Collector Module

This module is responsible for collecting and storing historical data
needed for backtesting the Option Hunter trading system.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
from pathlib import Path

class DataCollector:
    """
    Collects and stores historical data for backtesting purposes.
    
    This class is responsible for:
    - Collecting historical price data
    - Collecting historical option chain data
    - Collecting historical volatility regime data
    - Storing data in a structured format for backtesting
    """
    
    def __init__(self, config, tradier_api, drive_manager):
        """
        Initialize the DataCollector.
        
        Args:
            config (dict): System configuration
            tradier_api: TradierAPI instance
            drive_manager: GoogleDriveManager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        self.drive_manager = drive_manager
        
        # Data storage paths
        self.data_dir = "data/collected_options_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Queue for async data collection
        self.collection_queue = queue.Queue()
        self.collection_thread = None
        self.running = False
        
        # Set up data collection parameters
        self.tickers = config["ticker_list"]
        self.lookback_days = config["backtesting"].get("lookback_days", 90)
        self.batch_size = config["backtesting"].get("collection_batch_size", 5)
        self.collection_interval = config["backtesting"].get("collection_interval_seconds", 300)
        
        self.logger.info(f"DataCollector initialized with {len(self.tickers)} tickers and {self.lookback_days} days lookback")
    
    def start_collection(self):
        """Start the background data collection thread."""
        if self.collection_thread is not None and self.collection_thread.is_alive():
            self.logger.warning("Data collection is already running")
            return
        
        self.running = True
        self.collection_thread = threading.Thread(
            target=self._collection_worker,
            daemon=True,
            name="DataCollectionThread"
        )
        self.collection_thread.start()
        self.logger.info("Data collection thread started")
    
    def stop_collection(self):
        """Stop the background data collection thread."""
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=60)
            self.logger.info("Data collection thread stopped")
    
    def _collection_worker(self):
        """Background worker that processes the collection queue."""
        while self.running:
            try:
                # Process any items in the queue first
                while not self.collection_queue.empty():
                    collection_task = self.collection_queue.get()
                    self._process_collection_task(collection_task)
                    self.collection_queue.task_done()
                
                # Schedule regular collection of all tickers
                self._schedule_regular_collection()
                
                # Sleep between collection cycles
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in collection worker: {str(e)}")
                time.sleep(60)  # Sleep longer on error
    
    def _schedule_regular_collection(self):
        """Schedule regular collection of all tickers."""
        # Process tickers in batches to avoid overwhelming the API
        for i in range(0, len(self.tickers), self.batch_size):
            batch = self.tickers[i:i+self.batch_size]
            
            for ticker in batch:
                task = {
                    'type': 'full_collection',
                    'ticker': ticker,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.collection_queue.put(task)
            
            # Sleep between batches to avoid rate limiting
            if i + self.batch_size < len(self.tickers):
                time.sleep(5)
    
    def _process_collection_task(self, task):
        """
        Process a collection task.
        
        Args:
            task (dict): Collection task details
        """
        task_type = task.get('type')
        ticker = task.get('ticker')
        
        if task_type == 'full_collection':
            self.collect_historical_data(ticker)
            self.collect_option_chain_history(ticker)
            self.collect_market_regime_data(ticker)
        elif task_type == 'price_history':
            self.collect_historical_data(ticker)
        elif task_type == 'option_chains':
            self.collect_option_chain_history(ticker)
        elif task_type == 'market_regime':
            self.collect_market_regime_data(ticker)
        else:
            self.logger.warning(f"Unknown collection task type: {task_type}")
    
    def collect_historical_data(self, ticker, start_date=None, end_date=None):
        """
        Collect historical price data for a ticker.
        
        Args:
            ticker (str): Stock symbol
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Historical data
        """
        self.logger.info(f"Collecting historical data for {ticker}")
        
        # Set date range if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        try:
            # Get daily historical data
            daily_data = self.tradier_api.get_historical_data(
                ticker, interval='daily', start_date=start_date, end_date=end_date
            )
            
            if daily_data.empty:
                self.logger.warning(f"No historical data found for {ticker}")
                return pd.DataFrame()
            
            # Save data to file
            output_dir = f"{self.data_dir}/price_history"
            os.makedirs(output_dir, exist_ok=True)
            
            file_path = f"{output_dir}/{ticker}_daily.csv"
            daily_data.to_csv(file_path, index=False)
            
            self.logger.info(f"Saved historical data for {ticker} with {len(daily_data)} records")
            
            # Upload to Google Drive if configured
            if self.drive_manager:
                with open(file_path, 'r') as f:
                    self.drive_manager.upload_file(
                        f"backtest_data/{ticker}_daily.csv",
                        f.read(),
                        mime_type="text/csv"
                    )
            
            return daily_data
            
        except Exception as e:
            self.logger.error(f"Error collecting historical data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def collect_option_chain_history(self, ticker):
        """
        Collect historical option chain data for a ticker.
        
        Since historical option chains aren't directly available from Tradier,
        we'll collect current chains and save them daily, building history over time.
        
        Args:
            ticker (str): Stock symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Collecting option chain data for {ticker}")
        
        try:
            # Get available expirations
            expirations = self.tradier_api.get_option_expirations(ticker)
            
            if not expirations:
                self.logger.warning(f"No option expirations found for {ticker}")
                return False
            
            # Limit to reasonable number of expirations (next 4)
            expirations = expirations[:min(4, len(expirations))]
            
            # Get option chains for each expiration
            all_chains = []
            
            for expiration in expirations:
                chain_df = self.tradier_api.get_option_chains(ticker, expiration)
                
                if not chain_df.empty:
                    # Add collection date
                    chain_df['collection_date'] = datetime.now().strftime('%Y-%m-%d')
                    all_chains.append(chain_df)
                
                # Avoid rate limiting
                time.sleep(1)
            
            if not all_chains:
                self.logger.warning(f"No option chains collected for {ticker}")
                return False
            
            # Combine all chains
            combined_chains = pd.concat(all_chains, ignore_index=True)
            
            # Save data to file
            output_dir = f"{self.data_dir}/option_chains"
            os.makedirs(output_dir, exist_ok=True)
            
            today = datetime.now().strftime('%Y%m%d')
            file_path = f"{output_dir}/{ticker}_options_{today}.csv"
            combined_chains.to_csv(file_path, index=False)
            
            self.logger.info(f"Saved option chain data for {ticker} with {len(combined_chains)} contracts")
            
            # Upload to Google Drive if configured
            if self.drive_manager:
                with open(file_path, 'r') as f:
                    self.drive_manager.upload_file(
                        f"backtest_data/{ticker}_options_{today}.csv",
                        f.read(),
                        mime_type="text/csv"
                    )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting option chain data for {ticker}: {str(e)}")
            return False
    
    def collect_market_regime_data(self, ticker):
        """
        Collect market regime data for backtesting.
        
        This will use market state information from VIX and other indicators
        to classify historical days into different market regimes.
        
        Args:
            ticker (str): Stock symbol (typically an index like SPY)
            
        Returns:
            pandas.DataFrame: Market regime data
        """
        self.logger.info(f"Collecting market regime data using {ticker}")
        
        try:
            # Get VIX data if possible (as a proxy for market volatility)
            vix_data = self.tradier_api.get_historical_data(
                'VIX', 
                interval='daily', 
                start_date=(datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            # Get price data for the ticker
            price_data = self.tradier_api.get_historical_data(
                ticker,
                interval='daily', 
                start_date=(datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if price_data.empty:
                self.logger.warning(f"No price data found for {ticker}, cannot determine market regimes")
                return pd.DataFrame()
            
            # Calculate returns and volatility
            price_data = price_data.sort_values('date')
            price_data['return'] = price_data['close'].pct_change()
            
            # Calculate 10-day rolling volatility
            price_data['volatility_10d'] = price_data['return'].rolling(10).std() * np.sqrt(252)
            
            # Calculate 20-day moving average
            price_data['ma_20d'] = price_data['close'].rolling(20).mean()
            
            # Determine trend (above or below 20-day MA)
            price_data['trend'] = np.where(price_data['close'] > price_data['ma_20d'], 'up', 'down')
            
            # Merge with VIX data if available
            if not vix_data.empty:
                vix_data = vix_data.rename(columns={'close': 'vix'})
                merged_data = pd.merge(price_data, vix_data[['date', 'vix']], on='date', how='left')
            else:
                merged_data = price_data
                merged_data['vix'] = np.nan
            
            # Classify market regimes
            def classify_regime(row):
                if pd.isna(row['vix']):
                    # Use volatility if VIX is not available
                    high_vol = row['volatility_10d'] > 0.2
                else:
                    high_vol = row['vix'] > 20
                
                if high_vol and row['trend'] == 'up':
                    return 'volatile_bullish'
                elif high_vol and row['trend'] == 'down':
                    return 'volatile_bearish'
                elif not high_vol and row['trend'] == 'up':
                    return 'calm_bullish'
                else:
                    return 'calm_bearish'
            
            merged_data['market_regime'] = merged_data.apply(classify_regime, axis=1)
            
            # Save data to file
            output_dir = f"{self.data_dir}/market_regimes"
            os.makedirs(output_dir, exist_ok=True)
            
            file_path = f"{output_dir}/market_regimes.csv"
            merged_data.to_csv(file_path, index=False)
            
            self.logger.info(f"Saved market regime data with {len(merged_data)} records")
            
            # Upload to Google Drive if configured
            if self.drive_manager:
                with open(file_path, 'r') as f:
                    self.drive_manager.upload_file(
                        f"backtest_data/market_regimes.csv",
                        f.read(),
                        mime_type="text/csv"
                    )
            
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market regime data: {str(e)}")
            return pd.DataFrame()
    
    def load_historical_data(self, ticker, start_date=None, end_date=None):
        """
        Load historical price data for a ticker from storage.
        
        Args:
            ticker (str): Stock symbol
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Historical data
        """
        file_path = f"{self.data_dir}/price_history/{ticker}_daily.csv"
        
        if not os.path.exists(file_path):
            self.logger.warning(f"No stored historical data found for {ticker}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading historical data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def load_option_chain_history(self, ticker, collection_date=None):
        """
        Load historical option chain data for a ticker from storage.
        
        Args:
            ticker (str): Stock symbol
            collection_date (str, optional): Specific collection date (YYYYMMDD format)
            
        Returns:
            pandas.DataFrame: Option chain data
        """
        output_dir = f"{self.data_dir}/option_chains"
        
        if collection_date:
            file_path = f"{output_dir}/{ticker}_options_{collection_date}.csv"
            if not os.path.exists(file_path):
                self.logger.warning(f"No stored option chain data found for {ticker} on {collection_date}")
                return pd.DataFrame()
            
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                self.logger.error(f"Error loading option chain data for {ticker}: {str(e)}")
                return pd.DataFrame()
        else:
            # Load all available dates
            all_files = [f for f in os.listdir(output_dir) if f.startswith(f"{ticker}_options_") and f.endswith('.csv')]
            
            if not all_files:
                self.logger.warning(f"No stored option chain data found for {ticker}")
                return pd.DataFrame()
            
            try:
                dfs = []
                for file in all_files:
                    df = pd.read_csv(f"{output_dir}/{file}")
                    dfs.append(df)
                
                return pd.concat(dfs, ignore_index=True)
                
            except Exception as e:
                self.logger.error(f"Error loading option chain data for {ticker}: {str(e)}")
                return pd.DataFrame()
    
    def load_market_regime_data(self, start_date=None, end_date=None):
        """
        Load market regime data from storage.
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Market regime data
        """
        file_path = f"{self.data_dir}/market_regimes/market_regimes.csv"
        
        if not os.path.exists(file_path):
            self.logger.warning("No stored market regime data found")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]
                
            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading market regime data: {str(e)}")
            return pd.DataFrame()
    
    def get_data_availability_summary(self):
        """
        Get a summary of available historical data.
        
        Returns:
            dict: Summary of available data
        """
        summary = {
            'price_history': {},
            'option_chains': {},
            'market_regimes': False
        }
        
        # Check price history
        price_dir = f"{self.data_dir}/price_history"
        if os.path.exists(price_dir):
            price_files = [f for f in os.listdir(price_dir) if f.endswith('_daily.csv')]
            for file in price_files:
                ticker = file.split('_')[0]
                try:
                    df = pd.read_csv(f"{price_dir}/{file}")
                    if not df.empty:
                        summary['price_history'][ticker] = {
                            'records': len(df),
                            'date_range': [df['date'].min(), df['date'].max()] if 'date' in df.columns else None
                        }
                except:
                    pass
        
        # Check option chains
        option_dir = f"{self.data_dir}/option_chains"
        if os.path.exists(option_dir):
            # Group by ticker
            ticker_files = {}
            for file in os.listdir(option_dir):
                if '_options_' in file and file.endswith('.csv'):
                    ticker = file.split('_options_')[0]
                    if ticker not in ticker_files:
                        ticker_files[ticker] = []
                    ticker_files[ticker].append(file)
            
            for ticker, files in ticker_files.items():
                summary['option_chains'][ticker] = {
                    'collection_dates': len(files),
                    'latest_date': max(files).split('_options_')[1].split('.')[0]
                }
        
        # Check market regimes
        regime_file = f"{self.data_dir}/market_regimes/market_regimes.csv"
        if os.path.exists(regime_file):
            try:
                df = pd.read_csv(regime_file)
                if not df.empty:
                    summary['market_regimes'] = {
                        'records': len(df),
                        'date_range': [df['date'].min(), df['date'].max()] if 'date' in df.columns else None
                    }
            except:
                pass
        
        return summary
