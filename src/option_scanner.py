"""
Option Scanner Module

This module scans for undervalued options based on the pricing models
and identifies potential trading opportunities.
"""

import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
from src.option_pricing import OptionPricer, OptionType

class OptionScanner:
    """
    Scans option chains to find undervalued options that meet specific criteria.
    
    Uses the OptionPricer to calculate theoretical prices and compares them
    to market prices to identify potential opportunities.
    """
    
    def __init__(self, config, tradier_api, option_pricer):
        """
        Initialize the OptionScanner.
        
        Args:
            config (dict): Scanner configuration
            tradier_api: Tradier API instance for market data
            option_pricer: OptionPricer instance for calculating option prices
        """
        self.logger = logging.getLogger(__name__)
        self.config = config["option_scanner"]
        self.tradier_api = tradier_api
        self.option_pricer = option_pricer
        
        # Extract configuration
        self.min_days_to_expiration = self.config.get("min_days_to_expiration", 7)
        self.max_days_to_expiration = self.config.get("max_days_to_expiration", 45)
        self.min_open_interest = self.config.get("min_open_interest", 100)
        self.min_volume = self.config.get("min_volume", 50)
        self.max_price = self.config.get("max_price", 10.0)
        self.min_price = self.config.get("min_price", 0.10)
        self.focus_on_strikes = self.config.get("focus_on_strikes", [-3, -2, -1, 0, 1, 2, 3])
        
        self.logger.info("OptionScanner initialized")
    
    def scan_ticker(self, symbol, expiration=None):
        """
        Scan a single ticker for potential option opportunities.
        
        Args:
            symbol (str): Stock symbol to scan
            expiration (str, optional): Specific expiration date in YYYY-MM-DD format
            
        Returns:
            list: List of opportunity dictionaries
        """
        self.logger.info(f"Scanning options for {symbol}")
        
        try:
            # Get stock quote
            quote_df = self.tradier_api.get_quotes(symbol)
            if quote_df.empty:
                self.logger.warning(f"No quote found for {symbol}")
                return []
                
            current_price = quote_df.iloc[0]['last']
            
            # Get option expirations
            if not expiration:
                expirations = self.tradier_api.get_option_expirations(symbol)
                if not expirations:
                    self.logger.warning(f"No option expirations found for {symbol}")
                    return []
                
                # Filter expirations based on min/max days to expiration
                today = datetime.now().date()
                valid_expirations = []
                
                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                    days_to_exp = (exp_date - today).days
                    
                    if self.min_days_to_expiration <= days_to_exp <= self.max_days_to_expiration:
                        valid_expirations.append(exp)
                
                if not valid_expirations:
                    self.logger.info(f"No valid expirations found for {symbol} within {self.min_days_to_expiration}-{self.max_days_to_expiration} days")
                    return []
            else:
                valid_expirations = [expiration]
            
            # Get historical volatility (proxy for implied volatility if not available)
            historical_volatility = self._get_historical_volatility(symbol)
            
            # Scan each expiration
            all_opportunities = []
            
            for exp in valid_expirations:
                opportunities = self._scan_expiration(symbol, exp, current_price, historical_volatility)
                all_opportunities.extend(opportunities)
            
            self.logger.info(f"Found {len(all_opportunities)} potential opportunities for {symbol}")
            return all_opportunities
            
        except Exception as e:
            self.logger.error(f"Error scanning {symbol}: {str(e)}")
            return []
    
    def _scan_expiration(self, symbol, expiration, current_price, historical_volatility):
        """
        Scan a specific expiration date for a symbol.
        
        Args:
            symbol (str): Stock symbol
            expiration (str): Expiration date in YYYY-MM-DD format
            current_price (float): Current stock price
            historical_volatility (float): Historical volatility
            
        Returns:
            list: List of opportunity dictionaries
        """
        try:
            # Get option chain
            chain_df = self.tradier_api.get_option_chains(symbol, expiration)
            
            if chain_df.empty:
                self.logger.warning(f"No option chain found for {symbol} {expiration}")
                return []
            
            # Calculate time to expiration in years
            exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            today = datetime.now().date()
            days_to_exp = (exp_date - today).days
            T = days_to_exp / 365.0
            
            # Get risk-free rate (can use a fixed value or retrieve from a data source)
            r = self.option_pricer.config["black_scholes"]["parameters"].get("risk_free_rate", 0.05)
            
            # Filter strikes based on config
            atm_strike_idx = (chain_df['strike'] - current_price).abs().idxmin()
            chain_df['atm_distance'] = chain_df.index - atm_strike_idx
            
            filtered_chain = chain_df[
                (chain_df['atm_distance'].isin(self.focus_on_strikes)) &
                (chain_df['volume'] >= self.min_volume) &
                (chain_df['open_interest'] >= self.min_open_interest) &
                (chain_df['bid'] >= self.min_price) &
                (chain_df['ask'] <= self.max_price)
            ]
            
            if filtered_chain.empty:
                self.logger.info(f"No options meeting criteria for {symbol} {expiration}")
                return []
            
            # Evaluate each option
            opportunities = []
            
            for _, option in filtered_chain.iterrows():
                try:
                    # Extract option details
                    strike = option['strike']
                    option_type = OptionType.CALL if option['option_type'].lower() == 'call' else OptionType.PUT
                    
                    # Get mid price
                    bid = option['bid']
                    ask = option['ask']
                    mid_price = (bid + ask) / 2
                    
                    # Use implied volatility if available, otherwise use historical
                    if 'greeks' in option and option['greeks'] is not None:
                        if 'mid_iv' in option['greeks'] and option['greeks']['mid_iv'] > 0:
                            sigma = option['greeks']['mid_iv']
                        else:
                            sigma = historical_volatility
                    else:
                        sigma = historical_volatility
                    
                    # Use implied volatility to calculate model price for better accuracy
                    model_price_detailed = self.option_pricer.calculate_option_price(
                        current_price, strike, T, r, sigma, option_type, detailed=True
                    )
                    
                    model_price = model_price_detailed['weighted_price']
                    model_breakdown = model_price_detailed['model_prices']
                    
                    # Classify the option
                    classification, diff_percent = self.option_pricer.classify_option(mid_price, model_price)
                    
                    # Only consider undervalued options for potential trades
                    if classification == self.option_pricer.UNDERVALUED:
                        # Calculate Greeks
                        greeks = self.option_pricer.calculate_greeks(
                            current_price, strike, T, r, sigma, option_type
                        )
                        
                        # Create opportunity object
                        opportunity = {
                            'symbol': symbol,
                            'underlying_price': current_price,
                            'option_symbol': option['symbol'],
                            'expiration': expiration,
                            'days_to_expiration': days_to_exp,
                            'strike': strike,
                            'option_type': option_type.value,
                            'bid': bid,
                            'ask': ask,
                            'mid_price': mid_price,
                            'model_price': model_price,
                            'model_breakdown': model_breakdown,
                            'diff_percent': diff_percent,
                            'implied_volatility': sigma,
                            'volume': option['volume'],
                            'open_interest': option['open_interest'],
                            'delta': greeks['delta'],
                            'gamma': greeks['gamma'],
                            'theta': greeks['theta'],
                            'vega': greeks['vega'],
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        opportunities.append(opportunity)
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating option {option['symbol']}: {str(e)}")
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error scanning expiration {symbol} {expiration}: {str(e)}")
            return []
    
    def _get_historical_volatility(self, symbol, lookback_days=30):
        """
        Calculate historical volatility for a symbol.
        
        Args:
            symbol (str): Stock symbol
            lookback_days (int): Number of days to look back
            
        Returns:
            float: Annualized historical volatility
        """
        try:
            # Get historical daily data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days * 2)).strftime('%Y-%m-%d')
            
            historical_data = self.tradier_api.get_historical_data(
                symbol, interval='daily', start_date=start_date, end_date=end_date
            )
            
            if historical_data.empty or len(historical_data) < 5:
                self.logger.warning(f"Insufficient historical data for {symbol}, using default volatility")
                return 0.3  # Default volatility if data is insufficient
            
            # Calculate log returns
            historical_data = historical_data.sort_values('date')
            historical_data['log_return'] = np.log(historical_data['close'] / historical_data['close'].shift(1))
            
            # Calculate annualized standard deviation of log returns
            volatility = historical_data['log_return'].std() * np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility for {symbol}: {str(e)}")
            return 0.3  # Default volatility if calculation fails
    
    def scan_watchlist(self, symbols):
        """
        Scan multiple symbols in parallel.
        
        Args:
            symbols (list): List of stock symbols to scan
            
        Returns:
            list: Combined list of opportunities across all symbols
        """
        self.logger.info(f"Scanning watchlist with {len(symbols)} symbols")
        
        all_opportunities = []
        
        # Use ThreadPoolExecutor for parallel scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            future_to_symbol = {executor.submit(self.scan_ticker, symbol): symbol for symbol in symbols}
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    opportunities = future.result()
                    all_opportunities.extend(opportunities)
                except Exception as e:
                    self.logger.error(f"Error in watchlist scan for {symbol}: {str(e)}")
        
        # Sort by potential value (larger negative diff_percent means more undervalued)
        all_opportunities.sort(key=lambda x: x['diff_percent'])
        
        self.logger.info(f"Completed watchlist scan, found {len(all_opportunities)} total opportunities")
        return all_opportunities
    
    def run_continuous_scanner(self, symbols, callback=None, interval_seconds=None):
        """
        Run the scanner continuously at the specified interval.
        
        Args:
            symbols (list): List of stock symbols to scan
            callback (function, optional): Function to call with new opportunities
            interval_seconds (int, optional): Scan interval in seconds (overrides config)
            
        Returns:
            None
        """
        if interval_seconds is None:
            interval_seconds = self.config.get("scan_interval_seconds", 60)
        
        self.logger.info(f"Starting continuous scanner with {len(symbols)} symbols, interval: {interval_seconds}s")
        
        try:
            while True:
                start_time = time.time()
                
                # Scan watchlist
                opportunities = self.scan_watchlist(symbols)
                
                # Call callback if provided
                if callback and opportunities:
                    callback(opportunities)
                
                # Calculate sleep time to maintain consistent interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                
                if sleep_time > 0:
                    self.logger.debug(f"Sleeping for {sleep_time:.2f}s until next scan")
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Continuous scanner stopped by user")
        except Exception as e:
            self.logger.error(f"Error in continuous scanner: {str(e)}")
    
    def filter_opportunities(self, opportunities, max_results=10, min_volume=None, min_open_interest=None):
        """
        Filter and sort opportunities based on additional criteria.
        
        Args:
            opportunities (list): List of opportunity dictionaries
            max_results (int): Maximum number of results to return
            min_volume (int, optional): Minimum volume filter
            min_open_interest (int, optional): Minimum open interest filter
            
        Returns:
            list: Filtered list of opportunities
        """
        if not opportunities:
            return []
        
        # Apply additional filters if specified
        filtered = opportunities.copy()
        
        if min_volume is not None:
            filtered = [op for op in filtered if op['volume'] >= min_volume]
            
        if min_open_interest is not None:
            filtered = [op for op in filtered if op['open_interest'] >= min_open_interest]
        
        # Sort by value (more undervalued first)
        filtered.sort(key=lambda x: x['diff_percent'])
        
        # Limit results
        return filtered[:max_results]
