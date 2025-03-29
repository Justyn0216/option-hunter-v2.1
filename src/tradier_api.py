"""
Tradier API Integration Module

This module handles communication with the Tradier API for retrieving
market data, option chains, and executing paper trades.
"""

import logging
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

class TradierAPI:
    """
    Tradier API client for accessing market data and option information.
    
    Provides methods to:
    - Get option chains
    - Get market quotes
    - Get historical data
    - Get time & sales data
    - Execute paper trades
    """
    
    # API endpoints
    BASE_URL_SANDBOX = "https://sandbox.tradier.com/v1"
    BASE_URL_PRODUCTION = "https://api.tradier.com/v1"
    
    def __init__(self, api_key, account_id, sandbox_mode=True):
        """
        Initialize the Tradier API client.
        
        Args:
            api_key (str): Tradier API key
            account_id (str): Tradier account ID
            sandbox_mode (bool): If True, use sandbox environment
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.account_id = account_id
        self.sandbox_mode = sandbox_mode
        self.base_url = self.BASE_URL_SANDBOX if sandbox_mode else self.BASE_URL_PRODUCTION
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.25  # seconds between requests to avoid rate limiting
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }
        
        self.logger.info(f"Tradier API initialized in {'sandbox' if sandbox_mode else 'production'} mode")
    
    def _make_request(self, method, endpoint, params=None, data=None, retries=3):
        """
        Make an API request to Tradier with rate limiting and retries.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint path
            params (dict, optional): Query parameters
            data (dict, optional): Form data for POST requests
            retries (int): Number of retry attempts for failed requests
            
        Returns:
            dict: JSON response from the API
            
        Raises:
            Exception: If the request fails after all retries
        """
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(retries):
            try:
                self.last_request_time = time.time()
                
                if method.upper() == 'GET':
                    response = requests.get(url, headers=self.headers, params=params)
                elif method.upper() == 'POST':
                    response = requests.post(url, headers=self.headers, params=params, data=data)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    self.logger.error(f"Request failed after {retries} attempts: {str(e)}")
                    raise Exception(f"Tradier API request failed: {str(e)}")
                else:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
    
    def get_quotes(self, symbols):
        """
        Get current market quotes for one or more symbols.
        
        Args:
            symbols (str or list): Stock symbol(s) to get quotes for
            
        Returns:
            pandas.DataFrame: DataFrame with quote information
        """
        if isinstance(symbols, list):
            symbols = ','.join(symbols)
        
        params = {
            'symbols': symbols,
            'greeks': 'false'
        }
        
        try:
            response = self._make_request('GET', 'markets/quotes', params=params)
            
            if 'quotes' not in response or 'quote' not in response['quotes']:
                self.logger.warning(f"No quotes found for {symbols}")
                return pd.DataFrame()
            
            quotes = response['quotes']['quote']
            
            # Handle single quote response
            if not isinstance(quotes, list):
                quotes = [quotes]
                
            return pd.DataFrame(quotes)
            
        except Exception as e:
            self.logger.error(f"Error getting quotes for {symbols}: {str(e)}")
            return pd.DataFrame()
    
    def get_option_chains(self, symbol, expiration=None):
        """
        Get option chains for a symbol.
        
        Args:
            symbol (str): Stock symbol
            expiration (str, optional): Expiration date in YYYY-MM-DD format
                                        If None, gets nearest expiration
            
        Returns:
            pandas.DataFrame: DataFrame with option chain data
        """
        params = {
            'symbol': symbol,
            'greeks': 'true'
        }
        
        if expiration:
            params['expiration'] = expiration
        
        try:
            response = self._make_request('GET', 'markets/options/chains', params=params)
            
            if 'options' not in response or 'option' not in response['options']:
                self.logger.warning(f"No option chain found for {symbol}")
                return pd.DataFrame()
            
            options = response['options']['option']
            return pd.DataFrame(options)
            
        except Exception as e:
            self.logger.error(f"Error getting option chain for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_option_expirations(self, symbol):
        """
        Get available option expiration dates for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            list: List of expiration dates in YYYY-MM-DD format
        """
        params = {
            'symbol': symbol
        }
        
        try:
            response = self._make_request('GET', 'markets/options/expirations', params=params)
            
            if 'expirations' not in response or 'expiration' not in response['expirations']:
                self.logger.warning(f"No option expirations found for {symbol}")
                return []
            
            expirations = response['expirations']['expiration']
            if not isinstance(expirations, list):
                expirations = [expirations]
                
            # Return list of date strings
            return [exp['date'] for exp in expirations]
            
        except Exception as e:
            self.logger.error(f"Error getting option expirations for {symbol}: {str(e)}")
            return []
    
    def get_historical_data(self, symbol, interval='daily', start_date=None, end_date=None):
        """
        Get historical price data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            interval (str): Time interval ('daily', 'weekly', 'monthly' or 'minute')
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: DataFrame with historical data
        """
        # Default to last 30 days if no dates provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'start': start_date,
            'end': end_date
        }
        
        endpoint = 'markets/timesales' if interval == 'minute' else 'markets/history'
        
        try:
            response = self._make_request('GET', endpoint, params=params)
            
            # Handle different response formats for timesales vs history
            if interval == 'minute':
                if 'series' not in response or 'data' not in response['series']:
                    self.logger.warning(f"No timesales data found for {symbol}")
                    return pd.DataFrame()
                
                data = response['series']['data']
            else:
                if 'history' not in response or 'day' not in response['history']:
                    self.logger.warning(f"No historical data found for {symbol}")
                    return pd.DataFrame()
                
                data = response['history']['day']
            
            # Handle single data point response
            if not isinstance(data, list):
                data = [data]
                
            df = pd.DataFrame(data)
            
            # Convert date/time columns
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_time_and_sales(self, symbol, start=None, end=None, interval='1min'):
        """
        Get time and sales data (intraday trading data).
        
        Args:
            symbol (str): Stock symbol
            start (str, optional): Start time in format 'YYYY-MM-DD HH:MM'
            end (str, optional): End time in format 'YYYY-MM-DD HH:MM'
            interval (str): Interval ('1min', '5min', '15min')
            
        Returns:
            pandas.DataFrame: DataFrame with time and sales data
        """
        # Default to today if no dates provided
        if not start:
            start = datetime.now().replace(hour=9, minute=30).strftime('%Y-%m-%d %H:%M')
        if not end:
            end = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'start': start,
            'end': end,
            'session_filter': 'all'
        }
        
        try:
            response = self._make_request('GET', 'markets/timesales', params=params)
            
            if 'series' not in response or 'data' not in response['series']:
                self.logger.warning(f"No time and sales data found for {symbol}")
                return pd.DataFrame()
            
            data = response['series']['data']
            
            # Handle single data point response
            if not isinstance(data, list):
                data = [data]
                
            df = pd.DataFrame(data)
            
            # Convert time column
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting time and sales data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_option_strikes(self, symbol, expiration):
        """
        Get available option strike prices for a symbol and expiration.
        
        Args:
            symbol (str): Stock symbol
            expiration (str): Expiration date in YYYY-MM-DD format
            
        Returns:
            list: List of strike prices
        """
        params = {
            'symbol': symbol,
            'expiration': expiration
        }
        
        try:
            response = self._make_request('GET', 'markets/options/strikes', params=params)
            
            if 'strikes' not in response or 'strike' not in response['strikes']:
                self.logger.warning(f"No option strikes found for {symbol} {expiration}")
                return []
            
            strikes = response['strikes']['strike']
            if not isinstance(strikes, list):
                strikes = [strikes]
                
            return strikes
            
        except Exception as e:
            self.logger.error(f"Error getting option strikes for {symbol} {expiration}: {str(e)}")
            return []
    
    def get_option_quote(self, option_symbol):
        """
        Get quote for a specific option symbol.
        
        Args:
            option_symbol (str): Option symbol in Tradier format
            
        Returns:
            dict: Option quote data
        """
        params = {
            'symbols': option_symbol,
            'greeks': 'true'
        }
        
        try:
            response = self._make_request('GET', 'markets/quotes', params=params)
            
            if 'quotes' not in response or 'quote' not in response['quotes']:
                self.logger.warning(f"No quote found for option {option_symbol}")
                return {}
            
            quote = response['quotes']['quote']
            return quote
            
        except Exception as e:
            self.logger.error(f"Error getting quote for option {option_symbol}: {str(e)}")
            return {}
    
    def create_paper_order(self, symbol, side, quantity, order_type="market", price=None, stop=None, option_symbol=None):
        """
        Create a paper trading order.
        
        Args:
            symbol (str): Stock or option symbol
            side (str): 'buy' or 'sell'
            quantity (int): Number of shares/contracts
            order_type (str): 'market', 'limit', 'stop', 'stop_limit'
            price (float, optional): Limit price (required for limit and stop_limit orders)
            stop (float, optional): Stop price (required for stop and stop_limit orders)
            option_symbol (str, optional): Option symbol if trading options
            
        Returns:
            dict: Order status and information
        """
        if not self.sandbox_mode:
            self.logger.warning("Paper orders can only be created in sandbox mode")
            return {"status": "error", "message": "Not in sandbox mode"}
        
        data = {
            'class': 'option' if option_symbol else 'equity',
            'symbol': option_symbol if option_symbol else symbol,
            'side': side,
            'quantity': quantity,
            'type': order_type,
            'duration': 'day'
        }
        
        if price and order_type in ['limit', 'stop_limit']:
            data['price'] = price
            
        if stop and order_type in ['stop', 'stop_limit']:
            data['stop'] = stop
        
        try:
            response = self._make_request('POST', f'accounts/{self.account_id}/orders', data=data)
            return response
            
        except Exception as e:
            self.logger.error(f"Error creating paper order: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_account_positions(self):
        """
        Get current account positions.
        
        Returns:
            list: List of current positions
        """
        try:
            response = self._make_request('GET', f'accounts/{self.account_id}/positions')
            
            if 'positions' not in response:
                return []
                
            if response['positions'] == 'null' or 'position' not in response['positions']:
                return []
                
            positions = response['positions']['position']
            
            # Handle single position response
            if not isinstance(positions, list):
                positions = [positions]
                
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting account positions: {str(e)}")
            return []
    
    def get_account_balances(self):
        """
        Get account balances.
        
        Returns:
            dict: Account balance information
        """
        try:
            response = self._make_request('GET', f'accounts/{self.account_id}/balances')
            
            if 'balances' not in response:
                return {}
                
            return response['balances']
            
        except Exception as e:
            self.logger.error(f"Error getting account balances: {str(e)}")
            return {}
    
    def get_account_orders(self, status=None):
        """
        Get account orders.
        
        Args:
            status (str, optional): Filter by status ('open', 'filled', 'canceled')
            
        Returns:
            list: List of orders
        """
        params = {}
        if status:
            params['status'] = status
        
        try:
            response = self._make_request('GET', f'accounts/{self.account_id}/orders', params=params)
            
            if 'orders' not in response:
                return []
                
            if response['orders'] == 'null' or 'order' not in response['orders']:
                return []
                
            orders = response['orders']['order']
            
            # Handle single order response
            if not isinstance(orders, list):
                orders = [orders]
                
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting account orders: {str(e)}")
            return []
    
    def get_market_status(self):
        """
        Get the current market status.
        
        Returns:
            dict: Market status information
        """
        try:
            response = self._make_request('GET', 'markets/clock')
            
            if 'clock' not in response:
                return {}
                
            return response['clock']
            
        except Exception as e:
            self.logger.error(f"Error getting market status: {str(e)}")
            return {}
    
    def get_market_calendar(self, month=None, year=None):
        """
        Get the market calendar.
        
        Args:
            month (int, optional): Month (1-12)
            year (int, optional): Year (e.g., 2023)
            
        Returns:
            dict: Market calendar information
        """
        params = {}
        if month:
            params['month'] = month
        if year:
            params['year'] = year
        
        try:
            response = self._make_request('GET', 'markets/calendar', params=params)
            
            if 'calendar' not in response or 'days' not in response['calendar']:
                return {}
                
            return response['calendar']
            
        except Exception as e:
            self.logger.error(f"Error getting market calendar: {str(e)}")
            return {}
    
    def get_option_symbol(self, underlying, expiration_date, option_type, strike_price):
        """
        Generate a Tradier option symbol from components.
        
        Args:
            underlying (str): Underlying stock symbol
            expiration_date (str): Expiration date in format 'YYYY-MM-DD'
            option_type (str): 'call' or 'put'
            strike_price (float): Strike price
            
        Returns:
            str: Tradier option symbol
        """
        # Convert expiration date to OCC format (YYMMDD)
        expiration = datetime.strptime(expiration_date, '%Y-%m-%d')
        exp_format = expiration.strftime('%y%m%d')
        
        # Format strike price (multiply by 1000, no decimal)
        strike_format = str(int(float(strike_price) * 1000)).zfill(8)
        
        # Option type (C for call, P for put)
        opt_type = 'C' if option_type.lower() == 'call' else 'P'
        
        # Create the option symbol
        return f"{underlying}{exp_format}{opt_type}{strike_format}"
