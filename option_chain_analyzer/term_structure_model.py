"""
Term Structure Model Module

This module analyzes option term structure across expirations to 
detect anomalies and trading opportunities.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

class TermStructureModel:
    """
    Analyzes option term structure to detect anomalies and potential trades.
    """
    
    def __init__(self, tradier_api):
        """
        Initialize the TermStructureModel.
        
        Args:
            tradier_api: Instance of TradierAPI for market data
        """
        self.logger = logging.getLogger(__name__)
        self.tradier_api = tradier_api
        
        # Create directory for term structure visualizations
        os.makedirs("data/term_structures", exist_ok=True)
        
        # Constants for analysis
        self.min_expirations = 3  # Minimum expirations needed for analysis
        self.normal_term_slope_min = -0.0005  # Per day, expressed in IV points
        self.normal_term_slope_max = 0.001    # Per day, expressed in IV points
        
        self.logger.info("TermStructureModel initialized")
    
    def analyze_term_structure(self, symbol):
        """
        Analyze option term structure for a symbol.
        
        Args:
            symbol (str): Symbol to analyze
            
        Returns:
            dict: Term structure analysis
        """
        try:
            # Get option chain data
            option_chain = self.tradier_api.get_option_chains(symbol)
            
            if option_chain.empty:
                self.logger.warning(f"No option chain data available for {symbol}")
                return self._default_result(symbol)
            
            # Get underlying quote
            quotes = self.tradier_api.get_quotes(symbol)
            if quotes.empty:
                self.logger.warning(f"No quote data available for {symbol}")
                return self._default_result(symbol)
            
            underlying_price = quotes.iloc[0]['last']
            
            # Extract term structure data
            term_data = self._extract_term_structure(option_chain, underlying_price)
            
            if term_data is None or len(term_data) < self.min_expirations:
                self.logger.warning(f"Insufficient term structure data for {symbol}")
                return self._default_result(symbol)
            
            # Analyze the term structure
            basic_analysis = self._analyze_basic_term_structure(term_data)
            anomalies = self._detect_term_anomalies(term_data)
            calendar_opportunities = self._find_calendar_opportunities(term_data)
            
            # Generate term structure visualization
            plot_path = self._generate_term_plot(term_data, symbol, underlying_price)
            
            # Combine all analyses
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'underlying_price': underlying_price,
                'term_structure': basic_analysis,
                'anomalies': anomalies,
                'calendar_opportunities': calendar_opportunities,
                'plot_path': plot_path,
                'term_data': term_data
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing term structure for {symbol}: {str(e)}")
            return self._default_result(symbol)
    
    def _default_result(self, symbol):
        """
        Return default result when term structure analysis fails.
        
        Args:
            symbol (str): Symbol requested
            
        Returns:
            dict: Default analysis result
        """
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'underlying_price': 0.0,
            'term_structure': {
                'shape': 'unknown',
                'slope': 0.0,
                'steepness': 'flat'
            },
            'anomalies': [],
            'calendar_opportunities': [],
            'plot_path': None,
            'term_data': []
        }
    
    def _extract_term_structure(self, option_chain, underlying_price):
        """
        Extract term structure data from option chain.
        
        Args:
            option_chain (pd.DataFrame): Option chain data
            underlying_price (float): Current price of underlying
            
        Returns:
            list: Term structure data points
        """
        try:
            # Check if we have Greek data
            if 'greeks' not in option_chain.columns:
                self.logger.warning("No Greeks data in option chain")
                return None
            
            # Extract implied volatility from Greeks
            option_chain['iv'] = option_chain['greeks'].apply(
                lambda x: x.get('mid_iv', None) if isinstance(x, dict) and x is not None else None)
            
            # Drop rows with missing IV
            valid_options = option_chain.dropna(subset=['iv'])
            
            if valid_options.empty:
                return None
            
            # Add moneyness column (strike / underlying price)
            valid_options['moneyness'] = valid_options['strike'] / underlying_price
            
            # Add days to expiration column
            valid_options['days_to_expiration'] = valid_options['expiration'].apply(
                lambda x: (datetime.strptime(x, '%Y-%m-%d').date() - datetime.now().date()).days)
            
            # Get at-the-money options (closest to moneyness = 1)
            atm_options = []
            
            # Group by option type and expiration
            for (option_type, expiration), group in valid_options.groupby(['option_type', 'expiration']):
                # Filter to meaningful expirations (positive days to expiration)
                if group['days_to_expiration'].iloc[0] <= 0:
                    continue
                
                # Find closest to ATM
                atm_idx = (group['moneyness'] - 1).abs().idxmin()
                atm_option = group.loc[atm_idx].copy()
                
                # Only include if reasonably close to ATM
                if abs(atm_option['moneyness'] - 1) <= 0.1:
                    atm_options.append(atm_option)
            
            # Aggregate IVs by expiration (average call and put if both available)
            term_structure = {}
            
            for option in atm_options:
                days = option['days_to_expiration']
                expiration = option['expiration']
                iv = option['iv']
                option_type = option['option_type']
                
                if expiration not in term_structure:
                    term_structure[expiration] = {
                        'days': days,
                        'call_iv': None,
                        'put_iv': None,
                        'avg_iv': None,
                        'expiration': expiration
                    }
                
                if option_type == 'call':
                    term_structure[expiration]['call_iv'] = iv
                else:
                    term_structure[expiration]['put_iv'] = iv
                
                # Update average IV
                call_iv = term_structure[expiration]['call_iv']
                put_iv = term_structure[expiration]['put_iv']
                
                if call_iv is not None and put_iv is not None:
                    term_structure[expiration]['avg_iv'] = (call_iv + put_iv) / 2
                elif call_iv is not None:
                    term_structure[expiration]['avg_iv'] = call_iv
                elif put_iv is not None:
                    term_structure[expiration]['avg_iv'] = put_iv
            
            # Convert to list and sort by days to expiration
            term_data = list(term_structure.values())
            term_data = [t for t in term_data if t['avg_iv'] is not None]
            term_data.sort(key=lambda x: x['days'])
            
            return term_data
            
        except Exception as e:
            self.logger.error(f"Error extracting term structure: {str(e)}")
            return None
    
    def _analyze_basic_term_structure(self, term_data):
        """
        Perform basic analysis of term structure shape and slope.
        
        Args:
            term_data (list): Term structure data points
            
        Returns:
            dict: Basic term structure analysis
        """
        try:
            if not term_data or len(term_data) < 2:
                return {'shape': 'unknown', 'slope': 0.0, 'steepness': 'flat'}
            
            # Extract x and y values for regression
            x = np.array([point['days'] for point in term_data])
            y = np.array([point['avg_iv'] for point in term_data])
            
            # Calculate linear regression slope
            if len(x) >= 2:
                # Linear fit
                linear_fit = np.polyfit(x, y, 1)
                slope = linear_fit[0]  # IV change per day
                
                # Determine shape (linear, convex, concave)
                if len(x) >= 3:
                    # Quadratic fit to check for curvature
                    quad_fit = np.polyfit(x, y, 2)
                    curvature = quad_fit[0]  # Second derivative coefficient
                    
                    if abs(curvature) < 0.000001:
                        shape = 'linear'
                    elif curvature > 0:
                        shape = 'convex'  # Curves upward
                    else:
                        shape = 'concave'  # Curves downward
                else:
                    shape = 'linear'
                
                # Determine steepness based on slope
                slope_pct = slope * 100  # Convert to percentage points per day
                
                if abs(slope_pct) < 0.01:
                    steepness = 'flat'
                elif slope_pct >= 0.01 and slope_pct < 0.05:
                    steepness = 'gentle_up'
                elif slope_pct >= 0.05:
                    steepness = 'steep_up'
                elif slope_pct <= -0.01 and slope_pct > -0.05:
                    steepness = 'gentle_down'
                else:
                    steepness = 'steep_down'
                
                return {
                    'shape': shape,
                    'slope': slope_pct,  # Percentage points per day
                    'steepness': steepness,
                    'r_squared': self._calculate_r_squared(x, y, linear_fit)
                }
            else:
                return {'shape': 'unknown', 'slope': 0.0, 'steepness': 'flat'}
                
        except Exception as e:
            self.logger.error(f"Error analyzing basic term structure: {str(e)}")
            return {'shape': 'unknown', 'slope': 0.0, 'steepness': 'flat'}
    
    def _calculate_r_squared(self, x, y, fit):
        """
        Calculate R-squared for a linear fit.
        
        Args:
            x (ndarray): X values
            y (ndarray): Y values
            fit (ndarray): Linear fit coefficients
            
        Returns:
            float: R-squared value
        """
        # Calculate predicted y values
        y_pred = fit[0] * x + fit[1]
        
        # Calculate R-squared
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        
        if ss_tot == 0:
            return 0
        
        return 1 - (ss_res / ss_tot)
    
    def _detect_term_anomalies(self, term_data):
        """
        Detect anomalies in the term structure.
        
        Args:
            term_data (list): Term structure data points
            
        Returns:
            list: Detected anomalies
        """
        anomalies = []
        
        try:
            if not term_data or len(term_data) < 3:
                return anomalies
            
            # Calculate the moving average differences
            for i in range(1, len(term_data) - 1):
                prev_point = term_data[i-1]
                curr_point = term_data[i]
                next_point = term_data[i+1]
                
                # Calculate expected IV based on neighboring points
                days_diff_prev = curr_point['days'] - prev_point['days']
                days_diff_next = next_point['days'] - curr_point['days']
                
                # If days are evenly spaced, simple average is fine
                if abs(days_diff_prev - days_diff_next) < 5:
                    expected_iv = (prev_point['avg_iv'] + next_point['avg_iv']) / 2
                else:
                    # Weighted average based on days distance
                    weight_prev = 1 / days_diff_prev
                    weight_next = 1 / days_diff_next
                    expected_iv = (prev_point['avg_iv'] * weight_prev + next_point['avg_iv'] * weight_next) / (weight_prev + weight_next)
                
                # Calculate deviation from expected
                deviation = curr_point['avg_iv'] - expected_iv
                deviation_pct = deviation / expected_iv
                
                # Check if deviation is significant (more than 10%)
                if abs(deviation_pct) > 0.1:
                    anomalies.append({
                        'expiration': curr_point['expiration'],
                        'days': curr_point['days'],
                        'actual_iv': curr_point['avg_iv'],
                        'expected_iv': expected_iv,
                        'deviation': deviation,
                        'deviation_pct': deviation_pct * 100,  # Convert to percentage
                        'type': 'elevated' if deviation > 0 else 'depressed'
                    })
            
            # Check for inverted segments (decreasing IV with longer expiration)
            for i in range(len(term_data) - 1):
                curr_point = term_data[i]
                next_point = term_data[i+1]
                
                if next_point['avg_iv'] < curr_point['avg_iv']:
                    # Inversion detected
                    inversion_amount = curr_point['avg_iv'] - next_point['avg_iv']
                    inversion_pct = inversion_amount / curr_point['avg_iv'] * 100
                    
                    # Only report if significant (more than 5%)
                    if inversion_pct > 5:
                        anomalies.append({
                            'expiration_start': curr_point['expiration'],
                            'expiration_end': next_point['expiration'],
                            'days_start': curr_point['days'],
                            'days_end': next_point['days'],
                            'iv_start': curr_point['avg_iv'],
                            'iv_end': next_point['avg_iv'],
                            'inversion_amount': inversion_amount,
                            'inversion_pct': inversion_pct,
                            'type': 'inversion'
                        })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting term anomalies: {str(e)}")
            return []
    
    def _find_calendar_opportunities(self, term_data):
        """
        Find calendar spread opportunities in the term structure.
        
        Args:
            term_data (list): Term structure data points
            
        Returns:
            list: Potential calendar spread opportunities
        """
        opportunities = []
        
        try:
            if not term_data or len(term_data) < 2:
                return opportunities
            
            # Calculate IV per day for each adjacent pair of expirations
            for i in range(len(term_data) - 1):
                near_point = term_data[i]
                far_point = term_data[i+1]
                
                days_diff = far_point['days'] - near_point['days']
                iv_diff = far_point['avg_iv'] - near_point['avg_iv']
                
                # IV points per day
                iv_per_day = iv_diff / days_diff if days_diff > 0 else 0
                
                # Check for calendar opportunities
                
                # Bullish calendar spreads (sell near, buy far) when term structure is steep
                if iv_per_day > 0.001:  # More than 0.1% per day
                    opportunities.append({
                        'type': 'calendar_spread',
                        'direction': 'long_calendar',
                        'near_expiration': near_point['expiration'],
                        'far_expiration': far_point['expiration'],
                        'near_days': near_point['days'],
                        'far_days': far_point['days'],
                        'near_iv': near_point['avg_iv'],
                        'far_iv': far_point['avg_iv'],
                        'iv_diff': iv_diff,
                        'iv_per_day': iv_per_day,
                        'strength': min(1.0, iv_per_day / 0.002)  # Scale 0-1 based on steepness
                    })
                
                # Bearish calendar spreads (buy near, sell far) when term structure is inverted
                if iv_per_day < -0.0005:  # Less than -0.05% per day
                    opportunities.append({
                        'type': 'calendar_spread',
                        'direction': 'short_calendar',
                        'near_expiration': near_point['expiration'],
                        'far_expiration': far_point['expiration'],
                        'near_days': near_point['days'],
                        'far_days': far_point['days'],
                        'near_iv': near_point['avg_iv'],
                        'far_iv': far_point['avg_iv'],
                        'iv_diff': iv_diff,
                        'iv_per_day': iv_per_day,
                        'strength': min(1.0, abs(iv_per_day) / 0.001)  # Scale 0-1 based on steepness
                    })
            
            # Look for butterfly opportunities (when middle expiration is out of line)
            for i in range(1, len(term_data) - 1):
                prev_point = term_data[i-1]
                curr_point = term_data[i]
                next_point = term_data[i+1]
                
                # Calculate expected IV for middle point based on linear interpolation
                total_days_span = next_point['days'] - prev_point['days']
                days_from_prev = curr_point['days'] - prev_point['days']
                
                if total_days_span > 0:
                    weight = days_from_prev / total_days_span
                    expected_iv = prev_point['avg_iv'] + weight * (next_point['avg_iv'] - prev_point['avg_iv'])
                    
                    # Check deviation
                    deviation = curr_point['avg_iv'] - expected_iv
                    deviation_pct = deviation / expected_iv * 100
                    
                    # If middle is significantly higher than expected, short middle butterfly
                    if deviation_pct > 10:
                        opportunities.append({
                            'type': 'time_butterfly',
                            'direction': 'short_middle',
                            'short_expiration': prev_point['expiration'],
                            'middle_expiration': curr_point['expiration'],
                            'long_expiration': next_point['expiration'],
                            'short_days': prev_point['days'],
                            'middle_days': curr_point['days'],
                            'long_days': next_point['days'],
                            'deviation': deviation,
                            'deviation_pct': deviation_pct,
                            'strength': min(1.0, deviation_pct / 20)  # Scale 0-1 based on deviation
                        })
                    
                    # If middle is significantly lower than expected, long middle butterfly
                    elif deviation_pct < -10:
                        opportunities.append({
                            'type': 'time_butterfly',
                            'direction': 'long_middle',
                            'short_expiration': prev_point['expiration'],
                            'middle_expiration': curr_point['expiration'],
                            'long_expiration': next_point['expiration'],
                            'short_days': prev_point['days'],
                            'middle_days': curr_point['days'],
                            'long_days': next_point['days'],
                            'deviation': deviation,
                            'deviation_pct': deviation_pct,
                            'strength': min(1.0, abs(deviation_pct) / 20)  # Scale 0-1 based on deviation
                        })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding calendar opportunities: {str(e)}")
            return []
    
    def _generate_term_plot(self, term_data, symbol, underlying_price):
        """
        Generate a plot of the term structure.
        
        Args:
            term_data (list): Term structure data points
            symbol (str): Symbol being analyzed
            underlying_price (float): Current price of underlying
            
        Returns:
            str: Path to saved plot image, or None if generation failed
        """
        try:
            if not term_data or len(term_data) < 2:
                return None
            
            # Extract data for plotting
            days = [point['days'] for point in term_data]
            avg_ivs = [point['avg_iv'] for point in term_data]
            call_ivs = [point.get('call_iv', None) for point in term_data]
            put_ivs = [point.get('put_iv', None) for point in term_data]
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Plot average IV
            plt.plot(days, avg_ivs, 'k-', marker='o', label='Average IV', linewidth=2)
            
            # Plot call and put IVs if available
            valid_call_points = [(d, iv) for d, iv in zip(days, call_ivs) if iv is not None]
            valid_put_points = [(d, iv) for d, iv in zip(days, put_ivs) if iv is not None]
            
            if valid_call_points:
                call_days, call_ivs_valid = zip(*valid_call_points)
                plt.plot(call_days, call_ivs_valid, 'b--', marker='^', label='Call IV', alpha=0.7)
            
            if valid_put_points:
                put_days, put_ivs_valid = zip(*valid_put_points)
                plt.plot(put_days, put_ivs_valid, 'r--', marker='v', label='Put IV', alpha=0.7)
            
            # Add labels and title
            plt.xlabel('Days to Expiration')
            plt.ylabel('Implied Volatility')
            plt.title(f'Term Structure for {symbol} (Price: ${underlying_price:.2f})')
            
            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f"data/term_structures/{symbol}_term_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error generating term structure plot: {str(e)}")
            return None
    
    def get_historical_term_structure(self, symbol, days=30):
        """
        Get historical term structure data for a symbol.
        
        Args:
            symbol (str): Symbol to analyze
            days (int): Number of days of history
            
        Returns:
            pd.DataFrame: Historical term structure data
        """
        # In a real implementation, this would query a database of historical
        # term structures. Here we just return a placeholder.
        return pd.DataFrame()
    
    def compare_to_vix_term_structure(self, term_data):
        """
        Compare equity option term structure to VIX futures term structure.
        
        Args:
            term_data (list): Term structure data points
            
        Returns:
            dict: Comparison metrics
        """
        # In a real implementation, this would compare to VIX futures
        # from a data source. Here we just return a placeholder.
        return {
            'correlation': 0.5,
            'relative_steepness': 1.0
        }
    
    def evaluate_term_trades(self, term_data, lookback_days=30):
        """
        Evaluate past calendar spread trades based on term structure.
        
        Args:
            term_data (list): Current term structure data
            lookback_days (int): Days to look back for evaluation
            
        Returns:
            dict: Trade evaluation metrics
        """
        # In a real implementation, this would analyze past trades
        # from a database. Here we just return a placeholder.
        return {
            'avg_pnl': 0.0,
            'win_rate': 0.5,
            'avg_duration': 14
        }
