"""
Option Chain Pattern Detector Module

This module detects patterns across option chains that may indicate
future price movements or trading opportunities.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math

class PatternDetector:
    """
    Detects significant patterns in option chains that can signal
    trading opportunities or unusual activity.
    """
    
    def __init__(self, tradier_api):
        """
        Initialize the PatternDetector.
        
        Args:
            tradier_api: Instance of TradierAPI for market data
        """
        self.logger = logging.getLogger(__name__)
        self.tradier_api = tradier_api
        
        # Pattern detection thresholds
        self.unusual_volume_threshold = 3.0      # Volume > 3x average
        self.volume_skew_threshold = 2.0         # Volume ratio between strikes
        self.open_interest_change_threshold = 0.5 # 50% change in OI
        self.vertical_spread_threshold = 0.3     # 30% price difference for vertical spreads
        self.wing_skew_threshold = 0.5          # 50% difference in implied volatility
        
        # Known patterns and their interpretations
        self.known_patterns = {
            'whale_call_buying': 'bullish',
            'whale_put_buying': 'bearish',
            'call_wall': 'resistance',
            'put_wall': 'support',
            'synthetic_call': 'bullish',
            'synthetic_put': 'bearish',
            'iron_condor': 'range-bound',
            'call_butterfly': 'targeted_bullish',
            'put_butterfly': 'targeted_bearish',
            'risk_reversal_bullish': 'strongly_bullish',
            'risk_reversal_bearish': 'strongly_bearish',
            'long_strangle': 'volatile',
            'long_straddle': 'volatile'
        }
        
        # Cache for recent option chain data
        self.chain_cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=15)
        
        self.logger.info("PatternDetector initialized")
    
    def detect_patterns(self, symbol):
        """
        Detect option chain patterns for a symbol.
        
        Args:
            symbol (str): Symbol to analyze
            
        Returns:
            list: Detected patterns and their details
        """
        try:
            # Get option chain data
            option_chain = self._get_option_chain(symbol)
            
            if option_chain.empty:
                self.logger.warning(f"No option chain data available for {symbol}")
                return []
            
            # Get underlying data
            quotes = self.tradier_api.get_quotes(symbol)
            if quotes.empty:
                self.logger.warning(f"No quote data available for {symbol}")
                return []
            
            underlying_price = quotes.iloc[0]['last']
            
            # Run all pattern detection algorithms
            detected_patterns = []
            
            # High-volume patterns
            volume_patterns = self._detect_volume_patterns(option_chain, underlying_price)
            detected_patterns.extend(volume_patterns)
            
            # Strike clustering patterns
            strike_patterns = self._detect_strike_clustering(option_chain, underlying_price)
            detected_patterns.extend(strike_patterns)
            
            # Volatility smile/skew patterns
            vol_patterns = self._detect_volatility_patterns(option_chain, underlying_price)
            detected_patterns.extend(vol_patterns)
            
            # Spread patterns
            spread_patterns = self._detect_spread_patterns(option_chain, underlying_price)
            detected_patterns.extend(spread_patterns)
            
            return detected_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns for {symbol}: {str(e)}")
            return []
    
    def _get_option_chain(self, symbol):
        """
        Get option chain data, using cache if available.
        
        Args:
            symbol (str): Symbol to get data for
            
        Returns:
            pd.DataFrame: Option chain data
        """
        # Check cache
        if (symbol in self.chain_cache and 
            symbol in self.cache_expiry and 
            datetime.now() < self.cache_expiry[symbol]):
            return self.chain_cache[symbol]
        
        # Fetch new data
        option_chain = self.tradier_api.get_option_chains(symbol)
        
        # Update cache
        if not option_chain.empty:
            self.chain_cache[symbol] = option_chain
            self.cache_expiry[symbol] = datetime.now() + self.cache_duration
        
        return option_chain
    
    def _detect_volume_patterns(self, option_chain, underlying_price):
        """
        Detect patterns based on unusual option volume.
        
        Args:
            option_chain (pd.DataFrame): Option chain data
            underlying_price (float): Current underlying price
            
        Returns:
            list: Detected volume-based patterns
        """
        patterns = []
        
        # Skip if empty chain
        if option_chain.empty:
            return patterns
        
        try:
            # Filter to active options with volume
            active_options = option_chain[option_chain['volume'] > 0]
            
            if active_options.empty:
                return patterns
            
            # Group by expiration date and option type
            for expiration, exp_group in active_options.groupby('expiration'):
                for option_type, type_group in exp_group.groupby('option_type'):
                    # Skip if no data
                    if type_group.empty:
                        continue
                    
                    # Find options with unusual volume
                    avg_volume = type_group['volume'].mean()
                    high_volume = type_group[type_group['volume'] > avg_volume * self.unusual_volume_threshold]
                    
                    # Process high volume options
                    for _, option in high_volume.iterrows():
                        strike = option['strike']
                        volume = option['volume']
                        open_interest = option.get('open_interest', 0)
                        
                        # Calculate volume to open interest ratio
                        vol_oi_ratio = volume / open_interest if open_interest > 0 else float('inf')
                        
                        # Determine trade type (buy/sell) based on trade price if available
                        if 'bid' in option and 'ask' in option and 'last' in option:
                            mid_price = (option['bid'] + option['ask']) / 2
                            # If last trade closer to ask, likely buyer initiated
                            trade_type = 'buy' if option['last'] >= mid_price else 'sell'
                        else:
                            trade_type = 'unknown'
                        
                        # Check for whale patterns (large positions)
                        if volume >= 1000 and vol_oi_ratio >= 0.25:
                            pattern_type = ""
                            if option_type == 'call' and trade_type == 'buy':
                                pattern_type = 'whale_call_buying'
                                signal = 'bullish'
                            elif option_type == 'put' and trade_type == 'buy':
                                pattern_type = 'whale_put_buying'
                                signal = 'bearish'
                            elif option_type == 'call' and trade_type == 'sell':
                                pattern_type = 'whale_call_selling'
                                signal = 'bearish'
                            elif option_type == 'put' and trade_type == 'sell':
                                pattern_type = 'whale_put_selling'
                                signal = 'bullish'
                            
                            if pattern_type:
                                # Determine moneyness
                                if option_type == 'call':
                                    moneyness = "ITM" if strike < underlying_price else "OTM"
                                else:
                                    moneyness = "ITM" if strike > underlying_price else "OTM"
                                
                                patterns.append({
                                    'pattern': pattern_type,
                                    'signal': signal,
                                    'expiration': expiration,
                                    'strike': strike,
                                    'option_type': option_type,
                                    'volume': volume,
                                    'open_interest': open_interest,
                                    'vol_oi_ratio': vol_oi_ratio,
                                    'moneyness': moneyness,
                                    'confidence': self._calculate_confidence(volume, vol_oi_ratio, open_interest)
                                })
            
            # Check for volume walls (accumulation at specific strikes)
            for expiration, exp_group in active_options.groupby('expiration'):
                # Get call and put groups
                calls = exp_group[exp_group['option_type'] == 'call']
                puts = exp_group[exp_group['option_type'] == 'put']
                
                # Find call walls
                if not calls.empty:
                    call_volume_by_strike = calls.groupby('strike')['volume'].sum()
                    call_avg_volume = call_volume_by_strike.mean()
                    call_walls = call_volume_by_strike[call_volume_by_strike > call_avg_volume * 3]
                    
                    for strike, volume in call_walls.items():
                        # Call walls above price can act as resistance
                        if strike > underlying_price:
                            patterns.append({
                                'pattern': 'call_wall',
                                'signal': 'resistance',
                                'expiration': expiration,
                                'strike': strike,
                                'option_type': 'call',
                                'volume': volume,
                                'confidence': min(0.9, volume / (call_avg_volume * 5))
                            })
                
                # Find put walls
                if not puts.empty:
                    put_volume_by_strike = puts.groupby('strike')['volume'].sum()
                    put_avg_volume = put_volume_by_strike.mean()
                    put_walls = put_volume_by_strike[put_volume_by_strike > put_avg_volume * 3]
                    
                    for strike, volume in put_walls.items():
                        # Put walls below price can act as support
                        if strike < underlying_price:
                            patterns.append({
                                'pattern': 'put_wall',
                                'signal': 'support',
                                'expiration': expiration,
                                'strike': strike,
                                'option_type': 'put',
                                'volume': volume,
                                'confidence': min(0.9, volume / (put_avg_volume * 5))
                            })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting volume patterns: {str(e)}")
            return []
    
    def _detect_strike_clustering(self, option_chain, underlying_price):
        """
        Detect patterns based on clustering of activity at specific strikes.
        
        Args:
            option_chain (pd.DataFrame): Option chain data
            underlying_price (float): Current underlying price
            
        Returns:
            list: Detected clustering patterns
        """
        patterns = []
        
        # Skip if empty chain
        if option_chain.empty:
            return patterns
        
        try:
            # Filter to options with open interest
            active_options = option_chain[option_chain['open_interest'] > 0]
            
            if active_options.empty:
                return patterns
            
            # Group by expiration
            for expiration, exp_group in active_options.groupby('expiration'):
                # Get call and put groups
                calls = exp_group[exp_group['option_type'] == 'call']
                puts = exp_group[exp_group['option_type'] == 'put']
                
                # Check for max pain (strike with most combined OI * distance from strike)
                if not calls.empty and not puts.empty:
                    # Calculate dollar value exposed to expiration for each strike
                    all_strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
                    
                    pain_by_strike = {}
                    for potential_price in all_strikes:
                        total_pain = 0
                        
                        # Add call option pain (OI * distance ITM)
                        for _, call in calls.iterrows():
                            if potential_price > call['strike']:
                                # Call is ITM at this price, add pain
                                total_pain += call['open_interest'] * (potential_price - call['strike'])
                        
                        # Add put option pain (OI * distance ITM)
                        for _, put in puts.iterrows():
                            if potential_price < put['strike']:
                                # Put is ITM at this price, add pain
                                total_pain += put['open_interest'] * (put['strike'] - potential_price)
                        
                        pain_by_strike[potential_price] = total_pain
                    
                    # Find max pain point
                    if pain_by_strike:
                        max_pain_strike = min(pain_by_strike, key=pain_by_strike.get)
                        
                        # Only report if significant OI
                        if (not calls.empty and not puts.empty and
                            calls['open_interest'].sum() > 1000 and
                            puts['open_interest'].sum() > 1000):
                            
                            # Determine if price is expected to move toward max pain
                            signal = ""
                            if max_pain_strike < underlying_price:
                                signal = "bearish"
                            elif max_pain_strike > underlying_price:
                                signal = "bullish"
                            else:
                                signal = "neutral"
                            
                            patterns.append({
                                'pattern': 'max_pain',
                                'signal': signal,
                                'expiration': expiration,
                                'strike': max_pain_strike,
                                'current_price': underlying_price,
                                'distance_pct': abs(max_pain_strike / underlying_price - 1) * 100,
                                'total_oi': calls['open_interest'].sum() + puts['open_interest'].sum(),
                                'confidence': 0.7  # Base confidence for max pain
                            })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting strike clustering patterns: {str(e)}")
            return []
    
    def _detect_volatility_patterns(self, option_chain, underlying_price):
        """
        Detect patterns based on implied volatility across the chain.
        
        Args:
            option_chain (pd.DataFrame): Option chain data
            underlying_price (float): Current underlying price
            
        Returns:
            list: Detected volatility patterns
        """
        patterns = []
        
        # Skip if empty chain
        if option_chain.empty:
            return patterns
        
        try:
            # Check if we have IV data
            if 'greeks' not in option_chain.columns:
                return patterns
            
            # Extract IV from Greeks
            option_chain['iv'] = option_chain['greeks'].apply(
                lambda x: x.get('mid_iv', None) if isinstance(x, dict) and x is not None else None)
            
            # Filter to options with IV data
            iv_options = option_chain.dropna(subset=['iv'])
            
            if iv_options.empty:
                return patterns
            
            # Group by expiration date
            for expiration, exp_group in iv_options.groupby('expiration'):
                # Get calls and puts
                calls = exp_group[exp_group['option_type'] == 'call']
                puts = exp_group[exp_group['option_type'] == 'put']
                
                # Check for volatility smile
                # A normal volatility smile has higher IVs for OTM options
                if not calls.empty:
                    # Add moneyness column
                    calls = calls.copy()
                    calls['moneyness'] = calls['strike'] / underlying_price
                    
                    # Get ATM call
                    atm_call = calls.iloc[(calls['moneyness'] - 1).abs().argsort()[0]]
                    atm_iv = atm_call['iv']
                    
                    # Check for wing skew (OTM calls)
                    otm_calls = calls[calls['strike'] > underlying_price]
                    if not otm_calls.empty:
                        # Get furthest OTM call with decent open interest
                        far_otm_calls = otm_calls[otm_calls['open_interest'] > 100]
                        if not far_otm_calls.empty:
                            far_otm_call = far_otm_calls.iloc[far_otm_calls['moneyness'].argsort()[-1]]
                            far_otm_iv = far_otm_call['iv']
                            
                            # Check for significant skew
                            if far_otm_iv / atm_iv > (1 + self.wing_skew_threshold):
                                patterns.append({
                                    'pattern': 'call_wing_skew',
                                    'signal': 'volatile_upside',
                                    'expiration': expiration,
                                    'atm_iv': atm_iv,
                                    'otm_iv': far_otm_iv,
                                    'skew_ratio': far_otm_iv / atm_iv,
                                    'confidence': min(0.9, (far_otm_iv / atm_iv - 1) / self.wing_skew_threshold)
                                })
                
                # Check for volatility skew in puts
                if not puts.empty:
                    # Add moneyness column
                    puts = puts.copy()
                    puts['moneyness'] = puts['strike'] / underlying_price
                    
                    # Get ATM put
                    atm_put = puts.iloc[(puts['moneyness'] - 1).abs().argsort()[0]]
                    atm_iv = atm_put['iv']
                    
                    # Check for wing skew (OTM puts)
                    otm_puts = puts[puts['strike'] < underlying_price]
                    if not otm_puts.empty:
                        # Get furthest OTM put with decent open interest
                        far_otm_puts = otm_puts[otm_puts['open_interest'] > 100]
                        if not far_otm_puts.empty:
                            far_otm_put = far_otm_puts.iloc[far_otm_puts['moneyness'].argsort()[0]]
                            far_otm_iv = far_otm_put['iv']
                            
                            # Check for significant skew
                            if far_otm_iv / atm_iv > (1 + self.wing_skew_threshold):
                                patterns.append({
                                    'pattern': 'put_wing_skew',
                                    'signal': 'volatile_downside',
                                    'expiration': expiration,
                                    'atm_iv': atm_iv,
                                    'otm_iv': far_otm_iv,
                                    'skew_ratio': far_otm_iv / atm_iv,
                                    'confidence': min(0.9, (far_otm_iv / atm_iv - 1) / self.wing_skew_threshold)
                                })
                
                # Check for volatility smile (both wings elevated)
                if not calls.empty and not puts.empty:
                    # Get ATM options
                    atm_call = calls.iloc[(calls['moneyness'] - 1).abs().argsort()[0]]
                    atm_put = puts.iloc[(puts['moneyness'] - 1).abs().argsort()[0]]
                    atm_iv = (atm_call['iv'] + atm_put['iv']) / 2
                    
                    # Get OTM options
                    otm_calls = calls[calls['strike'] > underlying_price * 1.1]
                    otm_puts = puts[puts['strike'] < underlying_price * 0.9]
                    
                    if not otm_calls.empty and not otm_puts.empty:
                        avg_otm_call_iv = otm_calls['iv'].mean()
                        avg_otm_put_iv = otm_puts['iv'].mean()
                        
                        # If both wings are elevated, it's a volatility smile
                        if avg_otm_call_iv > atm_iv * 1.2 and avg_otm_put_iv > atm_iv * 1.2:
                            patterns.append({
                                'pattern': 'volatility_smile',
                                'signal': 'expecting_large_move',
                                'expiration': expiration,
                                'atm_iv': atm_iv,
                                'otm_call_iv': avg_otm_call_iv,
                                'otm_put_iv': avg_otm_put_iv,
                                'confidence': min(0.9, (min(avg_otm_call_iv, avg_otm_put_iv) / atm_iv - 1) / 0.2)
                            })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility patterns: {str(e)}")
            return []
    
    def _detect_spread_patterns(self, option_chain, underlying_price):
        """
        Detect common spread patterns that might indicate directional bets.
        
        Args:
            option_chain (pd.DataFrame): Option chain data
            underlying_price (float): Current underlying price
            
        Returns:
            list: Detected spread patterns
        """
        patterns = []
        
        # Skip if empty chain
        if option_chain.empty:
            return patterns
        
        try:
            # Filter to active options
            active_options = option_chain[(option_chain['volume'] > 0) & (option_chain['open_interest'] > 0)]
            
            if active_options.empty:
                return patterns
            
            # Group by expiration date
            for expiration, exp_group in active_options.groupby('expiration'):
                # Get calls and puts
                calls = exp_group[exp_group['option_type'] == 'call']
                puts = exp_group[exp_group['option_type'] == 'put']
                
                # Detect vertical spreads in calls
                call_spreads = self._detect_vertical_spreads(calls, underlying_price, 'call')
                patterns.extend(call_spreads)
                
                # Detect vertical spreads in puts
                put_spreads = self._detect_vertical_spreads(puts, underlying_price, 'put')
                patterns.extend(put_spreads)
                
                # Detect straddles/strangles (volatility bets)
                vol_spreads = self._detect_volatility_spreads(calls, puts, underlying_price, expiration)
                patterns.extend(vol_spreads)
                
                # Detect iron condors (range-bound bets)
                condors = self._detect_iron_condors(calls, puts, underlying_price, expiration)
                patterns.extend(condors)
                
                # Detect risk reversals (strong directional bets)
                risk_reversals = self._detect_risk_reversals(calls, puts, underlying_price, expiration)
                patterns.extend(risk_reversals)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting spread patterns: {str(e)}")
            return []
    
    def _detect_vertical_spreads(self, options, underlying_price, option_type):
        """
        Detect vertical spread patterns.
        
        Args:
            options (pd.DataFrame): Option data for a specific type and expiration
            underlying_price (float): Current underlying price
            option_type (str): 'call' or 'put'
            
        Returns:
            list: Detected vertical spread patterns
        """
        spreads = []
        
        # Skip if not enough options
        if len(options) < 2:
            return spreads
        
        try:
            # Sort by strike price
            options = options.sort_values('strike')
            
            # Look for adjacent strikes with similar volume and open interest
            for i in range(len(options) - 1):
                lower = options.iloc[i]
                upper = options.iloc[i + 1]
                
                # Check if volumes are similar (could be spread legs)
                vol_ratio = lower['volume'] / upper['volume'] if upper['volume'] > 0 else float('inf')
                vol_ratio = min(vol_ratio, 1/vol_ratio) if vol_ratio > 0 else 0
                
                # Check if open interest values are similar
                oi_ratio = lower['open_interest'] / upper['open_interest'] if upper['open_interest'] > 0 else float('inf')
                oi_ratio = min(oi_ratio, 1/oi_ratio) if oi_ratio > 0 else 0
                
                # Both need decent volume to be valid
                if lower['volume'] > 50 and upper['volume'] > 50:
                    # Check if this could be a vertical spread
                    if vol_ratio > 0.6 and abs(lower['volume'] - upper['volume']) < 100:
                        # Determine spread type and direction
                        if option_type == 'call':
                            # Bull call spread: buy lower strike, sell higher strike
                            if lower['volume'] > upper['volume'] * 0.9:
                                spread_type = 'bull_call_spread'
                                signal = 'bullish'
                            # Bear call spread: sell lower strike, buy higher strike
                            else:
                                spread_type = 'bear_call_spread'
                                signal = 'bearish'
                        else:  # put
                            # Bull put spread: sell lower strike, buy higher strike
                            if lower['volume'] < upper['volume'] * 1.1:
                                spread_type = 'bull_put_spread'
                                signal = 'bullish'
                            # Bear put spread: buy lower strike, sell higher strike
                            else:
                                spread_type = 'bear_put_spread'
                                signal = 'bearish'
                        
                        # Calculate moneyness
                        lower_moneyness = lower['strike'] / underlying_price
                        upper_moneyness = upper['strike'] / underlying_price
                        
                        # Calculate confidence based on volume and OI matching
                        confidence = (vol_ratio + oi_ratio) / 2
                        
                        spreads.append({
                            'pattern': spread_type,
                            'signal': signal,
                            'expiration': lower['expiration'],
                            'lower_strike': lower['strike'],
                            'upper_strike': upper['strike'],
                            'spread_width': upper['strike'] - lower['strike'],
                            'volume': min(lower['volume'], upper['volume']),
                            'lower_moneyness': lower_moneyness,
                            'upper_moneyness': upper_moneyness,
                            'confidence': min(0.85, confidence)
                        })
            
            return spreads
            
        except Exception as e:
            self.logger.error(f"Error detecting vertical spreads: {str(e)}")
            return []
    
    def _detect_volatility_spreads(self, calls, puts, underlying_price, expiration):
        """
        Detect straddle and strangle patterns (volatility bets).
        
        Args:
            calls (pd.DataFrame): Call option data for a specific expiration
            puts (pd.DataFrame): Put option data for same expiration
            underlying_price (float): Current underlying price
            expiration (str): Expiration date
            
        Returns:
            list: Detected volatility spread patterns
        """
        spreads = []
        
        # Skip if either calls or puts is empty
        if calls.empty or puts.empty:
            return spreads
        
        try:
            # Find ATM or near-ATM options
            calls['moneyness'] = abs(calls['strike'] / underlying_price - 1)
            puts['moneyness'] = abs(puts['strike'] / underlying_price - 1)
            
            # Get closest to ATM options
            atm_call = calls.iloc[calls['moneyness'].argsort()[0]]
            atm_put = puts.iloc[puts['moneyness'].argsort()[0]]
            
            # Check for straddle (same strike)
            if abs(atm_call['strike'] - atm_put['strike']) < 0.01 * underlying_price:
                # Check if volumes are similar (could be straddle legs)
                vol_ratio = atm_call['volume'] / atm_put['volume'] if atm_put['volume'] > 0 else float('inf')
                vol_ratio = min(vol_ratio, 1/vol_ratio) if vol_ratio > 0 else 0
                
                # Both need decent volume to be valid
                if atm_call['volume'] > 50 and atm_put['volume'] > 50 and vol_ratio > 0.5:
                    spreads.append({
                        'pattern': 'long_straddle',
                        'signal': 'volatile',
                        'expiration': expiration,
                        'strike': atm_call['strike'],
                        'call_volume': atm_call['volume'],
                        'put_volume': atm_put['volume'],
                        'confidence': min(0.8, vol_ratio)
                    })
            
            # Check for strangle (different strikes)
            else:
                # Find OTM call and put with similar volume
                otm_calls = calls[calls['strike'] > underlying_price]
                otm_puts = puts[puts['strike'] < underlying_price]
                
                if not otm_calls.empty and not otm_puts.empty:
                    # Find closest OTM options to make a reasonable strangle
                    otm_call = otm_calls.iloc[0] if len(otm_calls) > 0 else None
                    otm_put = otm_puts.iloc[-1] if len(otm_puts) > 0 else None
                    
                    if otm_call is not None and otm_put is not None:
                        # Check if volumes are similar (could be strangle legs)
                        vol_ratio = otm_call['volume'] / otm_put['volume'] if otm_put['volume'] > 0 else float('inf')
                        vol_ratio = min(vol_ratio, 1/vol_ratio) if vol_ratio > 0 else 0
                        
                        # Both need decent volume to be valid
                        if otm_call['volume'] > 50 and otm_put['volume'] > 50 and vol_ratio > 0.5:
                            spreads.append({
                                'pattern': 'long_strangle',
                                'signal': 'volatile',
                                'expiration': expiration,
                                'call_strike': otm_call['strike'],
                                'put_strike': otm_put['strike'],
                                'call_volume': otm_call['volume'],
                                'put_volume': otm_put['volume'],
                                'width': otm_call['strike'] - otm_put['strike'],
                                'confidence': min(0.75, vol_ratio)
                            })
            
            return spreads
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility spreads: {str(e)}")
            return []
    
    def _detect_iron_condors(self, calls, puts, underlying_price, expiration):
        """
        Detect iron condor patterns (range-bound bets).
        
        Args:
            calls (pd.DataFrame): Call option data for a specific expiration
            puts (pd.DataFrame): Put option data for same expiration
            underlying_price (float): Current underlying price
            expiration (str): Expiration date
            
        Returns:
            list: Detected iron condor patterns
        """
        condors = []
        
        # Skip if either calls or puts is empty
        if calls.empty or puts.empty:
            return condors
        
        try:
            # Need at least 2 calls and 2 puts for condors
            if len(calls) < 2 or len(puts) < 2:
                return condors
            
            # Sort by strike
            calls = calls.sort_values('strike')
            puts = puts.sort_values('strike')
            
            # For a condor, we need:
            # 1. OTM put spread (sell higher strike, buy lower strike)
            # 2. OTM call spread (sell lower strike, buy higher strike)
            
            # Look for put spreads below current price
            otm_puts = puts[puts['strike'] < underlying_price]
            if len(otm_puts) >= 2:
                for i in range(len(otm_puts) - 1):
                    lower_put = otm_puts.iloc[i]
                    upper_put = otm_puts.iloc[i + 1]
                    
                    # Potential put spread legs
                    if lower_put['volume'] > 20 and upper_put['volume'] > 20:
                        # Look for call spreads above current price
                        otm_calls = calls[calls['strike'] > underlying_price]
                        if len(otm_calls) >= 2:
                            for j in range(len(otm_calls) - 1):
                                lower_call = otm_calls.iloc[j]
                                upper_call = otm_calls.iloc[j + 1]
                                
                                # Potential call spread legs
                                if lower_call['volume'] > 20 and upper_call['volume'] > 20:
                                    # Check if all four options have similar volume (potential iron condor)
                                    volumes = [lower_put['volume'], upper_put['volume'], 
                                              lower_call['volume'], upper_call['volume']]
                                    
                                    # If volumes are reasonably close, it might be an iron condor
                                    if max(volumes) < min(volumes) * 3:
                                        condors.append({
                                            'pattern': 'iron_condor',
                                            'signal': 'range_bound',
                                            'expiration': expiration,
                                            'put_spread_width': upper_put['strike'] - lower_put['strike'],
                                            'call_spread_width': upper_call['strike'] - lower_call['strike'],
                                            'lower_range': upper_put['strike'],
                                            'upper_range': lower_call['strike'],
                                            'range_width': lower_call['strike'] - upper_put['strike'],
                                            'volumes': volumes,
                                            'confidence': 0.7  # Base confidence for condors
                                        })
            
            return condors
            
        except Exception as e:
            self.logger.error(f"Error detecting iron condors: {str(e)}")
            return []
    
    def _detect_risk_reversals(self, calls, puts, underlying_price, expiration):
        """
        Detect risk reversal patterns (strong directional bets).
        
        Args:
            calls (pd.DataFrame): Call option data for a specific expiration
            puts (pd.DataFrame): Put option data for same expiration
            underlying_price (float): Current underlying price
            expiration (str): Expiration date
            
        Returns:
            list: Detected risk reversal patterns
        """
        reversals = []
        
        # Skip if either calls or puts is empty
        if calls.empty or puts.empty:
            return reversals
        
        try:
            # Sort by moneyness
            calls['moneyness'] = calls['strike'] / underlying_price - 1
            puts['moneyness'] = 1 - puts['strike'] / underlying_price
            
            calls = calls.sort_values('moneyness')
            puts = puts.sort_values('moneyness')
            
            # Risk reversal is:
            # - Bullish: Sell OTM put, buy OTM call
            # - Bearish: Sell OTM call, buy OTM put
            
            # Look for similar OTM options (by % away from price)
            for call_idx in range(len(calls)):
                call = calls.iloc[call_idx]
                
                # Skip if not OTM
                if call['strike'] <= underlying_price:
                    continue
                
                # Only consider options with decent volume
                if call['volume'] < 30:
                    continue
                
                call_moneyness = call['moneyness']
                
                # Look for puts with similar moneyness
                for put_idx in range(len(puts)):
                    put = puts.iloc[put_idx]
                    
                    # Skip if not OTM
                    if put['strike'] >= underlying_price:
                        continue
                    
                    # Only consider options with decent volume
                    if put['volume'] < 30:
                        continue
                    
                    put_moneyness = put['moneyness']
                    
                    # Check if moneyness values are similar (within 10%)
                    if abs(call_moneyness - put_moneyness) < 0.1:
                        # Check volumes to determine likely direction
                        if call['volume'] > put['volume'] * 1.3:
                            # Buy call, sell put = bullish risk reversal
                            reversals.append({
                                'pattern': 'risk_reversal_bullish',
                                'signal': 'strongly_bullish',
                                'expiration': expiration,
                                'call_strike': call['strike'],
                                'put_strike': put['strike'],
                                'call_volume': call['volume'],
                                'put_volume': put['volume'],
                                'moneyness': (call_moneyness + put_moneyness) / 2,
                                'confidence': 0.7
                            })
                        elif put['volume'] > call['volume'] * 1.3:
                            # Buy put, sell call = bearish risk reversal
                            reversals.append({
                                'pattern': 'risk_reversal_bearish',
                                'signal': 'strongly_bearish',
                                'expiration': expiration,
                                'call_strike': call['strike'],
                                'put_strike': put['strike'],
                                'call_volume': call['volume'],
                                'put_volume': put['volume'],
                                'moneyness': (call_moneyness + put_moneyness) / 2,
                                'confidence': 0.7
                            })
            
            return reversals
            
        except Exception as e:
            self.logger.error(f"Error detecting risk reversals: {str(e)}")
            return []
    
    def _calculate_confidence(self, volume, vol_oi_ratio, open_interest):
        """
        Calculate confidence score for a pattern.
        
        Args:
            volume (int): Option volume
            vol_oi_ratio (float): Volume to open interest ratio
            open_interest (int): Option open interest
            
        Returns:
            float: Confidence score (0-1)
        """
        # Higher volume = higher confidence
        volume_score = min(0.5, volume / 2000)
        
        # Higher vol/oi ratio = higher confidence (unusual activity)
        ratio_score = min(0.3, vol_oi_ratio / 10)
        
        # Higher open interest = higher confidence (established position)
        oi_score = min(0.2, open_interest / 10000)
        
        return volume_score + ratio_score + oi_score
    
    def get_pattern_summary(self, patterns):
        """
        Get a summary of detected patterns.
        
        Args:
            patterns (list): List of detected patterns
            
        Returns:
            dict: Summary of patterns and signals
        """
        if not patterns:
            return {
                'count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'top_patterns': [],
                'overall_signal': 'neutral',
                'signal_strength': 0.0
            }
        
        # Count patterns by signal
        bullish_count = sum(1 for p in patterns if p.get('signal') in ['bullish', 'strongly_bullish'])
        bearish_count = sum(1 for p in patterns if p.get('signal') in ['bearish', 'strongly_bearish'])
        neutral_count = sum(1 for p in patterns if p.get('signal') not in ['bullish', 'strongly_bullish', 'bearish', 'strongly_bearish'])
        
        # Get top patterns by confidence
        top_patterns = sorted(patterns, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
        
        # Determine overall signal
        bullish_confidence = sum(p.get('confidence', 0) for p in patterns if p.get('signal') in ['bullish', 'strongly_bullish'])
        bearish_confidence = sum(p.get('confidence', 0) for p in patterns if p.get('signal') in ['bearish', 'strongly_bearish'])
        
        if bullish_confidence > bearish_confidence * 1.5:
            overall_signal = 'bullish'
            signal_strength = min(1.0, bullish_confidence / 5)
        elif bearish_confidence > bullish_confidence * 1.5:
            overall_signal = 'bearish'
            signal_strength = min(1.0, bearish_confidence / 5)
        else:
            overall_signal = 'neutral'
            signal_strength = 0.3  # Default for neutral
        
        return {
            'count': len(patterns),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'top_patterns': top_patterns,
            'overall_signal': overall_signal,
            'signal_strength': signal_strength
        }
