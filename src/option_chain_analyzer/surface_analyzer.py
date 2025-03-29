"""
Volatility Surface Analyzer Module

This module analyzes volatility surfaces to detect patterns and skews
that may indicate future price movements or trading opportunities.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import os

class SurfaceAnalyzer:
    """
    Analyzes option volatility surfaces and skews to identify trading signals.
    """
    
    def __init__(self, tradier_api):
        """
        Initialize the SurfaceAnalyzer.
        
        Args:
            tradier_api: Instance of TradierAPI for market data
        """
        self.logger = logging.getLogger(__name__)
        self.tradier_api = tradier_api
        
        # Create directory for surface visualizations
        os.makedirs("data/volatility_surfaces", exist_ok=True)
        
        self.logger.info("SurfaceAnalyzer initialized")
    
    def analyze_surface(self, symbol):
        """
        Analyze volatility surface for a symbol.
        
        Args:
            symbol (str): Symbol to analyze
            
        Returns:
            dict: Volatility surface analysis
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
            
            # Extract implied volatility values from option chain
            surface_data = self._extract_surface_data(option_chain, underlying_price)
            
            if surface_data.empty:
                self.logger.warning(f"Unable to extract volatility surface data for {symbol}")
                return self._default_result(symbol)
            
            # Analyze the surface
            term_structure = self._analyze_term_structure(surface_data)
            skew_analysis = self._analyze_volatility_skew(surface_data, underlying_price)
            wing_analysis = self._analyze_volatility_wings(surface_data, underlying_price)
            smile_analysis = self._analyze_volatility_smile(surface_data, underlying_price)
            
            # Generate 3D visualization
            surface_plot_path = self._generate_surface_plot(surface_data, symbol, underlying_price)
            
            # Combine all analyses
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'underlying_price': underlying_price,
                'term_structure': term_structure,
                'skew_analysis': skew_analysis,
                'wing_analysis': wing_analysis,
                'smile_analysis': smile_analysis,
                'surface_plot': surface_plot_path,
                'signals': self._determine_signals(term_structure, skew_analysis, wing_analysis, smile_analysis)
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility surface for {symbol}: {str(e)}")
            return self._default_result(symbol)
    
    def _default_result(self, symbol):
        """
        Return default result when surface analysis fails.
        
        Args:
            symbol (str): Symbol requested
            
        Returns:
            dict: Default analysis result
        """
        return {
            'symbol': symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'underlying_price': 0.0,
            'term_structure': {'slope': 0, 'shape': 'flat'},
            'skew_analysis': {'put_call_skew': 0, 'shape': 'neutral'},
            'wing_analysis': {'left_wing': 0, 'right_wing': 0},
            'smile_analysis': {'curvature': 0, 'smile_factor': 0},
            'surface_plot': None,
            'signals': []
        }
