"""
Option Pricing Module

This module implements multiple option pricing models:
- Black-Scholes
- Binomial Tree
- Merton Jump Diffusion
- Barone-Adesi-Whaley
- Monte Carlo Simulation

It uses weighted average pricing to determine if options are
undervalued, fairly priced, or overpriced.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import fsolve
from datetime import datetime
import math
import logging
from enum import Enum

class OptionType(Enum):
    """Enum for option types."""
    CALL = "call"
    PUT = "put"

class OptionPricer:
    """
    Multi-model option pricer that combines different pricing models
    to determine fair value and identify mispriced options.
    """
    
    # Constants for valuation categories
    UNDERVALUED = "undervalued"
    OVERVALUED = "overvalued"
    FAIRLY_PRICED = "fairly_priced"
    
    def __init__(self, config):
        """
        Initialize the OptionPricer with configuration.
        
        Args:
            config (dict): Configuration dictionary with pricing model settings
        """
        self.logger = logging.getLogger(__name__)
        self.config = config["pricing_models"]
        
        # Extract model weights and settings
        self.models = {}
        self.model_weights = {}
        
        if self.config["black_scholes"]["enabled"]:
            self.models["black_scholes"] = self.black_scholes
            self.model_weights["black_scholes"] = self.config["black_scholes"]["weight"]
            
        if self.config["binomial"]["enabled"]:
            self.models["binomial"] = self.binomial_tree
            self.model_weights["binomial"] = self.config["binomial"]["weight"]
            
        if self.config["merton_jump_diffusion"]["enabled"]:
            self.models["merton_jump_diffusion"] = self.merton_jump_diffusion
            self.model_weights["merton_jump_diffusion"] = self.config["merton_jump_diffusion"]["weight"]
            
        if self.config["barone_adesi_whaley"]["enabled"]:
            self.models["barone_adesi_whaley"] = self.barone_adesi_whaley
            self.model_weights["barone_adesi_whaley"] = self.config["barone_adesi_whaley"]["weight"]
            
        if self.config["monte_carlo"]["enabled"]:
            self.models["monte_carlo"] = self.monte_carlo
            self.model_weights["monte_carlo"] = self.config["monte_carlo"]["weight"]
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.model_weights.values())
        if weight_sum > 0:
            for model in self.model_weights:
                self.model_weights[model] /= weight_sum
        
        self.logger.info(f"OptionPricer initialized with {len(self.models)} models")
    
    def black_scholes(self, S, K, T, r, sigma, option_type):
        """
        Calculate option price using the Black-Scholes model.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (OptionType): Option type (CALL or PUT)
            
        Returns:
            float: Option price
        """
        # Add dividend yield from config if available
        q = self.config["black_scholes"]["parameters"].get("dividend_yield", 0.0)
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # PUT
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
        
        return price
    
    def binomial_tree(self, S, K, T, r, sigma, option_type):
        """
        Calculate option price using the Binomial Tree model.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (OptionType): Option type (CALL or PUT)
            
        Returns:
            float: Option price
        """
        # Extract parameters from config
        steps = self.config["binomial"]["parameters"].get("steps", 50)
        q = self.config["binomial"]["parameters"].get("dividend_yield", 0.0)
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Set up parameters for the binomial tree
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Initialize stock price tree
        stock_tree = np.zeros((steps + 1, steps + 1))
        for i in range(steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)
        
        # Initialize option value tree
        option_tree = np.zeros((steps + 1, steps + 1))
        
        # Calculate option value at expiration
        for j in range(steps + 1):
            if option_type == OptionType.CALL:
                option_tree[j, steps] = max(0, stock_tree[j, steps] - K)
            else:
                option_tree[j, steps] = max(0, K - stock_tree[j, steps])
        
        # Calculate option value at earlier times by backward induction
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                option_tree[j, i] = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
        
        return option_tree[0, 0]
    
    def merton_jump_diffusion(self, S, K, T, r, sigma, option_type):
        """
        Calculate option price using the Merton Jump Diffusion model.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (OptionType): Option type (CALL or PUT)
            
        Returns:
            float: Option price
        """
        # Extract parameters from config
        jump_intensity = self.config["merton_jump_diffusion"]["parameters"].get("jump_intensity", 0.1)
        jump_mean = self.config["merton_jump_diffusion"]["parameters"].get("jump_mean", -0.05)
        jump_std = self.config["merton_jump_diffusion"]["parameters"].get("jump_std", 0.1)
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Calculate jump component parameters
        lam = jump_intensity
        mu_j = jump_mean
        sigma_j = jump_std
        
        # Compute adjusted parameters
        gamma = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        r_adj = r - lam * gamma
        
        # Initialize price
        price = 0
        
        # Sum over number of jumps (truncate at reasonable number)
        max_jumps = 20
        for n in range(max_jumps):
            # Probability of n jumps
            p_n = np.exp(-lam * T) * (lam * T)**n / math.factorial(n)
            
            # Adjusted volatility for n jumps
            sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
            
            # Adjusted drift for n jumps
            r_n = r_adj + n * mu_j / T
            
            # Black-Scholes price for n jumps
            bs_price = self.black_scholes(S, K, T, r_n, sigma_n, option_type)
            
            # Add to total price
            price += p_n * bs_price
        
        return price
    
    def barone_adesi_whaley(self, S, K, T, r, sigma, option_type):
        """
        Calculate option price using the Barone-Adesi-Whaley model for American options.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (OptionType): Option type (CALL or PUT)
            
        Returns:
            float: Option price
        """
        # Extract parameters from config
        q = self.config["barone_adesi_whaley"]["parameters"].get("dividend_yield", 0.0)
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Get European option price using Black-Scholes
        european_price = self.black_scholes(S, K, T, r, sigma, option_type)
        
        # For very short-dated options, just use Black-Scholes
        if T < 0.1:
            return european_price
        
        # Constants for the quadratic approximation
        M = 2 * r / (sigma**2)
        N = 2 * (r - q) / (sigma**2)
        
        if option_type == OptionType.CALL:
            q2 = -(N - 1) + np.sqrt((N - 1)**2 + 4 * M / K)
            q2 = 0.5 * q2
            
            # Define critical stock price function for a call
            def call_critical_price(S_star):
                d1 = (np.log(S_star / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                return S_star - K - european_price + S_star * (1 - np.exp(-q * T) * stats.norm.cdf(d1)) / q2
            
            # Find critical stock price
            S_star = fsolve(call_critical_price, K)[0]
            
            # Calculate additional early exercise premium
            if S < S_star:
                return european_price
            else:
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                return S - K + (K - S_star) * (S / S_star)**q2
        
        else:  # PUT
            q1 = -(N - 1) - np.sqrt((N - 1)**2 + 4 * M / K)
            q1 = 0.5 * q1
            
            # Define critical stock price function for a put
            def put_critical_price(S_star):
                d1 = (np.log(S_star / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                return K - S_star - european_price - S_star * (1 - np.exp(-q * T) * stats.norm.cdf(-d1)) / q1
            
            # Find critical stock price
            S_star = fsolve(put_critical_price, K)[0]
            
            # Calculate additional early exercise premium
            if S > S_star:
                return european_price
            else:
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                return K - S + (S_star - K) * (S / S_star)**q1
    
    def monte_carlo(self, S, K, T, r, sigma, option_type):
        """
        Calculate option price using Monte Carlo simulation.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (OptionType): Option type (CALL or PUT)
            
        Returns:
            float: Option price
        """
        # Extract parameters from config
        simulations = self.config["monte_carlo"]["parameters"].get("simulations", 1000)
        q = self.config["monte_carlo"]["parameters"].get("dividend_yield", 0.0)
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate random paths
        Z = np.random.standard_normal(simulations)
        
        # Calculate terminal stock prices
        ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate option payoffs at expiration
        if option_type == OptionType.CALL:
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount payoffs to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        return option_price
    
    def calculate_option_price(self, S, K, T, r, sigma, option_type, detailed=False):
        """
        Calculate weighted average option price using all enabled pricing models.
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (OptionType or str): Option type (CALL or PUT)
            detailed (bool): If True, return breakdown by model
            
        Returns:
            float or dict: Weighted average option price or detailed breakdown
        """
        # Convert option_type from string if needed
        if isinstance(option_type, str):
            option_type = OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
        
        # Calculate price using each enabled model
        model_prices = {}
        
        try:
            for model_name, model_func in self.models.items():
                model_prices[model_name] = model_func(S, K, T, r, sigma, option_type)
            
            # Calculate weighted average
            weighted_price = 0
            for model_name, price in model_prices.items():
                weighted_price += price * self.model_weights[model_name]
            
            if detailed:
                return {
                    'weighted_price': weighted_price,
                    'model_prices': model_prices
                }
            else:
                return weighted_price
                
        except Exception as e:
            self.logger.error(f"Error calculating option price: {str(e)}")
            # Fall back to Black-Scholes if there's an error
            if 'black_scholes' in self.models:
                return self.black_scholes(S, K, T, r, sigma, option_type)
            raise
    
    def classify_option(self, market_price, model_price, threshold_percentage=5.0):
        """
        Classify an option as undervalued, overvalued, or fairly priced.
        
        Args:
            market_price (float): Current market price of the option
            model_price (float): Calculated theoretical price of the option
            threshold_percentage (float): Threshold for classification
            
        Returns:
            tuple: (classification, percentage_difference)
        """
        # Calculate percentage difference
        if model_price <= 0:
            return self.FAIRLY_PRICED, 0.0
            
        percentage_diff = ((market_price - model_price) / model_price) * 100
        
        # Classify based on threshold
        if percentage_diff < -threshold_percentage:
            return self.UNDERVALUED, percentage_diff
        elif percentage_diff > threshold_percentage:
            return self.OVERVALUED, percentage_diff
        else:
            return self.FAIRLY_PRICED, percentage_diff
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type):
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega).
        
        Args:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            sigma (float): Volatility
            option_type (OptionType or str): Option type (CALL or PUT)
            
        Returns:
            dict: Option Greeks
        """
        # Convert option_type from string if needed
        if isinstance(option_type, str):
            option_type = OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
        
        # Extract dividend yield from config
        q = self.config["black_scholes"]["parameters"].get("dividend_yield", 0.0)
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            return {
                'delta': 1.0 if option_type == OptionType.CALL else -1.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate Greeks
        if option_type == OptionType.CALL:
            delta = np.exp(-q * T) * stats.norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:  # PUT
            delta = -np.exp(-q * T) * stats.norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100
        
        # These are the same for both calls and puts
        gamma = np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta (per calendar day)
        term1 = -S * sigma * np.exp(-q * T) * stats.norm.pdf(d1) / (2 * np.sqrt(T))
        if option_type == OptionType.CALL:
            term2 = q * S * np.exp(-q * T) * stats.norm.cdf(d1)
            term3 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # PUT
            term2 = -q * S * np.exp(-q * T) * stats.norm.cdf(-d1)
            term3 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
        
        theta = (term1 + term2 + term3) / 365.0  # Convert to daily
        
        # Vega (for 1% change in volatility)
        vega = S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def implied_volatility(self, price, S, K, T, r, option_type, max_iterations=100, precision=0.00001):
        """
        Calculate implied volatility using the Newton-Raphson method.
        
        Args:
            price (float): Market price of the option
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to expiration in years
            r (float): Risk-free interest rate
            option_type (OptionType or str): Option type (CALL or PUT)
            max_iterations (int): Maximum number of iterations
            precision (float): Desired precision
            
        Returns:
            float: Implied volatility
        """
        # Convert option_type from string if needed
        if isinstance(option_type, str):
            option_type = OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT
        
        # Extract dividend yield from config
        q = self.config["black_scholes"]["parameters"].get("dividend_yield", 0.0)
        
        # Initial guess for implied volatility
        sigma = 0.2
        
        for i in range(max_iterations):
            # Calculate option price and vega
            price_diff = self.black_scholes(S, K, T, r, sigma, option_type) - price
            
            # If we're close enough to the target price, return current volatility
            if abs(price_diff) < precision:
                return sigma
            
            # Calculate vega
            vega = self.calculate_greeks(S, K, T, r, sigma, option_type)['vega'] * 100
            
            # Avoid division by zero
            if abs(vega) < 1e-10:
                sigma = sigma + 0.001
                continue
            
            # Update volatility estimate using Newton-Raphson
            sigma = sigma - price_diff / vega
            
            # Keep volatility in reasonable bounds
            sigma = max(0.001, min(sigma, 5.0))
        
        self.logger.warning(f"Implied volatility calculation did not converge after {max_iterations} iterations")
        return sigma
