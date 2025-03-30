"""
Risk Metrics Module

This module calculates various risk metrics for option trading strategies.
It provides tools for measuring and analyzing different types of risk,
including market risk, volatility risk, and tail risk.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class RiskMetrics:
    """
    Calculates and analyzes risk metrics for trading strategies.
    
    Features:
    - Value at Risk (VaR) calculation
    - Conditional Value at Risk (CVaR/Expected Shortfall)
    - Drawdown analysis
    - Tail risk analysis
    - Risk visualization
    """
    
    def __init__(self, config=None):
        """
        Initialize the RiskMetrics calculator.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.results_dir = "data/risk_metrics"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default parameters
        self.trading_days_per_year = self.config.get('trading_days_per_year', 252)
        
        self.logger.info("RiskMetrics initialized")
    
    def calculate_var(self, returns, confidence_level=0.95, method='historical'):
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns (pd.Series): Return series
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            method (str): Calculation method ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
            float: Value at Risk
        """
        if len(returns) < 10:
            self.logger.warning("Insufficient data for VaR calculation")
            return None
        
        alpha = 1 - confidence_level
        
        if method == 'historical':
            # Historical VaR (non-parametric)
            var = np.percentile(returns, alpha * 100)
            return abs(var)
            
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mu = returns.mean()
            sigma = returns.std()
            var = stats.norm.ppf(alpha, mu, sigma)
            return abs(var)
            
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            mu = returns.mean()
            sigma = returns.std()
            
            # Generate 10,000 random scenarios
            np.random.seed(42)
            simulations = 10000
            sim_returns = np.random.normal(mu, sigma, simulations)
            
            var = np.percentile(sim_returns, alpha * 100)
            return abs(var)
            
        else:
            self.logger.error(f"Unknown VaR method: {method}")
            return None
    
    def calculate_cvar(self, returns, confidence_level=0.95, method='historical'):
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns (pd.Series): Return series
            confidence_level (float): Confidence level (e.g., 0.95 for 95%)
            method (str): Calculation method ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
            float: Conditional Value at Risk
        """
        if len(returns) < 10:
            self.logger.warning("Insufficient data for CVaR calculation")
            return None
        
        alpha = 1 - confidence_level
        
        if method == 'historical':
            # Historical CVaR
            var = np.percentile(returns, alpha * 100)
            cvar = returns[returns <= var].mean()
            return abs(cvar)
            
        elif method == 'parametric':
            # Parametric CVaR (assuming normal distribution)
            mu = returns.mean()
            sigma = returns.std()
            var = stats.norm.ppf(alpha, mu, sigma)
            
            # Calculate CVaR
            cvar = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            return abs(cvar)
            
        elif method == 'monte_carlo':
            # Monte Carlo CVaR
            mu = returns.mean()
            sigma = returns.std()
            
            # Generate 10,000 random scenarios
            np.random.seed(42)
            simulations = 10000
            sim_returns = np.random.normal(mu, sigma, simulations)
            
            var = np.percentile(sim_returns, alpha * 100)
            cvar = sim_returns[sim_returns <= var].mean()
            return abs(cvar)
            
        else:
            self.logger.error(f"Unknown CVaR method: {method}")
            return None
    
    def calculate_drawdown_metrics(self, equity_curve):
        """
        Calculate drawdown metrics.
        
        Args:
            equity_curve (pd.Series): Equity curve
            
        Returns:
            dict: Drawdown metrics
        """
        if len(equity_curve) < 10:
            self.logger.warning("Insufficient data for drawdown calculation")
            return None
        
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        
        # No drawdowns
        if not is_drawdown.any():
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'avg_drawdown_duration': 0,
                'max_underwater_days': 0,
                'current_drawdown': 0.0,
                'time_to_recovery': 0.0
            }
        
        # Find start and end of each drawdown
        drawdown_start = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_end = ~is_drawdown & is_drawdown.shift(1).fillna(False)
        
        # Get start and end dates
        start_dates = equity_curve.index[drawdown_start]
        end_dates = equity_curve.index[drawdown_end]
        
        # If still in drawdown, add last date
        if len(start_dates) > len(end_dates):
            end_dates = pd.Index(list(end_dates) + [equity_curve.index[-1]])
        
        # Calculate drawdown durations
        durations = []
        max_drawdowns = []
        underwater_days = []
        
        for i in range(min(len(start_dates), len(end_dates))):
            start_date = start_dates[i]
            end_date = end_dates[i]
            
            # Get drawdown period
            period_drawdown = drawdown[start_date:end_date]
            
            # Calculate max drawdown in this period
            max_dd = period_drawdown.min()
            max_dd_date = period_drawdown.idxmin()
            
            # Duration
            duration = (end_date - start_date).days
            
            durations.append(duration)
            max_drawdowns.append(abs(max_dd))
            
            # Calculate underwater days from peak to recovery
            peak_date = running_max[running_max == running_max[max_dd_date]].index[0]
            underwater = (end_date - peak_date).days
            underwater_days.append(underwater)
        
        # Calculate metrics
        max_drawdown = abs(drawdown.min())
        avg_drawdown = np.mean(max_drawdowns)
        max_drawdown_duration = max(durations) if durations else 0
        avg_drawdown_duration = np.mean(durations) if durations else 0
        max_underwater = max(underwater_days) if underwater_days else 0
        
        # Calculate current drawdown
        current_drawdown = abs(drawdown.iloc[-1])
        
        # Calculate time to recovery based on average recovery rate
        if current_drawdown > 0:
            # Calculate average return during recovery periods
            recovery_returns = []
            
            for i in range(min(len(start_dates), len(end_dates))):
                end_date = end_dates[i]
                recovery_date = end_date + pd.Timedelta(days=1)
                
                if recovery_date in equity_curve.index:
                    # Calculate returns from drawdown end to recovery
                    recovery_period = 30  # days
                    recovery_end = min(recovery_date + pd.Timedelta(days=recovery_period), equity_curve.index[-1])
                    
                    if recovery_end > recovery_date:
                        recovery_return = (equity_curve[recovery_end] / equity_curve[recovery_date]) - 1
                        recovery_days = (recovery_end - recovery_date).days
                        
                        daily_recovery = recovery_return / recovery_days
                        recovery_returns.append(daily_recovery)
            
            avg_daily_recovery = np.mean(recovery_returns) if recovery_returns else 0.01  # default to 1% daily
            
            # Estimate time to recovery
            if avg_daily_recovery > 0:
                time_to_recovery = current_drawdown / avg_daily_recovery
            else:
                time_to_recovery = float('inf')
        else:
            time_to_recovery = 0.0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_underwater_days': max_underwater,
            'current_drawdown': current_drawdown,
            'time_to_recovery': time_to_recovery
        }
    
    def calculate_tail_risk_metrics(self, returns):
        """
        Calculate tail risk metrics.
        
        Args:
            returns (pd.Series): Return series
            
        Returns:
            dict: Tail risk metrics
        """
        if len(returns) < 30:
            self.logger.warning("Insufficient data for tail risk calculation")
            return None
        
        # Calculate moments
        mean = returns.mean()
        std = returns.std()
        skew = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Calculate downside deviation (semi-deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 1 else 0
        
        # Calculate VaR and CVaR at different confidence levels
        var_95 = self.calculate_var(returns, 0.95, 'historical')
        var_99 = self.calculate_var(returns, 0.99, 'historical')
        cvar_95 = self.calculate_cvar(returns, 0.95, 'historical')
        cvar_99 = self.calculate_cvar(returns, 0.99, 'historical')
        
        # Calculate tail ratio (ratio of right tail to left tail)
        right_tail = returns[returns > 0].std()
        left_tail = abs(downside_deviation)
        tail_ratio = right_tail / left_tail if left_tail > 0 else float('inf')
        
        # Calculate worst drawdown
        equity_curve = (1 + returns).cumprod()
        drawdown_metrics = self.calculate_drawdown_metrics(equity_curve)
        
        return {
            'mean': mean,
            'std': std,
            'skewness': skew,
            'kurtosis': kurtosis,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'tail_ratio': tail_ratio,
            'max_drawdown': drawdown_metrics['max_drawdown'] if drawdown_metrics else None
        }
    
    def calculate_risk_metrics(self, equity_curve, returns=None):
        """
        Calculate comprehensive risk metrics for a trading strategy.
        
        Args:
            equity_curve (pd.Series): Equity curve
            returns (pd.Series, optional): Return series (calculated from equity_curve if None)
            
        Returns:
            dict: Dictionary of risk metrics
        """
        self.logger.info("Calculating comprehensive risk metrics")
        
        # Convert equity curve to pandas Series if it's a dictionary
        if isinstance(equity_curve, dict):
            equity_curve = pd.Series(equity_curve)
            
        # Calculate returns if not provided
        if returns is None:
            returns = equity_curve.pct_change().dropna()
        
        # Calculate basic metrics
        metrics = {}
        
        # Volatility metrics
        metrics['volatility'] = returns.std() * np.sqrt(self.trading_days_per_year)
        metrics['annualized_downside_deviation'] = returns[returns < 0].std() * np.sqrt(self.trading_days_per_year) if len(returns[returns < 0]) > 1 else 0
        
        # VaR and CVaR at different confidence levels
        for cl in [0.95, 0.99]:
            metrics[f'daily_var_{int(cl*100)}'] = self.calculate_var(returns, cl, 'historical')
            metrics[f'daily_cvar_{int(cl*100)}'] = self.calculate_cvar(returns, cl, 'historical')
            
            # Annualized VaR and CVaR
            metrics[f'annual_var_{int(cl*100)}'] = metrics[f'daily_var_{int(cl*100)}'] * np.sqrt(self.trading_days_per_year)
            metrics[f'annual_cvar_{int(cl*100)}'] = metrics[f'daily_cvar_{int(cl*100)}'] * np.sqrt(self.trading_days_per_year)
        
        # Drawdown metrics
        drawdown_metrics = self.calculate_drawdown_metrics(equity_curve)
        if drawdown_metrics:
            metrics.update(drawdown_metrics)
        
        # Tail risk metrics
        tail_metrics = self.calculate_tail_risk_metrics(returns)
        if tail_metrics:
            # Only update with metrics not already calculated
            for key, value in tail_metrics.items():
                if key not in metrics:
                    metrics[key] = value
        
        # Calculate Ulcer Index
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        squared_drawdown = drawdown ** 2
        ulcer_index = np.sqrt(squared_drawdown.mean())
        metrics['ulcer_index'] = ulcer_index
        
        # Calculate Pain Index (average drawdown)
        pain_index = abs(drawdown.mean())
        metrics['pain_index'] = pain_index
        
        # Calculate Pain Ratio (return / pain)
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(equity_curve)) - 1
        pain_ratio = annualized_return / pain_index if pain_index > 0 else float('inf')
        metrics['pain_ratio'] = pain_ratio
        
        # Calculate Martin Ratio (return / ulcer)
        martin_ratio = annualized_return / ulcer_index if ulcer_index > 0 else float('inf')
        metrics['martin_ratio'] = martin_ratio
        
        # Visualize risk metrics
        self._visualize_risk_metrics(equity_curve, returns, metrics)
        
        self.logger.info("Risk metrics calculation completed")
        return metrics
    
    def _visualize_risk_metrics(self, equity_curve, returns, metrics):
        """
        Visualize risk metrics.
        
        Args:
            equity_curve (pd.Series): Equity curve
            returns (pd.Series): Return series
            metrics (dict): Risk metrics
            
        Returns:
            None
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(3, 2, figsize=(14, 18))
            
            # Plot equity curve with drawdowns
            running_max = equity_curve.cummax()
            axs[0, 0].plot(equity_curve.index, equity_curve.values, label='Equity')
            axs[0, 0].plot(running_max.index, running_max.values, 'r--', label='Running Maximum')
            axs[0, 0].set_title('Equity Curve and Maximum Drawdown', fontsize=14)
            axs[0, 0].set_xlabel('Date')
            axs[0, 0].set_ylabel('Equity')
            axs[0, 0].legend()
            
            # Plot drawdown
            drawdown = (equity_curve - running_max) / running_max
            axs[0, 1].fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
            axs[0, 1].axhline(y=-metrics['max_drawdown'], color='r', linestyle='--', 
                            label=f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            axs[0, 1].set_title('Drawdown', fontsize=14)
            axs[0, 1].set_xlabel('Date')
            axs[0, 1].set_ylabel('Drawdown (%)')
            axs[0, 1].legend()
            
            # Plot return distribution
            axs[1, 0].hist(returns.values, bins=50, alpha=0.75, color='skyblue')
            
            # Add normal distribution overlay
            x = np.linspace(returns.min(), returns.max(), 100)
            axs[1, 0].plot(x, stats.norm.pdf(x, returns.mean(), returns.std()) * len(returns) * (returns.max() - returns.min()) / 50,
                         'r-', linewidth=2)
            
            axs[1, 0].set_title('Return Distribution', fontsize=14)
            axs[1, 0].set_xlabel('Return')
            axs[1, 0].set_ylabel('Frequency')
            
            # Add VaR and CVaR lines
            var_95 = -metrics['daily_var_95']  # Negate to show on the left side
            cvar_95 = -metrics['daily_cvar_95']
            
            axs[1, 0].axvline(x=var_95, color='orange', linestyle='--', 
                           label=f"VaR 95%: {metrics['daily_var_95']:.2%}")
            axs[1, 0].axvline(x=cvar_95, color='red', linestyle='--', 
                           label=f"CVaR 95%: {metrics['daily_cvar_95']:.2%}")
            axs[1, 0].legend()
            
            # Plot QQ plot to check normality
            stats.probplot(returns.values, dist="norm", plot=axs[1, 1])
            axs[1, 1].set_title('Normal Q-Q Plot', fontsize=14)
            
            # Plot rolling volatility
            rolling_vol = returns.rolling(window=21).std() * np.sqrt(self.trading_days_per_year)
            axs[2, 0].plot(rolling_vol.index, rolling_vol.values)
            axs[2, 0].set_title('21-Day Rolling Annualized Volatility', fontsize=14)
            axs[2, 0].set_xlabel('Date')
            axs[2, 0].set_ylabel('Volatility')
            
            # Create a table with key risk metrics
            risk_table = {
                'Metric': [
                    'Annualized Volatility',
                    'Daily VaR (95%)',
                    'Daily CVaR (95%)',
                    'Maximum Drawdown',
                    'Avg Drawdown Duration',
                    'Ulcer Index',
                    'Pain Ratio',
                    'Skewness',
                    'Kurtosis'
                ],
                'Value': [
                    f"{metrics['volatility']:.2%}",
                    f"{metrics['daily_var_95']:.2%}",
                    f"{metrics['daily_cvar_95']:.2%}",
                    f"{metrics['max_drawdown']:.2%}",
                    f"{metrics['avg_drawdown_duration']:.1f} days",
                    f"{metrics['ulcer_index']:.2%}",
                    f"{metrics['pain_ratio']:.2f}",
                    f"{metrics['skewness']:.2f}",
                    f"{metrics['kurtosis']:.2f}"
                ]
            }
            
            table_df = pd.DataFrame(risk_table)
            axs[2, 1].axis('off')
            table = axs[2, 1].table(cellText=table_df.values, colLabels=table_df.columns,
                                 cellLoc='center', loc='center', bbox=[0.2, 0.2, 0.7, 0.7])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/risk_metrics_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Risk metrics visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing risk metrics: {str(e)}")
    
    def compare_risk_profiles(self, strategy_equity_curves, strategy_names=None, benchmark=None):
        """
        Compare risk profiles of multiple strategies.
        
        Args:
            strategy_equity_curves (dict): Dictionary mapping strategy names to equity curves
            strategy_names (list, optional): List of strategy names (if not provided, use dict keys)
            benchmark (pd.Series, optional): Benchmark equity curve
            
        Returns:
            dict: Comparison results
        """
        self.logger.info(f"Comparing risk profiles of {len(strategy_equity_curves)} strategies")
        
        # Use dict keys as strategy names if not provided
        if strategy_names is None:
            strategy_names = list(strategy_equity_curves.keys())
        
        # Calculate risk metrics for each strategy
        comparison = {}
        
        for i, (name, equity_curve) in enumerate(strategy_equity_curves.items()):
            if i < len(strategy_names):
                strategy_name = strategy_names[i]
            else:
                strategy_name = name
                
            metrics = self.calculate_risk_metrics(equity_curve)
            comparison[strategy_name] = metrics
        
        # Calculate benchmark metrics if provided
        if benchmark is not None:
            benchmark_metrics = self.calculate_risk_metrics(benchmark)
            comparison['Benchmark'] = benchmark_metrics
        
        # Create comparison visualization
        self._visualize_risk_comparison(comparison)
        
        return comparison
    
    def _visualize_risk_comparison(self, comparison):
        """
        Visualize risk profile comparison.
        
        Args:
            comparison (dict): Risk metrics comparison
            
        Returns:
            None
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 14))
            
            # Extract strategy names
            strategies = list(comparison.keys())
            
            # Key metrics to compare
            metrics_to_compare = {
                'volatility': 'Volatility',
                'max_drawdown': 'Max Drawdown',
                'daily_var_95': 'VaR (95%)',
                'daily_cvar_95': 'CVaR (95%)',
                'ulcer_index': 'Ulcer Index',
                'pain_ratio': 'Pain Ratio',
                'skewness': 'Skewness',
                'kurtosis': 'Kurtosis'
            }
            
            # Create DataFrames for comparison
            metrics_df = pd.DataFrame(index=strategies)
            
            for metric, label in metrics_to_compare.items():
                metrics_df[label] = [comparison[s].get(metric, np.nan) for s in strategies]
            
            # Plot volatility vs drawdown
            axs[0, 0].scatter(metrics_df['Volatility'], metrics_df['Max Drawdown'], s=100)
            
            # Add strategy labels
            for i, strategy in enumerate(strategies):
                axs[0, 0].annotate(strategy, 
                                 (metrics_df['Volatility'][i], metrics_df['Max Drawdown'][i]),
                                 xytext=(5, 5), textcoords='offset points')
            
            axs[0, 0].set_title('Volatility vs Max Drawdown', fontsize=14)
            axs[0, 0].set_xlabel('Volatility')
            axs[0, 0].set_ylabel('Max Drawdown')
            axs[0, 0].grid(True)
            
            # Plot VaR vs CVaR
            axs[0, 1].scatter(metrics_df['VaR (95%)'], metrics_df['CVaR (95%)'], s=100)
            
            # Add strategy labels
            for i, strategy in enumerate(strategies):
                axs[0, 1].annotate(strategy, 
                                 (metrics_df['VaR (95%)'][i], metrics_df['CVaR (95%)'][i]),
                                 xytext=(5, 5), textcoords='offset points')
            
            axs[0, 1].set_title('VaR vs CVaR', fontsize=14)
            axs[0, 1].set_xlabel('VaR (95%)')
            axs[0, 1].set_ylabel('CVaR (95%)')
            axs[0, 1].grid(True)
            
            # Plot pain ratio comparison
            metrics_df['Pain Ratio'].plot(kind='bar', ax=axs[1, 0])
            axs[1, 0].set_title('Pain Ratio Comparison', fontsize=14)
            axs[1, 0].set_ylabel('Pain Ratio')
            axs[1, 0].set_xticklabels(strategies, rotation=45, ha='right')
            
            # Add values
            for i, v in enumerate(metrics_df['Pain Ratio']):
                axs[1, 0].text(i, v + 0.1, f"{v:.2f}", ha='center')
            
            # Plot radar chart for risk metrics
            metrics_to_radar = ['Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)', 'Ulcer Index']
            
            # Number of variables
            N = len(metrics_to_radar)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Normalize values for radar chart (0 to 1, lower is better for risk)
            radar_df = metrics_df[metrics_to_radar].copy()
            
            # Invert so higher is better (safer)
            for col in radar_df.columns:
                radar_df[col] = 1 - (radar_df[col] / radar_df[col].max())
            
            # Set up axes
            ax = plt.subplot(2, 2, 4, polar=True)
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], metrics_to_radar, color='grey', size=10)
            
            # Draw radar chart for each strategy
            for i, strategy in enumerate(strategies):
                values = radar_df.loc[strategy].values.flatten().tolist()
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=strategy)
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_title('Risk Profile Comparison', fontsize=14)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/risk_comparison_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Risk comparison visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing risk comparison: {str(e)}")
