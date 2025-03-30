"""
Strategy Metrics Module

This module calculates advanced performance metrics for option trading strategies.
It provides tools for measuring risk-adjusted returns, drawdowns, and other key
performance indicators for evaluating trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class StrategyMetrics:
    """
    Calculates and analyzes trading strategy performance metrics.
    
    Features:
    - Risk-adjusted return metrics (Sharpe, Sortino, etc.)
    - Drawdown analysis
    - Win/loss statistics
    - Trade analysis
    - Performance visualization
    """
    
    def __init__(self, config=None):
        """
        Initialize the StrategyMetrics calculator.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.results_dir = "data/performance_metrics"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default parameters
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # Annual
        self.trading_days_per_year = self.config.get('trading_days_per_year', 252)
        
        self.logger.info("StrategyMetrics initialized")
    
    def calculate_metrics(self, equity_curve, trades=None, benchmark=None):
        """
        Calculate comprehensive performance metrics for a trading strategy.
        
        Args:
            equity_curve (dict or pd.Series): Strategy equity curve by date
            trades (list, optional): List of trade dictionaries
            benchmark (dict or pd.Series, optional): Benchmark equity curve
            
        Returns:
            dict: Dictionary of performance metrics
        """
        self.logger.info("Calculating strategy performance metrics")
        
        # Convert equity curve to pandas Series if it's a dictionary
        if isinstance(equity_curve, dict):
            equity_curve = pd.Series(equity_curve)
        
        # Convert benchmark to pandas Series if it's a dictionary and not None
        if benchmark is not None and isinstance(benchmark, dict):
            benchmark = pd.Series(benchmark)
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Calculate benchmark returns if available
        benchmark_returns = None
        if benchmark is not None:
            benchmark_returns = benchmark.pct_change().dropna()
        
        # Basic metrics
        metrics = {
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1,
            'annualized_return': self._calculate_annualized_return(returns),
            'volatility': self._calculate_volatility(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'calmar_ratio': self._calculate_calmar_ratio(returns, equity_curve),
            'omega_ratio': self._calculate_omega_ratio(returns),
            'win_rate': None,
            'profit_factor': None,
            'avg_win': None,
            'avg_loss': None,
            'avg_win_loss_ratio': None,
            'expectancy': None
        }
        
        # Add benchmark comparison if available
        if benchmark_returns is not None:
            metrics['benchmark_return'] = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
            metrics['benchmark_annualized_return'] = self._calculate_annualized_return(benchmark_returns)
            metrics['benchmark_volatility'] = self._calculate_volatility(benchmark_returns)
            metrics['alpha'] = self._calculate_alpha(returns, benchmark_returns)
            metrics['beta'] = self._calculate_beta(returns, benchmark_returns)
            metrics['information_ratio'] = self._calculate_information_ratio(returns, benchmark_returns)
        
        # Calculate trade-based metrics if trades are provided
        if trades is not None:
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.update(trade_metrics)
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(returns)
        metrics['monthly_returns'] = monthly_returns
        
        # Drawdown analysis
        drawdown_info = self._analyze_drawdowns(equity_curve)
        metrics['drawdowns'] = drawdown_info
        
        self.logger.info("Performance metrics calculation completed")
        return metrics
    
    def _calculate_annualized_return(self, returns):
        """
        Calculate annualized return from a series of returns.
        
        Args:
            returns (pd.Series): Daily returns
            
        Returns:
            float: Annualized return
        """
        if len(returns) < 2:
            return 0.0
            
        total_return = (1 + returns).prod() - 1
        years = len(returns) / self.trading_days_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        return annualized_return
    
    def _calculate_volatility(self, returns):
        """
        Calculate annualized volatility.
        
        Args:
            returns (pd.Series): Daily returns
            
        Returns:
            float: Annualized volatility
        """
        if len(returns) < 2:
            return 0.0
            
        return returns.std() * np.sqrt(self.trading_days_per_year)
    
    def _calculate_sharpe_ratio(self, returns):
        """
        Calculate Sharpe ratio.
        
        Args:
            returns (pd.Series): Daily returns
            
        Returns:
            float: Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
            
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + self.risk_free_rate) ** (1 / self.trading_days_per_year) - 1
        
        # Calculate excess returns
        excess_returns = returns - daily_risk_free
        
        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            return 0.0
            
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days_per_year)
        
        return sharpe
    
    def _calculate_sortino_ratio(self, returns):
        """
        Calculate Sortino ratio.
        
        Args:
            returns (pd.Series): Daily returns
            
        Returns:
            float: Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
            
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + self.risk_free_rate) ** (1 / self.trading_days_per_year) - 1
        
        # Calculate excess returns
        excess_returns = returns - daily_risk_free
        
        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0 if excess_returns.mean() <= 0 else float('inf')
            
        # Calculate Sortino ratio
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(self.trading_days_per_year)
        
        return sortino
    
    def _calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve (pd.Series): Equity curve
            
        Returns:
            float: Maximum drawdown as a positive percentage
        """
        if len(equity_curve) < 2:
            return 0.0
            
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Get maximum drawdown
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)  # Return as positive percentage
    
    def _calculate_calmar_ratio(self, returns, equity_curve):
        """
        Calculate Calmar ratio.
        
        Args:
            returns (pd.Series): Daily returns
            equity_curve (pd.Series): Equity curve
            
        Returns:
            float: Calmar ratio
        """
        if len(returns) < 2:
            return 0.0
            
        # Calculate annualized return
        annualized_return = self._calculate_annualized_return(returns)
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Avoid division by zero
        if max_drawdown == 0:
            return 0.0 if annualized_return <= 0 else float('inf')
            
        # Calculate Calmar ratio
        calmar = annualized_return / max_drawdown
        
        return calmar
    
    def _calculate_omega_ratio(self, returns, threshold=0.0):
        """
        Calculate Omega ratio.
        
        Args:
            returns (pd.Series): Daily returns
            threshold (float): Return threshold
            
        Returns:
            float: Omega ratio
        """
        if len(returns) < 2:
            return 0.0
            
        # Separate returns above and below threshold
        returns_above = returns[returns > threshold]
        returns_below = returns[returns <= threshold]
        
        # Calculate partial expectations
        if len(returns_above) == 0:
            return 0.0
            
        if len(returns_below) == 0:
            return float('inf')
            
        upside = returns_above.sum()
        downside = abs(returns_below.sum())
        
        # Avoid division by zero
        if downside == 0:
            return float('inf')
            
        # Calculate Omega ratio
        omega = upside / downside
        
        return omega
    
    def _calculate_alpha(self, returns, benchmark_returns):
        """
        Calculate Jensen's Alpha.
        
        Args:
            returns (pd.Series): Strategy returns
            benchmark_returns (pd.Series): Benchmark returns
            
        Returns:
            float: Alpha
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
            
        # Calculate beta
        beta = self._calculate_beta(returns, benchmark_returns)
        
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + self.risk_free_rate) ** (1 / self.trading_days_per_year) - 1
        
        # Calculate alpha (annualized)
        alpha = (returns.mean() - daily_risk_free) - beta * (benchmark_returns.mean() - daily_risk_free)
        alpha = alpha * self.trading_days_per_year
        
        return alpha
    
    def _calculate_beta(self, returns, benchmark_returns):
        """
        Calculate Beta.
        
        Args:
            returns (pd.Series): Strategy returns
            benchmark_returns (pd.Series): Benchmark returns
            
        Returns:
            float: Beta
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
            
        # Align returns (in case of different dates)
        aligned_returns = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_returns) < 2:
            return 0.0
            
        # Calculate covariance and variance
        covariance = aligned_returns['strategy'].cov(aligned_returns['benchmark'])
        variance = aligned_returns['benchmark'].var()
        
        # Avoid division by zero
        if variance == 0:
            return 0.0
            
        # Calculate beta
        beta = covariance / variance
        
        return beta
    
    def _calculate_information_ratio(self, returns, benchmark_returns):
        """
        Calculate Information Ratio.
        
        Args:
            returns (pd.Series): Strategy returns
            benchmark_returns (pd.Series): Benchmark returns
            
        Returns:
            float: Information Ratio
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
            
        # Align returns (in case of different dates)
        aligned_returns = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_returns) < 2:
            return 0.0
            
        # Calculate tracking error
        tracking_error = (aligned_returns['strategy'] - aligned_returns['benchmark']).std()
        
        # Avoid division by zero
        if tracking_error == 0:
            return 0.0
            
        # Calculate information ratio
        information_ratio = (aligned_returns['strategy'].mean() - aligned_returns['benchmark'].mean()) / tracking_error
        information_ratio = information_ratio * np.sqrt(self.trading_days_per_year)
        
        return information_ratio
    
    def _calculate_trade_metrics(self, trades):
        """
        Calculate trade-based performance metrics.
        
        Args:
            trades (list): List of trade dictionaries
            
        Returns:
            dict: Trade-based metrics
        """
        # Filter for completed trades
        completed_trades = [t for t in trades if t['type'] == 'exit']
        
        if not completed_trades:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_win_loss_ratio': 0.0,
                'expectancy': 0.0,
                'avg_trade_duration': 0.0
            }
        
        # Calculate win/loss metrics
        winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('pnl', 0) <= 0]
        
        total_trades = len(completed_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit metrics
        gross_profit = sum([t.get('pnl', 0) for t in winning_trades])
        gross_loss = sum([t.get('pnl', 0) for t in losing_trades])
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Calculate average win/loss
        avg_win = gross_profit / win_count if win_count > 0 else 0.0
        avg_loss = gross_loss / loss_count if loss_count > 0 else 0.0
        
        avg_win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calculate expectancy (average trade P&L)
        expectancy = (gross_profit + gross_loss) / total_trades
        
        # Calculate average trade duration (if entry and exit dates are available)
        durations = []
        for trade in completed_trades:
            if 'entry_date' in trade and 'date' in trade:  # 'date' is exit date
                entry_date = trade['entry_date']
                exit_date = trade['date']
                
                # Convert to datetime if they're strings
                if isinstance(entry_date, str):
                    entry_date = datetime.strptime(entry_date, '%Y-%m-%d' if len(entry_date) <= 10 else '%Y-%m-%d %H:%M:%S')
                
                if isinstance(exit_date, str):
                    exit_date = datetime.strptime(exit_date, '%Y-%m-%d' if len(exit_date) <= 10 else '%Y-%m-%d %H:%M:%S')
                
                duration = (exit_date - entry_date).total_seconds() / (60 * 60 * 24)  # in days
                durations.append(duration)
        
        avg_trade_duration = np.mean(durations) if durations else 0.0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'expectancy': expectancy,
            'avg_trade_duration': avg_trade_duration
        }
    
    def _calculate_monthly_returns(self, returns):
        """
        Calculate monthly returns from daily returns.
        
        Args:
            returns (pd.Series): Daily returns
            
        Returns:
            dict: Monthly returns by year and month
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            # Try to convert to DatetimeIndex
            try:
                returns.index = pd.to_datetime(returns.index)
            except:
                self.logger.warning("Unable to convert index to DatetimeIndex for monthly returns calculation")
                return {}
        
        # Group by year and month
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Convert to dictionary
        result = {}
        for (year, month), value in monthly_returns.items():
            if year not in result:
                result[year] = {}
            result[year][month] = value
        
        return result
    
    def _analyze_drawdowns(self, equity_curve, top_n=5):
        """
        Analyze drawdowns in detail.
        
        Args:
            equity_curve (pd.Series): Equity curve
            top_n (int): Number of top drawdowns to analyze
            
        Returns:
            dict: Drawdown analysis
        """
        if len(equity_curve) < 2:
            return {'max_drawdown': 0.0, 'top_drawdowns': []}
            
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown series
        drawdown = (equity_curve - running_max) / running_max
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        
        # No drawdowns
        if not is_drawdown.any():
            return {'max_drawdown': 0.0, 'top_drawdowns': []}
        
        # Find start and end of each drawdown
        drawdown_start = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_end = ~is_drawdown & is_drawdown.shift(1).fillna(False)
        
        # Get start and end dates
        start_dates = equity_curve.index[drawdown_start]
        end_dates = equity_curve.index[drawdown_end]
        
        # If still in drawdown, add last date
        if len(start_dates) > len(end_dates):
            end_dates = pd.Index(list(end_dates) + [equity_curve.index[-1]])
        
        # Calculate drawdown periods
        drawdown_periods = []
        
        for i in range(min(len(start_dates), len(end_dates))):
            start_date = start_dates[i]
            end_date = end_dates[i]
            
            # Get drawdown period
            period_equity = equity_curve[start_date:end_date]
            period_drawdown = drawdown[start_date:end_date]
            
            # Calculate max drawdown in this period
            max_dd = period_drawdown.min()
            max_dd_date = period_drawdown.idxmin()
            
            # Calculate recovery
            start_value = period_equity[0]
            lowest_value = period_equity[period_drawdown.idxmin()]
            end_value = period_equity[-1]
            
            # Duration
            duration = (end_date - start_date).days
            
            drawdown_periods.append({
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'duration_days': duration,
                'max_drawdown': max_dd,
                'max_drawdown_date': max_dd_date.strftime('%Y-%m-%d'),
                'recovery': (end_value / lowest_value) - 1 if end_value > lowest_value else 0.0
            })
        
        # Sort by max drawdown (largest first)
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])
        
        # Get top N drawdowns
        top_drawdowns = drawdown_periods[:top_n]
        
        return {
            'max_drawdown': abs(drawdown.min()),
            'top_drawdowns': top_drawdowns
        }
    
    def generate_performance_report(self, equity_curve, trades=None, benchmark=None, title="Strategy Performance Report"):
        """
        Generate a comprehensive performance report.
        
        Args:
            equity_curve (dict or pd.Series): Strategy equity curve by date
            trades (list, optional): List of trade dictionaries
            benchmark (dict or pd.Series, optional): Benchmark equity curve
            title (str): Report title
            
        Returns:
            dict: Performance report data
        """
        # Calculate metrics
        metrics = self.calculate_metrics(equity_curve, trades, benchmark)
        
        # Generate a report
        report = {
            'title': title,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'summary': self._generate_summary(metrics)
        }
        
        # Visualize performance
        self._visualize_performance(equity_curve, trades, benchmark, title)
        
        return report
    
    def _generate_summary(self, metrics):
        """
        Generate a natural language summary of performance metrics.
        
        Args:
            metrics (dict): Performance metrics
            
        Returns:
            str: Performance summary
        """
        summary = []
        
        # Basic performance
        summary.append(f"Total Return: {metrics['total_return']:.2%}")
        summary.append(f"Annualized Return: {metrics['annualized_return']:.2%}")
        summary.append(f"Volatility: {metrics['volatility']:.2%}")
        
        # Risk-adjusted returns
        summary.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        summary.append(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        summary.append(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        
        # Drawdown
        summary.append(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        
        # Trade statistics if available
        if metrics['win_rate'] is not None:
            summary.append(f"Win Rate: {metrics['win_rate']:.2%}")
            summary.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
            summary.append(f"Average Win: ${metrics['avg_win']:.2f}")
            summary.append(f"Average Loss: ${metrics['avg_loss']:.2f}")
            summary.append(f"Average Win/Loss Ratio: {metrics['avg_win_loss_ratio']:.2f}")
            summary.append(f"Expectancy: ${metrics['expectancy']:.2f}")
        
        # Benchmark comparison if available
        if 'benchmark_return' in metrics:
            summary.append(f"Benchmark Return: {metrics['benchmark_return']:.2%}")
            summary.append(f"Alpha: {metrics['alpha']:.2%}")
            summary.append(f"Beta: {metrics['beta']:.2f}")
            summary.append(f"Information Ratio: {metrics['information_ratio']:.2f}")
        
        return "\n".join(summary)
    
    def _visualize_performance(self, equity_curve, trades=None, benchmark=None, title="Strategy Performance"):
        """
        Create visualizations of strategy performance.
        
        Args:
            equity_curve (dict or pd.Series): Strategy equity curve by date
            trades (list, optional): List of trade dictionaries
            benchmark (dict or pd.Series, optional): Benchmark equity curve
            title (str): Plot title
            
        Returns:
            None
        """
        try:
            # Convert equity curve to pandas Series if it's a dictionary
            if isinstance(equity_curve, dict):
                equity_curve = pd.Series(equity_curve)
                equity_curve.index = pd.to_datetime(equity_curve.index)
            
            # Convert benchmark to pandas Series if it's a dictionary and not None
            if benchmark is not None:
                if isinstance(benchmark, dict):
                    benchmark = pd.Series(benchmark)
                    benchmark.index = pd.to_datetime(benchmark.index)
            
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot equity curve
            axs[0].plot(equity_curve.index, equity_curve.values, label='Strategy')
            
            if benchmark is not None:
                # Normalize benchmark to start at the same value as the strategy
                norm_factor = equity_curve.iloc[0] / benchmark.iloc[0]
                axs[0].plot(benchmark.index, benchmark.values * norm_factor, label='Benchmark', linestyle='--')
            
            axs[0].set_title(f"{title} - Equity Curve", fontsize=14)
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Equity ($)')
            axs[0].legend()
            axs[0].grid(True)
            
            # Plot drawdowns
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            
            axs[1].fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
            axs[1].set_title('Drawdowns', fontsize=14)
            axs[1].set_xlabel('Date')
            axs[1].set_ylabel('Drawdown (%)')
            axs[1].grid(True)
            
            # Plot monthly returns heatmap if we have enough data
            if len(equity_curve) > 30:
                returns = equity_curve.pct_change().dropna()
                
                if isinstance(returns.index, pd.DatetimeIndex):
                    # Calculate monthly returns
                    monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
                        lambda x: (1 + x).prod() - 1
                    ).unstack()
                    
                    # Plot heatmap
                    sns.heatmap(monthly_returns, annot=True, fmt=".1%", cmap="RdYlGn", center=0, ax=axs[2])
                    axs[2].set_title('Monthly Returns', fontsize=14)
                    axs[2].set_xlabel('Month')
                    axs[2].set_ylabel('Year')
                else:
                    axs[2].text(0.5, 0.5, "Monthly returns heatmap unavailable (datetime index required)",
                             horizontalalignment='center', verticalalignment='center')
            else:
                axs[2].text(0.5, 0.5, "Not enough data for monthly returns heatmap",
                         horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/performance_visualization_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Performance visualization saved to {plot_file}")
            
            # Create trade analysis visualization if trades are provided
            if trades and len(trades) > 0:
                self._visualize_trades(trades)
                
        except Exception as e:
            self.logger.error(f"Error visualizing performance: {str(e)}")
    
    def _visualize_trades(self, trades):
        """
        Create visualizations of trade statistics.
        
        Args:
            trades (list): List of trade dictionaries
            
        Returns:
            None
        """
        try:
            # Filter for completed trades
            completed_trades = [t for t in trades if t['type'] == 'exit']
            
            if not completed_trades:
                self.logger.warning("No completed trades to visualize")
                return
            
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            
            # Extract P&L values
            pnl_values = [t.get('pnl', 0) for t in completed_trades]
            
            # Plot trade P&L distribution
            sns.histplot(pnl_values, kde=True, ax=axs[0, 0])
            axs[0, 0].set_title('Trade P&L Distribution', fontsize=12)
            axs[0, 0].set_xlabel('P&L ($)')
            axs[0, 0].set_ylabel('Frequency')
            
            # Plot cumulative P&L
            cumulative_pnl = np.cumsum(pnl_values)
            axs[0, 1].plot(range(len(cumulative_pnl)), cumulative_pnl)
            axs[0, 1].set_title('Cumulative P&L', fontsize=12)
            axs[0, 1].set_xlabel('Trade #')
            axs[0, 1].set_ylabel('Cumulative P&L ($)')
            
            # Plot win/loss by trade type if available
            if all('option_type' in t for t in completed_trades if 'option_data' in t):
                # Extract trade types
                trade_types = []
                for trade in completed_trades:
                    if 'option_data' in trade and 'option_type' in trade['option_data']:
                        trade_types.append(trade['option_data']['option_type'])
                    else:
                        trade_types.append('unknown')
                
                # Calculate win rate by type
                trade_outcomes = ['win' if t.get('pnl', 0) > 0 else 'loss' for t in completed_trades]
                trade_df = pd.DataFrame({'type': trade_types, 'outcome': trade_outcomes})
                
                win_rate_by_type = trade_df.groupby('type')['outcome'].apply(
                    lambda x: (x == 'win').mean()
                ).reset_index()
                
                # Plot win rate by type
                axs[1, 0].bar(win_rate_by_type['type'], win_rate_by_type['outcome'])
                axs[1, 0].set_title('Win Rate by Option Type', fontsize=12)
                axs[1, 0].set_xlabel('Option Type')
                axs[1, 0].set_ylabel('Win Rate')
                axs[1, 0].set_ylim(0, 1)
                
                # Add win rate values
                for i, v in enumerate(win_rate_by_type['outcome']):
                    axs[1, 0].text(i, v + 0.02, f"{v:.2%}", ha='center')
            else:
                axs[1, 0].text(0.5, 0.5, "Win rate by type unavailable",
                             horizontalalignment='center', verticalalignment='center')
            
            # Plot trade duration distribution if available
            durations = []
            for trade in completed_trades:
                if 'entry_date' in trade and 'date' in trade:  # 'date' is exit date
                    entry_date = trade['entry_date']
                    exit_date = trade['date']
                    
                    # Convert to datetime if they're strings
                    if isinstance(entry_date, str):
                        entry_date = datetime.strptime(entry_date, '%Y-%m-%d' if len(entry_date) <= 10 else '%Y-%m-%d %H:%M:%S')
                    
                    if isinstance(exit_date, str):
                        exit_date = datetime.strptime(exit_date, '%Y-%m-%d' if len(exit_date) <= 10 else '%Y-%m-%d %H:%M:%S')
                    
                    duration = (exit_date - entry_date).total_seconds() / (60 * 60 * 24)  # in days
                    durations.append(duration)
            
            if durations:
                sns.histplot(durations, kde=True, ax=axs[1, 1])
                axs[1, 1].set_title('Trade Duration Distribution', fontsize=12)
                axs[1, 1].set_xlabel('Duration (days)')
                axs[1, 1].set_ylabel('Frequency')
            else:
                axs[1, 1].text(0.5, 0.5, "Trade duration data unavailable",
                             horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/trade_analysis_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Trade analysis visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing trades: {str(e)}")
    
    def compare_strategies(self, strategy_results, benchmark=None, title="Strategy Comparison"):
        """
        Compare multiple trading strategies.
        
        Args:
            strategy_results (dict): Dictionary mapping strategy names to their equity curves
            benchmark (dict or pd.Series, optional): Benchmark equity curve
            title (str): Plot title
            
        Returns:
            dict: Comparison results
        """
        self.logger.info(f"Comparing {len(strategy_results)} strategies")
        
        # Calculate metrics for each strategy
        comparison = {}
        
        for strategy_name, equity_curve in strategy_results.items():
            metrics = self.calculate_metrics(equity_curve, benchmark=benchmark)
            comparison[strategy_name] = metrics
        
        # Create comparison visualization
        self._visualize_strategy_comparison(strategy_results, benchmark, title)
        
        return comparison
    
    def _visualize_strategy_comparison(self, strategy_results, benchmark=None, title="Strategy Comparison"):
        """
        Create visualizations comparing multiple strategies.
        
        Args:
            strategy_results (dict): Dictionary mapping strategy names to their equity curves
            benchmark (dict or pd.Series, optional): Benchmark equity curve
            title (str): Plot title
            
        Returns:
            None
        """
        try:
            # Convert equity curves to pandas Series if needed
            processed_results = {}
            for name, curve in strategy_results.items():
                if isinstance(curve, dict):
                    series = pd.Series(curve)
                    series.index = pd.to_datetime(series.index)
                    processed_results[name] = series
                else:
                    processed_results[name] = curve
            
            # Convert benchmark if needed
            if benchmark is not None:
                if isinstance(benchmark, dict):
                    benchmark = pd.Series(benchmark)
                    benchmark.index = pd.to_datetime(benchmark.index)
            
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 2]})
            
            # Plot equity curves
            for name, curve in processed_results.items():
                axs[0].plot(curve.index, curve.values, label=name)
            
            if benchmark is not None:
                # Find a common starting point for normalization
                start_value = next(iter(processed_results.values())).iloc[0]
                norm_factor = start_value / benchmark.iloc[0]
                axs[0].plot(benchmark.index, benchmark.values * norm_factor, label='Benchmark', linestyle='--')
            
            axs[0].set_title(f"{title} - Equity Curves", fontsize=14)
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Equity ($)')
            axs[0].legend()
            axs[0].grid(True)
            
            # Plot drawdowns
            for name, curve in processed_results.items():
                running_max = curve.cummax()
                drawdown = (curve - running_max) / running_max
                axs[1].plot(drawdown.index, drawdown.values, label=name)
            
            axs[1].set_title('Drawdowns', fontsize=14)
            axs[1].set_xlabel('Date')
            axs[1].set_ylabel('Drawdown (%)')
            axs[1].legend()
            axs[1].grid(True)
            
            # Plot performance metrics comparison
            metrics_to_compare = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 
                               'sortino_ratio', 'max_drawdown', 'calmar_ratio']
            
            # Calculate metrics for each strategy
            metrics_data = {}
            for name, curve in processed_results.items():
                metrics = self.calculate_metrics(curve, benchmark=benchmark)
                metrics_data[name] = [metrics[m] for m in metrics_to_compare]
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics_data, index=metrics_to_compare)
            
            # Multiply percentage values by 100 for better display
            for metric in ['total_return', 'annualized_return', 'volatility', 'max_drawdown']:
                metrics_df.loc[metric] = metrics_df.loc[metric] * 100
            
            # Plot as bar chart
            metrics_df.T.plot(kind='bar', ax=axs[2])
            axs[2].set_title('Performance Metrics Comparison', fontsize=14)
            axs[2].set_xlabel('Strategy')
            axs[2].set_ylabel('Value')
            axs[2].legend(title='Metric')
            
            # Add value labels
            for container in axs[2].containers:
                axs[2].bar_label(container, fmt='%.2f')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/strategy_comparison_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Strategy comparison visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing strategy comparison: {str(e)}")
