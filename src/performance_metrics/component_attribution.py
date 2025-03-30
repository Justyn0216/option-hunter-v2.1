"""
Component Attribution Module

This module analyzes the performance attribution of different components
of a trading strategy, helping identify which parts contribute most to
overall performance.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import itertools

class ComponentAttribution:
    """
    Analyzes performance attribution across strategy components.
    
    Features:
    - Attribution analysis for different strategy components
    - Factor analysis of trading performance
    - Component contribution visualization
    - Performance decomposition
    """
    
    def __init__(self, config=None):
        """
        Initialize the ComponentAttribution analyzer.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.results_dir = "data/attribution_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info("ComponentAttribution initialized")
    
    def analyze_trade_components(self, trades, components=None):
        """
        Analyze the contribution of different components to trade performance.
        
        Args:
            trades (list): List of trade dictionaries
            components (list, optional): List of component names to analyze
            
        Returns:
            dict: Component attribution results
        """
        self.logger.info("Analyzing trade components")
        
        # Skip if no trades
        if not trades:
            self.logger.warning("No trades provided for component analysis")
            return {}
        
        # Filter for completed trades
        completed_trades = [t for t in trades if t.get('type') == 'exit']
        
        if not completed_trades:
            self.logger.warning("No completed trades for component analysis")
            return {}
        
        # Extract P&L values
        pnl_values = [t.get('pnl', 0) for t in completed_trades]
        
        # If components not specified, try to infer from trades
        if not components:
            # Look for potential component fields in the first trade
            candidate_fields = []
            for field in completed_trades[0].keys():
                if field not in ['id', 'symbol', 'type', 'date', 'price', 'quantity', 'value', 'commission', 'pnl']:
                    candidate_fields.append(field)
            
            components = candidate_fields
        
        # Check if we have any components to analyze
        if not components:
            self.logger.warning("No components identified for analysis")
            return {}
        
        # Extract component values from trades
        component_values = {}
        
        for component in components:
            values = []
            for trade in completed_trades:
                # Handle nested components (e.g., 'option_data.delta')
                if '.' in component:
                    parts = component.split('.')
                    value = trade
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                else:
                    value = trade.get(component)
                
                values.append(value)
            
            component_values[component] = values
        
        # Filter components that have missing data or no variation
        valid_components = {}
        for component, values in component_values.items():
            # Check if all values are numeric
            numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]
            
            if len(numeric_values) > 0.8 * len(values):  # At least 80% valid
                # Check if there's variation
                if len(set(numeric_values)) > 1:
                    valid_components[component] = numeric_values
                else:
                    self.logger.debug(f"Component {component} has no variation")
            else:
                self.logger.debug(f"Component {component} has too many non-numeric values")
        
        # Exit if no valid components
        if not valid_components:
            self.logger.warning("No valid numeric components for analysis")
            return {}
        
        # Create a pandas DataFrame for analysis
        analysis_df = pd.DataFrame(valid_components)
        analysis_df['pnl'] = pnl_values
        
        # Calculate correlation with P&L
        correlations = {}
        for component in valid_components.keys():
            correlations[component] = analysis_df[component].corr(analysis_df['pnl'])
        
        # Sort by absolute correlation
        correlations = {k: v for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)}
        
        # Linear regression analysis for top components (multivariate)
        top_components = list(correlations.keys())[:min(5, len(correlations))]
        
        X = analysis_df[top_components].copy()
        y = analysis_df['pnl']
        
        # Normalize features
        X_norm = (X - X.mean()) / X.std()
        
        # Linear regression
        model = LinearRegression()
        model.fit(X_norm, y)
        
        # Calculate attribution based on coefficients
        coeffs = dict(zip(top_components, model.coef_))
        
        # Scale coefficients by feature standard deviation to get impact
        impact = {}
        for component in top_components:
            impact[component] = coeffs[component] * X[component].std()
        
        # Normalize impact to get attribution percentages
        total_impact = sum(abs(val) for val in impact.values())
        attribution = {component: abs(impact[component]) / total_impact for component in top_components}
        
        # Sort by attribution
        attribution = {k: v for k, v in sorted(attribution.items(), key=lambda x: x[1], reverse=True)}
        
        # Calculate model accuracy
        y_pred = model.predict(X_norm)
        r2 = r2_score(y, y_pred)
        
        # Prepare results
        results = {
            'components': list(valid_components.keys()),
            'correlations': correlations,
            'top_components': top_components,
            'coefficients': coeffs,
            'impact': impact,
            'attribution': attribution,
            'model_r2': r2
        }
        
        # Visualize attribution
        self._visualize_attribution(results)
        
        self.logger.info(f"Component attribution analysis completed with {len(valid_components)} components")
        return results
    
    def _visualize_attribution(self, attribution_results):
        """
        Visualize component attribution results.
        
        Args:
            attribution_results (dict): Attribution analysis results
            
        Returns:
            None
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot correlations
            correlations = attribution_results['correlations']
            components = list(correlations.keys())
            corr_values = list(correlations.values())
            
            axs[0].barh(components, corr_values)
            axs[0].set_title('Component Correlations with P&L', fontsize=14)
            axs[0].set_xlabel('Correlation')
            axs[0].set_ylabel('Component')
            axs[0].grid(True)
            
            # Add correlation values as text
            for i, v in enumerate(corr_values):
                axs[0].text(v + 0.01 if v >= 0 else v - 0.08, i, f"{v:.2f}", va='center')
            
            # Plot attribution
            attribution = attribution_results['attribution']
            attr_components = list(attribution.keys())
            attr_values = list(attribution.values())
            
            axs[1].pie(attr_values, labels=attr_components, autopct='%1.1f%%',
                     startangle=90, explode=[0.05] * len(attr_components))
            axs[1].set_title('Component Attribution (%)', fontsize=14)
            axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/component_attribution_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Component attribution visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing attribution: {str(e)}")
    
    def analyze_entry_exit_attribution(self, trades):
        """
        Analyze the attribution of performance between entry and exit timing.
        
        Args:
            trades (list): List of trade dictionaries
            
        Returns:
            dict: Entry/exit attribution results
        """
        self.logger.info("Analyzing entry/exit attribution")
        
        # Skip if no trades
        if not trades:
            self.logger.warning("No trades provided for entry/exit analysis")
            return {}
        
        # Filter for completed trades
        completed_trades = [t for t in trades if t.get('type') == 'exit']
        
        if not completed_trades:
            self.logger.warning("No completed trades for entry/exit analysis")
            return {}
        
        # Extract entry and exit data
        entries = []
        exits = []
        symbols = []
        
        for trade in completed_trades:
            # Check if we have the necessary data
            if 'entry_date' in trade and 'date' in trade and 'entry_price' in trade and 'price' in trade:
                entry_date = trade['entry_date']
                exit_date = trade['date']
                entry_price = trade['entry_price']
                exit_price = trade['price']
                symbol = trade['symbol']
                
                # Convert dates to datetime if they're strings
                if isinstance(entry_date, str):
                    entry_date = datetime.strptime(entry_date, '%Y-%m-%d' if len(entry_date) <= 10 else '%Y-%m-%d %H:%M:%S')
                
                if isinstance(exit_date, str):
                    exit_date = datetime.strptime(exit_date, '%Y-%m-%d' if len(exit_date) <= 10 else '%Y-%m-%d %H:%M:%S')
                
                # Add to our lists
                entries.append({'date': entry_date, 'price': entry_price})
                exits.append({'date': exit_date, 'price': exit_price})
                symbols.append(symbol)
        
        if not entries or not exits:
            self.logger.warning("Insufficient entry/exit data for analysis")
            return {}
        
        # Calculate P&L for each trade
        pnl = []
        side = []
        
        for i in range(len(entries)):
            if 'side' in completed_trades[i]:
                trade_side = completed_trades[i]['side']
            else:
                # Infer side from option type if available
                if 'option_data' in completed_trades[i] and 'option_type' in completed_trades[i]['option_data']:
                    opt_type = completed_trades[i]['option_data']['option_type']
                    trade_side = 'long'  # Default to long
                else:
                    trade_side = 'long'
            
            side.append(trade_side)
            
            # Calculate P&L based on side
            if trade_side == 'long':
                trade_pnl = exits[i]['price'] - entries[i]['price']
            else:
                trade_pnl = entries[i]['price'] - exits[i]['price']
            
            pnl.append(trade_pnl)
        
        # Create a DataFrame for analysis
        analysis_df = pd.DataFrame({
            'symbol': symbols,
            'entry_date': [e['date'] for e in entries],
            'exit_date': [e['date'] for e in exits],
            'entry_price': [e['price'] for e in entries],
            'exit_price': [e['price'] for e in exits],
            'side': side,
            'pnl': pnl
        })
        
        # Group by symbol to calculate typical price movements
        symbol_stats = {}
        
        for symbol in analysis_df['symbol'].unique():
            symbol_trades = analysis_df[analysis_df['symbol'] == symbol]
            
            # Calculate average daily movement (as absolute percentage)
            durations = [(ex - en).total_seconds() / (60 * 60 * 24) for en, ex in 
                        zip(symbol_trades['entry_date'], symbol_trades['exit_date'])]
            price_changes = [abs(ex / en - 1) for en, ex in 
                           zip(symbol_trades['entry_price'], symbol_trades['exit_price'])]
            
            if durations and price_changes:
                avg_daily_movement = np.mean([pc / max(1, d) for pc, d in zip(price_changes, durations)])
                symbol_stats[symbol] = {'avg_daily_movement': avg_daily_movement}
        
        # Calculate hypothetical scenarios to attribute performance
        entry_attribution = []
        exit_attribution = []
        
        for i, row in analysis_df.iterrows():
            symbol = row['symbol']
            
            if symbol not in symbol_stats:
                continue
                
            avg_daily_movement = symbol_stats[symbol]['avg_daily_movement']
            
            # Duration in days
            duration = (row['exit_date'] - row['entry_date']).total_seconds() / (60 * 60 * 24)
            
            # Calculate hypothetical scenarios
            
            # Scenario 1: Perfect entry, actual exit
            # Assume perfect entry is better by avg_daily_movement * sqrt(duration) 
            # (assuming random walk, movement scales with sqrt of time)
            perfect_entry_advantage = avg_daily_movement * np.sqrt(duration)
            perfect_entry_price = row['entry_price'] * (1 - perfect_entry_advantage) if row['side'] == 'long' else row['entry_price'] * (1 + perfect_entry_advantage)
            
            if row['side'] == 'long':
                perfect_entry_pnl = row['exit_price'] - perfect_entry_price
            else:
                perfect_entry_pnl = perfect_entry_price - row['exit_price']
            
            # Scenario 2: Actual entry, perfect exit
            perfect_exit_advantage = avg_daily_movement * np.sqrt(duration)
            perfect_exit_price = row['exit_price'] * (1 + perfect_exit_advantage) if row['side'] == 'long' else row['exit_price'] * (1 - perfect_exit_advantage)
            
            if row['side'] == 'long':
                perfect_exit_pnl = perfect_exit_price - row['entry_price']
            else:
                perfect_exit_pnl = row['entry_price'] - perfect_exit_price
            
            # Calculate attribution
            actual_pnl = row['pnl']
            
            # Entry attribution: How much better would perfect entry be vs actual
            entry_improvement = perfect_entry_pnl - actual_pnl
            
            # Exit attribution: How much better would perfect exit be vs actual
            exit_improvement = perfect_exit_pnl - actual_pnl
            
            # Total potential improvement
            total_improvement = entry_improvement + exit_improvement
            
            # Calculate attribution percentages
            if total_improvement > 0:
                entry_attr = entry_improvement / total_improvement
                exit_attr = exit_improvement / total_improvement
            else:
                # If no improvement possible, split evenly
                entry_attr = exit_attr = 0.5
            
            entry_attribution.append(entry_attr)
            exit_attribution.append(exit_attr)
        
        # Calculate overall attribution
        avg_entry_attribution = np.mean(entry_attribution) if entry_attribution else 0.5
        avg_exit_attribution = np.mean(exit_attribution) if exit_attribution else 0.5
        
        # Normalize to ensure they sum to 1
        total_attr = avg_entry_attribution + avg_exit_attribution
        if total_attr > 0:
            avg_entry_attribution /= total_attr
            avg_exit_attribution /= total_attr
        else:
            avg_entry_attribution = avg_exit_attribution = 0.5
        
        # Prepare results
        results = {
            'entry_attribution': avg_entry_attribution,
            'exit_attribution': avg_exit_attribution,
            'trade_level_attribution': list(zip(entry_attribution, exit_attribution)),
            'symbol_stats': symbol_stats
        }
        
        # Visualize attribution
        self._visualize_entry_exit_attribution(results)
        
        self.logger.info(f"Entry/exit attribution analysis completed")
        return results
    
    def _visualize_entry_exit_attribution(self, attribution_results):
        """
        Visualize entry/exit attribution results.
        
        Args:
            attribution_results (dict): Attribution analysis results
            
        Returns:
            None
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot overall attribution
            labels = ['Entry Timing', 'Exit Timing']
            sizes = [attribution_results['entry_attribution'], attribution_results['exit_attribution']]
            explode = (0.1, 0)  # only "explode" the first slice
            
            axs[0].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                     shadow=True, startangle=90)
            axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            axs[0].set_title('Overall Attribution of Performance', fontsize=14)
            
            # Plot trade-level attribution
            trade_attributions = attribution_results['trade_level_attribution']
            if trade_attributions:
                entry_attr = [a[0] for a in trade_attributions]
                exit_attr = [a[1] for a in trade_attributions]
                
                trades = range(1, len(trade_attributions) + 1)
                
                axs[1].bar(trades, entry_attr, label='Entry Attribution')
                axs[1].bar(trades, exit_attr, bottom=entry_attr, label='Exit Attribution')
                
                axs[1].set_xlabel('Trade Number')
                axs[1].set_ylabel('Attribution Percentage')
                axs[1].set_title('Trade-Level Attribution', fontsize=14)
                axs[1].legend()
            else:
                axs[1].text(0.5, 0.5, "Insufficient data for trade-level attribution",
                         horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/entry_exit_attribution_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Entry/exit attribution visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing entry/exit attribution: {str(e)}")
    
    def analyze_factor_attribution(self, equity_curve, factor_data):
        """
        Analyze the attribution of performance to different market factors.
        
        Args:
            equity_curve (dict or pd.Series): Strategy equity curve by date
            factor_data (dict): Dictionary mapping factor names to their time series
            
        Returns:
            dict: Factor attribution results
        """
        self.logger.info("Analyzing factor attribution")
        
        # Convert equity curve to pandas Series if it's a dictionary
        if isinstance(equity_curve, dict):
            equity_curve = pd.Series(equity_curve)
            equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # Convert factor data to DataFrame if needed
        factor_df = None
        if isinstance(factor_data, dict):
            factor_df = pd.DataFrame(factor_data)
            if not isinstance(factor_df.index, pd.DatetimeIndex):
                # Try to convert to datetime index
                try:
                    factor_df.index = pd.to_datetime(factor_df.index)
                except:
                    self.logger.warning("Unable to convert factor data index to DatetimeIndex")
                    return {}
        else:
            factor_df = factor_data
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Align dates
        aligned_data = pd.concat([returns, factor_df], axis=1).dropna()
        
        if len(aligned_data) < 10:
            self.logger.warning("Insufficient aligned data for factor attribution")
            return {}
        
        # Rename columns
        aligned_data.columns = ['strategy_returns'] + list(factor_df.columns)
        
        # Prepare data for regression
        X = aligned_data.drop('strategy_returns', axis=1)
        y = aligned_data['strategy_returns']
        
        # Normalize features
        X_norm = (X - X.mean()) / X.std()
        
        # Linear regression to get factor loadings
        model = LinearRegression()
        model.fit(X_norm, y)
        
        # Calculate factor loadings (betas)
        factor_loadings = dict(zip(X.columns, model.coef_))
        
        # Calculate alpha (intercept)
        alpha = model.intercept_
        
        # Calculate factor contributions to return
        factor_contributions = {}
        for factor in X.columns:
            # Contribution = factor loading * factor mean
            factor_contributions[factor] = factor_loadings[factor] * X[factor].mean()
        
        # Calculate model R^2
        y_pred = model.predict(X_norm)
        r2 = r2_score(y, y_pred)
        
        # Calculate attribution percentages
        total_contribution = sum(abs(val) for val in factor_contributions.values()) + abs(alpha)
        
        factor_attribution = {factor: abs(contrib) / total_contribution 
                           for factor, contrib in factor_contributions.items()}
        
        # Add alpha attribution
        factor_attribution['alpha'] = abs(alpha) / total_contribution
        
        # Sort by attribution
        factor_attribution = {k: v for k, v in sorted(factor_attribution.items(), key=lambda x: x[1], reverse=True)}
        
        # Prepare results
        results = {
            'factors': list(X.columns),
            'factor_loadings': factor_loadings,
            'alpha': alpha,
            'factor_contributions': factor_contributions,
            'factor_attribution': factor_attribution,
            'model_r2': r2
        }
        
        # Visualize attribution
        self._visualize_factor_attribution(results)
        
        self.logger.info(f"Factor attribution analysis completed with {len(X.columns)} factors")
        return results
    
    def _visualize_factor_attribution(self, attribution_results):
        """
        Visualize factor attribution results.
        
        Args:
            attribution_results (dict): Attribution analysis results
            
        Returns:
            None
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot factor loadings
            factor_loadings = attribution_results['factor_loadings']
            factors = list(factor_loadings.keys())
            loadings = list(factor_loadings.values())
            
            axs[0].barh(factors, loadings)
            axs[0].set_title('Factor Loadings (Betas)', fontsize=14)
            axs[0].set_xlabel('Loading')
            axs[0].set_ylabel('Factor')
            axs[0].grid(True)
            
            # Add loading values as text
            for i, v in enumerate(loadings):
                axs[0].text(v + 0.01 if v >= 0 else v - 0.08, i, f"{v:.2f}", va='center')
            
            # Plot attribution
            attribution = attribution_results['factor_attribution']
            attr_factors = list(attribution.keys())
            attr_values = list(attribution.values())
            
            # Create pie chart colors, highlighting alpha
            colors = ['red' if factor == 'alpha' else 'skyblue' for factor in attr_factors]
            
            axs[1].pie(attr_values, labels=attr_factors, autopct='%1.1f%%',
                     startangle=90, colors=colors, explode=[0.05 if factor == 'alpha' else 0 for factor in attr_factors])
            axs[1].set_title('Factor Attribution (%)', fontsize=14)
            axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            plt.tight_layout()
            
            # Add model R^2 as text
            fig.text(0.5, 0.02, f"Model R² = {attribution_results['model_r2']:.2f}", ha='center', fontsize=12)
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/factor_attribution_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Factor attribution visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing factor attribution: {str(e)}")
    
    def analyze_strategy_components(self, component_returns):
        """
        Analyze the attribution of performance across different strategy components.
        
        Args:
            component_returns (dict): Dictionary mapping component names to their return series
            
        Returns:
            dict: Strategy component attribution results
        """
        self.logger.info(f"Analyzing strategy components: {list(component_returns.keys())}")
        
        # Ensure all components have the same length and dates
        aligned_data = pd.DataFrame(component_returns)
        
        if len(aligned_data) < 10:
            self.logger.warning("Insufficient data for strategy component analysis")
            return {}
        
        # Calculate component statistics
        component_stats = {}
        
        for component in aligned_data.columns:
            returns = aligned_data[component]
            
            # Calculate basic statistics
            component_stats[component] = {
                'mean_return': returns.mean(),
                'total_return': (1 + returns).prod() - 1,
                'volatility': returns.std(),
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'win_rate': (returns > 0).mean(),
                'max_return': returns.max(),
                'min_return': returns.min()
            }
        
        # Calculate correlation matrix
        correlation_matrix = aligned_data.corr()
        
        # Calculate contribution to overall performance
        # Assuming equal weighting of components
        total_returns = aligned_data.sum(axis=1)
        
        # Calculate attribution using regression
        X = aligned_data
        y = total_returns
        
        # Linear regression to get component weights
        model = LinearRegression(fit_intercept=False)  # No intercept as we want full attribution
        model.fit(X, y)
        
        # Calculate component loadings
        component_loadings = dict(zip(X.columns, model.coef_))
        
        # Calculate component contributions
        component_contributions = {}
        for component in X.columns:
            component_contributions[component] = component_loadings[component] * X[component].mean()
        
        # Calculate attribution percentages
        total_contribution = sum(abs(val) for val in component_contributions.values())
        
        component_attribution = {component: abs(contrib) / total_contribution 
                              for component, contrib in component_contributions.items()}
        
        # Sort by attribution
        component_attribution = {k: v for k, v in sorted(component_attribution.items(), key=lambda x: x[1], reverse=True)}
        
        # Calculate model R^2
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Prepare results
        results = {
            'components': list(aligned_data.columns),
            'component_stats': component_stats,
            'correlation_matrix': correlation_matrix.to_dict(),
            'component_loadings': component_loadings,
            'component_contributions': component_contributions,
            'component_attribution': component_attribution,
            'model_r2': r2
        }
        
        # Visualize attribution
        self._visualize_strategy_components(results)
        
        self.logger.info(f"Strategy component analysis completed with {len(aligned_data.columns)} components")
        return results
    
    def _visualize_strategy_components(self, attribution_results):
        """
        Visualize strategy component attribution results.
        
        Args:
            attribution_results (dict): Attribution analysis results
            
        Returns:
            None
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # Plot component attribution
            attribution = attribution_results['component_attribution']
            components = list(attribution.keys())
            attr_values = list(attribution.values())
            
            axs[0, 0].pie(attr_values, labels=components, autopct='%1.1f%%',
                        startangle=90, explode=[0.05] * len(components))
            axs[0, 0].set_title('Component Attribution (%)', fontsize=14)
            axs[0, 0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            # Plot component returns
            component_stats = attribution_results['component_stats']
            comp_returns = [component_stats[c]['total_return'] for c in components]
            
            axs[0, 1].bar(components, comp_returns)
            axs[0, 1].set_title('Component Total Returns', fontsize=14)
            axs[0, 1].set_ylabel('Return')
            axs[0, 1].set_xticklabels(components, rotation=45, ha='right')
            axs[0, 1].grid(True)
            
            # Add return values as text
            for i, v in enumerate(comp_returns):
                axs[0, 1].text(i, v + 0.01 if v >= 0 else v - 0.05, f"{v:.2%}", ha='center')
            
            # Plot correlation matrix
            corr_matrix = pd.DataFrame(attribution_results['correlation_matrix'])
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=.5, ax=axs[1, 0])
            axs[1, 0].set_title('Component Correlation Matrix', fontsize=14)
            
            # Plot component Sharpe ratios
            comp_sharpes = [component_stats[c]['sharpe'] for c in components]
            
            axs[1, 1].bar(components, comp_sharpes)
            axs[1, 1].set_title('Component Sharpe Ratios', fontsize=14)
            axs[1, 1].set_ylabel('Sharpe Ratio')
            axs[1, 1].set_xticklabels(components, rotation=45, ha='right')
            axs[1, 1].grid(True)
            
            # Add Sharpe values as text
            for i, v in enumerate(comp_sharpes):
                axs[1, 1].text(i, v + 0.1 if v >= 0 else v - 0.3, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/strategy_components_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Strategy component visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing strategy components: {str(e)}")
