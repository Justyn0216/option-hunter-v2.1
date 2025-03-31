"""
Throughput Monitor Module

This module provides functionality for monitoring system throughput and performance
for the Option Hunter trading system. It tracks API call rates, data flow volumes,
system bottlenecks, and overall processing efficiency.
"""

import logging
import os
import json
import time
import threading
import queue
import datetime
import collections
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

class ThroughputMonitor:
    """
    System throughput and performance monitoring module.
    
    Features:
    - API request and response tracking
    - Data flow volume monitoring
    - Processing time analysis
    - Bottleneck detection and alerts
    - Historical performance metrics
    - Trend analysis and visualization
    """
    
    def __init__(self, config):
        """
        Initialize the ThroughputMonitor.
        
        Args:
            config (dict): Throughput monitoring configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Extract configuration
        self.throughput_params = config.get("throughput_monitoring", {})
        
        # Default parameters
        self.enable_monitoring = self.throughput_params.get("enable_monitoring", True)
        self.monitoring_interval = self.throughput_params.get("monitoring_interval_seconds", 5)
        self.metrics_history_size = self.throughput_params.get("metrics_history_size", 1000)
        self.alert_thresholds = self.throughput_params.get("alert_thresholds", {})
        self.api_call_timeout = self.throughput_params.get("api_call_timeout_ms", 5000) / 1000.0  # Convert to seconds
        self.enable_visualization = self.throughput_params.get("enable_visualization", True)
        self.visualization_interval = self.throughput_params.get("visualization_interval_seconds", 3600)
        self.system_resource_tracking = self.throughput_params.get("system_resource_tracking", True)
        self.detailed_tracking = self.throughput_params.get("detailed_tracking", True)
        
        # Initialize metrics tracking
        self.api_calls = collections.defaultdict(int)
        self.api_errors = collections.defaultdict(int)
        self.api_latency = collections.defaultdict(list)
        self.data_volumes = collections.defaultdict(int)
        self.processing_times = collections.defaultdict(list)
        self.bottlenecks = []
        
        # Tracking for rate/throughput calculations
        self.last_reset_time = time.time()
        self.metrics_history = {
            "api_calls": collections.deque(maxlen=self.metrics_history_size),
            "api_errors": collections.deque(maxlen=self.metrics_history_size),
            "api_latency": collections.deque(maxlen=self.metrics_history_size),
            "data_volumes": collections.deque(maxlen=self.metrics_history_size),
            "processing_times": collections.deque(maxlen=self.metrics_history_size),
            "system_metrics": collections.deque(maxlen=self.metrics_history_size)
        }
        
        # Thread-safe queues for metrics recording
        self.api_call_queue = queue.Queue()
        self.data_volume_queue = queue.Queue()
        self.processing_time_queue = queue.Queue()
        
        # System metrics tracking
        self.system_metrics = {}
        
        # Alert counters
        self.alerts_triggered = collections.defaultdict(int)
        self.last_alert_time = collections.defaultdict(float)
        
        # Threading
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
        # Create directories
        self.logs_dir = "logs/throughput"
        self.visualizations_dir = "logs/throughput/visualizations"
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            self._start_monitoring()
            self.logger.info("ThroughputMonitor initialized and monitoring started")
        else:
            self.logger.info("ThroughputMonitor initialized (monitoring disabled)")
    
    def _start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self._monitoring_thread,
                daemon=True,
                name="ThroughputMonitorThread"
            )
            self.monitor_thread.start()
            self.logger.debug("Throughput monitoring thread started")
    
    def _monitoring_thread(self):
        """Background thread for processing metrics and generating reports."""
        last_visualization_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Process all queued metrics
                self._process_queued_metrics()
                
                # Check if it's time to calculate rates and record metrics
                current_time = time.time()
                elapsed = current_time - self.last_reset_time
                
                if elapsed >= self.monitoring_interval:
                    # Calculate and record metrics
                    self._calculate_metrics()
                    
                    # Check for threshold violations
                    self._check_thresholds()
                    
                    # Reset counters
                    self._reset_counters()
                    
                    # Update last reset time
                    self.last_reset_time = current_time
                
                # Check if it's time to generate visualizations
                if self.enable_visualization and current_time - last_visualization_time >= self.visualization_interval:
                    self._generate_visualizations()
                    last_visualization_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in throughput monitoring thread: {str(e)}")
            
            # Sleep for a short interval
            self.stop_event.wait(0.1)
    
    def _process_queued_metrics(self):
        """Process all queued metrics."""
        # Process API calls
        while not self.api_call_queue.empty():
            try:
                call_data = self.api_call_queue.get_nowait()
                
                # Record API call
                api_name = call_data["api"]
                self.api_calls[api_name] += 1
                
                # Record latency
                latency = call_data.get("latency")
                if latency is not None:
                    self.api_latency[api_name].append(latency)
                
                # Record errors
                if not call_data.get("success", True):
                    self.api_errors[api_name] += 1
                
                self.api_call_queue.task_done()
                
            except queue.Empty:
                break
        
        # Process data volumes
        while not self.data_volume_queue.empty():
            try:
                volume_data = self.data_volume_queue.get_nowait()
                
                # Record data volume
                data_type = volume_data["type"]
                self.data_volumes[data_type] += volume_data["bytes"]
                
                self.data_volume_queue.task_done()
                
            except queue.Empty:
                break
        
        # Process processing times
        while not self.processing_time_queue.empty():
            try:
                time_data = self.processing_time_queue.get_nowait()
                
                # Record processing time
                process_name = time_data["process"]
                self.processing_times[process_name].append(time_data["time"])
                
                self.processing_time_queue.task_done()
                
            except queue.Empty:
                break
    
    def _calculate_metrics(self):
        """Calculate metrics from current data and add to history."""
        timestamp = datetime.datetime.now()
        elapsed = time.time() - self.last_reset_time
        
        # Calculate API call rates
        api_call_rates = {
            api: count / elapsed for api, count in self.api_calls.items()
        }
        
        # Calculate API error rates
        api_error_rates = {
            api: self.api_errors[api] / max(1, self.api_calls[api]) for api in self.api_calls
        }
        
        # Calculate average API latency
        api_avg_latency = {
            api: np.mean(latencies) if latencies else 0 for api, latencies in self.api_latency.items()
        }
        
        # Calculate data volume rates
        data_volume_rates = {
            data_type: volume / elapsed for data_type, volume in self.data_volumes.items()
        }
        
        # Calculate average processing times
        avg_processing_times = {
            process: np.mean(times) if times else 0 for process, times in self.processing_times.items()
        }
        
        # Collect system metrics if enabled
        if self.system_resource_tracking:
            self.system_metrics = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_io": {k: v for k, v in psutil.disk_io_counters()._asdict().items()} if hasattr(psutil, 'disk_io_counters') else None,
                "network_io": {k: v for k, v in psutil.net_io_counters()._asdict().items()},
                "thread_count": threading.active_count()
            }
        
        # Add metrics to history
        self.metrics_history["api_calls"].append({
            "timestamp": timestamp,
            "rates": api_call_rates,
            "total": sum(self.api_calls.values()),
            "details": dict(self.api_calls) if self.detailed_tracking else None
        })
        
        self.metrics_history["api_errors"].append({
            "timestamp": timestamp,
            "rates": api_error_rates,
            "total": sum(self.api_errors.values()),
            "details": dict(self.api_errors) if self.detailed_tracking else None
        })
        
        self.metrics_history["api_latency"].append({
            "timestamp": timestamp,
            "averages": api_avg_latency,
            "weighted_avg": self._calculate_weighted_latency(),
            "details": {api: list(latencies) for api, latencies in self.api_latency.items()} if self.detailed_tracking else None
        })
        
        self.metrics_history["data_volumes"].append({
            "timestamp": timestamp,
            "rates": data_volume_rates,
            "total": sum(self.data_volumes.values()),
            "details": dict(self.data_volumes) if self.detailed_tracking else None
        })
        
        self.metrics_history["processing_times"].append({
            "timestamp": timestamp,
            "averages": avg_processing_times,
            "details": {process: list(times) for process, times in self.processing_times.items()} if self.detailed_tracking else None
        })
        
        self.metrics_history["system_metrics"].append({
            "timestamp": timestamp,
            "metrics": self.system_metrics
        })
        
        # Log metrics periodically
        if len(self.metrics_history["api_calls"]) % 10 == 0:
            self._log_metrics()
    
    def _calculate_weighted_latency(self):
        """Calculate weighted average latency across all APIs."""
        total_calls = sum(self.api_calls.values())
        if total_calls == 0:
            return 0
        
        weighted_sum = 0
        for api, latencies in self.api_latency.items():
            if not latencies:
                continue
            
            api_avg = np.mean(latencies)
            call_count = self.api_calls[api]
            weighted_sum += api_avg * call_count
        
        return weighted_sum / total_calls
    
    def _check_thresholds(self):
        """Check if any metrics exceed defined thresholds."""
        # Check API latency thresholds
        max_latency_threshold = self.alert_thresholds.get("max_api_latency_ms", 1000) / 1000.0  # Convert to seconds
        
        for api, latencies in self.api_latency.items():
            if not latencies:
                continue
                
            avg_latency = np.mean(latencies)
            if avg_latency > max_latency_threshold:
                self._trigger_alert(
                    "high_latency",
                    f"High latency detected for API {api}: {avg_latency*1000:.2f}ms (threshold: {max_latency_threshold*1000:.2f}ms)",
                    {"api": api, "latency_ms": avg_latency*1000, "threshold_ms": max_latency_threshold*1000}
                )
        
        # Check API error rate thresholds
        max_error_rate = self.alert_thresholds.get("max_error_rate", 0.05)  # 5% by default
        
        for api in self.api_calls:
            error_rate = self.api_errors[api] / max(1, self.api_calls[api])
            if error_rate > max_error_rate:
                self._trigger_alert(
                    "high_error_rate",
                    f"High error rate detected for API {api}: {error_rate:.2%} (threshold: {max_error_rate:.2%})",
                    {"api": api, "error_rate": error_rate, "threshold": max_error_rate}
                )
        
        # Check system resource thresholds
        if self.system_resource_tracking:
            # Check CPU usage
            max_cpu = self.alert_thresholds.get("max_cpu_percent", 90)
            cpu_percent = self.system_metrics.get("cpu_percent", 0)
            
            if cpu_percent > max_cpu:
                self._trigger_alert(
                    "high_cpu_usage",
                    f"High CPU usage detected: {cpu_percent:.2f}% (threshold: {max_cpu}%)",
                    {"cpu_percent": cpu_percent, "threshold": max_cpu}
                )
            
            # Check memory usage
            max_memory = self.alert_thresholds.get("max_memory_percent", 90)
            memory_percent = self.system_metrics.get("memory_percent", 0)
            
            if memory_percent > max_memory:
                self._trigger_alert(
                    "high_memory_usage",
                    f"High memory usage detected: {memory_percent:.2f}% (threshold: {max_memory}%)",
                    {"memory_percent": memory_percent, "threshold": max_memory}
                )
    
    def _trigger_alert(self, alert_type, message, details=None):
        """
        Trigger an alert based on threshold violation.
        
        Args:
            alert_type (str): Type of alert
            message (str): Alert message
            details (dict, optional): Additional alert details
        """
        # Avoid alert spam by limiting frequency
        current_time = time.time()
        alert_cooldown = self.throughput_params.get("alert_cooldown_seconds", 300)  # 5 minutes by default
        
        if current_time - self.last_alert_time[alert_type] < alert_cooldown:
            # Alert is in cooldown period
            return
        
        # Update alert counters
        self.alerts_triggered[alert_type] += 1
        self.last_alert_time[alert_type] = current_time
        
        # Log the alert
        self.logger.warning(f"THROUGHPUT ALERT: {message}")
        
        # Record bottleneck if it's a performance issue
        if alert_type in ["high_latency", "high_cpu_usage", "high_memory_usage"]:
            bottleneck = {
                "timestamp": datetime.datetime.now(),
                "type": alert_type,
                "message": message,
                "details": details or {}
            }
            self.bottlenecks.append(bottleneck)
            
            # Keep only recent bottlenecks (last 100)
            if len(self.bottlenecks) > 100:
                self.bottlenecks = self.bottlenecks[-100:]
        
        # Write alert to log file
        alert_log = os.path.join(self.logs_dir, "alerts.json")
        
        try:
            # Load existing alerts if available
            if os.path.exists(alert_log):
                with open(alert_log, 'r') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            # Add new alert
            alerts.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "type": alert_type,
                "message": message,
                "details": details or {}
            })
            
            # Keep only recent alerts (last a1000)
            if len(alerts) > 1000:
                alerts = alerts[-1000:]
            
            # Save alerts
            with open(alert_log, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving alert to log: {str(e)}")
    
    def _reset_counters(self):
        """Reset counters for the next interval."""
        self.api_calls = collections.defaultdict(int)
        self.api_errors = collections.defaultdict(int)
        self.api_latency = collections.defaultdict(list)
        self.data_volumes = collections.defaultdict(int)
        self.processing_times = collections.defaultdict(list)
    
    def _log_metrics(self):
        """Log metrics to file for historical analysis."""
        log_file = os.path.join(self.logs_dir, f"metrics_{datetime.datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            # Prepare metrics for serialization
            metrics = {
                "timestamp": datetime.datetime.now().isoformat(),
                "api_calls": {
                    "latest_rates": self.metrics_history["api_calls"][-1]["rates"] if self.metrics_history["api_calls"] else {},
                    "latest_total": self.metrics_history["api_calls"][-1]["total"] if self.metrics_history["api_calls"] else 0
                },
                "api_errors": {
                    "latest_rates": self.metrics_history["api_errors"][-1]["rates"] if self.metrics_history["api_errors"] else {},
                    "latest_total": self.metrics_history["api_errors"][-1]["total"] if self.metrics_history["api_errors"] else 0
                },
                "api_latency": {
                    "latest_averages": self.metrics_history["api_latency"][-1]["averages"] if self.metrics_history["api_latency"] else {},
                    "latest_weighted_avg": self.metrics_history["api_latency"][-1]["weighted_avg"] if self.metrics_history["api_latency"] else 0
                },
                "data_volumes": {
                    "latest_rates": self.metrics_history["data_volumes"][-1]["rates"] if self.metrics_history["data_volumes"] else {},
                    "latest_total": self.metrics_history["data_volumes"][-1]["total"] if self.metrics_history["data_volumes"] else 0
                },
                "system_metrics": self.system_metrics if self.system_resource_tracking else {},
                "bottlenecks": [b for b in self.bottlenecks[-5:]] if self.bottlenecks else [],
                "alerts": {alert_type: count for alert_type, count in self.alerts_triggered.items()}
            }
            
            # Load existing logs if available
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Add new metrics
            logs.append(metrics)
            
            # Keep last 1440 entries (1 day at 1-minute intervals)
            if len(logs) > 1440:
                logs = logs[-1440:]
            
            # Save logs
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging metrics: {str(e)}")
    
    def _generate_visualizations(self):
        """Generate visualizations of throughput metrics."""
        if not self.metrics_history["api_calls"]:
            # No data to visualize
            return
        
        try:
            # Set up plotting environment
            sns.set(style="whitegrid")
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create figure with subplots
            fig, axs = plt.subplots(3, 2, figsize=(15, 15))
            
            # Plot API call rates
            self._plot_api_calls(axs[0, 0])
            
            # Plot API latency
            self._plot_api_latency(axs[0, 1])
            
            # Plot error rates
            self._plot_error_rates(axs[1, 0])
            
            # Plot data volumes
            self._plot_data_volumes(axs[1, 1])
            
            # Plot system resources
            if self.system_resource_tracking:
                self._plot_system_resources(axs[2, 0])
            else:
                axs[2, 0].text(0.5, 0.5, "System resource tracking disabled",
                             ha='center', va='center', transform=axs[2, 0].transAxes)
            
            # Plot bottlenecks
            self._plot_bottlenecks(axs[2, 1])
            
            # Add title and adjust layout
            plt.suptitle(f"System Throughput Metrics - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save figure
            viz_file = os.path.join(self.visualizations_dir, f"throughput_{timestamp}.png")
            plt.savefig(viz_file, dpi=120)
            plt.close(fig)
            
            self.logger.info(f"Generated throughput visualization: {viz_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
    
    def _plot_api_calls(self, ax):
        """
        Plot API call rates.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
        """
        # Extract data
        timestamps = [m["timestamp"] for m in self.metrics_history["api_calls"]]
        
        # Get the top 5 APIs by call count
        all_apis = set()
        for m in self.metrics_history["api_calls"]:
            all_apis.update(m["rates"].keys())
        
        api_totals = {}
        for api in all_apis:
            api_totals[api] = sum(m["rates"].get(api, 0) for m in self.metrics_history["api_calls"])
        
        top_apis = sorted(api_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        top_api_names = [api for api, _ in top_apis]
        
        # Plot each API's call rate
        for api in top_api_names:
            rates = [m["rates"].get(api, 0) for m in self.metrics_history["api_calls"]]
            ax.plot(timestamps, rates, marker='.', markersize=3, linewidth=1, label=api)
        
        # Plot total call rate
        total_rates = [m["total"] / self.monitoring_interval for m in self.metrics_history["api_calls"]]
        ax.plot(timestamps, total_rates, 'k-', linewidth=2, label='Total')
        
        # Format plot
        ax.set_title('API Call Rates')
        ax.set_xlabel('Time')
        ax.set_ylabel('Calls per second')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.grid(True)
    
    def _plot_api_latency(self, ax):
        """
        Plot API latency.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
        """
        # Extract data
        timestamps = [m["timestamp"] for m in self.metrics_history["api_latency"]]
        weighted_avgs = [m["weighted_avg"] * 1000 for m in self.metrics_history["api_latency"]]  # Convert to ms
        
        # Get the top 5 APIs by latency
        all_apis = set()
        for m in self.metrics_history["api_latency"]:
            all_apis.update(m["averages"].keys())
        
        api_avg_latency = {}
        for api in all_apis:
            api_values = [m["averages"].get(api, 0) for m in self.metrics_history["api_latency"]]
            api_avg_latency[api] = np.mean([v for v in api_values if v > 0]) if api_values else 0
        
        top_apis = sorted(api_avg_latency.items(), key=lambda x: x[1], reverse=True)[:5]
        top_api_names = [api for api, _ in top_apis]
        
        # Plot each API's latency
        for api in top_api_names:
            latencies = [m["averages"].get(api, 0) * 1000 for m in self.metrics_history["api_latency"]]  # Convert to ms
            ax.plot(timestamps, latencies, marker='.', markersize=3, linewidth=1, label=api)
        
        # Plot weighted average latency
        ax.plot(timestamps, weighted_avgs, 'k-', linewidth=2, label='Weighted Avg')
        
        # Format plot
        ax.set_title('API Latency')
        ax.set_xlabel('Time')
        ax.set_ylabel('Latency (ms)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.grid(True)
    
    def _plot_error_rates(self, ax):
        """
        Plot API error rates.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
        """
        # Extract data
        timestamps = [m["timestamp"] for m in self.metrics_history["api_errors"]]
        
        # Get APIs with errors
        apis_with_errors = set()
        for m in self.metrics_history["api_errors"]:
            apis_with_errors.update([api for api, rate in m["rates"].items() if rate > 0])
        
        # Plot each API's error rate
        for api in apis_with_errors:
            error_rates = [m["rates"].get(api, 0) * 100 for m in self.metrics_history["api_errors"]]  # Convert to percentage
            ax.plot(timestamps, error_rates, marker='.', markersize=3, linewidth=1, label=api)
        
        # Format plot
        ax.set_title('API Error Rates')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error Rate (%)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.grid(True)
        ax.set_ylim(bottom=0)
    
    def _plot_data_volumes(self, ax):
        """
        Plot data volume rates.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
        """
        # Extract data
        timestamps = [m["timestamp"] for m in self.metrics_history["data_volumes"]]
        
        # Get data types
        data_types = set()
        for m in self.metrics_history["data_volumes"]:
            data_types.update(m["rates"].keys())
        
        # Plot each data type's volume rate
        for data_type in data_types:
            # Convert to KB/s
            rates = [m["rates"].get(data_type, 0) / 1024 for m in self.metrics_history["data_volumes"]]
            ax.plot(timestamps, rates, marker='.', markersize=3, linewidth=1, label=data_type)
        
        # Plot total volume rate
        total_rates = [m["total"] / self.monitoring_interval / 1024 for m in self.metrics_history["data_volumes"]]
        ax.plot(timestamps, total_rates, 'k-', linewidth=2, label='Total')
        
        # Format plot
        ax.set_title('Data Volume Rates')
        ax.set_xlabel('Time')
        ax.set_ylabel('KB per second')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.grid(True)
    
    def _plot_system_resources(self, ax):
        """
        Plot system resource usage.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
        """
        # Extract data
        timestamps = [m["timestamp"] for m in self.metrics_history["system_metrics"]]
        cpu_usage = [m["metrics"].get("cpu_percent", 0) for m in self.metrics_history["system_metrics"]]
        memory_usage = [m["metrics"].get("memory_percent", 0) for m in self.metrics_history["system_metrics"]]
        thread_counts = [m["metrics"].get("thread_count", 0) for m in self.metrics_history["system_metrics"]]
        
        # Plot CPU and memory usage on left y-axis
        ax.plot(timestamps, cpu_usage, 'r-', marker='.', markersize=3, linewidth=1, label='CPU Usage (%)')
        ax.plot(timestamps, memory_usage, 'b-', marker='.', markersize=3, linewidth=1, label='Memory Usage (%)')
        
        # Create right y-axis for thread count
        ax2 = ax.twinx()
        ax2.plot(timestamps, thread_counts, 'g-', marker='.', markersize=3, linewidth=1, label='Thread Count')
        
        # Format plot
        ax.set_title('System Resources')
        ax.set_xlabel('Time')
        ax.set_ylabel('Usage (%)')
        ax2.set_ylabel('Thread Count')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.grid(True)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_bottlenecks(self, ax):
        """
        Plot bottleneck occurrences.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
        """
        if not self.bottlenecks:
            ax.text(0.5, 0.5, "No bottlenecks detected", ha='center', va='center', transform=ax.transAxes)
            ax.set_title('System Bottlenecks')
            return
        
        # Group bottlenecks by type
        bottleneck_types = {}
        for bottleneck in self.bottlenecks:
            bottleneck_type = bottleneck['type']
            if bottleneck_type not in bottleneck_types:
                bottleneck_types[bottleneck_type] = []
            bottleneck_types[bottleneck_type].append(bottleneck)
        
        # Plot bottleneck occurrences over time
        timestamps = []
        types = []
        
        for bottleneck_type, bottlenecks in bottleneck_types.items():
            for bottleneck in bottlenecks:
                timestamps.append(bottleneck['timestamp'])
                types.append(bottleneck_type)
        
        # Convert to numeric representation for plotting
        unique_types = list(set(types))
        type_indices = [unique_types.index(t) for t in types]
        
        # Plot bottleneck points
        scatter = ax.scatter(timestamps, type_indices, c=type_indices, cmap='viridis', 
                           marker='s', s=100)
        
        # Format plot
        ax.set_title('System Bottlenecks')
        ax.set_xlabel('Time')
        ax.set_yticks(range(len(unique_types)))
        ax.set_yticklabels(unique_types)
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.grid(True, axis='x')
    
    def record_api_call(self, api_name, latency=None, success=True):
        """
        Record an API call for throughput monitoring.
        
        Args:
            api_name (str): Name of the API called
            latency (float, optional): Latency in seconds
            success (bool): Whether the call was successful
        """
        if not self.enable_monitoring:
            return
        
        self.api_call_queue.put({
            "api": api_name,
            "latency": latency,
            "success": success,
            "timestamp": time.time()
        })
    
    def record_data_volume(self, data_type, bytes_count):
        """
        Record data volume transferred.
        
        Args:
            data_type (str): Type of data
            bytes_count (int): Number of bytes transferred
        """
        if not self.enable_monitoring:
            return
        
        self.data_volume_queue.put({
            "type": data_type,
            "bytes": bytes_count,
            "timestamp": time.time()
        })
    
    def record_processing_time(self, process_name, processing_time):
        """
        Record time taken for a processing operation.
        
        Args:
            process_name (str): Name of the process
            processing_time (float): Time taken in seconds
        """
        if not self.enable_monitoring:
            return
        
        self.processing_time_queue.put({
            "process": process_name,
            "time": processing_time,
            "timestamp": time.time()
        })
    
    def start_timer(self, timer_name=None):
        """
        Start a timer for measuring processing time.
        
        Args:
            timer_name (str, optional): Name for the timer
            
        Returns:
            dict: Timer object
        """
        return {
            "name": timer_name,
            "start_time": time.time()
        }
    
    def stop_timer(self, timer):
        """
        Stop a timer and record the processing time.
        
        Args:
            timer (dict): Timer object from start_timer
            
        Returns:
            float: Elapsed time in seconds
        """
        if not isinstance(timer, dict) or "start_time" not in timer:
            self.logger.warning("Invalid timer object passed to stop_timer")
            return 0
        
        elapsed = time.time() - timer["start_time"]
        
        if "name" in timer and timer["name"]:
            self.record_processing_time(timer["name"], elapsed)
        
        return elapsed
    
    def measure_operation(self, operation_name):
        """
        Context manager for measuring operation time.
        
        Args:
            operation_name (str): Name of the operation
            
        Returns:
            OperationTimer: Context manager for timing
        """
        return OperationTimer(self, operation_name)
    
    def get_throughput_metrics(self):
        """
        Get the current throughput metrics.
        
        Returns:
            dict: Current throughput metrics
        """
        # Calculate rates from recent data
        current_time = time.time()
        elapsed = current_time - self.last_reset_time
        
        # API call metrics
        api_call_metrics = {
            "call_rates": {api: count / elapsed for api, count in self.api_calls.items()},
            "total_calls": sum(self.api_calls.values()),
            "calls_per_second": sum(self.api_calls.values()) / elapsed
        }
        
        # API error metrics
        api_error_metrics = {
            "error_rates": {api: self.api_errors[api] / max(1, self.api_calls[api]) for api in self.api_calls},
            "total_errors": sum(self.api_errors.values()),
            "error_percentage": sum(self.api_errors.values()) / max(1, sum(self.api_calls.values())) * 100
        }
        
        # API latency metrics
        all_latencies = []
        for latencies in self.api_latency.values():
            all_latencies.extend(latencies)
            
        api_latency_metrics = {
            "avg_latency": np.mean(all_latencies) if all_latencies else 0,
            "max_latency": np.max(all_latencies) if all_latencies else 0,
            "min_latency": np.min(all_latencies) if all_latencies else 0,
            "latency_95th": np.percentile(all_latencies, 95) if len(all_latencies) >= 20 else 0,
            "per_api": {api: np.mean(latencies) if latencies else 0 for api, latencies in self.api_latency.items()}
        }
        
        # Data volume metrics
        data_volume_metrics = {
            "volume_rates": {data_type: volume / elapsed for data_type, volume in self.data_volumes.items()},
            "total_volume": sum(self.data_volumes.values()),
            "bytes_per_second": sum(self.data_volumes.values()) / elapsed
        }
        
        # System metrics
        system_metrics = self.system_metrics if self.system_resource_tracking else {}
        
        # Combine all metrics
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "monitoring_interval_seconds": self.monitoring_interval,
            "elapsed_time": elapsed,
            "api_calls": api_call_metrics,
            "api_errors": api_error_metrics,
            "api_latency": api_latency_metrics,
            "data_volumes": data_volume_metrics,
            "system_metrics": system_metrics,
            "alerts": {
                "total_alerts": sum(self.alerts_triggered.values()),
                "by_type": dict(self.alerts_triggered)
            },
            "bottlenecks": len(self.bottlenecks)
        }
        
        return metrics
    
    def get_alert_summary(self):
        """
        Get a summary of triggered alerts.
        
        Returns:
            dict: Alert summary
        """
        return {
            "alerts_triggered": dict(self.alerts_triggered),
            "last_alert_times": {alert_type: datetime.datetime.fromtimestamp(timestamp).isoformat() 
                               for alert_type, timestamp in self.last_alert_time.items()},
            "bottlenecks": self.bottlenecks[-10:] if self.bottlenecks else []
        }
    
    def get_historical_metrics(self, metric_type=None, count=60):
        """
        Get historical metrics for analysis.
        
        Args:
            metric_type (str, optional): Type of metrics to get (api_calls, api_errors, api_latency, etc.)
            count (int): Number of data points to retrieve
            
        Returns:
            list: Historical metrics data
        """
        if not self.enable_monitoring:
            return []
        
        # If no specific type, return all metrics
        if metric_type is None:
            # Return a simplified version of all metrics
            return [
                {
                    "timestamp": m["timestamp"],
                    "api_calls_per_second": m["rates"].get("total", 0) if "rates" in m else 0,
                    "api_errors": m["total"] if metric_type == "api_errors" else 0,
                    "avg_latency": m["weighted_avg"] if metric_type == "api_latency" else 0,
                    "data_volume_per_second": m["rates"].get("total", 0) if metric_type == "data_volumes" else 0,
                    "cpu_percent": m["metrics"].get("cpu_percent", 0) if metric_type == "system_metrics" else 0,
                    "memory_percent": m["metrics"].get("memory_percent", 0) if metric_type == "system_metrics" else 0
                }
                for m in list(self.metrics_history["api_calls"])[-count:]
            ]
        
        # Return specific metric type if available
        if metric_type in self.metrics_history:
            return list(self.metrics_history[metric_type])[-count:]
        
        return []
    
    def generate_throughput_report(self):
        """
        Generate a comprehensive throughput report.
        
        Returns:
            dict: Throughput report
        """
        if not self.enable_monitoring:
            return {"status": "monitoring_disabled"}
        
        # Get current metrics
        current_metrics = self.get_throughput_metrics()
        
        # Calculate historical trends
        historical_api_calls = []
        historical_latency = []
        historical_error_rates = []
        historical_cpu = []
        historical_memory = []
        
        # Get metrics from the last hour (assuming 5-second intervals)
        history_count = min(720, len(self.metrics_history["api_calls"]))
        
        if history_count > 0:
            # API calls
            for i in range(history_count):
                idx = -(i + 1)
                if abs(idx) <= len(self.metrics_history["api_calls"]):
                    metric = self.metrics_history["api_calls"][idx]
                    total_calls = metric.get("total", 0)
                    historical_api_calls.append(total_calls / self.monitoring_interval)
            
            # Latency
            for i in range(history_count):
                idx = -(i + 1)
                if abs(idx) <= len(self.metrics_history["api_latency"]):
                    metric = self.metrics_history["api_latency"][idx]
                    weighted_avg = metric.get("weighted_avg", 0)
                    historical_latency.append(weighted_avg * 1000)  # Convert to ms
            
            # Error rates
            for i in range(history_count):
                idx = -(i + 1)
                if abs(idx) <= len(self.metrics_history["api_errors"]):
                    metric = self.metrics_history["api_errors"][idx]
                    total_errors = metric.get("total", 0)
                    error_rate = total_errors / max(1, self.metrics_history["api_calls"][idx].get("total", 1))
                    historical_error_rates.append(error_rate * 100)  # Convert to percentage
            
            # System metrics
            if self.system_resource_tracking:
                for i in range(history_count):
                    idx = -(i + 1)
                    if abs(idx) <= len(self.metrics_history["system_metrics"]):
                        metric = self.metrics_history["system_metrics"][idx]
                        metrics = metric.get("metrics", {})
                        historical_cpu.append(metrics.get("cpu_percent", 0))
                        historical_memory.append(metrics.get("memory_percent", 0))
        
        # Calculate trends
        api_call_trend = self._calculate_trend(historical_api_calls)
        latency_trend = self._calculate_trend(historical_latency)
        error_rate_trend = self._calculate_trend(historical_error_rates)
        cpu_trend = self._calculate_trend(historical_cpu)
        memory_trend = self._calculate_trend(historical_memory)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_metrics,
            api_call_trend,
            latency_trend,
            error_rate_trend,
            cpu_trend,
            memory_trend
        )
        
        # Compile report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "monitoring_period": f"{self.monitoring_interval} seconds",
            "metrics": current_metrics,
            "trends": {
                "api_calls": api_call_trend,
                "latency": latency_trend,
                "error_rate": error_rate_trend,
                "cpu_usage": cpu_trend,
                "memory_usage": memory_trend
            },
            "recommendations": recommendations,
            "bottlenecks": self.bottlenecks[-5:] if self.bottlenecks else [],
            "alerts": dict(self.alerts_triggered)
        }
        
        return report
    
    def _calculate_trend(self, values):
        """
        Calculate trend from a series of values.
        
        Args:
            values (list): List of values
            
        Returns:
            dict: Trend information
        """
        if not values or len(values) < 2:
            return {"direction": "stable", "change_percent": 0}
        
        # Use the first and last values to calculate trend
        first_value = values[-min(10, len(values))]  # Use up to 10 data points ago
        last_value = values[-1]
        
        # Calculate percent change
        if first_value == 0:
            if last_value == 0:
                change_percent = 0
            else:
                change_percent = 100  # Arbitrary large increase
        else:
            change_percent = ((last_value - first_value) / first_value) * 100
        
        # Determine trend direction
        if abs(change_percent) < 5:
            direction = "stable"
        elif change_percent > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {
            "direction": direction,
            "change_percent": change_percent
        }
    
    def _generate_recommendations(self, metrics, api_call_trend, latency_trend, error_rate_trend, cpu_trend, memory_trend):
        """
        Generate performance recommendations based on metrics and trends.
        
        Args:
            metrics (dict): Current metrics
            api_call_trend (dict): API call trend
            latency_trend (dict): Latency trend
            error_rate_trend (dict): Error rate trend
            cpu_trend (dict): CPU usage trend
            memory_trend (dict): Memory usage trend
            
        Returns:
            list: Recommendations
        """
        recommendations = []
        
        # Check high latency
        if metrics["api_latency"]["avg_latency"] > 0.5:  # More than 500ms
            recommendations.append({
                "type": "latency",
                "severity": "high" if metrics["api_latency"]["avg_latency"] > 1.0 else "medium",
                "message": f"High API latency detected ({metrics['api_latency']['avg_latency']*1000:.2f}ms). Consider optimizing API calls or adding caching."
            })
        
        # Check error rates
        if metrics["api_errors"]["error_percentage"] > 5:  # More than 5%
            recommendations.append({
                "type": "errors",
                "severity": "high" if metrics["api_errors"]["error_percentage"] > 10 else "medium",
                "message": f"High API error rate detected ({metrics['api_errors']['error_percentage']:.2f}%). Investigate error causes."
            })
        
        # Check system resources
        if self.system_resource_tracking:
            if metrics["system_metrics"].get("cpu_percent", 0) > 80:
                recommendations.append({
                    "type": "cpu",
                    "severity": "high" if metrics["system_metrics"].get("cpu_percent", 0) > 90 else "medium",
                    "message": f"High CPU usage detected ({metrics['system_metrics'].get('cpu_percent', 0):.2f}%). Consider optimizing processing or scaling up."
                })
                
            if metrics["system_metrics"].get("memory_percent", 0) > 80:
                recommendations.append({
                    "type": "memory",
                    "severity": "high" if metrics["system_metrics"].get("memory_percent", 0) > 90 else "medium",
                    "message": f"High memory usage detected ({metrics['system_metrics'].get('memory_percent', 0):.2f}%). Check for memory leaks or increase available memory."
                })
        
        # Check trends
        if latency_trend["direction"] == "increasing" and latency_trend["change_percent"] > 20:
            recommendations.append({
                "type": "trend",
                "severity": "medium",
                "message": f"Latency is increasing ({latency_trend['change_percent']:.2f}%). Monitor system performance and optimize response times."
            })
            
        if error_rate_trend["direction"] == "increasing" and error_rate_trend["change_percent"] > 20:
            recommendations.append({
                "type": "trend",
                "severity": "medium",
                "message": f"Error rate is increasing ({error_rate_trend['change_percent']:.2f}%). Investigate recent changes that might be causing errors."
            })
        
        return recommendations
    
    def shutdown(self):
        """Clean shutdown of the throughput monitor."""
        self.logger.info("Shutting down ThroughputMonitor")
        
        if not self.enable_monitoring:
            return
        
        # Signal thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        
        # Final logging of metrics
        self._log_metrics()
        
        # Generate final visualization
        if self.enable_visualization:
            self._generate_visualizations()


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor, operation_name):
        """
        Initialize the timer.
        
        Args:
            monitor (ThroughputMonitor): Monitor instance
            operation_name (str): Name of the operation
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting context."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.monitor.record_processing_time(self.operation_name, elapsed)
