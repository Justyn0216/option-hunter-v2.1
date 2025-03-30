"""
Latency Optimizer Module

This module provides functionality for optimizing system latency in the Option Hunter trading system.
It includes network optimization, request batching, caching, and monitoring
of system performance metrics.
"""

import logging
import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
import socket
import requests
from concurrent.futures import ThreadPoolExecutor
import psutil
import platform
import multiprocessing

class LatencyOptimizer:
    """
    System latency optimization and monitoring module.
    
    Features:
    - Request batching and throttling
    - Response caching
    - Network optimization
    - System resource monitoring
    - Performance benchmarking
    - Adaptive latency tuning
    """
    
    def __init__(self, config):
        """
        Initialize the LatencyOptimizer.
        
        Args:
            config (dict): Latency optimization configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Extract configuration
        self.latency_params = config.get("latency_optimization", {})
        
        # Default parameters
        self.enable_caching = self.latency_params.get("enable_caching", True)
        self.cache_ttl = self.latency_params.get("cache_ttl_seconds", 5)
        self.batch_requests = self.latency_params.get("batch_requests", True)
        self.max_batch_size = self.latency_params.get("max_batch_size", 100)
        self.throttle_requests = self.latency_params.get("throttle_requests", True)
        self.min_request_interval = self.latency_params.get("min_request_interval_ms", 100) / 1000.0  # Convert to seconds
        self.enable_monitoring = self.latency_params.get("enable_monitoring", True)
        self.monitoring_interval = self.latency_params.get("monitoring_interval_seconds", 60)
        self.adaptive_tuning = self.latency_params.get("adaptive_tuning", True)
        self.network_timeout = self.latency_params.get("network_timeout_seconds", 10)
        self.max_worker_threads = self.latency_params.get("max_worker_threads", 10)
        
        # Initialize caches
        self.response_cache = {}
        self.cache_timestamps = {}
        
        # Initialize request tracking
        self.request_timestamps = {}
        self.request_batch_queue = queue.Queue()
        self.batch_processing_thread = None
        self.stop_event = threading.Event()
        
        # Initialize monitoring
        self.latency_metrics = []
        self.system_metrics = []
        self.monitoring_thread = None
        
        # Initialize thread pool for parallel requests
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_worker_threads)
        
        # Create directory for latency metrics
        self.logs_dir = "logs/latency"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Start monitoring and batch processing if enabled
        if self.enable_monitoring:
            self._start_monitoring()
        
        if self.batch_requests:
            self._start_batch_processing()
        
        self.logger.info("LatencyOptimizer initialized")
    
    def _start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(
                target=self._monitor_system_performance,
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.debug("Monitoring thread started")
    
    def _start_batch_processing(self):
        """Start the batch processing thread."""
        if self.batch_processing_thread is None or not self.batch_processing_thread.is_alive():
            self.batch_processing_thread = threading.Thread(
                target=self._process_request_batch,
                daemon=True
            )
            self.batch_processing_thread.start()
            self.logger.debug("Batch processing thread started")
    
    def _monitor_system_performance(self):
        """Monitor system performance metrics in a background thread."""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Add to metrics history
                self.system_metrics.append(metrics)
                
                # Trim metrics history (keep last 24 hours)
                max_samples = int(86400 / self.monitoring_interval)  # 24 hours worth of samples
                if len(self.system_metrics) > max_samples:
                    self.system_metrics = self.system_metrics[-max_samples:]
                
                # Log metrics to file periodically (every 10 samples)
                if len(self.system_metrics) % 10 == 0:
                    self._log_metrics()
                
                # Adjust parameters if adaptive tuning is enabled
                if self.adaptive_tuning:
                    self._tune_parameters()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring thread: {str(e)}")
            
            # Sleep until next monitoring interval
            self.stop_event.wait(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """
        Collect system performance metrics.
        
        Returns:
            dict: System metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io": psutil.disk_io_counters() if hasattr(psutil, 'disk_io_counters') else None,
            "network_io": psutil.net_io_counters(),
            "request_latency": self._calculate_average_latency(),
            "cache_hit_ratio": self._calculate_cache_hit_ratio(),
            "batch_queue_size": self.request_batch_queue.qsize() if self.batch_requests else 0,
            "thread_pool_active": len([t for t in self.thread_pool._threads if t.is_alive()]),
            "thread_pool_max": self.max_worker_threads
        }
        
        return metrics
    
    def _calculate_average_latency(self):
        """
        Calculate average request latency from recent metrics.
        
        Returns:
            float: Average latency in milliseconds
        """
        if not self.latency_metrics:
            return 0
        
        # Get latencies from last minute
        recent_latencies = [
            metric["latency"] for metric in self.latency_metrics
            if (datetime.now() - datetime.fromisoformat(metric["timestamp"])).total_seconds() < 60
        ]
        
        if not recent_latencies:
            return 0
        
        return sum(recent_latencies) / len(recent_latencies)
    
    def _calculate_cache_hit_ratio(self):
        """
        Calculate cache hit ratio from recent metrics.
        
        Returns:
            float: Cache hit ratio (0-1)
        """
        # Count hits and misses from last hour
        hits = 0
        misses = 0
        
        for metric in self.latency_metrics:
            if (datetime.now() - datetime.fromisoformat(metric["timestamp"])).total_seconds() < 3600:
                if metric.get("cache_hit", False):
                    hits += 1
                else:
                    misses += 1
        
        total = hits + misses
        if total == 0:
            return 0
        
        return hits / total
    
    def _tune_parameters(self):
        """
        Adaptively tune system parameters based on performance metrics.
        """
        # Skip if not enough metrics collected
        if len(self.system_metrics) < 5:
            return
        
        # Get latest metrics
        latest = self.system_metrics[-1]
        
        # Adjust cache TTL based on market volatility and latency
        avg_latency = latest["request_latency"]
        if avg_latency > 500:  # If latency > 500ms
            # Increase cache TTL to reduce requests
            new_cache_ttl = min(self.cache_ttl * 1.5, 30)  # Max 30 seconds
            if new_cache_ttl != self.cache_ttl:
                self.logger.info(f"Adjusting cache TTL: {self.cache_ttl}s -> {new_cache_ttl}s due to high latency")
                self.cache_ttl = new_cache_ttl
        elif avg_latency < 100 and self.cache_ttl > 5:  # If latency is low and TTL is high
            # Decrease cache TTL for more fresh data
            new_cache_ttl = max(self.cache_ttl * 0.8, 1)  # Min 1 second
            if new_cache_ttl != self.cache_ttl:
                self.logger.info(f"Adjusting cache TTL: {self.cache_ttl}s -> {new_cache_ttl}s due to low latency")
                self.cache_ttl = new_cache_ttl
        
        # Adjust thread pool size based on CPU usage
        cpu_percent = latest["cpu_percent"]
        if cpu_percent > 80 and self.max_worker_threads > 4:
            # Reduce threads if CPU is overloaded
            new_thread_count = max(int(self.max_worker_threads * 0.8), 4)
            if new_thread_count != self.max_worker_threads:
                self.logger.info(f"Reducing thread pool: {self.max_worker_threads} -> {new_thread_count} due to high CPU usage")
                self.max_worker_threads = new_thread_count
                # Recreate thread pool
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = ThreadPoolExecutor(max_workers=self.max_worker_threads)
        elif cpu_percent < 50 and self.max_worker_threads < 20:
            # Increase threads if CPU has capacity
            new_thread_count = min(int(self.max_worker_threads * 1.25), 20)
            if new_thread_count != self.max_worker_threads:
                self.logger.info(f"Increasing thread pool: {self.max_worker_threads} -> {new_thread_count} due to low CPU usage")
                self.max_worker_threads = new_thread_count
                # Recreate thread pool
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = ThreadPoolExecutor(max_workers=self.max_worker_threads)
        
        # Adjust batch size based on queue backlog
        batch_queue_size = latest["batch_queue_size"]
        if batch_queue_size > self.max_batch_size * 2:
            # Increase batch size if queue is backing up
            new_batch_size = min(self.max_batch_size * 1.5, 200)
            if new_batch_size != self.max_batch_size:
                self.logger.info(f"Increasing batch size: {self.max_batch_size} -> {int(new_batch_size)} due to queue backlog")
                self.max_batch_size = int(new_batch_size)
        elif batch_queue_size < self.max_batch_size * 0.5 and self.max_batch_size > 50:
            # Decrease batch size if queue is small
            new_batch_size = max(self.max_batch_size * 0.8, 50)
            if new_batch_size != self.max_batch_size:
                self.logger.info(f"Decreasing batch size: {self.max_batch_size} -> {int(new_batch_size)} due to small queue")
                self.max_batch_size = int(new_batch_size)
    
    def _log_metrics(self):
        """Log performance metrics to file."""
        try:
            # Create filename based on date
            log_file = os.path.join(self.logs_dir, f"latency_metrics_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Prepare metrics for logging
            system_metrics_log = []
            for metric in self.system_metrics:
                # Convert non-serializable objects
                log_metric = {}
                for key, value in metric.items():
                    if key in ["disk_io", "network_io"]:
                        if value:
                            log_metric[key] = {k: v for k, v in value._asdict().items()} if hasattr(value, '_asdict') else str(value)
                        else:
                            log_metric[key] = None
                    else:
                        log_metric[key] = value
                system_metrics_log.append(log_metric)
            
            # Save to file
            with open(log_file, 'w') as f:
                json.dump({
                    "system_metrics": system_metrics_log,
                    "latency_metrics": self.latency_metrics[-100:],  # Only log recent latency metrics
                    "config": {
                        "cache_ttl": self.cache_ttl,
                        "max_worker_threads": self.max_worker_threads,
                        "max_batch_size": self.max_batch_size,
                        "min_request_interval": self.min_request_interval
                    }
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging metrics: {str(e)}")
    
    def _process_request_batch(self):
        """Process batched requests in a background thread."""
        while not self.stop_event.is_set():
            try:
                batch = []
                
                # Collect requests for the current batch
                while len(batch) < self.max_batch_size:
                    try:
                        # Get with timeout to allow checking stop_event
                        item = self.request_batch_queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        # If queue is empty, break out of collection loop
                        break
                
                # Process batch if not empty
                if batch:
                    self._execute_batch(batch)
                
            except Exception as e:
                self.logger.error(f"Error in batch processing thread: {str(e)}")
            
            # Short sleep to prevent high CPU usage if queue is empty
            if self.request_batch_queue.empty():
                self.stop_event.wait(0.1)
    
    def _execute_batch(self, batch):
        """
        Execute a batch of requests.
        
        Args:
            batch (list): List of request items containing (future, method, args, kwargs)
        """
        # Group by method to execute similar requests together
        method_groups = {}
        for future, method, args, kwargs in batch:
            method_key = id(method)  # Use method id as key
            if method_key not in method_groups:
                method_groups[method_key] = {
                    'method': method,
                    'items': []
                }
            method_groups[method_key]['items'].append((future, args, kwargs))
        
        # Process each method group
        for group_info in method_groups.values():
            method = group_info['method']
            items = group_info['items']
            
            try:
                # Check if method supports batch execution
                if hasattr(method, 'batch_execute'):
                    # Prepare batch arguments
                    batch_args = [args for _, args, _ in items]
                    batch_kwargs = [kwargs for _, _, kwargs in items]
                    
                    # Execute batch request
                    start_time = time.time()
                    batch_results = method.batch_execute(batch_args, batch_kwargs)
                    end_time = time.time()
                    
                    # Set results for each future
                    for i, (future, _, _) in enumerate(items):
                        if i < len(batch_results):
                            future.set_result((batch_results[i], end_time - start_time))
                        else:
                            future.set_exception(Exception("Batch execution returned fewer results than expected"))
                else:
                    # Execute individually
                    for future, args, kwargs in items:
                        self._execute_single_request(future, method, args, kwargs)
            except Exception as e:
                # Set exception for all futures in the group
                for future, _, _ in items:
                    future.set_exception(e)
    
    def _execute_single_request(self, future, method, args, kwargs):
        """
        Execute a single request.
        
        Args:
            future (concurrent.futures.Future): Future to set result on
            method (callable): Method to call
            args (tuple): Positional arguments
            kwargs (dict): Keyword arguments
        """
        try:
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            
            future.set_result((result, end_time - start_time))
        except Exception as e:
            future.set_exception(e)
    
    def optimized_request(self, method, *args, **kwargs):
        """
        Make an optimized request with caching, batching, and latency tracking.
        
        Args:
            method (callable): Method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            object: Result of the method call
        """
        # Generate cache key
        cache_key = self._generate_cache_key(method, args, kwargs)
        
        # Check cache if enabled
        if self.enable_caching and cache_key in self.response_cache:
            cache_timestamp = self.cache_timestamps.get(cache_key, 0)
            if time.time() - cache_timestamp < self.cache_ttl:
                # Cache hit
                self._record_latency(0, method.__name__, True)
                return self.response_cache[cache_key]
        
        # Apply request throttling if enabled
        if self.throttle_requests:
            self._throttle_request(method.__name__)
        
        # Use batch processing if enabled
        if self.batch_requests and not kwargs.get('bypass_batch', False):
            # Remove bypass_batch from kwargs if present
            if 'bypass_batch' in kwargs:
                kwargs.pop('bypass_batch')
                
            # Create future for this request
            future = self.thread_pool.submit(lambda: None)
            
            # Add to batch queue
            self.request_batch_queue.put((future, method, args, kwargs))
            
            try:
                # Wait for result with timeout
                result, latency = future.result(timeout=self.network_timeout)
                
                # Update cache
                if self.enable_caching:
                    self.response_cache[cache_key] = result
                    self.cache_timestamps[cache_key] = time.time()
                
                # Record latency
                self._record_latency(latency * 1000, method.__name__, False)
                
                return result
            except Exception as e:
                self.logger.error(f"Error in batched request to {method.__name__}: {str(e)}")
                raise
        else:
            # Direct execution
            start_time = time.time()
            try:
                result = method(*args, **kwargs)
                end_time = time.time()
                
                # Update cache
                if self.enable_caching:
                    self.response_cache[cache_key] = result
                    self.cache_timestamps[cache_key] = time.time()
                
                # Record latency
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                self._record_latency(latency, method.__name__, False)
                
                return result
            except Exception as e:
                self.logger.error(f"Error in direct request to {method.__name__}: {str(e)}")
                raise
    
    def _generate_cache_key(self, method, args, kwargs):
        """
        Generate a unique cache key for a method call.
        
        Args:
            method (callable): Method being called
            args (tuple): Positional arguments
            kwargs (dict): Keyword arguments
            
        Returns:
            str: Cache key
        """
        # Create key from method name and arguments
        method_name = method.__name__ if hasattr(method, '__name__') else str(method)
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        return f"{method_name}:{hash(args_str + kwargs_str)}"
    
    def _throttle_request(self, method_name):
        """
        Apply throttling to a request if needed.
        
        Args:
            method_name (str): Name of the method being called
        """
        current_time = time.time()
        
        # Check last request time for this method
        last_time = self.request_timestamps.get(method_name, 0)
        time_since_last = current_time - last_time
        
        # If request is too soon after the previous one, sleep
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        # Update timestamp
        self.request_timestamps[method_name] = time.time()
    
    def _record_latency(self, latency, method_name, cache_hit):
        """
        Record latency metrics for a request.
        
        Args:
            latency (float): Request latency in milliseconds
            method_name (str): Name of the method called
            cache_hit (bool): Whether the request was served from cache
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "method": method_name,
            "latency": latency,
            "cache_hit": cache_hit
        }
        
        self.latency_metrics.append(metric)
        
        # Trim metrics history (keep last hour)
        max_metrics = 3600  # 1 hour of per-second metrics
        if len(self.latency_metrics) > max_metrics:
            self.latency_metrics = self.latency_metrics[-max_metrics:]
    
    def clear_cache(self, method_name=None):
        """
        Clear the response cache.
        
        Args:
            method_name (str, optional): If provided, only clear cache for this method
        """
        if method_name:
            # Clear only for specific method
            keys_to_remove = []
            for key in self.response_cache:
                if key.startswith(f"{method_name}:"):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.response_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
            
            self.logger.debug(f"Cleared cache for method: {method_name}")
        else:
            # Clear entire cache
            self.response_cache = {}
            self.cache_timestamps = {}
            self.logger.debug("Cleared entire response cache")
    
    def get_network_stats(self):
        """
        Get network statistics and latency information.
        
        Returns:
            dict: Network statistics
        """
        # Basic network info
        hostname = socket.gethostname()
        network_info = {
            "hostname": hostname,
            "ip_address": socket.gethostbyname(hostname),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "net_connections": len(psutil.net_connections()),
            "net_io_counters": {k: v for k, v in psutil.net_io_counters()._asdict().items()} if hasattr(psutil, 'net_io_counters') else None
        }
        
        # Test ping to common services
        ping_targets = [
            "www.google.com",
            "www.amazon.com",
            "api.tradier.com"
        ]
        
        ping_results = {}
        for target in ping_targets:
            ping_results[target] = self._measure_ping(target)
        
        network_info["ping_results"] = ping_results
        
        # Add recent latency metrics
        if self.latency_metrics:
            recent_metrics = [
                metric for metric in self.latency_metrics
                if (datetime.now() - datetime.fromisoformat(metric["timestamp"])).total_seconds() < 300
            ]
            
            if recent_metrics:
                network_info["recent_api_latency"] = {
                    "average_ms": sum(m["latency"] for m in recent_metrics) / len(recent_metrics),
                    "min_ms": min(m["latency"] for m in recent_metrics),
                    "max_ms": max(m["latency"] for m in recent_metrics),
                    "samples": len(recent_metrics)
                }
        
        return network_info
    
    def _measure_ping(self, host, port=80, timeout=2):
        """
        Measure ping time to a host.
        
        Args:
            host (str): Host to ping
            port (int): Port to connect to
            timeout (float): Connection timeout in seconds
            
        Returns:
            float or None: Ping time in milliseconds, or None if failed
        """
        try:
            start_time = time.time()
            
            # Try socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            try:
                sock.connect((host, port))
                sock.shutdown(socket.SHUT_RDWR)
                return (time.time() - start_time) * 1000  # Convert to milliseconds
            finally:
                sock.close()
                
        except Exception as e:
            self.logger.debug(f"Ping to {host} failed: {str(e)}")
            return None
    
    def optimize_connection(self):
        """
        Optimize network connection settings.
        
        Returns:
            dict: Optimization results
        """
        results = {
            "optimizations_applied": [],
            "current_settings": {}
        }
        
        # Optimize socket settings if possible
        try:
            # Increase socket buffer sizes
            socket.setdefaulttimeout(self.network_timeout)
            results["optimizations_applied"].append("socket_timeout")
            results["current_settings"]["socket_timeout"] = self.network_timeout
            
            # Other optimizations are platform-specific and typically require root privileges
            
        except Exception as e:
            self.logger.error(f"Error optimizing socket settings: {str(e)}")
        
        # Optimize HTTP connection pooling for requests library
        try:
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.max_worker_threads,
                pool_maxsize=self.max_worker_threads * 2,
                max_retries=3
            )
            
            # Create a session with the adapter
            session = requests.Session()
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            # Store the session for future use
            self.http_session = session
            
            results["optimizations_applied"].append("http_connection_pooling")
            results["current_settings"]["pool_connections"] = self.max_worker_threads
            results["current_settings"]["pool_maxsize"] = self.max_worker_threads * 2
            
        except Exception as e:
            self.logger.error(f"Error optimizing HTTP connections: {str(e)}")
        
        # Log results
        self.logger.info(f"Connection optimization applied {len(results['optimizations_applied'])} settings")
        
        return results
    
    def benchmark_system(self, duration=10, api_method=None):
        """
        Run a system benchmark to measure performance.
        
        Args:
            duration (int): Benchmark duration in seconds
            api_method (callable, optional): API method to benchmark
            
        Returns:
            dict: Benchmark results
        """
        self.logger.info(f"Starting system benchmark for {duration} seconds")
        
        results = {
            "system_info": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": multiprocessing.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
            },
            "benchmark_configuration": {
                "duration": duration,
                "method": api_method.__name__ if api_method else "internal",
                "thread_pool_size": self.max_worker_threads,
                "cache_enabled": self.enable_caching,
                "batch_processing": self.batch_requests
            },
            "results": {}
        }
        
        # Start benchmark
        start_time = time.time()
        end_time = start_time + duration
        
        if api_method:
            # Benchmark specific API method
            requests_count = 0
            errors_count = 0
            latencies = []
            
            while time.time() < end_time:
                try:
                    # Make request and measure time
                    start_req = time.time()
                    api_method()
                    end_req = time.time()
                    
                    latency = (end_req - start_req) * 1000  # ms
                    latencies.append(latency)
                    requests_count += 1
                    
                    # Short sleep to prevent overwhelming the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    errors_count += 1
                    self.logger.debug(f"Benchmark request error: {str(e)}")
            
            # Calculate stats
            if latencies:
                results["results"] = {
                    "requests_count": requests_count,
                    "errors_count": errors_count,
                    "requests_per_second": requests_count / duration,
                    "average_latency_ms": sum(latencies) / len(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "error_rate": errors_count / (requests_count + errors_count) if (requests_count + errors_count) > 0 else 0
                }
            
        else:
            # Benchmark system performance
            cpu_samples = []
            memory_samples = []
            disk_io_samples = []
            net_io_samples = []
            
            while time.time() < end_time:
                # Sample system metrics
                cpu_percent = psutil.cpu_percent(interval=0.5)
                memory_percent = psutil.virtual_memory().percent
                
                cpu_samples.append(cpu_percent)
                memory_samples.append(memory_percent)
                
                if hasattr(psutil, 'disk_io_counters'):
                    disk_io = psutil.disk_io_counters()
                    disk_io_samples.append((disk_io.read_bytes, disk_io.write_bytes))
                
                net_io = psutil.net_io_counters()
                net_io_samples.append((net_io.bytes_sent, net_io.bytes_recv))
                
                # Short sleep between samples
                time.sleep(0.5)
            
            # Calculate disk and network throughput
            disk_throughput = None
            if disk_io_samples and len(disk_io_samples) > 1:
                first = disk_io_samples[0]
                last = disk_io_samples[-1]
                elapsed = duration
                disk_throughput = {
                    "read_bytes_per_sec": (last[0] - first[0]) / elapsed,
                    "write_bytes_per_sec": (last[1] - first[1]) / elapsed
                }
            
            net_throughput = None
            if net_io_samples and len(net_io_samples) > 1:
                first = net_io_samples[0]
                last = net_io_samples[-1]
                elapsed = duration
                net_throughput = {
                    "sent_bytes_per_sec": (last[0] - first[0]) / elapsed,
                    "recv_bytes_per_sec": (last[1] - first[1]) / elapsed
                }
            
            results["results"] = {
                "avg_cpu_percent": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
                "avg_memory_percent": sum(memory_samples) / len(memory_samples) if memory_samples else 0,
                "disk_throughput": disk_throughput,
                "network_throughput": net_throughput
            }
        
        self.logger.info(f"Benchmark completed in {time.time() - start_time:.2f} seconds")
        
        return results
    
    def shutdown(self):
        """Clean shutdown the optimizer."""
        self.logger.info("Shutting down LatencyOptimizer")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        
        if self.batch_processing_thread and self.batch_processing_thread.is_alive():
            self.batch_processing_thread.join(timeout=2)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Final log of metrics
        self._log_metrics()
