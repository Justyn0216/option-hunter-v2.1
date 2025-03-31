"""
Resource Allocator Module

This module provides intelligent resource allocation for the Option Hunter trading system.
It dynamically allocates CPU, memory, and network resources among different system components
based on current workload, priorities, and performance requirements.
"""

import logging
import os
import json
import time
import threading
import psutil
import multiprocessing
import queue
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

class ResourceAllocator:
    """
    Intelligent system resource allocation manager.
    
    Features:
    - Dynamic CPU thread/process allocation
    - Memory usage optimization
    - Network bandwidth allocation
    - Adaptive prioritization of system components
    - Load balancing between trading strategies
    - Performance feedback optimization
    """
    
    def __init__(self, config, throughput_monitor=None):
        """
        Initialize the ResourceAllocator.
        
        Args:
            config (dict): Resource allocation configuration
            throughput_monitor (ThroughputMonitor, optional): Throughput monitor for performance metrics
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.throughput_monitor = throughput_monitor
        
        # Extract configuration
        self.resource_params = config.get("resource_allocation", {})
        
        # Default parameters
        self.enable_allocation = self.resource_params.get("enable_allocation", True)
        self.monitoring_interval = self.resource_params.get("monitoring_interval_seconds", 10)
        self.cpu_reserved_percent = self.resource_params.get("cpu_reserved_percent", 10)  # Reserve CPU for system tasks
        self.memory_reserved_percent = self.resource_params.get("memory_reserved_percent", 20)  # Reserve memory
        self.component_priorities = self.resource_params.get("component_priorities", {})
        self.default_priority = self.resource_params.get("default_priority", 5)  # 1-10 scale
        self.adaptive_allocation = self.resource_params.get("adaptive_allocation", True)
        self.learning_rate = self.resource_params.get("learning_rate", 0.1)
        
        # Set up system resources
        self.total_cpu_count = multiprocessing.cpu_count()
        self.available_cpu_count = max(1, int(self.total_cpu_count * (1 - self.cpu_reserved_percent / 100)))
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = int(self.total_memory * (1 - self.memory_reserved_percent / 100))
        
        # Resource allocation tracking
        self.allocated_resources = {}
        self.allocation_history = deque(maxlen=1000)
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Thread pools and worker management
        self.thread_pools = {}
        self.active_workers = {}
        self.component_queues = {}
        
        # Components and their current resource allocations
        self.registered_components = {}
        
        # Performance metrics
        self.component_metrics = defaultdict(dict)
        
        # Threading
        self.stop_event = threading.Event()
        self.allocation_thread = None
        self.allocation_lock = threading.RLock()
        
        # Create logs directory
        self.logs_dir = "logs/resources"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Start allocation thread if enabled
        if self.enable_allocation:
            self._start_allocation_thread()
            self.logger.info("ResourceAllocator initialized and allocation started")
        else:
            self.logger.info("ResourceAllocator initialized (allocation disabled)")
    
    def _start_allocation_thread(self):
        """Start the resource allocation thread."""
        if self.allocation_thread is None or not self.allocation_thread.is_alive():
            self.allocation_thread = threading.Thread(
                target=self._allocation_thread,
                daemon=True,
                name="ResourceAllocationThread"
            )
            self.allocation_thread.start()
            self.logger.debug("Resource allocation thread started")
    
    def _allocation_thread(self):
        """Background thread for monitoring and adjusting resource allocations."""
        last_log_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Get current system resource usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                # Update available resources based on current usage
                self._update_available_resources(cpu_percent, memory_percent)
                
                # Perform resource allocation
                self._allocate_resources()
                
                # Update performance metrics and learn from them
                if self.adaptive_allocation:
                    self._update_adaptive_allocation()
                
                # Log resource allocations periodically
                current_time = time.time()
                if current_time - last_log_time >= 60:  # Log every minute
                    self._log_allocations()
                    last_log_time = current_time
                    
            except Exception as e:
                self.logger.error(f"Error in resource allocation thread: {str(e)}")
            
            # Sleep until next allocation cycle
            self.stop_event.wait(self.monitoring_interval)
    
    def _update_available_resources(self, cpu_percent, memory_percent):
        """
        Update available resources based on current system usage.
        
        Args:
            cpu_percent (float): Current CPU usage percentage
            memory_percent (float): Current memory usage percentage
        """
        with self.allocation_lock:
            # Calculate available CPU cores based on current usage
            if cpu_percent > 90:
                # System is under heavy load, reduce available cores
                self.available_cpu_count = max(1, int(self.total_cpu_count * 0.5))
            elif cpu_percent > 75:
                # System is under moderate load
                self.available_cpu_count = max(1, int(self.total_cpu_count * 0.7))
            else:
                # System has available capacity
                self.available_cpu_count = max(1, int(self.total_cpu_count * (1 - self.cpu_reserved_percent / 100)))
            
            # Calculate available memory based on current usage
            available_percent = 100 - memory_percent
            memory_usage_factor = max(0.1, min(1.0, available_percent / (100 - self.memory_reserved_percent)))
            self.available_memory = int(self.total_memory * (1 - self.memory_reserved_percent / 100) * memory_usage_factor)
    
    def _allocate_resources(self):
        """Allocate resources to registered components based on priorities and load."""
        if not self.registered_components:
            return
        
        with self.allocation_lock:
            # Calculate total priority points
            total_priority = sum(component["priority"] for component in self.registered_components.values())
            
            if total_priority == 0:
                # Equal allocation if no priorities
                equal_share_cpu = max(1, self.available_cpu_count // len(self.registered_components))
                equal_share_memory = self.available_memory // len(self.registered_components)
                
                for component_id, component in self.registered_components.items():
                    allocation = {
                        "cpu_threads": equal_share_cpu,
                        "memory_bytes": equal_share_memory,
                        "priority_level": component["priority"]
                    }
                    self.allocated_resources[component_id] = allocation
                    
                    # Notify component of allocation if callback provided
                    if "allocation_callback" in component and callable(component["allocation_callback"]):
                        try:
                            component["allocation_callback"](allocation)
                        except Exception as e:
                            self.logger.error(f"Error in allocation callback for component {component_id}: {str(e)}")
            else:
                # Proportional allocation based on priority
                for component_id, component in self.registered_components.items():
                    priority_share = component["priority"] / total_priority
                    
                    # Calculate CPU allocation (at least 1 thread)
                    cpu_allocation = max(1, int(self.available_cpu_count * priority_share))
                    
                    # Calculate memory allocation
                    memory_allocation = int(self.available_memory * priority_share)
                    
                    # Store allocation
                    allocation = {
                        "cpu_threads": cpu_allocation,
                        "memory_bytes": memory_allocation,
                        "priority_level": component["priority"]
                    }
                    self.allocated_resources[component_id] = allocation
                    
                    # Adjust thread pool size if needed
                    if component_id in self.thread_pools:
                        current_size = self.thread_pools[component_id]._max_workers
                        if current_size != cpu_allocation:
                            self._resize_thread_pool(component_id, cpu_allocation)
                    
                    # Notify component of allocation if callback provided
                    if "allocation_callback" in component and callable(component["allocation_callback"]):
                        try:
                            component["allocation_callback"](allocation)
                        except Exception as e:
                            self.logger.error(f"Error in allocation callback for component {component_id}: {str(e)}")
            
            # Record allocation history
            self.allocation_history.append({
                "timestamp": datetime.now().isoformat(),
                "allocations": self.allocated_resources.copy(),
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                    "available_cpu": self.available_cpu_count,
                    "available_memory": self.available_memory
                }
            })
    
    def _update_adaptive_allocation(self):
        """Update component priorities based on performance metrics."""
        if not self.component_metrics:
            return
        
        with self.allocation_lock:
            for component_id, metrics in self.component_metrics.items():
                if component_id not in self.registered_components:
                    continue
                
                # Get performance metrics
                latency = metrics.get("latency", 0)
                throughput = metrics.get("throughput", 0)
                error_rate = metrics.get("error_rate", 0)
                
                # Only adjust if we have performance data
                if latency == 0 and throughput == 0:
                    continue
                
                # Update performance history
                self.performance_history[component_id].append({
                    "timestamp": datetime.now().isoformat(),
                    "latency": latency,
                    "throughput": throughput,
                    "error_rate": error_rate,
                    "allocation": self.allocated_resources.get(component_id, {})
                })
                
                # Calculate performance score (lower is better)
                # - Low latency is good
                # - High throughput is good
                # - Low error rate is good
                normalized_latency = min(1.0, latency / 1000.0)
                normalized_throughput = 1.0 - min(1.0, throughput / 100.0) if throughput > 0 else 1.0
                normalized_error_rate = min(1.0, error_rate)
                
                performance_score = (normalized_latency + normalized_error_rate) / 2
                
                # Adjust priority based on performance
                current_priority = self.registered_components[component_id]["priority"]
                
                # If performance is poor, increase priority
                if performance_score > 0.7:  # High score = poor performance
                    new_priority = min(10, current_priority + self.learning_rate)
                # If performance is good, slightly decrease priority to free resources
                elif performance_score < 0.3:  # Low score = good performance
                    new_priority = max(1, current_priority - self.learning_rate * 0.5)
                else:
                    new_priority = current_priority
                
                # Update priority if changed
                if new_priority != current_priority:
                    self.registered_components[component_id]["priority"] = new_priority
                    self.logger.debug(f"Adjusted priority for {component_id}: {current_priority:.2f} -> {new_priority:.2f} (score: {performance_score:.2f})")
    
    def _log_allocations(self):
        """Log current resource allocations to file."""
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "system_resources": {
                    "total_cpu": self.total_cpu_count,
                    "available_cpu": self.available_cpu_count,
                    "total_memory": self.total_memory,
                    "available_memory": self.available_memory,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent
                },
                "component_allocations": self.allocated_resources,
                "component_metrics": {k: dict(v) for k, v in self.component_metrics.items()}
            }
            
            log_file = os.path.join(self.logs_dir, f"resource_allocation_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Append to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_data) + "\n")
                
            # Clean up old metrics after logging
            for component_id in self.component_metrics:
                self.component_metrics[component_id].clear()
                
        except Exception as e:
            self.logger.error(f"Error logging resource allocations: {str(e)}")
    
    def _resize_thread_pool(self, component_id, new_size):
        """
        Resize a component's thread pool.
        
        Args:
            component_id (str): Component identifier
            new_size (int): New thread pool size
        """
        try:
            # Create a new thread pool with the new size
            old_pool = self.thread_pools.get(component_id)
            
            # Create new pool
            new_pool = ThreadPoolExecutor(max_workers=new_size, thread_name_prefix=f"{component_id}_worker")
            self.thread_pools[component_id] = new_pool
            
            # Shutdown old pool gracefully if it exists
            if old_pool:
                # Don't wait for tasks to complete as they may take too long
                old_pool.shutdown(wait=False)
                
            self.logger.debug(f"Resized thread pool for {component_id}: {getattr(old_pool, '_max_workers', 0)} -> {new_size} threads")
            
        except Exception as e:
            self.logger.error(f"Error resizing thread pool for {component_id}: {str(e)}")
    
    def register_component(self, component_id, initial_priority=None, allocation_callback=None):
        """
        Register a component for resource allocation.
        
        Args:
            component_id (str): Unique identifier for the component
            initial_priority (float, optional): Initial priority level (1-10)
            allocation_callback (callable, optional): Function to call when allocation changes
            
        Returns:
            bool: True if registration was successful
        """
        try:
            with self.allocation_lock:
                # Set initial priority from config, parameter, or default
                if component_id in self.component_priorities:
                    priority = self.component_priorities[component_id]
                elif initial_priority is not None:
                    priority = max(1, min(10, initial_priority))
                else:
                    priority = self.default_priority
                
                # Create component entry
                component = {
                    "id": component_id,
                    "priority": priority,
                    "registration_time": datetime.now().isoformat()
                }
                
                # Add callback if provided
                if allocation_callback and callable(allocation_callback):
                    component["allocation_callback"] = allocation_callback
                
                # Create thread pool for this component
                initial_threads = max(1, self.available_cpu_count // (len(self.registered_components) + 1))
                thread_pool = ThreadPoolExecutor(max_workers=initial_threads, thread_name_prefix=f"{component_id}_worker")
                
                # Create task queue
                component_queue = queue.Queue()
                
                # Store component data
                self.registered_components[component_id] = component
                self.thread_pools[component_id] = thread_pool
                self.component_queues[component_id] = component_queue
                
                # Initial allocation
                allocation = {
                    "cpu_threads": initial_threads,
                    "memory_bytes": self.available_memory // (len(self.registered_components) + 1),
                    "priority_level": priority
                }
                self.allocated_resources[component_id] = allocation
                
                # Call allocation callback if provided
                if "allocation_callback" in component:
                    try:
                        component["allocation_callback"](allocation)
                    except Exception as e:
                        self.logger.error(f"Error in allocation callback for {component_id}: {str(e)}")
                
                self.logger.info(f"Registered component {component_id} with priority {priority}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error registering component {component_id}: {str(e)}")
            return False
    
    def unregister_component(self, component_id):
        """
        Unregister a component from resource allocation.
        
        Args:
            component_id (str): Component identifier
            
        Returns:
            bool: True if unregistration was successful
        """
        try:
            with self.allocation_lock:
                if component_id not in self.registered_components:
                    self.logger.warning(f"Component {component_id} not registered")
                    return False
                
                # Shutdown thread pool
                if component_id in self.thread_pools:
                    thread_pool = self.thread_pools[component_id]
                    thread_pool.shutdown(wait=False)
                    del self.thread_pools[component_id]
                
                # Remove from registered components
                del self.registered_components[component_id]
                
                # Remove from allocated resources
                if component_id in self.allocated_resources:
                    del self.allocated_resources[component_id]
                
                # Remove queue
                if component_id in self.component_queues:
                    del self.component_queues[component_id]
                
                self.logger.info(f"Unregistered component {component_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error unregistering component {component_id}: {str(e)}")
            return False
    
    def submit_task(self, component_id, task_function, *args, **kwargs):
        """
        Submit a task to be executed by a component's thread pool.
        
        Args:
            component_id (str): Component identifier
            task_function (callable): Function to execute
            *args, **kwargs: Arguments for the task function
            
        Returns:
            concurrent.futures.Future: Future object for the task, or None if submission failed
        """
        try:
            if component_id not in self.thread_pools:
                self.logger.warning(f"Component {component_id} does not have a thread pool")
                return None
            
            # Submit task to thread pool
            thread_pool = self.thread_pools[component_id]
            future = thread_pool.submit(task_function, *args, **kwargs)
            
            # Update active workers count
            self.active_workers[component_id] = self.active_workers.get(component_id, 0) + 1
            
            # Add completion callback to update worker count
            def done_callback(future):
                try:
                    self.active_workers[component_id] = max(0, self.active_workers.get(component_id, 0) - 1)
                    
                    # Check for exceptions
                    if future.exception():
                        self.logger.error(f"Error in task from {component_id}: {future.exception()}")
                except Exception as e:
                    self.logger.error(f"Error in task completion callback: {str(e)}")
            
            future.add_done_callback(done_callback)
            return future
            
        except Exception as e:
            self.logger.error(f"Error submitting task for {component_id}: {str(e)}")
            return None
    
    def update_priority(self, component_id, priority):
        """
        Update a component's priority.
        
        Args:
            component_id (str): Component identifier
            priority (float): New priority level (1-10)
            
        Returns:
            bool: True if update was successful
        """
        try:
            with self.allocation_lock:
                if component_id not in self.registered_components:
                    self.logger.warning(f"Component {component_id} not registered")
                    return False
                
                # Update priority
                self.registered_components[component_id]["priority"] = max(1, min(10, priority))
                self.logger.debug(f"Updated priority for {component_id} to {priority}")
                
                # Trigger reallocation in next cycle
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating priority for {component_id}: {str(e)}")
            return False
    
    def report_metrics(self, component_id, metrics):
        """
        Report performance metrics for a component.
        
        Args:
            component_id (str): Component identifier
            metrics (dict): Performance metrics (latency, throughput, error_rate, etc.)
            
        Returns:
            bool: True if metrics were recorded successfully
        """
        try:
            with self.allocation_lock:
                if component_id not in self.registered_components:
                    self.logger.warning(f"Component {component_id} not registered")
                    return False
                
                # Update metrics
                self.component_metrics[component_id].update(metrics)
                
                # Pass to throughput monitor if available
                if self.throughput_monitor:
                    self.throughput_monitor.report_component_metrics(component_id, metrics)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error reporting metrics for {component_id}: {str(e)}")
            return False
    
    def get_allocation(self, component_id):
        """
        Get current resource allocation for a component.
        
        Args:
            component_id (str): Component identifier
            
        Returns:
            dict: Current resource allocation or None if component not found
        """
        with self.allocation_lock:
            return self.allocated_resources.get(component_id)
    
    def get_allocation_history(self, component_id=None, last_n=None):
        """
        Get resource allocation history.
        
        Args:
            component_id (str, optional): Filter by component
            last_n (int, optional): Limit to last N entries
            
        Returns:
            list: Allocation history
        """
        with self.allocation_lock:
            history = list(self.allocation_history)
            
            # Filter by component if specified
            if component_id:
                history = [
                    entry for entry in history
                    if component_id in entry.get("allocations", {})
                ]
                
                # Extract just the component's allocations
                if history:
                    history = [
                        {
                            "timestamp": entry["timestamp"],
                            "allocation": entry["allocations"].get(component_id),
                            "system": entry.get("system", {})
                        }
                        for entry in history
                    ]
            
            # Limit to last N entries if specified
            if last_n and last_n > 0:
                history = history[-last_n:]
                
            return history
    
    def get_performance_history(self, component_id=None):
        """
        Get performance history for components.
        
        Args:
            component_id (str, optional): Filter by component
            
        Returns:
            dict: Performance history by component
        """
        with self.allocation_lock:
            if component_id:
                if component_id in self.performance_history:
                    return list(self.performance_history[component_id])
                return []
            
            # Return all components' history
            return {k: list(v) for k, v in self.performance_history.items()}
    
    def shutdown(self):
        """Shutdown the resource allocator and release resources."""
        self.logger.info("Shutting down ResourceAllocator")
        
        # Stop allocation thread
        self.stop_event.set()
        
        if self.allocation_thread and self.allocation_thread.is_alive():
            self.allocation_thread.join(timeout=5)
        
        # Shutdown all thread pools
        with self.allocation_lock:
            for component_id, thread_pool in list(self.thread_pools.items()):
                self.logger.debug(f"Shutting down thread pool for {component_id}")
                thread_pool.shutdown(wait=False)
            
            self.thread_pools.clear()
            self.registered_components.clear()
            self.allocated_resources.clear()
        
        self.logger.info("ResourceAllocator shutdown complete")
