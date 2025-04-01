"""
Batch Optimizer Module

This module provides batch processing optimization for the Option Hunter system.
It optimizes data batch sizes, chunk processing, and scheduling to maximize
throughput and minimize memory usage for data processing tasks.
"""

import logging
import time
import threading
import queue
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, deque
import psutil
import multiprocessing
import concurrent.futures
import math
import heapq
import os

class BatchOptimizer:
    """
    Batch processing optimizer for data processing tasks.
    
    Features:
    - Adaptive batch size optimization
    - Memory-efficient chunked processing
    - Workload scheduling and prioritization
    - Throughput monitoring and optimization
    - Task cancellation and rescheduling
    - Support for pandas, numpy, and custom data types
    """
    
    def __init__(self, config=None):
        """
        Initialize the BatchOptimizer.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration
        self.batch_params = self.config.get("batch_optimization", {})
        
        # Default parameters
        self.enable_optimization = self.batch_params.get("enable_optimization", True)
        self.min_batch_size = self.batch_params.get("min_batch_size", 64)
        self.max_batch_size = self.batch_params.get("max_batch_size", 8192)
        self.initial_batch_size = self.batch_params.get("initial_batch_size", 1024)
        self.memory_target_percent = self.batch_params.get("memory_target_percent", 70)
        self.throughput_window = self.batch_params.get("throughput_window", 10)
        self.learning_rate = self.batch_params.get("learning_rate", 0.2)
        self.prioritize_memory = self.batch_params.get("prioritize_memory", True)
        
        # Task execution tracking
        self.task_history = defaultdict(lambda: deque(maxlen=100))
        self.batch_sizes = {}
        self.throughput_history = {}
        self.processing_time_history = {}
        
        # Worker pool
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.batch_params.get("max_workers", max(4, multiprocessing.cpu_count())),
            thread_name_prefix="BatchOptimizerWorker"
        )
        
        # Task queues and management
        self.task_queues = defaultdict(lambda: queue.PriorityQueue())
        self.active_tasks = {}
        self.stop_event = threading.Event()
        self.worker_threads = {}
        self.task_locks = defaultdict(threading.RLock)
        
        # Create logs directory
        self.logs_dir = "logs/batch_optimizer"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Start worker threads for task types
        if self.enable_optimization:
            self._start_worker_thread("default")
        
        self.logger.info(f"BatchOptimizer initialized with initial batch size {self.initial_batch_size}")
    
    def _get_optimal_batch_size(self, task_type):
        """
        Get the current optimal batch size for a task type.
        
        Args:
            task_type (str): Type of task
            
        Returns:
            int: Optimal batch size
        """
        with self.task_locks[task_type]:
            # Use stored batch size if available
            if task_type in self.batch_sizes:
                return self.batch_sizes[task_type]
            
            # Otherwise use initial batch size
            self.batch_sizes[task_type] = self.initial_batch_size
            return self.initial_batch_size
    
    def _update_batch_size(self, task_type, batch_size, processing_time, memory_used, items_processed):
        """
        Update the optimal batch size based on performance metrics.
        
        Args:
            task_type (str): Type of task
            batch_size (int): Current batch size
            processing_time (float): Time taken to process the batch (seconds)
            memory_used (int): Memory used during processing (bytes)
            items_processed (int): Number of items processed in the batch
        """
        if not self.enable_optimization:
            return
        
        with self.task_locks[task_type]:
            # Calculate throughput (items per second)
            throughput = items_processed / max(0.001, processing_time)
            
            # Update history
            if task_type not in self.throughput_history:
                self.throughput_history[task_type] = deque(maxlen=self.throughput_window)
                self.processing_time_history[task_type] = deque(maxlen=self.throughput_window)
            
            self.throughput_history[task_type].append((batch_size, throughput))
            self.processing_time_history[task_type].append((batch_size, processing_time))
            
            # Only optimize after collecting enough data
            if len(self.throughput_history[task_type]) < 3:
                return
            
            # Get current system memory usage
            system_memory_percent = psutil.virtual_memory().percent
            
            # Determine whether to adjust batch size
            if system_memory_percent > self.memory_target_percent + 10:
                # Memory pressure, reduce batch size
                adjustment_factor = 0.8
                reason = "memory pressure"
            elif system_memory_percent < self.memory_target_percent - 10:
                # Memory available, try increasing batch size
                adjustment_factor = 1.2
                reason = "memory available"
            else:
                # Check if throughput is improving with current batch size
                recent_throughputs = [t for _, t in self.throughput_history[task_type]]
                
                if len(recent_throughputs) >= 3:
                    # Check if throughput is increasing or decreasing
                    throughput_trend = recent_throughputs[-1] - recent_throughputs[-3]
                    
                    if throughput_trend > 0:
                        # Throughput is improving, continue in same direction
                        previous_adjustment = self.batch_sizes[task_type] / batch_size
                        adjustment_factor = previous_adjustment
                        reason = "improving throughput"
                    else:
                        # Throughput is decreasing, reverse direction
                        previous_adjustment = self.batch_sizes[task_type] / batch_size
                        adjustment_factor = 1.0 / previous_adjustment
                        reason = "declining throughput"
                else:
                    # Not enough data yet
                    adjustment_factor = 1.0
                    reason = "insufficient data"
            
            # Apply learning rate to adjustment
            adjustment_factor = 1.0 + (adjustment_factor - 1.0) * self.learning_rate
            
            # Calculate new batch size
            new_batch_size = int(batch_size * adjustment_factor)
            
            # Ensure batch size stays within bounds
            new_batch_size = max(self.min_batch_size, min(self.max_batch_size, new_batch_size))
            
            # Update stored batch size
            self.batch_sizes[task_type] = new_batch_size
            
            self.logger.debug(f"Updated batch size for {task_type}: {batch_size} -> {new_batch_size} ({reason})")
    
    def _start_worker_thread(self, task_type):
        """
        Start a worker thread for a task type.
        
        Args:
            task_type (str): Type of task
        """
        if task_type in self.worker_threads and self.worker_threads[task_type].is_alive():
            return
        
        worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(task_type,),
            daemon=True,
            name=f"BatchOptimizer_{task_type}"
        )
        self.worker_threads[task_type] = worker_thread
        worker_thread.start()
        
        self.logger.debug(f"Started worker thread for task type {task_type}")
    
    def _worker_loop(self, task_type):
        """
        Main loop for processing batched tasks.
        
        Args:
            task_type (str): Type of task
        """
        while not self.stop_event.is_set():
            try:
                # Check if there are tasks in the queue
                if self.task_queues[task_type].empty():
                    # No tasks, sleep and check again
                    time.sleep(0.1)
                    continue
                
                # Get optimal batch size
                batch_size = self._get_optimal_batch_size(task_type)
                
                # Collect up to batch_size tasks
                tasks = []
                priorities = []
                
                try:
                    # Collect tasks with a timeout to allow checking stop_event
                    start_time = time.time()
                    collection_timeout = 0.5  # Maximum time to spend collecting tasks
                    
                    while (len(tasks) < batch_size and 
                           time.time() - start_time < collection_timeout and
                           not self.stop_event.is_set()):
                        try:
                            # Get task with small timeout
                            priority, task_data = self.task_queues[task_type].get(timeout=0.01)
                            tasks.append(task_data)
                            priorities.append(priority)
                            
                            # Mark as in-progress
                            self.task_queues[task_type].task_done()
                        except queue.Empty:
                            # No more tasks available immediately
                            break
                except Exception as e:
                    self.logger.error(f"Error collecting tasks for {task_type}: {str(e)}")
                
                # Process the batch if there are any tasks
                if tasks:
                    self._process_task_batch(task_type, tasks, priorities, batch_size)
                
            except Exception as e:
                self.logger.error(f"Error in worker loop for {task_type}: {str(e)}")
                time.sleep(1)  # Sleep to avoid tight loop on error
    
    def _process_task_batch(self, task_type, tasks, priorities, batch_size):
        """
        Process a batch of tasks.
        
        Args:
            task_type (str): Type of task
            tasks (list): List of tasks to process
            priorities (list): List of task priorities
            batch_size (int): Target batch size
        """
        try:
            # Measure memory before
            memory_before = psutil.Process(os.getpid()).memory_info().rss
            
            # Start timing
            start_time = time.time()
            
            # Group similar tasks if possible
            grouped_tasks = self._group_similar_tasks(tasks)
            
            # Process each group
            results = []
            
            for group in grouped_tasks:
                # Check if this is a batched task
                if 'batch_processor' in group[0] and callable(group[0]['batch_processor']):
                    # This is a batch-compatible task, process all at once
                    batch_data = [task['data'] for task in group]
                    batch_args = group[0].get('batch_args', [])
                    batch_kwargs = group[0].get('batch_kwargs', {})
                    
                    # Track batch size and processor
                    actual_batch_size = len(batch_data)
                    processor_name = group[0].get('processor_name', 'unknown')
                    
                    try:
                        # Process the batch
                        batch_result = group[0]['batch_processor'](batch_data, *batch_args, **batch_kwargs)
                        
                        # Match results to original tasks
                        for i, task in enumerate(group):
                            if i < len(batch_result):
                                result = {
                                    'success': True,
                                    'result': batch_result[i],
                                    'task_id': task.get('task_id')
                                }
                            else:
                                result = {
                                    'success': False,
                                    'error': "Batch result index out of range",
                                    'task_id': task.get('task_id')
                                }
                            
                            results.append((task, result))
                    except Exception as e:
                        # Batch processing failed, process individually
                        self.logger.warning(f"Batch processing failed for {processor_name}, falling back to individual processing: {str(e)}")
                        
                        for task in group:
                            try:
                                if 'processor' in task and callable(task['processor']):
                                    # Individual processing
                                    indiv_result = task['processor'](task['data'], *task.get('args', []), **task.get('kwargs', {}))
                                    
                                    result = {
                                        'success': True,
                                        'result': indiv_result,
                                        'task_id': task.get('task_id')
                                    }
                                else:
                                    result = {
                                        'success': False,
                                        'error': "No processor function available",
                                        'task_id': task.get('task_id')
                                    }
                            except Exception as task_e:
                                result = {
                                    'success': False,
                                    'error': str(task_e),
                                    'task_id': task.get('task_id')
                                }
                            
                            results.append((task, result))
                else:
                    # Process tasks individually
                    for task in group:
                        try:
                            if 'processor' in task and callable(task['processor']):
                                indiv_result = task['processor'](task['data'], *task.get('args', []), **task.get('kwargs', {}))
                                
                                result = {
                                    'success': True,
                                    'result': indiv_result,
                                    'task_id': task.get('task_id')
                                }
                            else:
                                result = {
                                    'success': False,
                                    'error': "No processor function available",
                                    'task_id': task.get('task_id')
                                }
                        except Exception as e:
                            result = {
                                'success': False,
                                'error': str(e),
                                'task_id': task.get('task_id')
                            }
                        
                        results.append((task, result))
            
            # Measure processing time
            processing_time = time.time() - start_time
            
            # Measure memory after
            memory_after = psutil.Process(os.getpid()).memory_info().rss
            memory_used = memory_after - memory_before
            
            # Update task history
            for task, result in results:
                task_id = task.get('task_id')
                if task_id:
                    self.task_history[task_type].append({
                        'task_id': task_id,
                        'success': result['success'],
                        'processing_time': processing_time,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Call completion callbacks
            for task, result in results:
                if 'callback' in task and callable(task['callback']):
                    try:
                        if result['success']:
                            task['callback'](result['result'])
                        else:
                            task['callback'](None, error=result.get('error'))
                    except Exception as e:
                        self.logger.error(f"Error in task callback: {str(e)}")
            
            # Update batch size optimization
            actual_items = len(tasks)
            self._update_batch_size(task_type, batch_size, processing_time, memory_used, actual_items)
            
            # Log batch processing
            self.logger.debug(f"Processed batch of {actual_items} items for {task_type} in {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing batch for {task_type}: {str(e)}")
    
    def _group_similar_tasks(self, tasks):
        """
        Group similar tasks that can be processed together.
        
        Args:
            tasks (list): List of tasks
            
        Returns:
            list: List of task groups
        """
        if not tasks:
            return []
        
        # Simple grouping by processor name
        groups = defaultdict(list)
        
        for task in tasks:
            processor_name = task.get('processor_name', task.get('processor', 'unknown'))
            batch_compatible = 'batch_processor' in task and callable(task['batch_processor'])
            
            if batch_compatible:
                groups[f"batch_{processor_name}"].append(task)
            else:
                # Individual processing - don't group
                groups[f"indiv_{id(task)}"].append(task)
        
        return list(groups.values())
    
    def submit_task(self, processor, data, task_type="default", priority=0, callback=None, 
                   batch_processor=None, processor_name=None, *args, **kwargs):
        """
        Submit a task for optimized batch processing.
        
        Args:
            processor (callable): Function to process a single data item
            data: Data to process
            task_type (str): Type of task for grouping
            priority (int): Task priority (lower number = higher priority)
            callback (callable, optional): Function to call with the result
            batch_processor (callable, optional): Function that can process a batch of data
            processor_name (str, optional): Name of the processor for grouping
            *args, **kwargs: Additional arguments for the processor
            
        Returns:
            str: Task ID
        """
        if not self.enable_optimization:
            # Direct processing without batching
            try:
                result = processor(data, *args, **kwargs)
                if callback and callable(callback):
                    callback(result)
                return "direct_" + str(id(data))
            except Exception as e:
                self.logger.error(f"Error in direct processing: {str(e)}")
                if callback and callable(callback):
                    callback(None, error=str(e))
                return None
        
        try:
            # Create task ID
            task_id = f"{task_type}_{id(data)}_{time.time()}"
            
            # Create task data
            task_data = {
                'task_id': task_id,
                'processor': processor,
                'data': data,
                'args': args,
                'kwargs': kwargs,
                'callback': callback,
                'submission_time': time.time(),
                'processor_name': processor_name or processor.__name__ if hasattr(processor, '__name__') else None
            }
            
            # Add batch processor if provided
            if batch_processor and callable(batch_processor):
                task_data['batch_processor'] = batch_processor
                task_data['batch_args'] = args
                task_data['batch_kwargs'] = kwargs
            
            # Submit to queue
            self.task_queues[task_type].put((priority, task_data))
            
            # Ensure worker thread is running
            if task_type not in self.worker_threads or not self.worker_threads[task_type].is_alive():
                self._start_worker_thread(task_type)
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error submitting task: {str(e)}")
            if callback and callable(callback):
                callback(None, error=str(e))
            return None
    
    def process_batch(self, items, processor, batch_size=None, task_type="default", priority=0, 
                     callback=None, batch_processor=None, processor_name=None, *args, **kwargs):
        """
        Process a batch of items with optimal batching.
        
        Args:
            items (list): List of data items to process
            processor (callable): Function to process a single data item
            batch_size (int, optional): Force specific batch size, or None for automatic
            task_type (str): Type of task for grouping
            priority (int): Task priority
            callback (callable, optional): Function to call with all results
            batch_processor (callable, optional): Function that can process a batch of data
            processor_name (str, optional): Name of the processor for grouping
            *args, **kwargs: Additional arguments for the processor
            
        Returns:
            list: List of task IDs
        """
        if not items:
            if callback and callable(callback):
                callback([])
            return []
        
        # Determine batch size
        if batch_size is None:
            batch_size = self._get_optimal_batch_size(task_type)
        else:
            batch_size = max(1, min(self.max_batch_size, batch_size))
        
        # Handle direct processing if optimization is disabled
        if not self.enable_optimization:
            try:
                if batch_processor and callable(batch_processor):
                    results = batch_processor(items, *args, **kwargs)
                else:
                    results = [processor(item, *args, **kwargs) for item in items]
                
                if callback and callable(callback):
                    callback(results)
                
                return [f"direct_{i}" for i in range(len(items))]
            except Exception as e:
                self.logger.error(f"Error in direct batch processing: {str(e)}")
                if callback and callable(callback):
                    callback(None, error=str(e))
                return []
        
        # Split into chunks for processing
        task_ids = []
        results = [None] * len(items)
        completed = [False] * len(items)
        
        # Create lock for results
        results_lock = threading.RLock()
        
        # Create callback for each chunk
        def chunk_callback(chunk_results, chunk_indices):
            nonlocal results, completed
            with results_lock:
                for i, result in zip(chunk_indices, chunk_results):
                    if i < len(results):
                        results[i] = result
                        completed[i] = True
                
                # Check if all completed
                if all(completed):
                    if callback and callable(callback):
                        callback(results)
        
        # Split data into chunks and submit
        for i in range(0, len(items), batch_size):
            chunk = items[i:i+batch_size]
            chunk_indices = list(range(i, min(i+batch_size, len(items))))
            
            # Create callback for this chunk
            def create_chunk_callback(indices):
                return lambda r, error=None: chunk_callback(r if error is None else [None]*len(indices), indices)
            
            # Submit the chunk
            if len(chunk) == 1:
                # Single item, submit as individual task
                task_id = self.submit_task(
                    processor=processor,
                    data=chunk[0],
                    task_type=task_type,
                    priority=priority,
                    callback=create_chunk_callback(chunk_indices),
                    processor_name=processor_name,
                    *args, **kwargs
                )
                task_ids.append(task_id)
            elif batch_processor and callable(batch_processor):
                # Use batch processor for multi-item chunks
                batch_task_id = f"{task_type}_batch_{i}_{time.time()}"
                
                # Create batch task
                task_data = {
                    'task_id': batch_task_id,
                    'processor': processor,  # For fallback
                    'batch_processor': batch_processor,
                    'data': chunk,  # The whole chunk
                    'args': args,
                    'kwargs': kwargs,
                    'callback': create_chunk_callback(chunk_indices),
                    'submission_time': time.time(),
                    'processor_name': processor_name or (
                        batch_processor.__name__ if hasattr(batch_processor, '__name__') else None
                    ),
                    'batch_args': args,
                    'batch_kwargs': kwargs
                }
                
                # Submit batch task
                self.task_queues[task_type].put((priority, task_data))
                task_ids.append(batch_task_id)
            else:
                # Submit individual tasks for the chunk
                for j, item in enumerate(chunk):
                    task_id = self.submit_task(
                        processor=processor,
                        data=item,
                        task_type=task_type,
                        priority=priority,
                        callback=create_chunk_callback([chunk_indices[j]]),
                        processor_name=processor_name,
                        *args, **kwargs
                    )
                    task_ids.append(task_id)
        
        # Ensure worker thread is running
        if task_type not in self.worker_threads or not self.worker_threads[task_type].is_alive():
            self._start_worker_thread(task_type)
            
        return task_ids
    
    def optimize_dataframe_processing(self, dataframe, processor, task_type="dataframe", 
                                     priority=0, callback=None, *args, **kwargs):
        """
        Optimize batch processing for pandas DataFrame.
        
        Args:
            dataframe (pd.DataFrame): DataFrame to process
            processor (callable): Function to process DataFrame chunks
            task_type (str): Type of task
            priority (int): Task priority
            callback (callable, optional): Function to call with combined result
            *args, **kwargs: Additional arguments for the processor
            
        Returns:
            str: Batch task ID
        """
        if dataframe is None or dataframe.empty:
            if callback and callable(callback):
                callback(pd.DataFrame())
            return None
            
        # Determine optimal chunk size in rows
        batch_size = self._get_optimal_batch_size(task_type)
        
        # Define batch processor function
        def dataframe_batch_processor(df_chunks):
            return pd.concat([processor(chunk, *args, **kwargs) for chunk in df_chunks])
        
        # Split dataframe into chunks
        chunks = []
        for i in range(0, len(dataframe), batch_size):
            chunks.append(dataframe.iloc[i:i+batch_size])
        
        # Submit batch processing
        task_id = f"{task_type}_dataframe_{time.time()}"
        
        # Create dataframe task
        task_data = {
            'task_id': task_id,
            'processor': processor,
            'batch_processor': dataframe_batch_processor,
            'data': chunks,
            'args': args,
            'kwargs': kwargs,
            'callback': callback,
            'submission_time': time.time(),
            'processor_name': 'dataframe_processor'
        }
        
        # Submit to queue
        self.task_queues[task_type].put((priority, task_data))
        
        # Ensure worker thread is running
        if task_type not in self.worker_threads or not self.worker_threads[task_type].is_alive():
            self._start_worker_thread(task_type)
            
        return task_id
    
    def optimize_numpy_processing(self, array, processor, axis=0, task_type="numpy", 
                                priority=0, callback=None, *args, **kwargs):
        """
        Optimize batch processing for numpy arrays.
        
        Args:
            array (np.ndarray): NumPy array to process
            processor (callable): Function to process array chunks
            axis (int): Axis along which to split the array
            task_type (str): Type of task
            priority (int): Task priority
            callback (callable, optional): Function to call with combined result
            *args, **kwargs: Additional arguments for the processor
            
        Returns:
            str: Batch task ID
        """
        if array is None or array.size == 0:
            if callback and callable(callback):
                callback(np.array([]))
            return None
            
        # Determine optimal chunk size
        batch_size = self._get_optimal_batch_size(task_type)
        
        # Ensure batch size is appropriate for array dimensions
        array_size_along_axis = array.shape[axis]
        batch_size = min(batch_size, array_size_along_axis)
        
        # Define batch processor function
        def numpy_batch_processor(array_chunks):
            processed_chunks = [processor(chunk, *args, **kwargs) for chunk in array_chunks]
            
            # Try to combine results based on their shapes
            try:
                if all(isinstance(chunk, np.ndarray) for chunk in processed_chunks):
                    # Try concatenating along the same axis
                    return np.concatenate(processed_chunks, axis=axis)
                else:
                    # Return as list if concatenation isn't possible
                    return processed_chunks
            except Exception as e:
                self.logger.warning(f"Could not combine numpy results: {str(e)}")
                return processed_chunks
        
        # Split array into chunks
        chunks = np.array_split(array, math.ceil(array_size_along_axis / batch_size), axis=axis)
        
        # Submit batch processing
        task_id = f"{task_type}_numpy_{time.time()}"
        
        # Create numpy task
        task_data = {
            'task_id': task_id,
            'processor': processor,
            'batch_processor': numpy_batch_processor,
            'data': chunks,
            'args': args,
            'kwargs': kwargs,
            'callback': callback,
            'submission_time': time.time(),
            'processor_name': 'numpy_processor'
        }
        
        # Submit to queue
        self.task_queues[task_type].put((priority, task_data))
        
        # Ensure worker thread is running
        if task_type not in self.worker_threads or not self.worker_threads[task_type].is_alive():
            self._start_worker_thread(task_type)
            
        return task_id
    
    def get_optimal_batch_sizes(self):
        """
        Get the current optimal batch sizes for all task types.
        
        Returns:
            dict: Dictionary of task types and their optimal batch sizes
        """
        with self.task_locks["default"]:  # Use default lock for this operation
            return self.batch_sizes.copy()
    
    def get_task_history(self, task_type=None, limit=None):
        """
        Get task execution history.
        
        Args:
            task_type (str, optional): Filter by task type
            limit (int, optional): Limit the number of history entries
            
        Returns:
            dict: Task history by task type
        """
        if task_type:
            history = list(self.task_history.get(task_type, []))
            if limit:
                history = history[-limit:]
            return {task_type: history}
        else:
            result = {}
            for t_type, history in self.task_history.items():
                history_list = list(history)
                if limit:
                    history_list = history_list[-limit:]
                result[t_type] = history_list
            return result
    
    def get_throughput_metrics(self, task_type=None):
        """
        Get throughput metrics for task types.
        
        Args:
            task_type (str, optional): Filter by task type
            
        Returns:
            dict: Throughput metrics by task type
        """
        result = {}
        
        if task_type:
            types_to_check = [task_type]
        else:
            types_to_check = list(self.throughput_history.keys())
        
        for t_type in types_to_check:
            if t_type in self.throughput_history and self.throughput_history[t_type]:
                throughput_data = self.throughput_history[t_type]
                processing_data = self.processing_time_history.get(t_type, [])
                
                # Calculate metrics
                batch_sizes = [bs for bs, _ in throughput_data]
                throughputs = [tp for _, tp in throughput_data]
                
                result[t_type] = {
                    'avg_throughput': np.mean(throughputs) if throughputs else 0,
                    'max_throughput': max(throughputs) if throughputs else 0,
                    'current_batch_size': self.batch_sizes.get(t_type, self.initial_batch_size),
                    'avg_batch_size': np.mean(batch_sizes) if batch_sizes else 0,
                    'avg_processing_time': np.mean([t for _, t in processing_data]) if processing_data else 0,
                    'sample_count': len(throughputs)
                }
        
        return result
    
    def shutdown(self):
        """Shutdown the batch optimizer and release resources."""
        self.logger.info("Shutting down BatchOptimizer")
        
        # Stop worker threads
        self.stop_event.set()
        
        # Wait for threads to finish
        for task_type, thread in self.worker_threads.items():
            if thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    self.logger.warning(f"Worker thread for {task_type} did not shut down cleanly")
        
        # Shutdown worker pool
        self.worker_pool.shutdown(wait=False)
        
        # Clear task queues
        for task_type, task_queue in self.task_queues.items():
            while not task_queue.empty():
                try:
                    task_queue.get_nowait()
                    task_queue.task_done()
                except queue.Empty:
                    break
        
        self.logger.info("BatchOptimizer shutdown complete")
