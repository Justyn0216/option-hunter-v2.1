"""
GPU Manager Module

This module provides GPU resource management for ML models in the Option Hunter system.
It handles GPU allocation, memory management, and workload scheduling to optimize GPU
utilization for machine learning model training and inference.
"""

import logging
import os
import json
import time
import threading
import queue
import numpy as np
from datetime import datetime
from collections import defaultdict, deque

# Try importing GPU-related libraries, with graceful fallback if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

class GPUManager:
    """
    GPU resource manager for optimizing ML model training and inference.
    
    Features:
    - GPU availability detection and selection
    - Memory management and allocation
    - Optimized model placement on GPUs
    - Batch scheduling for efficient processing
    - Workload balancing across multiple GPUs
    - Support for TensorFlow, PyTorch, and CuPy
    """
    
    def __init__(self, config=None):
        """
        Initialize the GPUManager.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration
        self.gpu_params = self.config.get("gpu_acceleration", {})
        
        # Default parameters
        self.enable_gpu = self.gpu_params.get("enable_gpu", True)
        self.memory_fraction = self.gpu_params.get("memory_fraction", 0.8)
        self.allow_growth = self.gpu_params.get("allow_growth", True)
        self.preferred_gpu_ids = self.gpu_params.get("gpu_devices", [])
        self.enable_mixed_precision = self.gpu_params.get("enable_mixed_precision", True)
        
        # State tracking
        self.available_gpus = []
        self.gpu_status = {}
        self.allocated_models = {}
        self.allocation_lock = threading.RLock()
        
        # Queue for GPU tasks
        self.task_queue = queue.PriorityQueue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Create logs directory
        self.logs_dir = "logs/gpu"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize GPU resources if enabled
        if self.enable_gpu:
            self._initialize_gpu_resources()
            self._start_worker_thread()
            self.logger.info(f"GPUManager initialized with {len(self.available_gpus)} available GPUs")
        else:
            self.logger.info("GPUManager initialized (GPU acceleration disabled)")
    
    def _initialize_gpu_resources(self):
        """Detect and initialize available GPU resources."""
        self.available_gpus = []
        self.gpu_status = {}
        
        # Detect GPUs using PyTorch
        if HAS_TORCH:
            try:
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    self.logger.info(f"PyTorch detected {gpu_count} CUDA devices")
                    for i in range(gpu_count):
                        device_name = torch.cuda.get_device_name(i)
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        self.available_gpus.append({
                            "id": i,
                            "name": device_name,
                            "total_memory": total_memory,
                            "backend": "pytorch"
                        })
                        self.gpu_status[i] = {
                            "used_memory": 0,
                            "allocated_models": [],
                            "utilization": 0.0
                        }
            except Exception as e:
                self.logger.warning(f"Error detecting PyTorch GPUs: {str(e)}")
        
        # Detect GPUs using TensorFlow
        if not self.available_gpus and HAS_TF:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    self.logger.info(f"TensorFlow detected {len(gpus)} GPU devices")
                    for i, gpu in enumerate(gpus):
                        # Configure memory growth if enabled
                        if self.allow_growth:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        
                        # Set memory limit if specified
                        if self.memory_fraction < 1.0:
                            tf.config.experimental.set_virtual_device_configuration(
                                gpu,
                                [tf.config.experimental.VirtualDeviceConfiguration(
                                    memory_limit=int(self.memory_fraction * 1024)
                                )]
                            )
                        
                        self.available_gpus.append({
                            "id": i,
                            "name": gpu.name,
                            "total_memory": None,  # TF doesn't provide this easily
                            "backend": "tensorflow"
                        })
                        self.gpu_status[i] = {
                            "used_memory": 0,
                            "allocated_models": [],
                            "utilization": 0.0
                        }
            except Exception as e:
                self.logger.warning(f"Error detecting TensorFlow GPUs: {str(e)}")
        
        # Enable mixed precision if available
        if self.enable_mixed_precision:
            if HAS_TF:
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    self.logger.info("TensorFlow mixed precision enabled")
                except Exception as e:
                    self.logger.warning(f"Error enabling TensorFlow mixed precision: {str(e)}")
            
            if HAS_TORCH:
                try:
                    # PyTorch mixed precision will be enabled per model
                    pass
                except Exception as e:
                    self.logger.warning(f"Error with PyTorch mixed precision: {str(e)}")
        
        # If specific GPU IDs are preferred, filter to those
        if self.preferred_gpu_ids:
            self.available_gpus = [gpu for gpu in self.available_gpus if gpu["id"] in self.preferred_gpu_ids]
            self.gpu_status = {k: v for k, v in self.gpu_status.items() if k in self.preferred_gpu_ids}
            self.logger.info(f"Filtered to {len(self.available_gpus)} preferred GPUs")
    
    def _start_worker_thread(self):
        """Start the GPU task worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self._worker_thread_loop,
                daemon=True,
                name="GPUWorkerThread"
            )
            self.worker_thread.start()
            self.logger.debug("GPU worker thread started")
    
    def _worker_thread_loop(self):
        """Main loop for processing GPU tasks."""
        while not self.stop_event.is_set():
            try:
                # Get task with a timeout to allow checking stop_event
                try:
                    priority, task_data = self.task_queue.get(timeout=1.0)
                    model_id = task_data.get("model_id")
                    task_fn = task_data.get("task_fn")
                    args = task_data.get("args", [])
                    kwargs = task_data.get("kwargs", {})
                    callback = task_data.get("callback")
                    
                    # Process the task
                    start_time = time.time()
                    
                    try:
                        # Get assigned GPU for this model
                        gpu_id = self._get_model_gpu(model_id)
                        
                        # Set the appropriate GPU context
                        self._set_gpu_context(gpu_id)
                        
                        # Execute the task
                        result = task_fn(*args, **kwargs)
                        success = True
                    except Exception as e:
                        self.logger.error(f"Error executing GPU task for model {model_id}: {str(e)}")
                        result = None
                        success = False
                    
                    # Record task execution time
                    execution_time = time.time() - start_time
                    
                    # Record performance
                    self._record_task_performance(model_id, execution_time, success)
                    
                    # Call the callback with the result
                    if callback and callable(callback):
                        try:
                            callback(result, success)
                        except Exception as e:
                            self.logger.error(f"Error in task callback for model {model_id}: {str(e)}")
                    
                    # Mark task as done
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    # No tasks in queue, just continue
                    pass
                
            except Exception as e:
                self.logger.error(f"Error in GPU worker thread: {str(e)}")
                time.sleep(1)  # Sleep to avoid tight loop on error
    
    def _set_gpu_context(self, gpu_id):
        """
        Set the current GPU context for the executing thread.
        
        Args:
            gpu_id (int): GPU ID to use
        """
        if gpu_id is None:
            return
        
        # Set PyTorch device
        if HAS_TORCH:
            try:
                torch.cuda.set_device(gpu_id)
            except Exception as e:
                self.logger.warning(f"Error setting PyTorch GPU device {gpu_id}: {str(e)}")
        
        # Set TensorFlow device
        if HAS_TF:
            try:
                # TF 2.x way to set visible devices for this thread
                tf.config.set_visible_devices(
                    tf.config.list_physical_devices('GPU')[gpu_id], 
                    'GPU'
                )
            except Exception as e:
                self.logger.warning(f"Error setting TensorFlow GPU device {gpu_id}: {str(e)}")
        
        # Set CuPy device
        if HAS_CUPY:
            try:
                cp.cuda.Device(gpu_id).use()
            except Exception as e:
                self.logger.warning(f"Error setting CuPy GPU device {gpu_id}: {str(e)}")
    
    def _record_task_performance(self, model_id, execution_time, success):
        """
        Record performance metrics for a task.
        
        Args:
            model_id (str): Model identifier
            execution_time (float): Task execution time in seconds
            success (bool): Whether the task completed successfully
        """
        with self.allocation_lock:
            self.performance_history[model_id].append({
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "success": success
            })
    
    def _get_model_gpu(self, model_id):
        """
        Get the GPU ID assigned to a model.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            int: GPU ID or None if the model is not allocated
        """
        with self.allocation_lock:
            return self.allocated_models.get(model_id, {}).get("gpu_id")
    
    def _select_gpu_for_model(self, model_id, memory_required=None, model_type=None):
        """
        Select the best GPU for a model based on availability and load.
        
        Args:
            model_id (str): Model identifier
            memory_required (int, optional): Required GPU memory in bytes
            model_type (str, optional): Type of model (e.g., 'tensorflow', 'pytorch')
            
        Returns:
            int: Selected GPU ID or None if no suitable GPU is available
        """
        with self.allocation_lock:
            if not self.available_gpus:
                return None
            
            # If model is already allocated, return its GPU
            if model_id in self.allocated_models:
                return self.allocated_models[model_id]["gpu_id"]
            
            # Calculate GPU scores based on utilization and memory
            gpu_scores = []
            
            for gpu in self.available_gpus:
                gpu_id = gpu["id"]
                status = self.gpu_status[gpu_id]
                
                # Calculate utilization score (lower is better)
                utilization_score = status["utilization"]
                
                # Calculate memory score (lower is better)
                if memory_required and gpu["total_memory"]:
                    memory_score = status["used_memory"] / gpu["total_memory"]
                else:
                    memory_score = len(status["allocated_models"]) / (5 + len(status["allocated_models"]))
                
                # Combine scores (lower is better)
                score = 0.7 * utilization_score + 0.3 * memory_score
                
                gpu_scores.append((gpu_id, score))
            
            # Sort by score (lowest first)
            gpu_scores.sort(key=lambda x: x[1])
            
            # Return the best GPU ID
            if gpu_scores:
                return gpu_scores[0][0]
            
            return None
    
    def register_model(self, model_id, model_object=None, memory_required=None, model_type=None, priority=0):
        """
        Register a model for GPU execution.
        
        Args:
            model_id (str): Unique identifier for the model
            model_object (object, optional): The model itself
            memory_required (int, optional): Estimated memory required in bytes
            model_type (str, optional): Type of model (e.g., 'tensorflow', 'pytorch')
            priority (int): Priority level (higher values get higher priority)
            
        Returns:
            dict: Allocation information or None if registration failed
        """
        if not self.enable_gpu or not self.available_gpus:
            return None
        
        try:
            with self.allocation_lock:
                # Select GPU for the model
                gpu_id = self._select_gpu_for_model(model_id, memory_required, model_type)
                
                if gpu_id is None:
                    self.logger.warning(f"No suitable GPU available for model {model_id}")
                    return None
                
                # Register model allocation
                allocation = {
                    "model_id": model_id,
                    "gpu_id": gpu_id,
                    "memory_required": memory_required,
                    "model_type": model_type,
                    "registration_time": datetime.now().isoformat(),
                    "priority": priority
                }
                
                self.allocated_models[model_id] = allocation
                
                # Update GPU status
                if memory_required and self.gpu_status[gpu_id]["used_memory"] is not None:
                    self.gpu_status[gpu_id]["used_memory"] += memory_required
                
                self.gpu_status[gpu_id]["allocated_models"].append(model_id)
                
                self.logger.info(f"Registered model {model_id} on GPU {gpu_id}")
                
                # Move the model to GPU if provided
                if model_object is not None:
                    self._move_model_to_gpu(model_object, gpu_id, model_type)
                
                return allocation
                
        except Exception as e:
            self.logger.error(f"Error registering model {model_id}: {str(e)}")
            return None
    
    def _move_model_to_gpu(self, model, gpu_id, model_type=None):
        """
        Move a model to the specified GPU.
        
        Args:
            model: Model object
            gpu_id (int): GPU ID
            model_type (str, optional): Type of model (e.g., 'tensorflow', 'pytorch')
            
        Returns:
            object: Updated model object or None if operation failed
        """
        try:
            # Auto-detect model type if not specified
            if model_type is None:
                if HAS_TORCH and isinstance(model, torch.nn.Module):
                    model_type = "pytorch"
                elif HAS_TF and isinstance(model, tf.keras.Model):
                    model_type = "tensorflow"
            
            # Move model based on its type
            if model_type == "pytorch" and HAS_TORCH:
                device = torch.device(f"cuda:{gpu_id}")
                model = model.to(device)
                
                # Enable mixed precision if configured
                if self.enable_mixed_precision:
                    # Will be done when running the model
                    pass
                
            elif model_type == "tensorflow" and HAS_TF:
                # TensorFlow models should automatically use the visible GPU
                # based on the context set in the worker thread
                pass
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error moving model to GPU {gpu_id}: {str(e)}")
            return model  # Return original model on failure
    
    def unregister_model(self, model_id):
        """
        Unregister a model from GPU execution.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            bool: True if unregistration was successful
        """
        if not self.enable_gpu:
            return True
        
        try:
            with self.allocation_lock:
                if model_id not in self.allocated_models:
                    self.logger.warning(f"Model {model_id} not registered")
                    return False
                
                allocation = self.allocated_models[model_id]
                gpu_id = allocation["gpu_id"]
                
                # Update GPU status
                if allocation.get("memory_required") and self.gpu_status[gpu_id]["used_memory"] is not None:
                    self.gpu_status[gpu_id]["used_memory"] -= allocation["memory_required"]
                
                if model_id in self.gpu_status[gpu_id]["allocated_models"]:
                    self.gpu_status[gpu_id]["allocated_models"].remove(model_id)
                
                # Remove model allocation
                del self.allocated_models[model_id]
                
                self.logger.info(f"Unregistered model {model_id} from GPU {gpu_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error unregistering model {model_id}: {str(e)}")
            return False
    
    def submit_task(self, model_id, task_fn, args=None, kwargs=None, callback=None, priority=0):
        """
        Submit a task for execution on GPU.
        
        Args:
            model_id (str): Model identifier
            task_fn (callable): Function to execute
            args (list, optional): Arguments for the task function
            kwargs (dict, optional): Keyword arguments for the task function
            callback (callable, optional): Function to call with the result
            priority (int): Task priority (higher values = higher priority)
            
        Returns:
            bool: True if task was submitted successfully
        """
        if not self.enable_gpu or not self.available_gpus:
            return False
        
        try:
            # Check if model is registered
            with self.allocation_lock:
                if model_id not in self.allocated_models:
                    self.logger.warning(f"Model {model_id} not registered for GPU execution")
                    return False
            
            # Create task data
            task_data = {
                "model_id": model_id,
                "task_fn": task_fn,
                "args": args or [],
                "kwargs": kwargs or {},
                "callback": callback
            }
            
            # Get model priority and combine with task priority
            model_priority = self.allocated_models[model_id].get("priority", 0)
            combined_priority = -(model_priority + priority)  # Negative because PriorityQueue puts lowest first
            
            # Submit to queue
            self.task_queue.put((combined_priority, task_data))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting task for model {model_id}: {str(e)}")
            return False
    
    def execute_on_gpu(self, model_id, input_data, batch_size=None):
        """
        Execute model inference on GPU.
        
        Args:
            model_id (str): Model identifier
            input_data: Input data for inference
            batch_size (int, optional): Batch size for inference
            
        Returns:
            object: Inference result or None if execution failed
        """
        if not self.enable_gpu or not self.available_gpus:
            return None
        
        try:
            # Check if model is registered
            with self.allocation_lock:
                if model_id not in self.allocated_models:
                    self.logger.warning(f"Model {model_id} not registered for GPU execution")
                    return None
            
            # Create result placeholder
            result_queue = queue.Queue()
            
            # Define task function
            def inference_task(data, bs=None):
                try:
                    # Get model
                    if model_id in self.allocated_models:
                        allocation = self.allocated_models[model_id]
                        model_type = allocation.get("model_type")
                        gpu_id = allocation.get("gpu_id")
                        
                        # Perform inference based on model type
                        if model_type == "pytorch" and HAS_TORCH:
                            # Convert input to tensor if needed
                            if not isinstance(data, torch.Tensor):
                                if isinstance(data, np.ndarray):
                                    data = torch.from_numpy(data)
                                else:
                                    data = torch.tensor(data)
                            
                            # Move data to GPU
                            device = torch.device(f"cuda:{gpu_id}")
                            data = data.to(device)
                            
                            # Run inference (placeholder - actual method depends on model)
                            # In actual usage, you'd use your model directly
                            # result = model(data)
                            
                            # Placeholder result
                            result = data  # Replace with actual inference
                            return result
                            
                        elif model_type == "tensorflow" and HAS_TF:
                            # Run inference (placeholder - actual method depends on model)
                            # result = model(data)
                            
                            # Placeholder result
                            result = data  # Replace with actual inference
                            return result
                            
                        else:
                            return None
                    else:
                        return None
                        
                except Exception as e:
                    self.logger.error(f"Error in inference task for model {model_id}: {str(e)}")
                    return None
            
            # Define callback
            def callback(result, success):
                result_queue.put((result, success))
            
            # Submit task
            submitted = self.submit_task(
                model_id=model_id,
                task_fn=inference_task,
                args=[input_data, batch_size],
                callback=callback
            )
            
            if not submitted:
                return None
            
            # Wait for result
            result, success = result_queue.get(timeout=60.0)
            
            if not success:
                return None
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing model {model_id} on GPU: {str(e)}")
            return None
    
    def get_gpu_status(self):
        """
        Get current GPU status information.
        
        Returns:
            dict: GPU status information
        """
        with self.allocation_lock:
            return {
                "available_gpus": self.available_gpus.copy(),
                "gpu_status": {k: v.copy() for k, v in self.gpu_status.items()},
                "allocated_models": {k: v.copy() for k, v in self.allocated_models.items()},
                "queue_size": self.task_queue.qsize(),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_performance_metrics(self, model_id=None):
        """
        Get performance metrics for models.
        
        Args:
            model_id (str, optional): Filter by model ID
            
        Returns:
            dict: Performance metrics
        """
        with self.allocation_lock:
            if model_id:
                if model_id in self.performance_history:
                    history = list(self.performance_history[model_id])
                    
                    # Calculate metrics
                    if history:
                        execution_times = [entry["execution_time"] for entry in history]
                        success_rate = sum(1 for entry in history if entry["success"]) / len(history)
                        
                        return {
                            "model_id": model_id,
                            "avg_execution_time": np.mean(execution_times),
                            "min_execution_time": min(execution_times),
                            "max_execution_time": max(execution_times),
                            "success_rate": success_rate,
                            "sample_count": len(history)
                        }
                return {}
            
            # Return metrics for all models
            metrics = {}
            for mid, history in self.performance_history.items():
                history_list = list(history)
                if history_list:
                    execution_times = [entry["execution_time"] for entry in history_list]
                    success_rate = sum(1 for entry in history_list if entry["success"]) / len(history_list)
                    
                    metrics[mid] = {
                        "avg_execution_time": np.mean(execution_times),
                        "min_execution_time": min(execution_times),
                        "max_execution_time": max(execution_times),
                        "success_rate": success_rate,
                        "sample_count": len(history_list)
                    }
            
            return metrics
    
    def clear_gpu_cache(self, gpu_id=None):
        """
        Clear GPU memory cache.
        
        Args:
            gpu_id (int, optional): Specific GPU to clear, or all if None
            
        Returns:
            bool: True if operation was successful
        """
        if not self.enable_gpu:
            return False
        
        try:
            if HAS_TORCH:
                if gpu_id is not None:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
                    
                self.logger.info(f"Cleared PyTorch GPU cache for {'all GPUs' if gpu_id is None else f'GPU {gpu_id}'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing GPU cache: {str(e)}")
            return False
    
    def shutdown(self):
        """Shutdown the GPU manager and release resources."""
        self.logger.info("Shutting down GPUManager")
        
        # Stop worker thread
        self.stop_event.set()
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        # Unregister all models
        with self.allocation_lock:
            for model_id in list(self.allocated_models.keys()):
                self.unregister_model(model_id)
        
        # Clear GPU caches
        if HAS_TORCH:
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        self.logger.info("GPUManager shutdown complete")
