def _task_completed_callback(self, future, task_id):
        """
        Callback for when a task completes.
        
        Args:
            future (Future): Future object
            task_id (int): Task ID
        """
        try:
            # Get task data
            with self.task_lock:
                if task_id not in self.active_tasks:
                    return
                
                task_data = self.active_tasks[task_id]
            
            # Check for exception
            exc = future.exception(timeout=0.1)
            if exc:
                # Task raised an exception
                error = str(exc)
                result = None
            else:
                # Get result
                try:
                    result_tuple = future.result(timeout=0.1)
                    _, result, error = result_tuple
                except Exception as e:
                    result = None
                    error = str(e)
            
            # Update task status
            with self.task_lock:
                end_time = time.time()
                duration = end_time - task_data.get("start_time", end_time)
                
                task_data["end_time"] = end_time
                task_data["duration"] = duration
                task_data["status"] = "completed" if error is None else "failed"
                task_data["error"] = error
                
                # Remove future to avoid reference cycles
                if "future" in task_data:
                    del task_data["future"]
                
                # Move to results
                self.task_results[task_id] = task_data
                
                # Add to history
                task_type = task_data.get("task_type", "default")
                self.execution_history[task_type].append({
                    "task_id": task_id,
                    "status": task_data["status"],
                    "duration": duration,
                    "error": error,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Remove from active tasks
                del self.active_tasks[task_id]
            
            # Call completion callback if provided
            callback = task_data.get("callback")
            if callback and callable(callback):
                try:
                    callback(result, error)
                except Exception as e:
                    self.logger.error(f"Error in task completion callback for task {task_id}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in task completion callback: {str(e)}")
    
    def _clean_completed_tasks(self):
        """Clean up old completed tasks."""
        try:
            # Check for timed out tasks
            current_time = time.time()
            
            with self.task_lock:
                # Find timed out tasks
                timed_out_tasks = []
                
                for task_id, task_data in self.active_tasks.items():
                    start_time = task_data.get("start_time", 0)
                    timeout = self.thread_timeout if task_data.get("executor_type") == "thread" else self.process_timeout
                    
                    if current_time - start_time > timeout:
                        timed_out_tasks.append(task_id)
                
                # Cancel timed out tasks
                for task_id in timed_out_tasks:
                    self._cancel_task(task_id, "timeout")
                
                # Clean up old results
                result_age_limit = current_time - 3600  # 1 hour
                old_results = [tid for tid, data in self.task_results.items() 
                              if data.get("end_time", 0) < result_age_limit]
                
                for task_id in old_results:
                    del self.task_results[task_id]
                
        except Exception as e:
            self.logger.error(f"Error cleaning completed tasks: {str(e)}")
    
    def _cancel_task(self, task_id, reason="user_request"):
        """
        Cancel a running task.
        
        Args:
            task_id (int): Task ID
            reason (str): Cancellation reason
            
        Returns:
            bool: True if task was cancelled
        """
        with self.task_lock:
            if task_id not in self.active_tasks:
                return False
            
            task_data = self.active_tasks[task_id]
            future = task_data.get("future")
            
            if future and not future.done():
                success = future.cancel()
                
                if success:
                    # Update task data
                    end_time = time.time()
                    duration = end_time - task_data.get("start_time", end_time)
                    
                    task_data["end_time"] = end_time
                    task_data["duration"] = duration
                    task_data["status"] = "cancelled"
                    task_data["error"] = f"Task cancelled: {reason}"
                    
                    # Remove future to avoid reference cycles
                    del task_data["future"]
                    
                    # Move to results
                    self.task_results[task_id] = task_data
                    
                    # Remove from active tasks
                    del self.active_tasks[task_id]
                    
                    # Call cancellation callback if provided
                    callback = task_data.get("cancel_callback")
                    if callback and callable(callback):
                        try:
                            callback(reason)
                        except Exception as e:
                            self.logger.error(f"Error in task cancellation callback: {str(e)}")
                    
                    return True
            
            return False
    
    def _update_task_progress(self, task_id, current, total):
        """
        Update progress for a task.
        
        Args:
            task_id (int): Task ID
            current (int): Current progress
            total (int): Total items to process
        """
        try:
            with self.task_lock:
                if task_id not in self.active_tasks:
                    return
                
                task_data = self.active_tasks[task_id]
                task_data["progress"] = {
                    "current": current,
                    "total": total,
                    "percentage": int(100 * current / max(1, total))
                }
                
                # Call progress callback if provided
                progress_callback = self.progress_callbacks.get(task_id)
                if progress_callback and callable(progress_callback):
                    try:
                        progress_callback(current, total)
                    except Exception as e:
                        self.logger.error(f"Error in progress callback for task {task_id}: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error updating task progress: {str(e)}")
    
    def _get_next_task_id(self):
        """Get the next available task ID."""
        with self.task_lock:
            task_id = self.next_task_id
            self.next_task_id += 1
            return task_id
    
    def submit_task(self, function, use_process=None, priority=0, task_type="default",
                   callback=None, cancel_callback=None, progress_callback=None,
                   total_items=None, *args, **kwargs):
        """
        Submit a task for parallel execution.
        
        Args:
            function (callable): Function to execute
            use_process (bool, optional): Whether to use process-based parallelism
            priority (int): Task priority (lower number = higher priority)
            task_type (str): Type of task for grouping
            callback (callable, optional): Function to call when task completes
            cancel_callback (callable, optional): Function to call if task is cancelled
            progress_callback (callable, optional): Function to call with progress updates
            total_items (int, optional): Total items for progress tracking
            *args, **kwargs: Arguments for the function
            
        Returns:
            int: Task ID
        """
        try:
            # Determine whether to use process or thread
            if use_process is None:
                # Auto-select based on configuration and task characteristics
                use_process = not self.thread_preference
            
            # Create task ID
            task_id = self._get_next_task_id()
            
            # Add progress tracking
            if total_items is not None:
                kwargs['_total_items'] = total_items
                
                # Store progress callback
                if progress_callback and callable(progress_callback):
                    self.progress_callbacks[task_id] = progress_callback
            
            # Create task data
            task_data = {
                "task_id": task_id,
                "function": function,
                "args": args,
                "kwargs": kwargs,
                "priority": priority,
                "task_type": task_type,
                "submission_time": time.time(),
                "callback": callback,
                "cancel_callback": cancel_callback,
                "status": "queued"
            }
            
            # Add to appropriate queue
            if use_process:
                self.process_queue.put((priority, task_data))
            else:
                self.thread_queue.put((priority, task_data))
            
            # Store task data
            with self.task_lock:
                self.active_tasks[task_id] = task_data
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error submitting task: {str(e)}")
            return None
    
    def map(self, function, items, use_process=None, chunk_size=None, ordered=True,
           task_type="map", callback=None, progress_callback=None, *args, **kwargs):
        """
        Apply a function to each item in a collection in parallel.
        
        Args:
            function (callable): Function to apply to each item
            items (list): Items to process
            use_process (bool, optional): Whether to use process-based parallelism
            chunk_size (int, optional): Size of chunks for processing
            ordered (bool): Whether to maintain order of results
            task_type (str): Type of task for grouping
            callback (callable, optional): Function to call with all results
            progress_callback (callable, optional): Function to call with progress updates
            *args, **kwargs: Additional arguments for the function
            
        Returns:
            int: Task ID
        """
        if not items:
            if callback:
                callback([], None)
            return None
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = min(self.default_chunk_size, max(1, len(items) // self.max_threads))
        else:
            chunk_size = max(1, chunk_size)
        
        # Create result placeholders
        result_queue = queue.Queue()
        results = [None] * len(items)
        completed = [False] * len(items)
        errors = []
        
        # Create lock for results
        results_lock = threading.Lock()
        
        # Create completion callback for each chunk
        def chunk_completion_callback(chunk_results, chunk_indices, error=None):
            with results_lock:
                if error:
                    errors.append(error)
                else:
                    for i, result in zip(chunk_indices, chunk_results):
                        if i < len(results):
                            results[i] = result
                            completed[i] = True
                
                # Check if all completed
                if all(completed):
                    if callback:
                        callback(results, errors[0] if errors else None)
        
        # Create progress tracking
        total_items = len(items)
        processed_items = [0]  # Use list for mutable reference
        
        # Create master progress callback
        def master_progress_callback(current, total):
            processed_items[0] = current
            if progress_callback:
                progress_callback(current, total_items)
        
        # Split into chunks and submit
        task_ids = []
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i+chunk_size]
            chunk_indices = list(range(i, min(i+chunk_size, len(items))))
            
            # Create processing function for this chunk
            def process_chunk(chunk_items, indices, *inner_args, progress_callback=None, **inner_kwargs):
                try:
                    # Process each item in the chunk
                    chunk_results = []
                    
                    for j, item in enumerate(chunk_items):
                        # Call function on item
                        result = function(item, *inner_args, **inner_kwargs)
                        chunk_results.append(result)
                        
                        # Update progress
                        if progress_callback and callable(progress_callback):
                            progress_callback(j + 1, len(chunk_items))
                    
                    return chunk_results
                except Exception as e:
                    raise Exception(f"Error processing chunk: {str(e)}")
            
            # Create completion callback for this chunk
            def create_chunk_callback(indices):
                return lambda results, error: chunk_completion_callback(
                    results, indices, error
                )
            
            # Submit the chunk
            task_id = self.submit_task(
                process_chunk,
                use_process=use_process,
                priority=0,
                task_type=task_type,
                callback=create_chunk_callback(chunk_indices),
                progress_callback=master_progress_callback,
                total_items=len(chunk),
                args=(chunk, chunk_indices) + args,
                kwargs=kwargs
            )
            
            task_ids.append(task_id)
        
        return task_ids
    
    def apply_dataframe(self, function, dataframe, use_process=None, chunk_size=None,
                       task_type="dataframe", callback=None, progress_callback=None,
                       *args, **kwargs):
        """
        Apply a function to chunks of a pandas DataFrame in parallel.
        
        Args:
            function (callable): Function to apply to each chunk
            dataframe (pd.DataFrame): DataFrame to process
            use_process (bool, optional): Whether to use process-based parallelism
            chunk_size (int, optional): Size of chunks in rows
            task_type (str): Type of task for grouping
            callback (callable, optional): Function to call with combined result
            progress_callback (callable, optional): Function to call with progress updates
            *args, **kwargs: Additional arguments for the function
            
        Returns:
            int: Task ID
        """
        if dataframe is None or dataframe.empty:
            if callback:
                callback(pd.DataFrame(), None)
            return None
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = min(self.default_chunk_size, max(1, len(dataframe) // self.max_threads))
        else:
            chunk_size = max(1, chunk_size)
        
        # Create result placeholders
        chunks_processed = [0]
        results = []
        errors = []
        
        # Create lock for results
        results_lock = threading.Lock()
        
        # Calculate number of chunks
        num_chunks = (len(dataframe) + chunk_size - 1) // chunk_size
        
        # Create completion callback for all chunks
        def final_callback():
            if not results:
                if callback:
                    callback(pd.DataFrame(), errors[0] if errors else "No results returned")
                return
            
            try:
                # Combine results
                combined_df = pd.concat(results, ignore_index=True)
                
                if callback:
                    callback(combined_df, errors[0] if errors else None)
            except Exception as e:
                if callback:
                    callback(pd.DataFrame(), str(e))
        
        # Create completion callback for each chunk
        def chunk_completion_callback(chunk_result, error=None):
            with results_lock:
                if error:
                    errors.append(error)
                else:
                    results.append(chunk_result)
                
                chunks_processed[0] += 1
                
                # Check if all completed
                if chunks_processed[0] >= num_chunks:
                    final_callback()
        
        # Create progress tracking
        total_rows = len(dataframe)
        processed_rows = [0]  # Use list for mutable reference
        
        # Create master progress callback
        def master_progress_callback(current, total):
            processed_rows[0] = current
            if progress_callback:
                progress_callback(processed_rows[0], total_rows)
        
        # Split DataFrame into chunks and submit
        task_ids = []
        
        for i in range(0, len(dataframe), chunk_size):
            chunk_df = dataframe.iloc[i:i+chunk_size]
            
            # Submit the chunk
            task_id = self.submit_task(
                function,
                use_process=use_process,
                priority=0,
                task_type=task_type,
                callback=chunk_completion_callback,
                progress_callback=master_progress_callback,
                total_items=len(chunk_df),
                args=(chunk_df,) + args,
                kwargs=kwargs
            )
            
            task_ids.append(task_id)
        
        return task_ids
    
    def apply_batches(self, function, items, batch_size=None, max_concurrency=None,
                     task_type="batch", callback=None, progress_callback=None,
                     use_process=None, *args, **kwargs):
        """
        Process items in batches with controlled concurrency.
        
        Args:
            function (callable): Function to apply to each batch
            items (list): Items to process
            batch_size (int, optional): Size of each batch
            max_concurrency (int, optional): Maximum number of concurrent batches
            task_type (str): Type of task for grouping
            callback (callable, optional): Function to call with all results
            progress_callback (callable, optional): Function to call with progress updates
            use_process (bool, optional): Whether to use process-based parallelism
            *args, **kwargs: Additional arguments for the function
            
        Returns:
            list: List of task IDs
        """
        if not items:
            if callback:
                callback([], None)
            return []
        
        # Determine batch size
        if batch_size is None:
            batch_size = min(self.default_chunk_size, max(1, len(items) // self.max_threads))
        else:
            batch_size = max(1, batch_size)
        
        # Determine max concurrency
        if max_concurrency is None:
            max_concurrency = max(1, min(self.max_threads, (len(items) + batch_size - 1) // batch_size))
        else:
            max_concurrency = max(1, max_concurrency)
        
        # Create batches
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i+batch_size])
        
        # Create result tracking
        batch_results = []
        completed_batches = 0
        errors = []
        
        # Create lock for results
        results_lock = threading.Lock()
        processing_semaphore = threading.Semaphore(max_concurrency)
        
        # Create progress tracking
        total_items = len(items)
        processed_items = [0]  # Use list for mutable reference
        
        # Create master progress callback
        def master_progress_callback(current, total):
            with results_lock:
                processed_items[0] += current
                if progress_callback:
                    progress_callback(min(processed_items[0], total_items), total_items)
        
        # Create completion callback for each batch
        def batch_completion_callback(batch_result, error=None):
            nonlocal completed_batches
            
            with results_lock:
                if error:
                    errors.append(error)
                else:
                    batch_results.append(batch_result)
                
                completed_batches += 1
                processing_semaphore.release()
                
                # Check if all completed
                if completed_batches >= len(batches):
                    if callback:
                        callback(batch_results, errors[0] if errors else None)
        
        # Process batches with controlled concurrency
        task_ids = []
        
        for i, batch in enumerate(batches):
            # Wait for semaphore
            processing_semaphore.acquire()
            
            # Submit the batch
            task_id = self.submit_task(
                function,
                use_process=use_process,
                priority=0,
                task_type=task_type,
                callback=batch_completion_callback,
                progress_callback=master_progress_callback,
                total_items=len(batch),
                args=(batch,) + args,
                kwargs=kwargs
            )
            
            task_ids.append(task_id)
        
        return task_ids
    
    def execute_pipeline(self, pipeline_stages, initial_input=None, task_type="pipeline",
                        callback=None, progress_callback=None, use_process=None,
                        error_handling="stop", *args, **kwargs):
        """
        Execute a pipeline of functions, where output of one stage becomes input to the next.
        
        Args:
            pipeline_stages (list): List of functions to execute in sequence
            initial_input: Initial input to the first stage
            task_type (str): Type of task for grouping
            callback (callable, optional): Function to call with the final result
            progress_callback (callable, optional): Function to call with progress updates
            use_process (bool, optional): Whether to use process-based parallelism
            error_handling (str): How to handle errors ('stop' or 'continue')
            *args, **kwargs: Additional arguments for all functions
            
        Returns:
            int: Pipeline task ID
        """
        if not pipeline_stages:
            if callback:
                callback(initial_input, None)
            return None
        
        # Create pipeline state
        pipeline_id = f"pipeline_{self._get_next_task_id()}"
        current_stage = [0]  # Use list for mutable reference
        pipeline_result = [initial_input]  # Use list for mutable reference
        pipeline_error = [None]  # Use list for mutable reference
        
        # Create progress tracking
        total_stages = len(pipeline_stages)
        
        # Create stage completion callback
        def stage_completion_callback(result, error=None):
            if error:
                pipeline_error[0] = error
                
                # Handle error based on strategy
                if error_handling == "stop":
                    # Stop pipeline execution
                    if callback:
                        callback(None, error)
                    return
            
            # Store result
            pipeline_result[0] = result
            
            # Update progress
            current_stage[0] += 1
            if progress_callback:
                progress_callback(current_stage[0], total_stages)
            
            # Check if pipeline is complete
            if current_stage[0] >= total_stages:
                if callback:
                    callback(pipeline_result[0], pipeline_error[0])
                return
            
            # Execute next stage
            execute_next_stage()
        
        # Function to execute the next stage
        def execute_next_stage():
            stage_idx = current_stage[0]
            if stage_idx >= len(pipeline_stages):
                return
            
            # Get function for this stage
            stage_function = pipeline_stages[stage_idx]
            
            # Get input for this stage
            stage_input = pipeline_result[0]
            
            # Submit task for this stage
            self.submit_task(
                stage_function,
                use_process=use_process,
                priority=0,
                task_type=f"{task_type}_stage{stage_idx}",
                callback=stage_completion_callback,
                args=(stage_input,) + args,
                kwargs=kwargs
            )
        
        # Start pipeline execution
        execute_next_stage()
        
        return pipeline_id
    
    def cancel_task(self, task_id):
        """
        Cancel a running task.
        
        Args:
            task_id (int): Task ID
            
        Returns:
            bool: True if task was cancelled
        """
        return self._cancel_task(task_id, "user_request")
    
    def get_task_status(self, task_id):
        """
        Get the status of a task.
        
        Args:
            task_id (int): Task ID
            
        Returns:
            dict: Task status information or None if not found
        """
        with self.task_lock:
            # Check active tasks
            if task_id in self.active_tasks:
                task_data = self.active_tasks[task_id].copy()
                
                # Remove future to avoid serialization issues
                if "future" in task_data:
                    del task_data["future"]
                
                return task_data
            
            # Check completed tasks
            if task_id in self.task_results:
                return self.task_results[task_id].copy()
            
            return None
    
    def get_all_tasks(self, task_type=None, status=None):
        """
        Get all tasks, optionally filtered by type and status.
        
        Args:
            task_type (str, optional): Filter by task type
            status (str, optional): Filter by status
            
        Returns:
            list: List of task data dictionaries
        """
        with self.task_lock:
            # Combine active and completed tasks
            all_tasks = {}
            
            # Add active tasks
            for task_id, task_data in self.active_tasks.items():
                # Skip if filtering by task type
                if task_type and task_data.get("task_type") != task_type:
                    continue
                
                # Skip if filtering by status
                if status and task_data.get("status") != status:
                    continue
                
                # Create copy to avoid serialization issues
                task_copy = task_data.copy()
                if "future" in task_copy:
                    del task_copy["future"]
                
                all_tasks[task_id] = task_copy
            
            # Add completed tasks
            for task_id, task_data in self.task_results.items():
                # Skip if already added
                if task_id in all_tasks:
                    continue
                
                # Skip if filtering by task type
                if task_type and task_data.get("task_type") != task_type:
                    continue
                
                # Skip if filtering by status
                if status and task_data.get("status") != status:
                    continue
                
                all_tasks[task_id] = task_data.copy()
            
            return list(all_tasks.values())
    
    def get_execution_history(self, task_type=None, limit=None):
        """
        Get execution history.
        
        Args:
            task_type (str, optional): Filter by task type
            limit (int, optional): Limit the number of history entries
            
        Returns:
            dict: Execution history by task type
        """
        with self.task_lock:
            if task_type:
                history = list(self.execution_history.get(task_type, []))
                if limit:
                    history = history[-limit:]
                return {task_type: history}
            else:
                result = {}
                for t_type, history in self.execution_history.items():
                    history_list = list(history)
                    if limit:
                        history_list = history_list[-limit:]
                    result[t_type] = history_list
                return result
    
    def wait_for_task(self, task_id, timeout=None):
        """
        Wait for a task to complete.
        
        Args:
            task_id (int): Task ID
            timeout (float, optional): Maximum time to wait in seconds
            
        Returns:
            tuple: (result, error) or (None, "timeout") if timeout occurs
        """
        start_time = time.time()
        result_queue = queue.Queue()
        
        # Create waiting callback
        def wait_callback(result, error=None):
            result_queue.put((result, error))
        
        # Check if task exists and register callback
        with self.task_lock:
            if task_id in self.active_tasks:
                # Task is still active, register callback
                self.active_tasks[task_id]["callback"] = wait_callback
            elif task_id in self.task_results:
                # Task is already complete
                task_data = self.task_results[task_id]
                wait_callback(None, task_data.get("error"))
            else:
                # Task not found
                return None, "Task not found"
        
        # Wait for result
        try:
            result, error = result_queue.get(timeout=timeout)
            return result, error
        except queue.Empty:
            return None, "timeout"
    
    def wait_for_tasks(self, task_ids, timeout=None, return_when="ALL_COMPLETED"):
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids (list): List of task IDs
            timeout (float, optional): Maximum time to wait in seconds
            return_when (str): When to return ('ALL_COMPLETED', 'FIRST_COMPLETED', 'FIRST_EXCEPTION')
            
        Returns:
            dict: Dictionary of task results {task_id: (result, error)}
        """
        start_time = time.time()
        result_dict = {}
        completed_event = threading.Event()
        
        # Create lock for results
        results_lock = threading.Lock()
        
        # Create waiting callback
        def wait_callback(task_id):
            def callback(result, error=None):
                with results_lock:
                    result_dict[task_id] = (result, error)
                    
                    # Check completion condition
                    if return_when == "FIRST_COMPLETED":
                        completed_event.set()
                    elif return_when == "FIRST_EXCEPTION" and error:
                        completed_event.set()
                    elif return_when == "ALL_COMPLETED" and len(result_dict) >= len(task_ids):
                        completed_event.set()
            
            return callback
        
        # Check tasks and register callbacks
        with self.task_lock:
            for task_id in task_ids:
                if task_id in self.active_tasks:
                    # Task is still active, register callback
                    self.active_tasks[task_id]["callback"] = wait_callback(task_id)
                elif task_id in self.task_results:
                    # Task is already complete
                    task_data = self.task_results[task_id]
                    result_dict[task_id] = (None, task_data.get("error"))
                else:
                    # Task not found
                    result_dict[task_id] = (None, "Task not found")
            
            # Check if we're already done
            if return_when == "FIRST_COMPLETED" and result_dict:
                return result_dict
            elif return_when == "FIRST_EXCEPTION" and any(error for _, error in result_dict.values()):
                return result_dict
            elif return_when == "ALL_COMPLETED" and len(result_dict) >= len(task_ids):
                return result_dict
        
        # Wait for completion
        if timeout is not None:
            completed = completed_event.wait(timeout=timeout)
            if not completed:
                # Timeout occurred
                with results_lock:
                    # Add timeout result for incomplete tasks
                    for task_id in task_ids:
                        if task_id not in result_dict:
                            result_dict[task_id] = (None, "timeout")
        else:
            completed_event.wait()
        
        return result_dict
    
    def shutdown(self, wait=True):
        """
        Shutdown the parallel executor and release resources.
        
        Args:
            wait (bool): Whether to wait for tasks to complete
        """
        self.logger.info(f"Shutting down ParallelExecutor (wait={wait})")
        
        # Stop scheduler thread
        self.stop_event.set()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
            if self.scheduler_thread.is_alive():
                self.logger.warning("Scheduler thread did not shut down cleanly")
        
        # Cancel all active tasks
        with self.task_lock:
            active_task_ids = list(self.active_tasks.keys())
        
        for task_id in active_task_ids:
            self._cancel_task(task_id, "shutdown")
        
        # Shutdown executor pools
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)
        
        # Clear queues
        while not self.thread_queue.empty():
            try:
                self.thread_queue.get_nowait()
                self.thread_queue.task_done()
            except queue.Empty:
                break
                
        while not self.process_queue.empty():
            try:
                self.process_queue.get_nowait()
                self.process_queue.task_done()
            except queue.Empty:
                break
        
        self.logger.info("ParallelExecutor shutdown complete")

# Decorator for progress tracking
def track_progress(func):
    """
    Decorator to add progress tracking to a function.
    
    The decorated function will receive a progress_callback parameter
    that can be called to update progress.
    
    Usage:
        @track_progress
        def process_data(data, progress_callback=None):
            # Process data and report progress
            for i, item in enumerate(data):
                # Process item...
                if progress_callback:
                    progress_callback(i+1, len(data))
    """
    def wrapper(*args, progress_callback=None, **kwargs):
        # Pass progress_callback to the function
        kwargs['progress_callback'] = progress_callback
        return func(*args, **kwargs)
    
    # Store reference to original function
    wrapper.__wrapped__ = func
    
    return wrapper"""
Parallel Execution Module

This module provides advanced parallel execution capabilities for the Option Hunter system.
It manages parallel data processing, asynchronous operations, and distributed workloads
for different components of the trading system.
"""

import logging
import os
import time
import threading
import queue
import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import signal
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, deque

class ParallelExecutor:
    """
    Advanced parallel execution manager for high-performance trading operations.
    
    Features:
    - Automatic selection between thread and process parallelism
    - Smart task distribution and load balancing
    - Priority-based execution scheduling
    - Pipeline processing for streaming data
    - Async/await pattern integration
    - Progress tracking and monitoring
    - Graceful cancellation and exception handling
    """
    
    def __init__(self, config=None):
        """
        Initialize the ParallelExecutor.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration
        self.parallel_params = self.config.get("parallel_execution", {})
        
        # Default parameters
        self.max_threads = self.parallel_params.get("max_threads", multiprocessing.cpu_count() * 2)
        self.max_processes = self.parallel_params.get("max_processes", max(1, multiprocessing.cpu_count() - 1))
        self.thread_preference = self.parallel_params.get("thread_preference", True)  # Prefer threads over processes
        self.default_chunk_size = self.parallel_params.get("default_chunk_size", 1000)
        self.thread_timeout = self.parallel_params.get("thread_timeout", 300)  # 5 minutes
        self.process_timeout = self.parallel_params.get("process_timeout", 600)  # 10 minutes
        
        # Executor pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_threads,
            thread_name_prefix="ParallelExecutor_Thread"
        )
        
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.max_processes
        )
        
        # Task tracking
        self.active_tasks = {}
        self.task_results = {}
        self.execution_history = defaultdict(lambda: deque(maxlen=100))
        self.next_task_id = 0
        self.task_lock = threading.RLock()
        
        # Priority queues for task scheduling
        self.thread_queue = queue.PriorityQueue()
        self.process_queue = queue.PriorityQueue()
        
        # Threading controls
        self.stop_event = threading.Event()
        self.scheduler_thread = None
        
        # Progress tracking
        self.progress_callbacks = {}
        
        # Create logs directory
        self.logs_dir = "logs/parallel"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Start scheduler thread
        self._start_scheduler()
        
        self.logger.info(f"ParallelExecutor initialized with {self.max_threads} threads and {self.max_processes} processes")
    
    def _start_scheduler(self):
        """Start the task scheduler thread."""
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="ParallelExecutor_Scheduler"
            )
            self.scheduler_thread.start()
            self.logger.debug("Task scheduler thread started")
    
    def _scheduler_loop(self):
        """Main loop for task scheduling."""
        while not self.stop_event.is_set():
            try:
                # Process thread queue
                self._process_thread_queue()
                
                # Process process queue
                self._process_process_queue()
                
                # Clean up completed tasks
                self._clean_completed_tasks()
                
                # Sleep briefly to avoid tight loop
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(1)  # Sleep longer on error
    
    def _process_thread_queue(self):
        """Process tasks in the thread queue."""
        try:
            # Check if thread pool has available workers
            active_threads = len([t for t in self.active_tasks.values() if t.get("executor_type") == "thread"])
            
            if active_threads >= self.max_threads:
                return  # Thread pool is full
            
            # Process tasks from queue
            while not self.thread_queue.empty() and active_threads < self.max_threads:
                try:
                    # Get task with a small timeout
                    priority, task_data = self.thread_queue.get(timeout=0.01)
                    
                    # Submit to thread pool
                    future = self.thread_pool.submit(
                        self._execute_task_wrapper,
                        task_data["task_id"],
                        task_data["function"],
                        task_data["args"],
                        task_data["kwargs"]
                    )
                    
                    # Set up completion callback
                    future.add_done_callback(
                        lambda f, tid=task_data["task_id"]: self._task_completed_callback(f, tid)
                    )
                    
                    # Update task data
                    with self.task_lock:
                        task_data["future"] = future
                        task_data["start_time"] = time.time()
                        task_data["status"] = "running"
                        task_data["executor_type"] = "thread"
                        self.active_tasks[task_data["task_id"]] = task_data
                    
                    # Mark as processed
                    self.thread_queue.task_done()
                    
                    # Update count
                    active_threads += 1
                    
                except queue.Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing thread queue task: {str(e)}")
                    self.thread_queue.task_done()
                
        except Exception as e:
            self.logger.error(f"Error in thread queue processing: {str(e)}")
    
    def _process_process_queue(self):
        """Process tasks in the process queue."""
        try:
            # Check if process pool has available workers
            active_processes = len([t for t in self.active_tasks.values() if t.get("executor_type") == "process"])
            
            if active_processes >= self.max_processes:
                return  # Process pool is full
            
            # Process tasks from queue
            while not self.process_queue.empty() and active_processes < self.max_processes:
                try:
                    # Get task with a small timeout
                    priority, task_data = self.process_queue.get(timeout=0.01)
                    
                    # Submit to process pool
                    future = self.process_pool.submit(
                        self._execute_task_wrapper,
                        task_data["task_id"],
                        task_data["function"],
                        task_data["args"],
                        task_data["kwargs"]
                    )
                    
                    # Set up completion callback
                    future.add_done_callback(
                        lambda f, tid=task_data["task_id"]: self._task_completed_callback(f, tid)
                    )
                    
                    # Update task data
                    with self.task_lock:
                        task_data["future"] = future
                        task_data["start_time"] = time.time()
                        task_data["status"] = "running"
                        task_data["executor_type"] = "process"
                        self.active_tasks[task_data["task_id"]] = task_data
                    
                    # Mark as processed
                    self.process_queue.task_done()
                    
                    # Update count
                    active_processes += 1
                    
                except queue.Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing process queue task: {str(e)}")
                    self.process_queue.task_done()
                
        except Exception as e:
            self.logger.error(f"Error in process queue processing: {str(e)}")
    
    def _execute_task_wrapper(self, task_id, function, args, kwargs):
        """
        Wrapper to execute a task and handle exceptions.
        
        Args:
            task_id (int): Task ID
            function (callable): Function to execute
            args (tuple): Positional arguments
            kwargs (dict): Keyword arguments
            
        Returns:
            tuple: (task_id, result or None, error or None)
        """
        try:
            # Initialize progress tracking
            total_items = kwargs.pop('_total_items', None)
            
            # Check if this is a progress-tracked task
            if total_items is not None and hasattr(function, '__wrapped__'):
                # This is a progress-tracked function
                
                # Create progress tracking callback
                def progress_callback(current, total=total_items):
                    self._update_task_progress(task_id, current, total)
                
                # Add progress callback to kwargs
                kwargs['progress_callback'] = progress_callback
            
            # Execute function
            result = function(*args, **kwargs)
            
            # Final progress update if needed
            if total_items is not None:
                self._update_task_progress(task_id, total_items, total_items)
            
            return (task_id, result, None)
            
        except Exception as e:
            self.logger.error(f"Error executing task {task_id}: {str(e)}")
            return (task_id, None, str(e))
    
    def _task_completed_callback(self, future, task_id):
        """
        Callback for when a task compl
