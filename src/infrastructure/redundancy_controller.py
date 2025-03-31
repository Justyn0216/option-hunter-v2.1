"""
Redundancy Controller Module

This module provides redundancy management functionality for the Option Hunter trading system.
It includes failover mechanisms, state synchronization between instances, 
data replication, and coordinated decision making between primary and backup instances.
"""

import logging
import os
import json
import time
import threading
import socket
import pickle
import queue
import hashlib
import uuid
import random
from datetime import datetime, timedelta
import redis
import zmq
import numpy as np
import pandas as pd

class RedundancyController:
    """
    System redundancy controller that manages redundant operation between instances.
    
    Features:
    - Primary/backup instance management
    - Automated failover
    - State synchronization
    - Heartbeat monitoring
    - Configuration replication
    - Trading state coordination
    """
    
    # Instance roles
    ROLE_PRIMARY = "primary"
    ROLE_BACKUP = "backup"
    ROLE_INITIALIZING = "initializing"
    ROLE_FAILED = "failed"
    
    def __init__(self, config, instance_id=None):
        """
        Initialize the RedundancyController.
        
        Args:
            config (dict): Redundancy configuration
            instance_id (str, optional): Unique ID for this instance (auto-generated if not provided)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Extract configuration
        self.redundancy_params = config.get("redundancy", {})
        
        # Default parameters
        self.enable_redundancy = self.redundancy_params.get("enable_redundancy", False)
        self.auto_failover = self.redundancy_params.get("auto_failover", True)
        self.primary_host = self.redundancy_params.get("primary_host", "localhost")
        self.primary_port = self.redundancy_params.get("primary_port", 5555)
        self.backup_host = self.redundancy_params.get("backup_host", "localhost")
        self.backup_port = self.redundancy_params.get("backup_port", 5556)
        self.heartbeat_interval = self.redundancy_params.get("heartbeat_interval_seconds", 5)
        self.failover_timeout = self.redundancy_params.get("failover_timeout_seconds", 15)
        self.state_sync_interval = self.redundancy_params.get("state_sync_interval_seconds", 60)
        self.use_redis = self.redundancy_params.get("use_redis", False)
        self.redis_host = self.redundancy_params.get("redis_host", "localhost")
        self.redis_port = self.redundancy_params.get("redis_port", 6379)
        self.redis_password = self.redundancy_params.get("redis_password", None)
        
        # Instance identification
        self.instance_id = instance_id or str(uuid.uuid4())
        self.hostname = socket.gethostname()
        
        # State variables
        self.current_role = self.ROLE_INITIALIZING
        self.primary_instance_id = None
        self.last_heartbeat_sent = time.time()
        self.last_heartbeat_received = None
        self.last_state_sync = None
        self.is_running = False
        self.startup_time = time.time()
        self.failover_count = 0
        
        # Communication channels
        self.context = None
        self.socket = None
        self.redis_client = None
        
        # Synchronization objects
        self.sync_lock = threading.RLock()
        self.state_data = {}
        self.message_queue = queue.Queue()
        
        # Threads
        self.threads = []
        self.stop_event = threading.Event()
        
        # Create logs directory
        self.logs_dir = "logs/redundancy"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize redundancy system
        if self.enable_redundancy:
            self._initialize_redundancy()
            self.logger.info("RedundancyController initialized")
        else:
            self.current_role = self.ROLE_PRIMARY  # Default to primary if redundancy disabled
            self.logger.info("Redundancy disabled, operating as standalone primary instance")
    
    def _initialize_redundancy(self):
        """Initialize the redundancy system."""
        try:
            # Set up messaging context
            self.context = zmq.Context()
            
            # Initialize Redis if enabled
            if self.use_redis:
                self._initialize_redis()
            
            # Determine initial role
            self._determine_initial_role()
            
            # Set up communication based on role
            self._setup_communication()
            
            # Start background threads
            self._start_threads()
            
        except Exception as e:
            self.logger.error(f"Error initializing redundancy system: {str(e)}")
            self.current_role = self.ROLE_FAILED
    
    def _initialize_redis(self):
        """Initialize Redis connection for state storage and coordination."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Connected to Redis server")
            
        except Exception as e:
            self.logger.error(f"Error connecting to Redis: {str(e)}")
            self.redis_client = None
    
    def _determine_initial_role(self):
        """Determine the initial role for this instance."""
        # If Redis is available, use it for coordination
        if self.use_redis and self.redis_client:
            try:
                # Try to get current primary from Redis
                primary_id = self.redis_client.get("primary_instance_id")
                
                if primary_id:
                    # Check if primary is responsive
                    last_heartbeat = self.redis_client.get(f"heartbeat:{primary_id}")
                    
                    if last_heartbeat:
                        last_time = float(last_heartbeat)
                        if time.time() - last_time < self.failover_timeout:
                            # Primary is active
                            self.current_role = self.ROLE_BACKUP
                            self.primary_instance_id = primary_id
                            self.logger.info(f"Taking backup role, primary is {primary_id}")
                            return
                
                # No active primary, attempt to become primary
                if self.redis_client.setnx("primary_instance_id", self.instance_id):
                    # Successfully claimed primary role
                    self.current_role = self.ROLE_PRIMARY
                    self.logger.info("Taking primary role (via Redis)")
                else:
                    # Someone else claimed it first
                    self.current_role = self.ROLE_BACKUP
                    self.primary_instance_id = self.redis_client.get("primary_instance_id")
                    self.logger.info(f"Taking backup role, primary is {self.primary_instance_id}")
                
                return
                
            except Exception as e:
                self.logger.error(f"Error determining role via Redis: {str(e)}")
                # Fall back to network discovery
        
        # Use network discovery to determine role
        try:
            # Try to connect to primary
            socket = self.context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 1000)
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            socket.connect(f"tcp://{self.primary_host}:{self.primary_port}")
            
            # Send ping to check if primary is responsive
            socket.send_json({
                "action": "ping",
                "instance_id": self.instance_id,
                "hostname": self.hostname,
                "timestamp": time.time()
            })
            
            # Wait for response
            try:
                response = socket.recv_json()
                if response.get("status") == "success":
                    # Primary is active
                    self.current_role = self.ROLE_BACKUP
                    self.primary_instance_id = response.get("instance_id")
                    self.logger.info(f"Taking backup role, primary is {self.primary_instance_id}")
                else:
                    # Got response but something's wrong, become primary
                    self.current_role = self.ROLE_PRIMARY
                    self.logger.info("Taking primary role (unexpected primary response)")
            except zmq.Again:
                # No response, become primary
                self.current_role = self.ROLE_PRIMARY
                self.logger.info("Taking primary role (no primary response)")
            
            socket.close()
            
        except Exception as e:
            # Error connecting, become primary
            self.logger.warning(f"Error checking primary status: {str(e)}")
            self.current_role = self.ROLE_PRIMARY
            self.logger.info("Taking primary role (connection error)")
    
    def _setup_communication(self):
        """Set up communication channels based on current role."""
        # Close existing socket if any
        if self.socket:
            self.socket.close()
            self.socket = None
        
        try:
            if self.current_role == self.ROLE_PRIMARY:
                # Primary instance listens for connections from backups
                self.socket = self.context.socket(zmq.REP)
                bind_address = f"tcp://*:{self.primary_port}"
                self.socket.bind(bind_address)
                self.logger.info(f"Primary instance listening on port {self.primary_port}")
                
                # Register as primary in Redis if available
                if self.use_redis and self.redis_client:
                    self.redis_client.set("primary_instance_id", self.instance_id)
                    self.redis_client.set(f"primary_host", self.hostname)
                    self.redis_client.set(f"heartbeat:{self.instance_id}", time.time())
                
            elif self.current_role == self.ROLE_BACKUP:
                # Backup instance connects to primary
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.LINGER, 1000)
                self.socket.setsockopt(zmq.RCVTIMEO, self.heartbeat_interval * 1000)
                primary_address = f"tcp://{self.primary_host}:{self.primary_port}"
                self.socket.connect(primary_address)
                self.logger.info(f"Backup instance connected to primary at {primary_address}")
                
                # Register as backup in Redis if available
                if self.use_redis and self.redis_client:
                    self.redis_client.sadd("backup_instances", self.instance_id)
                    self.redis_client.set(f"backup:{self.instance_id}:host", self.hostname)
                    self.redis_client.set(f"heartbeat:{self.instance_id}", time.time())
        
        except Exception as e:
            self.logger.error(f"Error setting up communication: {str(e)}")
            self.current_role = self.ROLE_FAILED
    
    def _start_threads(self):
        """Start background threads for redundancy management."""
        # Clear any existing threads
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1)
        
        self.threads = []
        self.stop_event.clear()
        
        # Heartbeat thread for both primary and backup
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_thread,
            name="HeartbeatThread",
            daemon=True
        )
        self.threads.append(heartbeat_thread)
        
        # State synchronization thread for backup
        if self.current_role == self.ROLE_BACKUP:
            sync_thread = threading.Thread(
                target=self._state_sync_thread,
                name="StateSyncThread",
                daemon=True
            )
            self.threads.append(sync_thread)
        
        # Message processing thread for primary
        if self.current_role == self.ROLE_PRIMARY:
            message_thread = threading.Thread(
                target=self._message_processing_thread,
                name="MessageThread",
                daemon=True
            )
            self.threads.append(message_thread)
        
        # Start all threads
        for thread in self.threads:
            thread.start()
        
        self.is_running = True
        self.logger.info(f"Started {len(self.threads)} redundancy management threads")
    
    def _heartbeat_thread(self):
        """Background thread for sending and receiving heartbeats."""
        self.logger.debug(f"Heartbeat thread started for {self.current_role} instance")
        
        while not self.stop_event.is_set():
            try:
                if self.current_role == self.ROLE_PRIMARY:
                    # Primary listens for messages from backups
                    try:
                        # Non-blocking receive with timeout
                        if self.socket.poll(100, zmq.POLLIN):
                            message = self.socket.recv_json()
                            self._handle_backup_message(message)
                    except Exception as e:
                        self.logger.error(f"Error receiving backup message: {str(e)}")
                    
                    # Update heartbeat in Redis if available
                    if self.use_redis and self.redis_client:
                        self.redis_client.set(f"heartbeat:{self.instance_id}", time.time())
                        
                elif self.current_role == self.ROLE_BACKUP:
                    # Backup sends heartbeat to primary
                    current_time = time.time()
                    
                    # Send heartbeat every interval
                    if current_time - self.last_heartbeat_sent >= self.heartbeat_interval:
                        self._send_heartbeat()
                        self.last_heartbeat_sent = current_time
                    
                    # Check if primary is responsive
                    if self.last_heartbeat_received is None or \
                       current_time - self.last_heartbeat_received > self.failover_timeout:
                        # Primary may be down, initiate failover if auto-failover is enabled
                        if self.auto_failover:
                            self._initiate_failover()
                    
                    # Update heartbeat in Redis if available
                    if self.use_redis and self.redis_client:
                        self.redis_client.set(f"heartbeat:{self.instance_id}", time.time())
            
            except Exception as e:
                self.logger.error(f"Error in heartbeat thread: {str(e)}")
            
            # Sleep briefly to prevent high CPU usage
            self.stop_event.wait(0.1)
    
    def _send_heartbeat(self):
        """Send heartbeat message from backup to primary."""
        try:
            heartbeat_message = {
                "action": "heartbeat",
                "instance_id": self.instance_id,
                "hostname": self.hostname,
                "timestamp": time.time(),
                "role": self.current_role
            }
            
            self.socket.send_json(heartbeat_message)
            
            # Wait for response with timeout
            response = self.socket.recv_json()
            
            if response.get("status") == "success":
                self.last_heartbeat_received = time.time()
                
                # Update primary instance id if it changed
                if "instance_id" in response:
                    self.primary_instance_id = response["instance_id"]
            else:
                self.logger.warning(f"Received error response from primary: {response.get('error', 'Unknown error')}")
                
        except zmq.Again:
            # Timeout waiting for response
            self.logger.warning("Timeout waiting for heartbeat response from primary")
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {str(e)}")
    
    def _handle_backup_message(self, message):
        """
        Handle message received from backup instance.
        
        Args:
            message (dict): Message from backup
        """
        action = message.get("action")
        sender_id = message.get("instance_id", "unknown")
        
        response = {
            "status": "success",
            "instance_id": self.instance_id,
            "timestamp": time.time()
        }
        
        try:
            if action == "heartbeat":
                # Just acknowledge heartbeat
                pass
                
            elif action == "ping":
                # Simple ping to check if primary is alive
                pass
                
            elif action == "sync_request":
                # Add to message queue for processing
                self.message_queue.put(message)
                response["message"] = "Sync request queued"
                
            elif action == "get_state":
                # Return requested state variables
                keys = message.get("keys", [])
                state_data = {}
                
                with self.sync_lock:
                    for key in keys:
                        if key in self.state_data:
                            state_data[key] = self.state_data[key]
                
                response["state_data"] = state_data
                
            else:
                # Unknown action
                response["status"] = "error"
                response["error"] = f"Unknown action: {action}"
                
        except Exception as e:
            self.logger.error(f"Error handling message from backup {sender_id}: {str(e)}")
            response["status"] = "error"
            response["error"] = str(e)
        
        # Send response
        self.socket.send_json(response)
    
    def _message_processing_thread(self):
        """Thread for processing queued messages in primary instance."""
        self.logger.debug("Message processing thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get message from queue with timeout
                try:
                    message = self.message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process message
                action = message.get("action")
                sender_id = message.get("instance_id", "unknown")
                
                if action == "sync_request":
                    # Handle state synchronization request
                    self.logger.debug(f"Processing sync request from {sender_id}")
                    # No immediate response needed as the original message already got a response
                    # The backup will request specific state data via get_state
                
                # Mark message as processed
                self.message_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in message processing thread: {str(e)}")
            
            # Sleep briefly to prevent high CPU usage if queue is empty
            if self.message_queue.empty():
                self.stop_event.wait(0.1)
    
    def _state_sync_thread(self):
        """Thread for periodic state synchronization in backup instance."""
        self.logger.debug("State synchronization thread started")
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Perform state sync at configured interval
                if self.last_state_sync is None or \
                   current_time - self.last_state_sync >= self.state_sync_interval:
                    
                    self._sync_state_from_primary()
                    self.last_state_sync = current_time
                
            except Exception as e:
                self.logger.error(f"Error in state sync thread: {str(e)}")
            
            # Sleep until next sync
            next_sync = self.state_sync_interval
            if self.last_state_sync:
                elapsed = time.time() - self.last_state_sync
                next_sync = max(0.1, self.state_sync_interval - elapsed)
            
            self.stop_event.wait(next_sync)
    
    def _sync_state_from_primary(self):
        """Synchronize state data from primary instance."""
        if self.current_role != self.ROLE_BACKUP:
            return
        
        try:
            # Request sync
            sync_request = {
                "action": "sync_request",
                "instance_id": self.instance_id,
                "timestamp": time.time()
            }
            
            self.socket.send_json(sync_request)
            
            # Wait for response
            response = self.socket.recv_json()
            
            if response.get("status") != "success":
                self.logger.warning(f"Sync request failed: {response.get('error', 'Unknown error')}")
                return
            
            # Get keys to sync
            keys_to_sync = list(self.state_data.keys())
            
            # Request state data
            get_state_request = {
                "action": "get_state",
                "instance_id": self.instance_id,
                "timestamp": time.time(),
                "keys": keys_to_sync
            }
            
            self.socket.send_json(get_state_request)
            
            # Wait for response
            response = self.socket.recv_json()
            
            if response.get("status") != "success":
                self.logger.warning(f"Get state failed: {response.get('error', 'Unknown error')}")
                return
            
            # Update local state with data from primary
            state_data = response.get("state_data", {})
            
            with self.sync_lock:
                for key, value in state_data.items():
                    self.state_data[key] = value
            
            self.logger.debug(f"Synchronized {len(state_data)} state variables from primary")
            
        except zmq.Again:
            self.logger.warning("Timeout waiting for state sync response from primary")
        except Exception as e:
            self.logger.error(f"Error synchronizing state: {str(e)}")
    
    def _initiate_failover(self):
        """Initiate failover process, converting this backup to primary."""
        self.logger.warning("Initiating failover: backup taking over as primary")
        
        try:
            # Check Redis first if available
            if self.use_redis and self.redis_client:
                # Check if primary is still reporting heartbeats
                primary_id = self.redis_client.get("primary_instance_id")
                if primary_id:
                    last_heartbeat = self.redis_client.get(f"heartbeat:{primary_id}")
                    if last_heartbeat:
                        last_time = float(last_heartbeat)
                        if time.time() - last_time < self.failover_timeout:
                            # Primary is still active, abort failover
                            self.logger.info("Aborting failover: primary is still active according to Redis")
                            return
                
                # Attempt to claim primary role
                if not self.redis_client.set("primary_instance_id", self.instance_id, nx=False):
                    # Could not update primary_instance_id
                    self.logger.warning("Could not claim primary role in Redis")
            
            # Change role to primary
            old_role = self.current_role
            self.current_role = self.ROLE_PRIMARY
            
            # Reconfigure communication
            self._setup_communication()
            
            # Restart threads
            self._start_threads()
            
            # Increment failover count
            self.failover_count += 1
            
            # Log the failover
            failover_info = {
                "timestamp": datetime.now().isoformat(),
                "instance_id": self.instance_id,
                "hostname": self.hostname,
                "previous_role": old_role,
                "new_role": self.current_role,
                "previous_primary": self.primary_instance_id,
                "failover_count": self.failover_count
            }
            
            # Save to file
            log_file = os.path.join(self.logs_dir, "failovers.json")
            
            try:
                # Load existing logs if available
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                else:
                    log_data = []
                
                # Add new failover event
                log_data.append(failover_info)
                
                # Save log file
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
            except Exception as e:
                self.logger.error(f"Error logging failover: {str(e)}")
            
            self.logger.critical(f"FAILOVER COMPLETED: Instance {self.instance_id} is now PRIMARY")
            
        except Exception as e:
            self.logger.error(f"Error during failover: {str(e)}")
            self.current_role = self.ROLE_FAILED
    
    def is_primary(self):
        """
        Check if this instance is the primary.
        
        Returns:
            bool: True if this instance is the primary
        """
        return self.current_role == self.ROLE_PRIMARY
    
    def is_backup(self):
        """
        Check if this instance is a backup.
        
        Returns:
            bool: True if this instance is a backup
        """
        return self.current_role == self.ROLE_BACKUP
    
    def get_instance_info(self):
        """
        Get information about this instance.
        
        Returns:
            dict: Instance information
        """
        return {
            "instance_id": self.instance_id,
            "hostname": self.hostname,
            "role": self.current_role,
            "primary_id": self.primary_instance_id,
            "uptime": time.time() - self.startup_time,
            "failover_count": self.failover_count,
            "last_heartbeat_sent": self.last_heartbeat_sent,
            "last_heartbeat_received": self.last_heartbeat_received,
            "last_state_sync": self.last_state_sync,
            "redundancy_enabled": self.enable_redundancy,
            "auto_failover": self.auto_failover,
            "timestamp": time.time()
        }
    
    def update_state(self, key, value):
        """
        Update state data that will be synchronized between instances.
        
        Args:
            key (str): State variable name
            value: State variable value
        """
        with self.sync_lock:
            self.state_data[key] = value
        
        # Also update in Redis if enabled
        if self.use_redis and self.redis_client:
            try:
                # Use pickle to serialize complex objects
                serialized = pickle.dumps(value)
                self.redis_client.set(f"state:{key}", serialized)
            except Exception as e:
                self.logger.error(f"Error updating state in Redis: {str(e)}")
    
    def get_state(self, key, default=None):
        """
        Get state variable value.
        
        Args:
            key (str): State variable name
            default: Default value if key is not found
            
        Returns:
            Value of state variable or default
        """
        # First check local state
        with self.sync_lock:
            if key in self.state_data:
                return self.state_data[key]
        
        # If not found locally and Redis is enabled, check there
        if self.use_redis and self.redis_client:
            try:
                serialized = self.redis_client.get(f"state:{key}")
                if serialized:
                    return pickle.loads(serialized)
            except Exception as e:
                self.logger.error(f"Error getting state from Redis: {str(e)}")
        
        return default
    
    def get_backup_instances(self):
        """
        Get list of active backup instances.
        
        Returns:
            list: Active backup instances
        """
        if not self.is_primary():
            self.logger.warning("Only primary instance can get backup instances")
            return []
        
        backup_instances = []
        
        # Check Redis if available
        if self.use_redis and self.redis_client:
            try:
                # Get all backup instance IDs
                instance_ids = self.redis_client.smembers("backup_instances")
                
                for instance_id in instance_ids:
                    # Check if instance is still active
                    last_heartbeat = self.redis_client.get(f"heartbeat:{instance_id}")
                    
                    if last_heartbeat:
                        last_time = float(last_heartbeat)
                        if time.time() - last_time < self.failover_timeout:
                            # Instance is active
                            hostname = self.redis_client.get(f"backup:{instance_id}:host") or "unknown"
                            
                            backup_instances.append({
                                "instance_id": instance_id,
                                "hostname": hostname,
                                "last_heartbeat": last_time
                            })
                
                return backup_instances
                
            except Exception as e:
                self.logger.error(f"Error getting backup instances from Redis: {str(e)}")
        
        # If Redis not available or failed, return empty list
        # Could be enhanced to track backups in local state
        return []
    
    def coordinate_decision(self, decision_key, value=None, timeout=5):
        """
        Coordinate a decision between instances.
        
        Args:
            decision_key (str): Decision identifier
            value: Value proposed by this instance
            timeout (int): Timeout in seconds
            
        Returns:
            tuple: (success, result)
        """
        if not self.enable_redundancy:
            # If redundancy is disabled, make decision locally
            return (True, value)
        
        # Use Redis if available
        if self.use_redis and self.redis_client:
            try:
                # Generate a unique key for this decision
                redis_key = f"decision:{decision_key}:{time.time()}"
                
                if self.is_primary():
                    # Primary makes the decision and publishes it
                    self.redis_client.set(redis_key, pickle.dumps(value))
                    self.redis_client.expire(redis_key, 300)  # Keep for 5 minutes
                    return (True, value)
                    
                else:
                    # Backup gets decision from primary
                    start_time = time.time()
                    
                    while time.time() - start_time < timeout:
                        # Find the most recent decision key that matches
                        keys = self.redis_client.keys(f"decision:{decision_key}:*")
                        
                        if keys:
                            # Get most recent key (highest timestamp)
                            latest_key = sorted(keys)[-1]
                            result = self.redis_client.get(latest_key)
                            
                            if result:
                                return (True, pickle.loads(result))
                        
                        # Wait and try again
                        time.sleep(0.5)
                    
                    # Timeout waiting for decision
                    return (False, None)
                    
            except Exception as e:
                self.logger.error(f"Error coordinating decision via Redis: {str(e)}")
        
        # Fallback: primary decides, backup uses local value
        if self.is_primary():
            return (True, value)
        else:
            return (False, value)  # Backup couldn't coordinate
    
    def replicate_configuration(self, config_data):
        """
        Replicate configuration to all instances.
        
        Args:
            config_data (dict): Configuration data
            
        Returns:
            bool: True if successfully replicated
        """
        if not self.enable_redundancy:
            return True
        
        if not self.is_primary():
            self.logger.warning("Only primary instance can replicate configuration")
            return False
        
        try:
            # Use Redis if available
            if self.use_redis and self.redis_client:
                # Store configuration in Redis
                serialized = pickle.dumps(config_data)
                self.redis_client.set("config:latest", serialized)
                self.redis_client.set("config:timestamp", time.time())
                
                self.logger.info("Configuration replicated to Redis")
                return True
            else:
                # No Redis, can't replicate
                self.logger.warning("Cannot replicate configuration without Redis")
                return False
                
        except Exception as e:
            self.logger.error(f"Error replicating configuration: {str(e)}")
            return False
    
    def get_replicated_configuration(self):
        """
        Get configuration that was replicated from primary.
        
        Returns:
            dict or None: Replicated configuration or None if not available
        """
        if not self.enable_redundancy or self.is_primary():
            return None
        
        try:
            # Try to get from Redis
            if self.use_redis and self.redis_client:
                serialized = self.redis_client.get("config:latest")
                
                if serialized:
                    config_data = pickle.loads(serialized)
                    return config_data
                    
        except Exception as e:
            self.logger.error(f"Error getting replicated configuration: {str(e)}")
            
        return None
    
    def log_redundancy_event(self, event_type, details=None):
        """
        Log a redundancy-related event.
        
        Args:
            event_type (str): Type of event
            details (dict, optional): Event details
            
        Returns:
            bool: True if event was logged successfully
        """
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "instance_id": self.instance_id,
                "hostname": self.hostname,
                "role": self.current_role,
                "event_type": event_type,
                "details": details or {}
            }
            
            # Log to file
            log_file = os.path.join(self.logs_dir, f"events_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Load existing logs if available
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            # Add new event
            log_data.append(event)
            
            # Save log file
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            # Also log to Redis if available
            if self.use_redis and self.redis_client:
                event_id = f"{time.time()}_{random.randint(1000, 9999)}"
                self.redis_client.set(f"event:{event_id}", json.dumps(event))
                self.redis_client.expire(f"event:{event_id}", 86400)  # Expire after 1 day
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging redundancy event: {str(e)}")
            return False
    
    def get_redundancy_status(self):
        """
        Get comprehensive status of the redundancy system.
        
        Returns:
            dict: Redundancy status information
        """
        status = {
            "instance_id": self.instance_id,
            "hostname": self.hostname,
            "role": self.current_role,
            "primary_id": self.primary_instance_id,
            "redundancy_enabled": self.enable_redundancy,
            "is_running": self.is_running,
            "uptime": time.time() - self.startup_time,
            "failover_count": self.failover_count,
            "redis_connected": bool(self.redis_client) if self.use_redis else False,
            "heartbeat": {
                "last_sent": self.last_heartbeat_sent,
                "last_received": self.last_heartbeat_received,
                "interval": self.heartbeat_interval
            },
            "state_sync": {
                "last_sync": self.last_state_sync,
                "interval": self.state_sync_interval,
                "state_vars": len(self.state_data)
            },
            "threads": {
                "count": len(self.threads),
                "active": sum(1 for t in self.threads if t.is_alive())
            },
            "timestamp": time.time()
        }
        
        # Add additional info based on role
        if self.is_primary():
            # Get backup instances
            backup_instances = self.get_backup_instances()
            status["backup_instances"] = {
                "count": len(backup_instances),
                "instances": backup_instances
            }
            
            # Add message queue info
            status["message_queue"] = {
                "size": self.message_queue.qsize()
            }
            
        elif self.is_backup():
            # Add primary connection info
            status["primary_connection"] = {
                "host": self.primary_host,
                "port": self.primary_port,
                "responsive": self.last_heartbeat_received is not None and \
                              (time.time() - self.last_heartbeat_received < self.failover_timeout)
            }
        
        return status
    
    def perform_maintenance_task(self, task_name):
        """
        Perform a maintenance task on appropriate instance.
        
        Args:
            task_name (str): Name of maintenance task
            
        Returns:
            dict: Task result
        """
        # Default response
        result = {
            "status": "error",
            "task": task_name,
            "instance": self.instance_id,
            "role": self.current_role,
            "error": "Unknown task or not implemented"
        }
        
        # Verify instance can perform this task
        if task_name in ["cleanup_stale_data", "check_consistency"]:
            # These tasks should only be run on primary
            if not self.is_primary():
                result["error"] = f"Task '{task_name}' can only be run on primary instance"
                return result
                
        elif task_name in ["sync_data"]:
            # These tasks should only be run on backup
            if not self.is_backup():
                result["error"] = f"Task '{task_name}' can only be run on backup instance"
                return result
        
        # Perform requested task
        try:
            if task_name == "cleanup_stale_data":
                result = self._perform_cleanup()
                
            elif task_name == "check_consistency":
                result = self._check_data_consistency()
                
            elif task_name == "sync_data":
                result = self._perform_full_sync()
                
            elif task_name == "test_failover":
                result = self._perform_test_failover()
                
            else:
                # Unknown task
                pass
                
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.logger.error(f"Error performing maintenance task '{task_name}': {str(e)}")
        
        return result
    
    def _perform_cleanup(self):
        """
        Clean up stale data in Redis and local storage.
        
        Returns:
            dict: Cleanup results
        """
        results = {
            "status": "success",
            "task": "cleanup_stale_data",
            "cleanup_actions": []
        }
        
        try:
            # Clean up Redis if available
            if self.use_redis and self.redis_client:
                # Remove stale heartbeats
                all_keys = self.redis_client.keys("heartbeat:*")
                removed = 0
                
                for key in all_keys:
                    try:
                        heartbeat_time = float(self.redis_client.get(key) or 0)
                        if time.time() - heartbeat_time > 3600:  # Older than 1 hour
                            self.redis_client.delete(key)
                            removed += 1
                    except:
                        pass
                
                results["cleanup_actions"].append({
                    "target": "redis_heartbeats",
                    "removed": removed
                })
                
                # Remove old events
                event_keys = self.redis_client.keys("event:*")
                self.redis_client.delete(*event_keys) if event_keys else None
                
                results["cleanup_actions"].append({
                    "target": "redis_events",
                    "removed": len(event_keys)
                })
            
            # Clean up old log files
            log_files = os.listdir(self.logs_dir)
            removed_files = 0
            
            for filename in log_files:
                if filename.startswith("events_"):
                    try:
                        # Parse date from filename (format: events_YYYYMMDD.json)
                        date_str = filename.split("_")[1].split(".")[0]
                        file_date = datetime.strptime(date_str, "%Y%m%d").date()
                        
                        # Remove files older than 30 days
                        if (datetime.now().date() - file_date).days > 30:
                            os.remove(os.path.join(self.logs_dir, filename))
                            removed_files += 1
                    except:
                        pass
            
            results["cleanup_actions"].append({
                "target": "log_files",
                "removed": removed_files
            })
            
            # Log the cleanup
            self.log_redundancy_event("cleanup", {
                "actions": results["cleanup_actions"]
            })
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
    
    def _check_data_consistency(self):
        """
        Check consistency of data between primary and backups.
        
        Returns:
            dict: Consistency check results
        """
        results = {
            "status": "success",
            "task": "check_consistency",
            "checks": []
        }
        
        try:
            # Only works with Redis
            if not (self.use_redis and self.redis_client):
                results["status"] = "error"
                results["error"] = "Consistency check requires Redis"
                return results
            
            # Get all backup instances
            backup_instances = self.get_backup_instances()
            
            if not backup_instances:
                results["status"] = "warning"
                results["message"] = "No backup instances found for consistency check"
                return results
            
            # Check state variable consistency
            state_keys = [key for key in self.redis_client.keys("state:*")]
            
            for key in state_keys:
                var_name = key.split(":", 1)[1]
                
                # Get primary's value
                primary_value = pickle.loads(self.redis_client.get(key) or b'')
                primary_hash = hashlib.md5(pickle.dumps(primary_value)).hexdigest()
                
                # Compare with local state
                local_value = None
                with self.sync_lock:
                    local_value = self.state_data.get(var_name)
                
                local_hash = hashlib.md5(pickle.dumps(local_value)).hexdigest() if local_value is not None else None
                
                results["checks"].append({
                    "variable": var_name,
                    "primary_hash": primary_hash,
                    "local_matches_redis": primary_hash == local_hash,
                    "hash_difference": primary_hash != local_hash
                })
            
            # Log the consistency check
            self.log_redundancy_event("consistency_check", {
                "variables_checked": len(results["checks"]),
                "differences_found": sum(1 for c in results["checks"] if c.get("hash_difference"))
            })
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
    
    def _perform_full_sync(self):
        """
        Perform a full state synchronization from primary (backup only).
        
        Returns:
            dict: Sync results
        """
        results = {
            "status": "success",
            "task": "sync_data",
            "synced_variables": 0
        }
        
        if not self.is_backup():
            results["status"] = "error"
            results["error"] = "Full sync can only be performed by backup instance"
            return results
        
        try:
            # Force a state sync from primary
            self._sync_state_from_primary()
            
            # If using Redis, also sync from there
            if self.use_redis and self.redis_client:
                state_keys = [key for key in self.redis_client.keys("state:*")]
                
                for key in state_keys:
                    var_name = key.split(":", 1)[1]
                    value = pickle.loads(self.redis_client.get(key) or b'')
                    
                    with self.sync_lock:
                        self.state_data[var_name] = value
                        results["synced_variables"] += 1
            
            # Update last sync time
            self.last_state_sync = time.time()
            
            # Log the sync
            self.log_redundancy_event("full_sync", {
                "variables_synced": results["synced_variables"]
            })
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
    
    def _perform_test_failover(self):
        """
        Perform a test failover to verify redundancy.
        
        Returns:
            dict: Test results
        """
        results = {
            "status": "success",
            "task": "test_failover",
            "original_role": self.current_role
        }
        
        try:
            if self.is_primary():
                # Primary: temporarily yield to backup
                old_role = self.current_role
                
                # Set a flag in Redis
                if self.use_redis and self.redis_client:
                    self.redis_client.set("test_failover", time.time())
                    self.redis_client.expire("test_failover", 60)  # Auto-expire after 1 minute
                
                # Change role temporarily
                self.current_role = self.ROLE_BACKUP
                results["new_role"] = self.current_role
                
                # Log a planned failover
                self.log_redundancy_event("test_failover_yield", {
                    "previous_role": old_role,
                    "new_role": self.current_role
                })
                
                # Wait a few seconds
                time.sleep(5)
                
                # Return to primary role
                self.current_role = self.ROLE_PRIMARY
                
                # Reconfigure
                self._setup_communication()
                self._start_threads()
                
            elif self.is_backup():
                # Backup: temporarily become primary if test is in progress
                if self.use_redis and self.redis_client:
                    test_time = self.redis_client.get("test_failover")
                    
                    if test_time:
                        old_role = self.current_role
                        
                        # Change role temporarily
                        self.current_role = self.ROLE_PRIMARY
                        results["new_role"] = self.current_role
                        
                        # Log a test failover
                        self.log_redundancy_event("test_failover_assume", {
                            "previous_role": old_role,
                            "new_role": self.current_role
                        })
                        
                        # Wait a few seconds
                        time.sleep(5)
                        
                        # Return to backup role
                        self.current_role = self.ROLE_BACKUP
                        
                        # Reconfigure
                        self._setup_communication()
                        self._start_threads()
                    else:
                        results["status"] = "warning"
                        results["message"] = "No test failover in progress"
                else:
                    results["status"] = "error"
                    results["error"] = "Test failover requires Redis"
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
    
    def shutdown(self):
        """Clean shutdown of the redundancy controller."""
        self.logger.info("Shutting down RedundancyController")
        
        # Signal threads to stop
        self.stop_event.set()
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)
        
        # Notify of role change if primary
        if self.current_role == self.ROLE_PRIMARY:
            if self.use_redis and self.redis_client:
                try:
                    # Clear primary marker if we're the primary
                    primary_id = self.redis_client.get("primary_instance_id")
                    if primary_id == self.instance_id:
                        self.redis_client.delete("primary_instance_id")
                    
                    # Add shutdown timestamp
                    self.redis_client.set(f"shutdown:{self.instance_id}", time.time())
                    
                    # Log the shutdown
                    self.log_redundancy_event("shutdown", {
                        "role": self.current_role,
                        "primary_id": self.primary_instance_id
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error updating Redis during shutdown: {str(e)}")
        
        # Close ZeroMQ socket and context
        if self.socket:
            self.socket.close()
            self.socket = None
            
        if self.context:
            self.context.term()
            self.context = None
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
            self.redis_client = None
