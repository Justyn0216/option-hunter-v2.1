"""
Colocation Manager Module

This module provides functionality for managing colocation services and optimizing
connection to exchange data centers for the Option Hunter trading system.
It includes setup, monitoring, and management of colocated server instances.
"""

import logging
import os
import json
import requests
import time
import socket
import subprocess
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import paramiko
from cryptography.fernet import Fernet
import yaml

class ColocationManager:
    """
    Manages colocation services for high-frequency trading.
    
    Features:
    - Colocation server management
    - Secure configuration of remote services
    - Network latency monitoring
    - Failover management
    - Status monitoring and reporting
    - SSH and secure command execution
    """
    
    def __init__(self, config):
        """
        Initialize the ColocationManager.
        
        Args:
            config (dict): Colocation configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Extract configuration
        self.colocation_params = config.get("colocation", {})
        
        # Default parameters
        self.enable_colocation = self.colocation_params.get("enable_colocation", False)
        self.primary_server = self.colocation_params.get("primary_server", None)
        self.backup_servers = self.colocation_params.get("backup_servers", [])
        self.auth_key_path = self.colocation_params.get("ssh_key_path", None)
        self.auth_username = self.colocation_params.get("ssh_username", None)
        self.monitoring_interval = self.colocation_params.get("monitoring_interval_seconds", 60)
        self.failover_threshold = self.colocation_params.get("failover_threshold_ms", 100)
        self.heartbeat_interval = self.colocation_params.get("heartbeat_interval_seconds", 30)
        self.encryption_key = self._load_or_create_encryption_key()
        
        # State tracking
        self.active_server = None
        self.server_statuses = {}
        self.last_failover = None
        self.ssh_clients = {}
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Create logs directory
        self.logs_dir = "logs/colocation"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize encrypted credentials store
        self.credentials_store = {}
        self._load_credentials()
        
        # Check if colocation is enabled
        if not self.enable_colocation:
            self.logger.info("Colocation services are disabled in configuration")
            return
        
        # Validate required configuration
        if not self.primary_server:
            self.logger.error("Primary server not configured, colocation services disabled")
            self.enable_colocation = False
            return
        
        # Initialize and test connections
        if not self._initialize_connections():
            self.logger.error("Failed to initialize colocation connections, services disabled")
            self.enable_colocation = False
            return
        
        # Start monitoring
        self._start_monitoring()
        
        self.logger.info("ColocationManager initialized successfully")
    
    def _load_or_create_encryption_key(self):
        """
        Load or create encryption key for secure credential storage.
        
        Returns:
            bytes: Encryption key
        """
        key_file = self.colocation_params.get("encryption_key_file", "data/colocation/encryption_key.bin")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            
            # Try to load existing key
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    key = f.read()
                    return key
            
            # Create new key if none exists
            key = Fernet.generate_key()
            
            # Save key to file with restricted permissions
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restricted permissions on Unix systems
            try:
                os.chmod(key_file, 0o600)  # Only owner can read/write
            except:
                pass  # Ignore on non-Unix systems
                
            return key
            
        except Exception as e:
            self.logger.error(f"Error managing encryption key: {str(e)}")
            # Fallback to a session-only key
            return Fernet.generate_key()
    
    def _load_credentials(self):
        """
        Load encrypted credentials from storage.
        """
        credentials_file = self.colocation_params.get("credentials_file", "data/colocation/credentials.enc")
        
        try:
            if os.path.exists(credentials_file):
                # Create cipher using key
                cipher = Fernet(self.encryption_key)
                
                # Read and decrypt credentials
                with open(credentials_file, 'rb') as f:
                    encrypted_data = f.read()
                    
                decrypted_data = cipher.decrypt(encrypted_data)
                self.credentials_store = json.loads(decrypted_data)
                
                self.logger.debug(f"Loaded credentials for {len(self.credentials_store)} servers")
                
        except Exception as e:
            self.logger.error(f"Error loading credentials: {str(e)}")
            # Reset to empty if there's an error
            self.credentials_store = {}
    
    def _save_credentials(self):
        """
        Save encrypted credentials to storage.
        """
        credentials_file = self.colocation_params.get("credentials_file", "data/colocation/credentials.enc")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(credentials_file), exist_ok=True)
            
            # Create cipher using key
            cipher = Fernet(self.encryption_key)
            
            # Encrypt credentials
            data_json = json.dumps(self.credentials_store)
            encrypted_data = cipher.encrypt(data_json.encode())
            
            # Save to file with restricted permissions
            with open(credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restricted permissions on Unix systems
            try:
                os.chmod(credentials_file, 0o600)  # Only owner can read/write
            except:
                pass  # Ignore on non-Unix systems
                
            self.logger.debug("Saved encrypted credentials")
            
        except Exception as e:
            self.logger.error(f"Error saving credentials: {str(e)}")
    
    def store_server_credential(self, server, username, password=None, key_path=None):
        """
        Securely store server credentials.
        
        Args:
            server (str): Server address
            username (str): Username
            password (str, optional): Password (only stored if key_path not provided)
            key_path (str, optional): Path to SSH key file
        """
        self.credentials_store[server] = {
            "username": username,
        }
        
        if key_path:
            self.credentials_store[server]["key_path"] = key_path
        elif password:
            self.credentials_store[server]["password"] = password
        
        # Save credentials
        self._save_credentials()
        
        self.logger.info(f"Stored credentials for server {server}")
    
    def _initialize_connections(self):
        """
        Initialize connections to all configured servers.
        
        Returns:
            bool: True if at least one server connection succeeded
        """
        # Check primary server
        primary_status = self._check_server_connection(self.primary_server)
        self.server_statuses[self.primary_server] = primary_status
        
        if primary_status["status"] == "connected":
            self.active_server = self.primary_server
            self.logger.info(f"Connected to primary colocation server: {self.primary_server}")
        else:
            self.logger.warning(f"Could not connect to primary server: {self.primary_server}")
            
            # Try backup servers
            for server in self.backup_servers:
                backup_status = self._check_server_connection(server)
                self.server_statuses[server] = backup_status
                
                if backup_status["status"] == "connected":
                    self.active_server = server
                    self.logger.info(f"Connected to backup colocation server: {server}")
                    break
        
        return self.active_server is not None
    
    def _check_server_connection(self, server):
        """
        Check connection to a colocation server.
        
        Args:
            server (str): Server address
            
        Returns:
            dict: Server status information
        """
        status = {
            "server": server,
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "latency_ms": None,
            "error": None
        }
        
        try:
            # Check basic connectivity with ping
            ping_ms = self._ping_server(server)
            
            if ping_ms is None:
                status["status"] = "unreachable"
                status["error"] = "Ping failed"
                return status
            
            status["latency_ms"] = ping_ms
            
            # Try SSH connection
            ssh_client = self._get_ssh_connection(server)
            
            if ssh_client:
                # Execute simple command to verify connection
                stdin, stdout, stderr = ssh_client.exec_command("uptime")
                uptime_output = stdout.read().decode().strip()
                
                if uptime_output:
                    status["status"] = "connected"
                    status["uptime"] = uptime_output
                else:
                    status["status"] = "error"
                    status["error"] = stderr.read().decode().strip()
                
                # Don't close the connection - keep it for reuse
                self.ssh_clients[server] = ssh_client
            else:
                status["status"] = "auth_failed"
                status["error"] = "SSH authentication failed"
            
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
            self.logger.error(f"Error checking server {server}: {str(e)}")
        
        return status
    
    def _ping_server(self, server, count=3, timeout=2):
        """
        Ping a server to check connectivity and measure latency.
        
        Args:
            server (str): Server address
            count (int): Number of pings to send
            timeout (int): Timeout in seconds
            
        Returns:
            float or None: Average ping time in milliseconds, or None if failed
        """
        try:
            # First try socket connection for a quick check
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            start_time = time.time()
            sock.connect((server, 22))  # Try SSH port
            sock.close()
            
            socket_latency = (time.time() - start_time) * 1000  # ms
            
            # On Unix-like systems, try the ping command for more accuracy
            if os.name != 'nt':  # Not Windows
                try:
                    # Execute ping command
                    ping_cmd = f"ping -c {count} -W {timeout} {server}"
                    result = subprocess.run(ping_cmd, shell=True, capture_output=True, text=True, timeout=timeout*count+1)
                    
                    if result.returncode == 0:
                        # Parse ping output to extract average time
                        output = result.stdout
                        avg_line = [line for line in output.split('\n') if 'avg' in line]
                        
                        if avg_line:
                            # Extract average from something like "rtt min/avg/max/mdev = 0.123/0.456/0.789/0.100 ms"
                            avg_ms = float(avg_line[0].split('=')[1].strip().split('/')[1])
                            return avg_ms
                    
                    # Fallback to socket latency if ping command fails
                    return socket_latency
                    
                except (subprocess.SubprocessError, IndexError, ValueError):
                    # Fallback to socket latency
                    return socket_latency
            
            return socket_latency
            
        except Exception as e:
            self.logger.debug(f"Ping to {server} failed: {str(e)}")
            return None
    
    def _get_ssh_connection(self, server):
        """
        Get or create an SSH connection to a server.
        
        Args:
            server (str): Server address
            
        Returns:
            paramiko.SSHClient or None: SSH client or None if connection failed
        """
        # Check if we already have a connection
        if server in self.ssh_clients:
            client = self.ssh_clients[server]
            
            # Check if connection is still active
            try:
                stdin, stdout, stderr = client.exec_command("echo test")
                if stdout.channel.recv_exit_status() == 0:
                    return client
            except:
                # Connection is dead, remove it
                del self.ssh_clients[server]
        
        # Create new connection
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Get credentials
            if server in self.credentials_store:
                creds = self.credentials_store[server]
                username = creds.get("username", self.auth_username)
                
                if "key_path" in creds:
                    # Use key authentication
                    key_path = creds["key_path"]
                    client.connect(server, username=username, key_filename=key_path, timeout=10)
                elif "password" in creds:
                    # Use password authentication
                    password = creds["password"]
                    client.connect(server, username=username, password=password, timeout=10)
                else:
                    # No specific credentials, use global config
                    client.connect(server, username=self.auth_username, key_filename=self.auth_key_path, timeout=10)
            else:
                # Use global config
                client.connect(server, username=self.auth_username, key_filename=self.auth_key_path, timeout=10)
            
            return client
            
        except Exception as e:
            self.logger.error(f"SSH connection to {server} failed: {str(e)}")
            return None
    
    def _start_monitoring(self):
        """Start the monitoring thread."""
        if not self.enable_colocation:
            return
            
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(
                target=self._monitor_colocation,
                daemon=True
            )
            self.monitoring_thread.start()
            self.logger.debug("Colocation monitoring thread started")
    
    def _monitor_colocation(self):
        """Monitor colocation servers in a background thread."""
        while not self.stop_event.is_set():
            try:
                # Check all servers
                all_servers = [self.primary_server] + self.backup_servers
                for server in all_servers:
                    if server:
                        status = self._check_server_connection(server)
                        self.server_statuses[server] = status
                
                # Check if we need to failover
                self._check_failover_condition()
                
                # Log status
                self._log_server_status()
                
            except Exception as e:
                self.logger.error(f"Error in colocation monitoring thread: {str(e)}")
            
            # Sleep until next monitoring interval
            self.stop_event.wait(self.monitoring_interval)
    
    def _check_failover_condition(self):
        """
        Check if failover to a backup server is needed.
        
        Returns:
            bool: True if failover was performed
        """
        if not self.enable_colocation or not self.active_server:
            return False
        
        # Check status of active server
        active_status = self.server_statuses.get(self.active_server, {})
        active_latency = active_status.get("latency_ms")
        
        # If active server is unreachable or has high latency
        if (active_status.get("status") != "connected" or 
            (active_latency and active_latency > self.failover_threshold)):
            
            # Look for a better server
            best_server = None
            best_latency = float('inf')
            
            # Check backup servers first
            for server in self.backup_servers:
                if server:
                    status = self.server_statuses.get(server, {})
                    latency = status.get("latency_ms")
                    
                    if status.get("status") == "connected" and latency and latency < best_latency:
                        best_server = server
                        best_latency = latency
            
            # Check primary server if no good backup found or if active is a backup
            if (not best_server or self.active_server != self.primary_server):
                status = self.server_statuses.get(self.primary_server, {})
                latency = status.get("latency_ms")
                
                if status.get("status") == "connected" and latency and latency < best_latency:
                    best_server = self.primary_server
                    best_latency = latency
            
            # Perform failover if better server found
            if best_server and best_server != self.active_server:
                self._perform_failover(best_server)
                return True
                
        return False
    
    def _perform_failover(self, new_server):
        """
        Perform failover to a new server.
        
        Args:
            new_server (str): New server to failover to
        """
        old_server = self.active_server
        self.active_server = new_server
        self.last_failover = datetime.now()
        
        # Log the failover
        self.logger.warning(f"Performing failover from {old_server} to {new_server}")
        
        # Record in failover log
        log_file = os.path.join(self.logs_dir, "failovers.json")
        
        try:
            # Load existing logs if available
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            # Add new failover event
            log_data.append({
                "timestamp": datetime.now().isoformat(),
                "from_server": old_server,
                "to_server": new_server,
                "reason": self.server_statuses.get(old_server, {}).get("status"),
                "old_latency": self.server_statuses.get(old_server, {}).get("latency_ms"),
                "new_latency": self.server_statuses.get(new_server, {}).get("latency_ms")
            })
            
            # Save log file
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging failover: {str(e)}")
        
        # Notify about failover (can be extended to send alerts)
        self.logger.critical(
            f"FAILOVER COMPLETED: Now using {new_server} instead of {old_server}. "
            f"Latency improved from "
            f"{self.server_statuses.get(old_server, {}).get('latency_ms', 'unknown')}ms to "
            f"{self.server_statuses.get(new_server, {}).get('latency_ms', 'unknown')}ms"
        )
    
    def _log_server_status(self):
        """Log server status information."""
        log_file = os.path.join(self.logs_dir, f"server_status_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            # Prepare status for logging
            status_log = {
                "timestamp": datetime.now().isoformat(),
                "active_server": self.active_server,
                "servers": self.server_statuses,
                "last_failover": self.last_failover.isoformat() if self.last_failover else None
            }
            
            # Load existing logs if available
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            # Add new status
            log_data.append(status_log)
            
            # Keep only the last 1000 entries to prevent file growth
            if len(log_data) > 1000:
                log_data = log_data[-1000:]
            
            # Save log file
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging server status: {str(e)}")
    
    def execute_command(self, command, server=None, timeout=30):
        """
        Execute a command on the active colocation server.
        
        Args:
            command (str): Command to execute
            server (str, optional): Specific server to use (defaults to active server)
            timeout (int): Command timeout in seconds
            
        Returns:
            dict: Command execution results
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        try:
            # Get SSH connection
            ssh_client = self._get_ssh_connection(target_server)
            
            if not ssh_client:
                return {"status": "error", "error": f"Could not connect to server {target_server}"}
            
            # Execute command
            stdin, stdout, stderr = ssh_client.exec_command(command, timeout=timeout)
            exit_status = stdout.channel.recv_exit_status()
            
            # Get output
            stdout_text = stdout.read().decode().strip()
            stderr_text = stderr.read().decode().strip()
            
            return {
                "status": "success" if exit_status == 0 else "error",
                "exit_code": exit_status,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "server": target_server
            }
            
        except Exception as e:
            self.logger.error(f"Error executing command on {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def transfer_file(self, local_path, remote_path, direction="upload", server=None):
        """
        Transfer a file to or from the colocation server.
        
        Args:
            local_path (str): Local file path
            remote_path (str): Remote file path
            direction (str): 'upload' or 'download'
            server (str, optional): Specific server to use (defaults to active server)
            
        Returns:
            dict: File transfer results
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        try:
            # Get SSH connection
            ssh_client = self._get_ssh_connection(target_server)
            
            if not ssh_client:
                return {"status": "error", "error": f"Could not connect to server {target_server}"}
            
            # Open SFTP connection
            sftp = ssh_client.open_sftp()
            
            # Transfer file
            start_time = time.time()
            
            if direction == "upload":
                sftp.put(local_path, remote_path)
                action = "Uploaded"
            else:  # download
                sftp.get(remote_path, local_path)
                action = "Downloaded"
            
            transfer_time = time.time() - start_time
            
            # Close SFTP connection
            sftp.close()
            
            # Get file size
            file_size = os.path.getsize(local_path)
            
            return {
                "status": "success",
                "direction": direction,
                "file_size_bytes": file_size,
                "transfer_time_seconds": transfer_time,
                "transfer_rate_kbps": (file_size / 1024) / transfer_time if transfer_time > 0 else 0,
                "local_path": local_path,
                "remote_path": remote_path,
                "server": target_server,
                "message": f"{action} {file_size} bytes in {transfer_time:.2f} seconds"
            }
            
        except Exception as e:
            self.logger.error(f"Error transferring file to/from {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def deploy_trading_code(self, local_dir, remote_dir=None, server=None):
        """
        Deploy trading code to the colocation server.
        
        Args:
            local_dir (str): Local directory containing code
            remote_dir (str, optional): Remote directory (defaults to configured path)
            server (str, optional): Specific server to use (defaults to active server)
            
        Returns:
            dict: Deployment results
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        # Use configured remote directory if not specified
        if not remote_dir:
            remote_dir = self.colocation_params.get("deploy_directory", "/opt/trading")
        
        try:
            # Get SSH connection
            ssh_client = self._get_ssh_connection(target_server)
            
            if not ssh_client:
                return {"status": "error", "error": f"Could not connect to server {target_server}"}
            
            # Create remote directory if it doesn't exist
            mkdir_cmd = f"mkdir -p {remote_dir}"
            stdin, stdout, stderr = ssh_client.exec_command(mkdir_cmd)
            if stdout.channel.recv_exit_status() != 0:
                return {"status": "error", "error": f"Failed to create remote directory: {stderr.read().decode()}"}
            
            # Open SFTP connection
            sftp = ssh_client.open_sftp()
            
            # Track files and sizes
            transferred_files = []
            total_bytes = 0
            start_time = time.time()
            
            # Walk through local directory and upload files
            for root, dirs, files in os.walk(local_dir):
                # Create relative path from local_dir
                rel_path = os.path.relpath(root, local_dir)
                remote_root = os.path.join(remote_dir, rel_path) if rel_path != '.' else remote_dir
                
                # Create remote directories
                for dir_name in dirs:
                    remote_path = os.path.join(remote_root, dir_name)
                    try:
                        sftp.mkdir(remote_path)
                    except:
                        # Directory might already exist
                        pass
                
                # Upload files
                for file_name in files:
                    # Skip certain files
                    if file_name.startswith('.') or file_name.endswith('.pyc'):
                        continue
                        
                    local_file = os.path.join(root, file_name)
                    remote_file = os.path.join(remote_root, file_name)
                    
                    try:
                        sftp.put(local_file, remote_file)
                        file_size = os.path.getsize(local_file)
                        total_bytes += file_size
                        transferred_files.append({
                            "local_path": local_file,
                            "remote_path": remote_file,
                            "size_bytes": file_size
                        })
                    except Exception as e:
                        self.logger.error(f"Error transferring {local_file}: {str(e)}")
            
            # Close SFTP connection
            sftp.close()
            
            # Calculate total time
            transfer_time = time.time() - start_time
            
            # Return deployment info
            return {
                "status": "success",
                "server": target_server,
                "files_transferred": len(transferred_files),
                "total_bytes": total_bytes,
                "transfer_time_seconds": transfer_time,
                "transfer_rate_kbps": (total_bytes / 1024) / transfer_time if transfer_time > 0 else 0,
                "file_list": transferred_files,
                "remote_dir": remote_dir
            }
            
        except Exception as e:
            self.logger.error(f"Error deploying code to {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def start_trading_service(self, script_name=None, args=None, server=None):
        """
        Start a trading service on the colocation server.
        
        Args:
            script_name (str, optional): Script to run (defaults to configured script)
            args (str, optional): Command line arguments
            server (str, optional): Specific server to use (defaults to active server)
            
        Returns:
            dict: Service start results
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        # Use configured script if not specified
        if not script_name:
            script_name = self.colocation_params.get("trading_script", "main.py")
        
        # Build command
        remote_dir = self.colocation_params.get("deploy_directory", "/opt/trading")
        cmd_args = args or self.colocation_params.get("script_args", "")
        
        # Use nohup to keep process running after SSH session ends
        command = f"cd {remote_dir} && nohup python {script_name} {cmd_args} > trading.log 2>&1 &"
        
        try:
            # Execute start command
            result = self.execute_command(command, server=target_server)
            
            if result["status"] == "success":
                # Check if process is running
                check_command = f"ps aux | grep '{script_name}' | grep -v grep"
                check_result = self.execute_command(check_command, server=target_server)
                
                if check_result["status"] == "success" and check_result["stdout"]:
                    # Extract PID from ps output (first number)
                    try:
                        pid = check_result["stdout"].split()[1]
                        result["pid"] = pid
                        result["message"] = f"Trading service started with PID {pid}"
                    except:
                        result["message"] = "Trading service started, but couldn't determine PID"
                else:
                    result["status"] = "error"
                    result["error"] = "Failed to start trading service"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error starting trading service on {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def stop_trading_service(self, script_name=None, server=None):
        """
        Stop a trading service on the colocation server.
        
        Args:
            script_name (str, optional): Script to stop (defaults to configured script)
            server (str, optional): Specific server to use (defaults to active server)
            
        Returns:
            dict: Service stop results
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        # Use configured script if not specified
        if not script_name:
            script_name = self.colocation_params.get("trading_script", "main.py")
        
        # Find PIDs
        find_command = f"ps aux | grep '{script_name}' | grep -v grep | awk '{{print $2}}'"
        
        try:
            # Find process IDs
            find_result = self.execute_command(find_command, server=target_server)
            
            if find_result["status"] != "success" or not find_result["stdout"]:
                return {"status": "warning", "message": "No trading service found running", "server": target_server}
            
            # Get list of PIDs
            pids = find_result["stdout"].strip().split('\n')
            
            # Kill each process
            killed_pids = []
            for pid in pids:
                kill_command = f"kill -15 {pid}"  # SIGTERM for graceful shutdown
                kill_result = self.execute_command(kill_command, server=target_server)
                
                if kill_result["status"] == "success":
                    killed_pids.append(pid)
                
            # Check if processes are gone
            time.sleep(2)  # Give processes time to terminate
            check_result = self.execute_command(find_command, server=target_server)
            
            if check_result["status"] == "success" and not check_result["stdout"]:
                return {
                    "status": "success",
                    "message": f"Stopped {len(killed_pids)} trading services",
                    "killed_pids": killed_pids,
                    "server": target_server
                }
            else:
                # Force kill if graceful shutdown didn't work
                remaining_pids = check_result["stdout"].strip().split('\n') if check_result["stdout"] else []
                force_killed = []
                
                for pid in remaining_pids:
                    force_command = f"kill -9 {pid}"  # SIGKILL for forced termination
                    force_result = self.execute_command(force_command, server=target_server)
                    
                    if force_result["status"] == "success":
                        force_killed.append(pid)
                
                return {
                    "status": "warning",
                    "message": f"Force-stopped {len(force_killed)} remaining services",
                    "graceful_stop": killed_pids,
                    "force_stop": force_killed,
                    "server": target_server
                }
            
        except Exception as e:
            self.logger.error(f"Error stopping trading service on {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def check_trading_status(self, script_name=None, server=None):
        """
        Check the status of trading services.
        
        Args:
            script_name (str, optional): Script to check (defaults to configured script)
            server (str, optional): Specific server to use (defaults to active server)
            
        Returns:
            dict: Service status information
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        # Use configured script if not specified
        if not script_name:
            script_name = self.colocation_params.get("trading_script", "main.py")
        
        # Command to check process
        process_command = f"ps aux | grep '{script_name}' | grep -v grep"
        
        # Command to get recent log entries
        remote_dir = self.colocation_params.get("deploy_directory", "/opt/trading")
        log_command = f"tail -n 20 {remote_dir}/trading.log 2>/dev/null || echo 'Log file not found'"
        
        try:
            # Check process status
            process_result = self.execute_command(process_command, server=target_server)
            
            # Get log entries
            log_result = self.execute_command(log_command, server=target_server)
            
            # Determine running status
            is_running = process_result["status"] == "success" and process_result["stdout"].strip() != ""
            
            # Extract resource usage if running
            resource_usage = None
            if is_running:
                try:
                    # Parse process info
                    process_lines = process_result["stdout"].strip().split('\n')
                    process_info = []
                    
                    for line in process_lines:
                        parts = line.split()
                        if len(parts) >= 10:
                            process_info.append({
                                "user": parts[0],
                                "pid": parts[1],
                                "cpu_percent": parts[2],
                                "memory_percent": parts[3],
                                "start_time": parts[8],
                                "command": " ".join(parts[10:])
                            })
                    
                    resource_usage = process_info
                except:
                    resource_usage = "Could not parse process information"
            
            # Combine results
            return {
                "status": "success",
                "is_running": is_running,
                "process_count": len(process_result["stdout"].strip().split('\n')) if is_running else 0,
                "resource_usage": resource_usage,
                "server": target_server,
                "log_entries": log_result["stdout"].strip().split('\n') if log_result["status"] == "success" else [],
                "raw_process_info": process_result["stdout"] if is_running else None
            }
            
        except Exception as e:
            self.logger.error(f"Error checking trading status on {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def get_server_status(self, server=None):
        """
        Get detailed status of a colocation server.
        
        Args:
            server (str, optional): Specific server to check (defaults to active server)
            
        Returns:
            dict: Server status information
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        # Commands to gather system information
        commands = {
            "uptime": "uptime",
            "memory": "free -m",
            "disk": "df -h",
            "load": "cat /proc/loadavg",
            "network": "netstat -tuln",
            "processes": "ps aux --sort=-%cpu | head -10",
            "kernel": "uname -a",
            "last_login": "last | head -5"
        }
        
        results = {}
        
        try:
            # Execute each command
            for name, cmd in commands.items():
                result = self.execute_command(cmd, server=target_server)
                if result["status"] == "success":
                    results[name] = result["stdout"]
                else:
                    results[name] = f"Error: {result.get('error', 'Unknown error')}"
            
            # Get connection latency
            latency = self._ping_server(target_server)
            
            # Compile status report
            status_report = {
                "status": "success",
                "server": target_server,
                "is_active": target_server == self.active_server,
                "connection": {
                    "latency_ms": latency,
                    "status": "connected" if latency else "error",
                    "ssh_connected": target_server in self.ssh_clients
                },
                "system_info": results,
                "timestamp": datetime.now().isoformat()
            }
            
            return status_report
            
        except Exception as e:
            self.logger.error(f"Error getting server status for {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def get_all_servers_status(self):
        """
        Get status of all configured colocation servers.
        
        Returns:
            dict: Status information for all servers
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        all_servers = []
        if self.primary_server:
            all_servers.append(self.primary_server)
        all_servers.extend([s for s in self.backup_servers if s])
        
        results = {}
        
        for server in all_servers:
            # Get basic status for each server
            status = self.server_statuses.get(server, {})
            if not status:
                status = self._check_server_connection(server)
                self.server_statuses[server] = status
            
            results[server] = {
                "is_active": server == self.active_server,
                "connection_status": status.get("status", "unknown"),
                "latency_ms": status.get("latency_ms"),
                "last_check": status.get("timestamp"),
                "error": status.get("error")
            }
        
        return {
            "status": "success",
            "active_server": self.active_server,
            "servers": results,
            "last_failover": self.last_failover.isoformat() if self.last_failover else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def configure_remote_service(self, config_file, remote_path=None, server=None):
        """
        Configure a service on the remote server.
        
        Args:
            config_file (str): Local configuration file
            remote_path (str, optional): Remote path (defaults to configured path)
            server (str, optional): Specific server to use (defaults to active server)
            
        Returns:
            dict: Configuration results
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        # Use configured remote directory if remote path not specified
        if not remote_path:
            remote_dir = self.colocation_params.get("deploy_directory", "/opt/trading")
            remote_path = f"{remote_dir}/config.json"
        
        try:
            # Transfer configuration file
            transfer_result = self.transfer_file(config_file, remote_path, direction="upload", server=target_server)
            
            if transfer_result["status"] != "success":
                return transfer_result
            
            # Verify file was transferred correctly
            verify_command = f"cat {remote_path}"
            verify_result = self.execute_command(verify_command, server=target_server)
            
            if verify_result["status"] != "success" or not verify_result["stdout"]:
                return {
                    "status": "error",
                    "error": "Configuration file transfer failed verification",
                    "server": target_server
                }
            
            return {
                "status": "success",
                "message": f"Configuration file deployed to {remote_path}",
                "file_size": transfer_result["file_size_bytes"],
                "server": target_server
            }
            
        except Exception as e:
            self.logger.error(f"Error configuring remote service on {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def setup_new_server(self, server, username, password=None, key_path=None, install_dependencies=True):
        """
        Set up a new colocation server.
        
        Args:
            server (str): Server address
            username (str): SSH username
            password (str, optional): SSH password
            key_path (str, optional): Path to SSH key file
            install_dependencies (bool): Whether to install dependencies
            
        Returns:
            dict: Setup results
        """
        # Store credentials
        self.store_server_credential(server, username, password, key_path)
        
        # Test connection
        test_connection = self._check_server_connection(server)
        
        if test_connection["status"] != "connected":
            return {
                "status": "error",
                "error": f"Could not connect to server: {test_connection.get('error', 'Unknown error')}",
                "server": server
            }
        
        results = {
            "status": "success",
            "server": server,
            "connection_test": test_connection,
            "setup_steps": []
        }
        
        try:
            # Create trading directory
            remote_dir = self.colocation_params.get("deploy_directory", "/opt/trading")
            mkdir_cmd = f"mkdir -p {remote_dir}"
            mkdir_result = self.execute_command(mkdir_cmd, server=server)
            results["setup_steps"].append({"action": "create_directory", "result": mkdir_result})
            
            # Install dependencies if requested
            if install_dependencies:
                # Install Python and pip if needed
                python_cmd = "which python3 || (apt-get update && apt-get install -y python3 python3-pip)"
                python_result = self.execute_command(python_cmd, server=server)
                results["setup_steps"].append({"action": "install_python", "result": python_result})
                
                # Install required Python packages
                packages = self.colocation_params.get("required_packages", [
                    "numpy", "pandas", "requests", "websockets", "pytz"
                ])
                
                if packages:
                    packages_str = " ".join(packages)
                    pip_cmd = f"pip3 install {packages_str}"
                    pip_result = self.execute_command(pip_cmd, server=server)
                    results["setup_steps"].append({"action": "install_packages", "result": pip_result})
            
            # Add server to available servers list
            if server not in [self.primary_server] + self.backup_servers:
                if not self.primary_server:
                    self.primary_server = server
                    results["added_as"] = "primary"
                else:
                    self.backup_servers.append(server)
                    results["added_as"] = "backup"
            
            # All steps completed
            results["message"] = f"Server {server} setup completed successfully"
            return results
            
        except Exception as e:
            self.logger.error(f"Error setting up server {server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": server, "steps_completed": results["setup_steps"]}
    
    def get_latency_metrics(self, server=None, samples=10):
        """
        Get detailed latency metrics for a server.
        
        Args:
            server (str, optional): Specific server to check (defaults to active server)
            samples (int): Number of ping samples to collect
            
        Returns:
            dict: Latency metrics
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        latencies = []
        errors = 0
        
        # Collect samples
        for _ in range(samples):
            latency = self._ping_server(target_server)
            if latency is not None:
                latencies.append(latency)
            else:
                errors += 1
            time.sleep(0.5)  # Add delay between pings
        
        if not latencies:
            return {
                "status": "error",
                "error": "All ping attempts failed",
                "server": target_server
            }
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        jitter = np.std(latencies) if len(latencies) > 1 else 0
        
        return {
            "status": "success",
            "server": target_server,
            "average_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "jitter_ms": jitter,
            "samples_collected": len(latencies),
            "samples_failed": errors,
            "packet_loss": errors / samples if samples > 0 else 0,
            "raw_latencies": latencies,
            "timestamp": datetime.now().isoformat()
        }
    
    def synchronize_time(self, server=None):
        """
        Synchronize time on the colocation server.
        
        Args:
            server (str, optional): Specific server to use (defaults to active server)
            
        Returns:
            dict: Synchronization results
        """
        if not self.enable_colocation:
            return {"status": "error", "error": "Colocation services disabled"}
        
        target_server = server or self.active_server
        
        if not target_server:
            return {"status": "error", "error": "No active server available"}
        
        # Commands to check and set time
        check_cmd = "date '+%Y-%m-%d %H:%M:%S'"
        ntp_cmd = "sudo service ntp restart || sudo service ntpd restart || sudo systemctl restart systemd-timesyncd || echo 'No NTP service found'"
        sync_cmd = "sudo ntpdate pool.ntp.org || sudo chronyd -Q 'server pool.ntp.org iburst' || echo 'Cannot synchronize time'"
        
        try:
            # Check current time
            before_result = self.execute_command(check_cmd, server=target_server)
            before_time = before_result["stdout"].strip() if before_result["status"] == "success" else "unknown"
            
            # Try to restart NTP service
            ntp_result = self.execute_command(ntp_cmd, server=target_server)
            
            # Force sync
            sync_result = self.execute_command(sync_cmd, server=target_server)
            
            # Check time after sync
            time.sleep(2)  # Wait for sync to complete
            after_result = self.execute_command(check_cmd, server=target_server)
            after_time = after_result["stdout"].strip() if after_result["status"] == "success" else "unknown"
            
            # Check time on local machine
            local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "status": "success",
                "server": target_server,
                "before_sync": before_time,
                "after_sync": after_time,
                "local_time": local_time,
                "sync_output": sync_result.get("stdout", ""),
                "ntp_output": ntp_result.get("stdout", ""),
                "message": f"Time synchronized on {target_server}"
            }
            
        except Exception as e:
            self.logger.error(f"Error synchronizing time on {target_server}: {str(e)}")
            return {"status": "error", "error": str(e), "server": target_server}
    
    def shutdown(self):
        """Clean shutdown of the colocation manager."""
        self.logger.info("Shutting down ColocationManager")
        
        # Signal monitoring thread to stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        
        # Close SSH connections
        for server, client in list(self.ssh_clients.items()):
            try:
                client.close()
                self.logger.debug(f"Closed SSH connection to {server}")
            except:
                pass
        
        self.ssh_clients = {}
