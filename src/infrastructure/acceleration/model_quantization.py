"""
Model Quantization Module

This module provides model quantization capabilities for the Option Hunter system.
It converts full-precision ML models to lower precision formats for faster inference
and reduced memory usage without significant accuracy loss.
"""

import logging
import os
import json
import time
import numpy as np
from datetime import datetime
import threading

# Try importing ML-related libraries with graceful fallback
try:
    import torch
    from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

class ModelQuantizer:
    """
    ML model quantizer for optimizing inference performance.
    
    Features:
    - Post-training quantization for TensorFlow and PyTorch models
    - Dynamic and static quantization options
    - Int8 and float16 precision support
    - Quantization-aware training helpers
    - Accuracy validation after quantization
    - Memory usage and performance benchmarking
    """
    
    def __init__(self, config=None):
        """
        Initialize the ModelQuantizer.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration
        self.quant_params = self.config.get("model_quantization", {})
        
        # Default parameters
        self.enable_quantization = self.quant_params.get("enable_quantization", True)
        self.default_precision = self.quant_params.get("default_precision", "int8")  # 'int8', 'float16'
        self.calibration_size = self.quant_params.get("calibration_size", 100)
        self.accuracy_threshold = self.quant_params.get("accuracy_threshold", 0.99)  # 99% of original accuracy
        self.benchmark_iterations = self.quant_params.get("benchmark_iterations", 20)
        
        # Quantized model storage
        self.quantized_models = {}
        self.model_lock = threading.RLock()
        
        # Performance tracking
        self.benchmark_results = {}
        
        # Create logs directory
        self.logs_dir = "logs/quantization"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Check available frameworks
        self.available_frameworks = []
        if HAS_TORCH:
            self.available_frameworks.append("pytorch")
        if HAS_TF:
            self.available_frameworks.append("tensorflow")
        if HAS_ONNX:
            self.available_frameworks.append("onnx")
        
        self.logger.info(f"ModelQuantizer initialized with frameworks: {', '.join(self.available_frameworks)}")
    
    def quantize_pytorch_model(self, model, model_id, calibration_data=None, precision=None):
        """
        Quantize a PyTorch model.
        
        Args:
            model (torch.nn.Module): PyTorch model to quantize
            model_id (str): Unique identifier for the model
            calibration_data (tuple, optional): Input data for calibration (x_cal, y_cal)
            precision (str, optional): Target precision ('int8' or 'float16')
            
        Returns:
            torch.nn.Module: Quantized model or None if quantization failed
        """
        if not HAS_TORCH:
            self.logger.error("PyTorch not available for quantization")
            return None
        
        if not self.enable_quantization:
            return model
        
        if precision is None:
            precision = self.default_precision
        
        self.logger.info(f"Quantizing PyTorch model {model_id} to {precision} precision")
        
        try:
            # Make a copy of the model to avoid modifying the original
            original_model = model
            model = type(original_model)()
            model.load_state_dict(original_model.state_dict())
            
            # Move to CPU for quantization
            model = model.cpu()
            model.eval()
            
            # Quantize based on precision
            if precision == "int8":
                # Dynamic quantization (no calibration data needed)
                qconfig = torch.quantization.default_dynamic_qconfig
                
                # Prepare the model for quantization
                model_prepared = torch.quantization.prepare(model, inplace=False)
                
                # If calibration data provided, use it
                if calibration_data is not None:
                    x_cal, _ = calibration_data
                    if isinstance(x_cal, np.ndarray):
                        x_cal = torch.tensor(x_cal)
                    
                    # Run calibration data through the model
                    with torch.no_grad():
                        for i in range(min(len(x_cal), self.calibration_size)):
                            model_prepared(x_cal[i:i+1])
                
                # Convert to quantized model
                quantized_model = torch.quantization.convert(model_prepared, inplace=False)
                
            elif precision == "float16":
                # Use PyTorch's AMP (automatic mixed precision)
                # This doesn't actually quantize the model, but enables float16 during inference
                quantized_model = model  # Original model, will use with torch.cuda.amp
                
                # Store to indicate float16 inference should be used
                with self.model_lock:
                    self.quantized_models[model_id] = {
                        "model": quantized_model,
                        "precision": "float16",
                        "framework": "pytorch",
                        "original_size": self._get_model_size(original_model),
                        "quantized_size": self._get_model_size(quantized_model),
                        "amp_enabled": True  # Flag to enable AMP during inference
                    }
                
                return quantized_model
            else:
                self.logger.error(f"Unsupported precision: {precision}")
                return model
            
            # Benchmark to compare performance
            benchmark_results = self._benchmark_pytorch_models(
                original_model, quantized_model, calibration_data
            )
            
            # Store quantized model
            with self.model_lock:
                self.quantized_models[model_id] = {
                    "model": quantized_model,
                    "precision": precision,
                    "framework": "pytorch",
                    "original_size": self._get_model_size(original_model),
                    "quantized_size": self._get_model_size(quantized_model),
                    "benchmark": benchmark_results
                }
            
            self.logger.info(f"Successfully quantized PyTorch model {model_id}")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Error quantizing PyTorch model {model_id}: {str(e)}")
            return model
    
    def quantize_tensorflow_model(self, model, model_id, calibration_data=None, precision=None):
        """
        Quantize a TensorFlow model.
        
        Args:
            model (tf.keras.Model): TensorFlow model to quantize
            model_id (str): Unique identifier for the model
            calibration_data (tuple, optional): Input data for calibration (x_cal, y_cal)
            precision (str, optional): Target precision ('int8' or 'float16')
            
        Returns:
            tf.keras.Model: Quantized model or None if quantization failed
        """
        if not HAS_TF:
            self.logger.error("TensorFlow not available for quantization")
            return None
        
        if not self.enable_quantization:
            return model
        
        if precision is None:
            precision = self.default_precision
        
        self.logger.info(f"Quantizing TensorFlow model {model_id} to {precision} precision")
        
        try:
            original_model = model
            
            if precision == "int8":
                # TensorFlow Lite quantization
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Use representative dataset if provided
                if calibration_data is not None:
                    x_cal, _ = calibration_data
                    
                    def representative_dataset():
                        for i in range(min(len(x_cal), self.calibration_size)):
                            if isinstance(x_cal, np.ndarray):
                                yield [x_cal[i:i+1].astype(np.float32)]
                            else:
                                # Handle TensorFlow tensors
                                yield [tf.cast(x_cal[i:i+1], tf.float32)]
                    
                    converter.representative_dataset = representative_dataset
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                
                # Convert model to TFLite format
                tflite_model = converter.convert()
                
                # Create an interpreter for the quantized model
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                interpreter.allocate_tensors()
                
                # Wrap the interpreter in a callable class for compatibility
                class QuantizedModel:
                    def __init__(self, interpreter):
                        self.interpreter = interpreter
                        self.input_details = interpreter.get_input_details()
                        self.output_details = interpreter.get_output_details()
                    
                    def __call__(self, inputs):
                        if isinstance(inputs, tf.Tensor):
                            inputs = inputs.numpy()
                        
                        # Resize input tensor if needed
                        for i, input_detail in enumerate(self.input_details):
                            required_shape = list(input_detail['shape'])
                            if len(required_shape) == 4:  # NHWC format
                                required_shape[0] = len(inputs)
                                self.interpreter.resize_tensor_input(input_detail['index'], required_shape)
                        
                        self.interpreter.allocate_tensors()
                        
                        # Set input tensor
                        self.interpreter.set_tensor(self.input_details[0]['index'], inputs)
                        
                        # Run inference
                        self.interpreter.invoke()
                        
                        # Get output tensor
                        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                        return output_data
                
                quantized_model = QuantizedModel(interpreter)
                
                # Store the TFLite model content for later use
                tflite_model_content = tflite_model
                
            elif precision == "float16":
                # Mixed precision using TF's mixed precision API
                from tensorflow.keras import mixed_precision
                
                # Save original policy
                original_policy = mixed_precision.global_policy()
                
                # Set global policy to mixed precision
                mixed_precision.set_global_policy('mixed_float16')
                
                # Clone the model in mixed precision
                config = original_model.get_config()
                quantized_model = type(original_model).from_config(config)
                quantized_model.set_weights(original_model.get_weights())
                
                # Restore original policy
                mixed_precision.set_global_policy(original_policy)
                
                # Store policy info for later use
                tflite_model_content = None
                
            else:
                self.logger.error(f"Unsupported precision: {precision}")
                return model
            
            # Benchmark to compare performance
            benchmark_results = self._benchmark_tensorflow_models(
                original_model, quantized_model, calibration_data
            )
            
            # Store quantized model
            with self.model_lock:
                self.quantized_models[model_id] = {
                    "model": quantized_model,
                    "precision": precision,
                    "framework": "tensorflow",
                    "original_size": self._get_model_size(original_model),
                    "quantized_size": self._get_model_size(quantized_model),
                    "tflite_model": tflite_model_content,
                    "benchmark": benchmark_results
                }
            
            self.logger.info(f"Successfully quantized TensorFlow model {model_id}")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Error quantizing TensorFlow model {model_id}: {str(e)}")
            return model
    
    def quantize_to_onnx(self, model, model_id, input_shape, framework, calibration_data=None, precision=None):
        """
        Quantize a model by converting to ONNX format.
        
        Args:
            model: Original model (PyTorch or TensorFlow)
            model_id (str): Unique identifier for the model
            input_shape (tuple): Shape of input data
            framework (str): Original framework ('pytorch' or 'tensorflow')
            calibration_data (tuple, optional): Input data for calibration (x_cal, y_cal)
            precision (str, optional): Target precision ('int8' or 'float16')
            
        Returns:
            object: ONNX Runtime session or None if conversion failed
        """
        if not HAS_ONNX:
            self.logger.error("ONNX Runtime not available for quantization")
            return None
        
        if not self.enable_quantization:
            return model
        
        if precision is None:
            precision = self.default_precision
        
        # Check if framework is supported
        if framework not in ['pytorch', 'tensorflow']:
            self.logger.error(f"Unsupported framework for ONNX conversion: {framework}")
            return None
        
        if framework == 'pytorch' and not HAS_TORCH:
            self.logger.error("PyTorch not available for ONNX conversion")
            return None
            
        if framework == 'tensorflow' and not HAS_TF:
            self.logger.error("TensorFlow not available for ONNX conversion")
            return None
        
        self.logger.info(f"Converting and quantizing {framework} model {model_id} to ONNX")
        
        try:
            # Create directory for ONNX models
            onnx_dir = os.path.join(self.logs_dir, "onnx_models")
            os.makedirs(onnx_dir, exist_ok=True)
            
            # Base filenames
            base_onnx_filename = os.path.join(onnx_dir, f"{model_id}")
            orig_onnx_path = f"{base_onnx_filename}.onnx"
            quant_onnx_path = f"{base_onnx_filename}_quantized.onnx"
            
            # Export to ONNX format
            if framework == 'pytorch':
                # PyTorch to ONNX
                dummy_input = torch.randn(input_shape)
                torch.onnx.export(model, dummy_input, orig_onnx_path,
                                 export_params=True, opset_version=12)
            else:
                # TensorFlow to ONNX (requires tf2onnx)
                try:
                    import tf2onnx
                    
                    # Create dummy input
                    dummy_input = tf.random.normal(input_shape)
                    
                    # Convert to ONNX
                    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(input_shape, tf.float32)])
                    
                    # Save the ONNX model
                    with open(orig_onnx_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                except ImportError:
                    self.logger.error("tf2onnx package not found. Install it with: pip install tf2onnx")
                    return None
            
            # Load ONNX model
            onnx_model = onnx.load(orig_onnx_path)
            
            # Perform quantization based on precision
            if precision == 'int8':
                # Use ONNX Runtime quantization
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                # Dynamic quantization to int8
                quantize_dynamic(orig_onnx_path, quant_onnx_path, weight_type=QuantType.QInt8)
                
                # Create ONNX Runtime session with the quantized model
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                quantized_model = ort.InferenceSession(quant_onnx_path, sess_options)
                
            elif precision == 'float16':
                # Convert to float16
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                # Float16 quantization
                quantize_dynamic(orig_onnx_path, quant_onnx_path, weight_type=QuantType.QFloat16)
                
                # Create ONNX Runtime session with the float16 model
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                quantized_model = ort.InferenceSession(quant_onnx_path, sess_options)
                
            else:
                self.logger.error(f"Unsupported precision for ONNX quantization: {precision}")
                return None
            
            # Create a wrapper for the ONNX Runtime session
            class ONNXModel:
                def __init__(self, session):
                    self.session = session
                    self.input_name = session.get_inputs()[0].name
                    self.output_name = session.get_outputs()[0].name
                
                def __call__(self, inputs):
                    # Convert inputs to numpy if necessary
                    if framework == 'pytorch' and isinstance(inputs, torch.Tensor):
                        inputs = inputs.cpu().numpy()
                    elif framework == 'tensorflow' and isinstance(inputs, tf.Tensor):
                        inputs = inputs.numpy()
                    
                    # Run inference
                    outputs = self.session.run([self.output_name], {self.input_name: inputs})
                    return outputs[0]
            
            wrapped_model = ONNXModel(quantized_model)
            
            # Benchmark to compare performance
            if framework == 'pytorch':
                benchmark_results = self._benchmark_pytorch_models(
                    model, wrapped_model, calibration_data
                )
            else:
                benchmark_results = self._benchmark_tensorflow_models(
                    model, wrapped_model, calibration_data
                )
            
            # Get file sizes for comparison
            original_size = os.path.getsize(orig_onnx_path)
            quantized_size = os.path.getsize(quant_onnx_path)
            
            # Store quantized model
            with self.model_lock:
                self.quantized_models[model_id] = {
                    "model": wrapped_model,
                    "precision": precision,
                    "framework": "onnx",
                    "original_framework": framework,
                    "original_size": original_size,
                    "quantized_size": quantized_size,
                    "original_path": orig_onnx_path,
                    "quantized_path": quant_onnx_path,
                    "benchmark": benchmark_results
                }
            
            self.logger.info(f"Successfully converted and quantized model {model_id} to ONNX")
            return wrapped_model
            
        except Exception as e:
            self.logger.error(f"Error converting/quantizing to ONNX: {str(e)}")
            return model
    
    def _benchmark_pytorch_models(self, original_model, quantized_model, data=None):
        """
        Benchmark PyTorch models to compare performance.
        
        Args:
            original_model (torch.nn.Module): Original model
            quantized_model (torch.nn.Module): Quantized model
            data (tuple, optional): Benchmark data (x, y)
            
        Returns:
            dict: Benchmark results
        """
        if not HAS_TORCH:
            return {}
        
        # Prepare benchmark data
        if data is None or len(data) < 2 or data[0] is None:
            # Generate synthetic data
            dummy_input = torch.randn(16, *original_model.input_shape[1:])
            x, y = dummy_input, None
        else:
            x, y = data
            if isinstance(x, np.ndarray):
                x = torch.tensor(x)
        
        # Prepare models for evaluation
        original_model.eval()
        if hasattr(quantized_model, 'eval'):
            quantized_model.eval()
        
        original_device = next(original_model.parameters()).device
        
        try:
            # Move inputs to the same device as the model
            x = x.to(original_device)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(5):
                    _ = original_model(x[:1])
                    if hasattr(quantized_model, '__call__'):
                        _ = quantized_model(x[:1])
            
            # Benchmark original model
            original_times = []
            original_memory = []
            
            with torch.no_grad():
                for _ in range(self.benchmark_iterations):
                    # Measure memory before
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    # Time the inference
                    start_time = time.time()
                    _ = original_model(x)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    original_times.append(time.time() - start_time)
                    
                    # Measure memory after
                    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    original_memory.append(memory_after - memory_before)
            
            # Benchmark quantized model
            quantized_times = []
            quantized_memory = []
            
            with torch.no_grad():
                for _ in range(self.benchmark_iterations):
                    # Measure memory before
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    # Time the inference
                    start_time = time.time()
                    _ = quantized_model(x)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    quantized_times.append(time.time() - start_time)
                    
                    # Measure memory after
                    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    quantized_memory.append(memory_after - memory_before)
            
            # Compare accuracy if we have labels
            accuracy_comparison = None
            if y is not None:
                y = y.to(original_device)
                
                with torch.no_grad():
                    original_out = original_model(x)
                    quantized_out = quantized_model(x)
                
                # Check if outputs are in the same format
                if hasattr(original_out, 'shape') and hasattr(quantized_out, 'shape') and original_out.shape == quantized_out.shape:
                    # Calculate mean squared error
                    mse = torch.mean((original_out - quantized_out) ** 2).item()
                    
                    # Calculate original accuracy and quantized accuracy (for classification)
                    if original_out.ndim > 1 and original_out.shape[1] > 1:
                        # Classification task
                        original_preds = torch.argmax(original_out, dim=1)
                        quantized_preds = torch.argmax(quantized_out, dim=1)
                        
                        # Calculate accuracy
                        original_acc = (original_preds == y).float().mean().item()
                        quantized_acc = (quantized_preds == y).float().mean().item()
                        
                        accuracy_comparison = {
                            'original_accuracy': original_acc,
                            'quantized_accuracy': quantized_acc,
                            'accuracy_ratio': quantized_acc / max(original_acc, 1e-8),
                            'mse': mse
                        }
                    else:
                        # Regression task
                        accuracy_comparison = {
                            'mse': mse
                        }
            
            # Calculate speedup and memory reduction
            avg_original_time = np.mean(original_times)
            avg_quantized_time = np.mean(quantized_times)
            speedup = avg_original_time / max(avg_quantized_time, 1e-8)
            
            avg_original_memory = np.mean(original_memory)
            avg_quantized_memory = np.mean(quantized_memory)
            memory_reduction = avg_original_memory / max(avg_quantized_memory, 1)
            
            # Compile results
            benchmark_results = {
                'original_inference_time': avg_original_time,
                'quantized_inference_time': avg_quantized_time,
                'speedup': speedup,
                'original_memory': avg_original_memory,
                'quantized_memory': avg_quantized_memory,
                'memory_reduction': memory_reduction,
                'accuracy_comparison': accuracy_comparison
            }
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Error during PyTorch benchmark: {str(e)}")
            return {}
    
    def _benchmark_tensorflow_models(self, original_model, quantized_model, data=None):
        """
        Benchmark TensorFlow models to compare performance.
        
        Args:
            original_model (tf.keras.Model): Original model
            quantized_model: Quantized model
            data (tuple, optional): Benchmark data (x, y)
            
        Returns:
            dict: Benchmark results
        """
        if not HAS_TF:
            return {}
        
        # Prepare benchmark data
        if data is None or len(data) < 2 or data[0] is None:
            # Generate synthetic data
            if hasattr(original_model, 'input_shape'):
                input_shape = original_model.input_shape
                if isinstance(input_shape, tuple) and input_shape[0] is None:
                    # Add batch dimension
                    input_shape = (16,) + input_shape[1:]
                dummy_input = tf.random.normal(input_shape)
            else:
                # Guess a reasonable input shape
                dummy_input = tf.random.normal((16, 32, 32, 3))
            x, y = dummy_input, None
        else:
            x, y = data
            if isinstance(x, np.ndarray):
                x = tf.convert_to_tensor(x)
        
        try:
            # Warm-up
            for _ in range(5):
                _ = original_model(x[:1])
                if hasattr(quantized_model, '__call__'):
                    _ = quantized_model(x[:1])
            
            # Benchmark original model
            original_times = []
            
            for _ in range(self.benchmark_iterations):
                start_time = time.time()
                _ = original_model(x)
                original_times.append(time.time() - start_time)
            
            # Benchmark quantized model
            quantized_times = []
            
            for _ in range(self.benchmark_iterations):
                start_time = time.time()
                _ = quantized_model(x)
                quantized_times.append(time.time() - start_time)
            
            # Compare accuracy if we have labels
            accuracy_comparison = None
            if y is not None:
                original_out = original_model(x)
                quantized_out = quantized_model(x)
                
                # Check if outputs are in the same format
                if hasattr(original_out, 'shape') and hasattr(quantized_out, 'shape'):
                    # Convert to numpy for consistency
                    if hasattr(original_out, 'numpy'):
                        original_out = original_out.numpy()
                    if hasattr(quantized_out, 'numpy'):
                        quantized_out = quantized_out.numpy()
                    if hasattr(y, 'numpy'):
                        y = y.numpy()
                    
                    # Calculate mean squared error
                    mse = np.mean((original_out - quantized_out) ** 2)
                    
                    # Calculate original accuracy and quantized accuracy (for classification)
                    if len(original_out.shape) > 1 and original_out.shape[1] > 1:
                        # Classification task
                        original_preds = np.argmax(original_out, axis=1)
                        quantized_preds = np.argmax(quantized_out, axis=1)
                        
                        # Calculate accuracy
                        original_acc = np.mean(original_preds == y)
                        quantized_acc = np.mean(quantized_preds == y)
                        
                        accuracy_comparison = {
                            'original_accuracy': original_acc,
                            'quantized_accuracy': quantized_acc,
                            'accuracy_ratio': quantized_acc / max(original_acc, 1e-8),
                            'mse': mse
                        }
                    else:
                        # Regression task
                        accuracy_comparison = {
                            'mse': mse
                        }
            
            # Calculate speedup (TensorFlow memory usage is harder to track)
            avg_original_time = np.mean(original_times)
            avg_quantized_time = np.mean(quantized_times)
            speedup = avg_original_time / max(avg_quantized_time, 1e-8)
            
            # Estimate model size reduction (TensorFlow-specific)
            original_size = self._get_model_size(original_model)
            quantized_size = self._get_model_size(quantized_model)
            size_reduction = original_size / max(quantized_size, 1)
            
            # Compile results
            benchmark_results = {
                'original_inference_time': avg_original_time,
                'quantized_inference_time': avg_quantized_time,
                'speedup': speedup,
                'original_model_size': original_size,
                'quantized_model_size': quantized_size,
                'size_reduction': size_reduction,
                'accuracy_comparison': accuracy_comparison
            }
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Error during TensorFlow benchmark: {str(e)}")
            return {}
    
    def _get_model_size(self, model):
        """
        Estimate model size in bytes.
        
        Args:
            model: ML model
            
        Returns:
            int: Estimated model size in bytes
        """
        try:
            if HAS_TORCH and isinstance(model, torch.nn.Module):
                # PyTorch model size estimation
                size_bytes = 0
                for param in model.parameters():
                    size_bytes += param.nelement() * param.element_size()
                return size_bytes
                
            elif HAS_TF and isinstance(model, tf.keras.Model):
                # TensorFlow model size estimation
                size_bytes = 0
                for weight in model.weights:
                    size_bytes += weight.numpy().nbytes
                return size_bytes
                
            elif HAS_ONNX and hasattr(model, 'session'):
                # ONNX model size estimation
                # Use file size if available, otherwise estimate
                for key, value in self.quantized_models.items():
                    if value.get('model') == model:
                        return value.get('quantized_size', 0)
                return 0
                
            else:
                # Generic fallback
                memory_size = 0
                
                # Try to get size from pickle serialization
                import io
                import pickle
                
                try:
                    buffer = io.BytesIO()
                    pickle.dump(model, buffer)
                    memory_size = buffer.getbuffer().nbytes
                except:
                    pass
                
                return memory_size
                
        except Exception as e:
            self.logger.error(f"Error estimating model size: {str(e)}")
            return 0
    
    def get_quantized_model(self, model_id):
        """
        Get a quantized model by ID.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            object: Quantized model or None if not found
        """
        with self.model_lock:
            if model_id in self.quantized_models:
                return self.quantized_models[model_id].get("model")
            return None
    
    def get_quantization_info(self, model_id=None):
        """
        Get information about quantized models.
        
        Args:
            model_id (str, optional): Filter by model ID
            
        Returns:
            dict: Quantization information
        """
        with self.model_lock:
            if model_id:
                if model_id in self.quantized_models:
                    info = self.quantized_models[model_id].copy()
                    
                    # Remove the actual model object from the info
                    if "model" in info:
                        info["model"] = str(type(info["model"]))
                    
                    # Remove any large binary data
                    if "tflite_model" in info:
                        info["tflite_model"] = f"<TFLite Model: {len(info['tflite_model'])} bytes>"
                    
                    return info
                return None
            
            # Return info for all models
            result = {}
            for mid, info in self.quantized_models.items():
                model_info = info.copy()
                
                # Remove the actual model object from the info
                if "model" in model_info:
                    model_info["model"] = str(type(model_info["model"]))
                
                # Remove any large binary data
                if "tflite_model" in model_info:
                    model_info["tflite_model"] = f"<TFLite Model: {len(model_info['tflite_model'])} bytes>"
                
                result[mid] = model_info
            
            return result
    
    def run_inference(self, model_id, inputs, use_quantized=True):
        """
        Run inference using a quantized model.
        
        Args:
            model_id (str): Model identifier
            inputs: Input data for inference
            use_quantized (bool): Whether to use the quantized model
            
        Returns:
            object: Inference result or None if error
        """
        try:
            # Get the appropriate model
            with self.model_lock:
                if not use_quantized or model_id not in self.quantized_models:
                    # Use original model (if available)
                    return None
                
                model_info = self.quantized_models[model_id]
                model = model_info.get("model")
                framework = model_info.get("framework")
                precision = model_info.get("precision")
            
            if model is None:
                return None
            
            # Run inference based on framework
            if framework == "pytorch" and HAS_TORCH:
                # Convert inputs to tensor if needed
                if not isinstance(inputs, torch.Tensor):
                    if isinstance(inputs, np.ndarray):
                        inputs = torch.from_numpy(inputs)
                    else:
                        inputs = torch.tensor(inputs)
                
                # Use mixed precision if applicable
                if precision == "float16" and model_info.get("amp_enabled", False) and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        return model(inputs)
                else:
                    return model(inputs)
                
            elif framework == "tensorflow" and HAS_TF:
                # Convert inputs if needed
                if not isinstance(inputs, tf.Tensor) and not isinstance(inputs, np.ndarray):
                    inputs = np.array(inputs)
                
                # Run inference
                return model(inputs)
                
            elif framework == "onnx" and HAS_ONNX:
                # ONNX models are wrapped in a callable class
                return model(inputs)
                
            else:
                # Generic fallback
                return model(inputs)
                
        except Exception as e:
            self.logger.error(f"Error running inference with model {model_id}: {str(e)}")
            return None
    
    def save_quantized_model(self, model_id, output_path=None):
        """
        Save a quantized model to file.
        
        Args:
            model_id (str): Model identifier
            output_path (str, optional): Output file path
            
        Returns:
            str: Path to saved model or None if error
        """
        try:
            with self.model_lock:
                if model_id not in self.quantized_models:
                    self.logger.error(f"Model {model_id} not found in quantized models")
                    return None
                
                model_info = self.quantized_models[model_id]
                model = model_info.get("model")
                framework = model_info.get("framework")
                precision = model_info.get("precision")
            
            if model is None:
                return None
            
            # Create default output path if not provided
            if output_path is None:
                if not os.path.exists(self.logs_dir):
                    os.makedirs(self.logs_dir, exist_ok=True)
                output_path = os.path.join(self.logs_dir, f"{model_id}_{framework}_{precision}.quantized")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save based on framework
            if framework == "pytorch" and HAS_TORCH:
                # Save PyTorch model
                torch.save(model, output_path)
                
            elif framework == "tensorflow" and HAS_TF:
                # For TFLite models, save the binary content
                if "tflite_model" in model_info:
                    with open(output_path, 'wb') as f:
                        f.write(model_info["tflite_model"])
                else:
                    # Standard model
                    tf.keras.models.save_model(model, output_path)
                
            elif framework == "onnx" and HAS_ONNX:
                # For ONNX, copy the already-saved file
                if "quantized_path" in model_info:
                    import shutil
                    shutil.copy2(model_info["quantized_path"], output_path)
                
            else:
                # Generic fallback using pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(model, f)
            
            self.logger.info(f"Saved quantized model {model_id} to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving quantized model {model_id}: {str(e)}")
            return None
    
    def load_quantized_model(self, model_path, model_id, framework):
        """
        Load a quantized model from file.
        
        Args:
            model_path (str): Path to model file
            model_id (str): Model identifier
            framework (str): Model framework ('pytorch', 'tensorflow', 'onnx')
            
        Returns:
            object: Loaded model or None if error
        """
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return None
            
            # Load based on framework
            if framework == "pytorch" and HAS_TORCH:
                # Load PyTorch model
                model = torch.load(model_path)
                
            elif framework == "tensorflow" and HAS_TF:
                # Check if this is a TFLite model
                if model_path.endswith('.tflite'):
                    # Load TFLite model
                    with open(model_path, 'rb') as f:
                        tflite_model = f.read()
                    
                    # Create an interpreter
                    interpreter = tf.lite.Interpreter(model_content=tflite_model)
                    interpreter.allocate_tensors()
                    
                    # Wrap in a callable class (same as in quantize_tensorflow_model)
                    class QuantizedModel:
                        def __init__(self, interpreter):
                            self.interpreter = interpreter
                            self.input_details = interpreter.get_input_details()
                            self.output_details = interpreter.get_output_details()
                        
                        def __call__(self, inputs):
                            if isinstance(inputs, tf.Tensor):
                                inputs = inputs.numpy()
                            
                            # Resize input tensor if needed
                            for i, input_detail in enumerate(self.input_details):
                                required_shape = list(input_detail['shape'])
                                if len(required_shape) == 4:  # NHWC format
                                    required_shape[0] = len(inputs)
                                    self.interpreter.resize_tensor_input(input_detail['index'], required_shape)
                            
                            self.interpreter.allocate_tensors()
                            
                            # Set input tensor
                            self.interpreter.set_tensor(self.input_details[0]['index'], inputs)
                            
                            # Run inference
                            self.interpreter.invoke()
                            
                            # Get output tensor
                            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                            return output_data
                    
                    model = QuantizedModel(interpreter)
                    
                    # Store the TFLite model content
                    tflite_model_content = tflite_model
                else:
                    # Standard TensorFlow model
                    model = tf.keras.models.load_model(model_path)
                    tflite_model_content = None
                
            elif framework == "onnx" and HAS_ONNX:
                # Load ONNX model
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                onnx_session = ort.InferenceSession(model_path, sess_options)
                
                # Wrap in callable class
                class ONNXModel:
                    def __init__(self, session):
                        self.session = session
                        self.input_name = session.get_inputs()[0].name
                        self.output_name = session.get_outputs()[0].name
                    
                    def __call__(self, inputs):
                        # Convert inputs to numpy if necessary
                        if framework == 'pytorch' and isinstance(inputs, torch.Tensor):
                            inputs = inputs.cpu().numpy()
                        elif framework == 'tensorflow' and isinstance(inputs, tf.Tensor):
                            inputs = inputs.numpy()
                        
                        # Run inference
                        outputs = self.session.run([self.output_name], {self.input_name: inputs})
                        return outputs[0]
                
                model = ONNXModel(onnx_session)
                
            else:
                # Generic fallback using pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Store in registry
            with self.model_lock:
                self.quantized_models[model_id] = {
                    "model": model,
                    "framework": framework,
                    "precision": "unknown",  # Can't determine from file
                    "original_size": 0,      # Can't determine from file
                    "quantized_size": os.path.getsize(model_path),
                    "file_path": model_path
                }
                
                # Add TFLite content if applicable
                if framework == "tensorflow" and 'tflite_model_content' in locals():
                    self.quantized_models[model_id]["tflite_model"] = tflite_model_content
            
            self.logger.info(f"Loaded quantized model {model_id} from {model_path}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading quantized model: {str(e)}")
            return None
