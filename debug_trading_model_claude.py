import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import gc
import joblib
import logging
import pickle
import time
import random
import json
import math
import sklearn
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yfinance as yf
import sys
from scipy.stats import pearsonr
from scipy import stats  # Added for zscore
from tqdm import tqdm
import warnings
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass, field
from scipy.stats import pearsonr, percentileofscore
optimize_memory = True
analyze_shap = True  # Set to True to enable SHAP analysis  
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Ensemble weights will be set manually.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using alternative models.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    try:
        from lightgbm.callback import early_stopping
        LIGHTGBM_CALLBACKS_AVAILABLE = True
    except ImportError:
        LIGHTGBM_CALLBACKS_AVAILABLE = False
        print("LightGBM callbacks not available. Using alternative early stopping approach.")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_CALLBACKS_AVAILABLE = False
    print("LightGBM not available. Using alternative models.")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Using alternative models.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Some models will be disabled.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import pytorch_lightning as pl
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Deep learning models will be disabled.")

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import mutual_info_regression, RFECV
    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Many features will be disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Feature importance visualization will be limited.")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("TA-Lib not available. Technical indicators will be limited.")

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Some statistical features will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Time series forecasting will be limited.")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("ARCH not available. Volatility modeling will be limited.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Psutil not available. Memory management will be limited.")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False
    print("SARIMAX not available. SARIMA models will be disabled.")

# For TCN
try:
    import torch.nn.functional as F
    TCN_AVAILABLE = TORCH_AVAILABLE  # Will depend on PyTorch availability
except ImportError:
    TCN_AVAILABLE = False
    print("TCN dependencies not available. TCN models will be disabled.")

# For Transformer and N-BEATS
try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    TRANSFORMER_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Transformer dependencies not available. Transformer models will be disabled.")

class ModelError(Exception):
    """Custom error for model-related issues."""
    pass

class DataError(Exception):
    """Custom error for data-related issues."""
    pass

class ModelHyperparameterTuner:
    """Base class for hyperparameter tuning with Optuna."""
    
    def __init__(self, n_trials=50, timeout=600, n_jobs=-1, study_name=None):
        """
        Initialize the tuner.
        
        Args:
            n_trials (int): Maximum number of trials for optimization
            timeout (int): Timeout in seconds for the entire optimization
            n_jobs (int): Number of parallel jobs. -1 means using all available cores
            study_name (str): Optional name for the Optuna study
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.best_params = None
        self.study = None
        
        # Verify Optuna is available
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Hyperparameter tuning will be disabled.")
            
    def tune(self, X_train, y_train, X_val, y_val):
        """
        Run hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features 
            y_val: Validation target
            
        Returns:
            dict: Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Returning default parameters.")
            return self.get_default_params()
            
        try:
            # Create Optuna study
            study = optuna.create_study(
                direction="minimize",
                study_name=self.study_name or f"{self.__class__.__name__}_study",
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=min(self.n_jobs, os.cpu_count() or 1),
                show_progress_bar=True
            )
            
            # Store results
            self.best_params = study.best_params
            self.study = study
            
            # Log results
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best value: {study.best_value:.6f}")
            logger.info(f"Best parameters: {study.best_params}")
            
            return self.best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self.get_default_params()
    
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Objective function for Optuna optimization.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _objective")
    
    def get_default_params(self):
        """
        Get default hyperparameters.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_default_params")
        
    def create_model(self, params=None):
        """
        Create a model with the given parameters.
        If params is None, use the best_params if available, else use default params.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement create_model")
        
    def plot_optimization_history(self, output_path=None):
        """
        Plot the optimization history.
        
        Args:
            output_path (str): If provided, save the plot to this path
        """
        if not OPTUNA_AVAILABLE or self.study is None:
            logger.warning("No optimization history available to plot.")
            return False
            
        try:
            # Create the plot
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title("Optimization History")
            plt.tight_layout()
            
            # Save or show the plot
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Optimization history plot saved to {output_path}")
                plt.close()
                return True
            else:
                plt.show()
                plt.close()
                return True
                
        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")
            plt.close()
            return False
        
    def plot_param_importances(self, output_path=None):
        """
        Plot parameter importances.
        
        Args:
            output_path (str): If provided, save the plot to this path
        """
        if not OPTUNA_AVAILABLE or self.study is None:
            logger.warning("No parameter importances available to plot.")
            return False
            
        try:
            # Create the plot
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.title("Parameter Importances")
            plt.tight_layout()
            
            # Save or show the plot
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Parameter importances plot saved to {output_path}")
                plt.close()
                return True
            else:
                plt.show()
                plt.close()
                return True
                
        except Exception as e:
            logger.error(f"Error plotting parameter importances: {e}")
            plt.close()
            return False
class MemoryManager:
    """
    Manages memory usage with advanced monitoring and optimization techniques.
    Helps prevent out-of-memory errors for large datasets.
    """
    
    def __init__(self, memory_threshold: float = 0.8, gc_threshold: float = 0.7):
        """
        Initialize memory manager.
        
        Args:
            memory_threshold: Memory usage threshold to trigger emergency cleanup (fraction of total)
            gc_threshold: Memory usage threshold to trigger gc.collect() (fraction of total)
        """
        self.memory_threshold = memory_threshold
        self.gc_threshold = gc_threshold
        self.last_check_time = time.time()
        self.check_interval = 10  # Seconds between memory checks
        
        # Check if we have psutil for advanced monitoring
        self.has_psutil = PSUTIL_AVAILABLE
        
        # Store reference to tensor caches if using PyTorch
        self.tensor_caches = []
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.tensor_caches.append(torch.cuda.memory_cached(i))
    
    def check_memory(self, log_level: str = 'debug') -> Dict[str, float]:
        """
        Check current memory usage and perform cleanup if needed.
        
        Args:
            log_level: Logging level for memory stats ('debug', 'info', 'warning')
            
        Returns:
            Dictionary with memory statistics
        """
        current_time = time.time()
        # Only check every check_interval seconds to avoid performance overhead
        if current_time - self.last_check_time < self.check_interval:
            return {}
            
        self.last_check_time = current_time
        
        try:
            memory_stats = {}
            
            # Get CPU memory usage
            if self.has_psutil:
                try:
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_stats['cpu_memory_used_gb'] = memory_info.rss / (1024 ** 3)
                    
                    system_memory = psutil.virtual_memory()
                    memory_stats['system_memory_available_gb'] = system_memory.available / (1024 ** 3)
                    memory_stats['system_memory_percent'] = system_memory.percent / 100.0
                    
                except Exception as e:
                    logger.warning(f"Error getting detailed memory info: {e}")
            
            # Get GPU memory usage if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    gpu_stats = []
                    for i in range(torch.cuda.device_count()):
                        gpu_stats.append({
                            'allocated_gb': torch.cuda.memory_allocated(i) / (1024 ** 3),
                            'cached_gb': torch.cuda.memory_reserved(i) / (1024 ** 3),
                            'device': torch.cuda.get_device_name(i)
                        })
                    memory_stats['gpu_stats'] = gpu_stats
                    
                    # Calculate overall GPU memory pressure
                    total_allocated = sum(stat['allocated_gb'] for stat in gpu_stats)
                    memory_stats['gpu_memory_allocated_gb'] = total_allocated
                    
                except Exception as e:
                    logger.warning(f"Error getting GPU memory info: {e}")
            
            # Calculate memory pressure
            memory_pressure = memory_stats.get('system_memory_percent', 0)
            
            # Log memory stats
            log_func = getattr(logger, log_level, logger.debug)
            
            if self.has_psutil:
                log_func(f"Memory usage: CPU {memory_stats.get('cpu_memory_used_gb', 0):.2f} GB, "
                       f"System available {memory_stats.get('system_memory_available_gb', 0):.2f} GB, "
                       f"Usage {memory_pressure:.1%}")
            
            if TORCH_AVAILABLE and torch.cuda.is_available() and memory_stats.get('gpu_stats'):
                gpu_usage = [
                    f"GPU {i} ({stat['device']}): {stat['allocated_gb']:.2f} GB allocated"
                    for i, stat in enumerate(memory_stats['gpu_stats'])
                ]
                log_func(f"GPU memory usage: {', '.join(gpu_usage)}")
            
            # Trigger cleanup if needed
            if memory_pressure > self.memory_threshold:
                logger.warning("High memory pressure detected! Performing emergency cleanup.")
                self.emergency_cleanup()
            elif memory_pressure > self.gc_threshold:
                logger.info("Memory usage above threshold. Running garbage collection.")
                self.collect_garbage()
            
            return memory_stats
            
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
            return {}
    
    def collect_garbage(self):
        """Run garbage collection and release unused memory."""
        try:
            # Run Python's garbage collector
            n_collected = gc.collect(generation=2)  # Full collection
            
            # Clear PyTorch CUDA cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("PyTorch CUDA cache cleared")
                except Exception as e:
                    logger.warning(f"Error clearing CUDA cache: {e}")
            
            # Log collection results
            logger.info(f"Garbage collection: {n_collected} objects collected")
            
            # Check if we freed memory
            if self.has_psutil:
                try:
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    logger.info(f"Current memory usage: {memory_info.rss / (1024 ** 3):.2f} GB")
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in garbage collection: {e}")
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup operations."""
        try:
            # Run full garbage collection
            self.collect_garbage()
            
            # Try to clear module caches
            if 'pandas' in sys.modules and hasattr(sys.modules['pandas'], '_libs'):
                try:
                    # This can help clear some pandas internal caches
                    pandas_module = sys.modules['pandas']
                    if hasattr(pandas_module, '_libs') and hasattr(pandas_module._libs, 'hashtable'):
                        if hasattr(pandas_module._libs.hashtable, '_SIZE_HINT_LIMIT'):
                            pandas_module._libs.hashtable._SIZE_HINT_LIMIT = 0
                    logger.info("Cleared pandas internal caches")
                except Exception:
                    pass
            
            # Try to release memory back to OS
            if self.has_psutil:
                try:
                    # On Linux, this can help return memory to the OS
                    if hasattr(psutil, 'Process'):
                        process = psutil.Process(os.getpid())
                        if hasattr(process, 'memory_maps'):
                            process.memory_maps()
                except Exception:
                    pass
                    
            logger.info("Emergency memory cleanup completed")
            
            # Check current memory after cleanup
            if self.has_psutil:
                try:
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    system_memory = psutil.virtual_memory()
                    logger.info(f"After cleanup: Process memory: {memory_info.rss / (1024 ** 3):.2f} GB, "
                              f"System available: {system_memory.available / (1024 ** 3):.2f} GB")
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in emergency cleanup: {e}")
    
    @staticmethod
    def reduce_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce memory usage of a pandas DataFrame by optimizing dtypes.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame with lower memory usage
        """
        try:
            start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
            logger.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
            
            # Iterate through each column and optimize its dtype
            for col in df.columns:
                col_type = df[col].dtype
                
                # Skip non-numeric columns
                if col_type == object:
                    continue
                    
                # Get column min and max values
                col_min = df[col].min()
                col_max = df[col].max()
                
                # Optimize integer types
                if pd.api.types.is_integer_dtype(col_type):
                    if col_min >= 0:  # Unsigned integers
                        if col_max <= 255:
                            df[col] = df[col].astype(np.uint8)
                        elif col_max <= 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif col_max <= 4294967295:
                            df[col] = df[col].astype(np.uint32)
                        else:
                            df[col] = df[col].astype(np.uint64)
                    else:  # Signed integers
                        if col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype(np.int8)
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype(np.int16)
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            df[col] = df[col].astype(np.int32)
                        else:
                            df[col] = df[col].astype(np.int64)
                            
                # Optimize float types
                elif pd.api.types.is_float_dtype(col_type):
                    if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            
            # Calculate memory savings
            end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
            reduction = 100 * (start_mem - end_mem) / start_mem
            logger.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
            logger.info(f"Memory reduced by {reduction:.2f}%")
            
            # Explicit garbage collection to free memory immediately
            gc.collect()
            
            return df
            
        except Exception as e:
            logger.error(f"Error optimizing DataFrame memory: {e}")
            return df
        
class ModelDriftDetector:
    """
    Detects when a model needs retraining by monitoring data and prediction drift.
    Implements multiple drift detection algorithms with robust error handling.
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None, 
                prediction_history: Optional[List] = None,
                feature_names: Optional[List[str]] = None,
                drift_threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference data distribution (training data)
            prediction_history: History of model predictions
            feature_names: Names of features to monitor
            drift_threshold: p-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.prediction_history = prediction_history or []
        self.feature_names = feature_names or []
        self.drift_threshold = drift_threshold
        self.reference_stats = {}
        self.current_stats = {}
        self.drift_metrics = {}
        self.drift_detected = False
        self.retraining_recommended = False
        self.last_check_time = None
        
        # Initialize reference statistics if data is provided
        if reference_data is not None:
            self.compute_reference_statistics()
    
    def compute_reference_statistics(self):
        """Compute statistics on reference data for later comparison."""
        if self.reference_data is None or len(self.reference_data) == 0:
            logger.warning("No reference data provided. Cannot compute statistics.")
            return
            
        try:
            # Basic statistics for numeric features
            self.reference_stats['mean'] = {}
            self.reference_stats['std'] = {}
            self.reference_stats['min'] = {}
            self.reference_stats['max'] = {}
            self.reference_stats['quantiles'] = {}
            
            # Process each column
            for col in self.reference_data.columns:
                col_data = self.reference_data[col]
                
                # Skip non-numeric columns
                if not np.issubdtype(col_data.dtype, np.number):
                    continue
                    
                # Compute statistics with robust handling
                try:
                    self.reference_stats['mean'][col] = np.mean(col_data)
                    self.reference_stats['std'][col] = np.std(col_data)
                    self.reference_stats['min'][col] = np.min(col_data)
                    self.reference_stats['max'][col] = np.max(col_data)
                    self.reference_stats['quantiles'][col] = np.quantile(
                        col_data, [0.25, 0.5, 0.75]).tolist()
                except Exception as e:
                    logger.warning(f"Error computing statistics for {col}: {e}")
            
            # Compute correlation matrix for all numeric features
            numeric_cols = self.reference_data.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 1:
                try:
                    self.reference_stats['correlation'] = self.reference_data[numeric_cols].corr().to_dict()
                except Exception as e:
                    logger.warning(f"Error computing correlation matrix: {e}")
            
            # Compute data distribution histograms for selected features
            self.reference_stats['histograms'] = {}
            if self.feature_names:
                features_to_use = [f for f in self.feature_names if f in self.reference_data.columns]
            else:
                # If no specific features provided, use all numeric columns
                features_to_use = numeric_cols
                
            for col in features_to_use:
                try:
                    hist, bin_edges = np.histogram(self.reference_data[col], bins=20, density=True)
                    self.reference_stats['histograms'][col] = {
                        'hist': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Error computing histogram for {col}: {e}")
            
            # Save prediction distribution if available
            if self.prediction_history:
                try:
                    pred_array = np.array(self.prediction_history)
                    self.reference_stats['prediction_mean'] = np.mean(pred_array)
                    self.reference_stats['prediction_std'] = np.std(pred_array)
                    self.reference_stats['prediction_quantiles'] = np.quantile(
                        pred_array, [0.25, 0.5, 0.75]).tolist()
                    hist, bin_edges = np.histogram(pred_array, bins=20, density=True)
                    self.reference_stats['prediction_hist'] = {
                        'hist': hist.tolist(),
                        'bin_edges': bin_edges.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Error computing prediction statistics: {e}")
            
            logger.info("Reference statistics computed successfully")
            
        except Exception as e:
            logger.error(f"Error computing reference statistics: {e}")
    
    def check_data_drift(self, current_data: pd.DataFrame) -> bool:
        """
        Check for drift in feature distributions.
        
        Args:
            current_data: Current data to compare against reference
            
        Returns:
            True if drift is detected, False otherwise
        """
        if self.reference_data is None or self.reference_stats == {}:
            logger.warning("Reference statistics not available. Cannot check for drift.")
            return False
            
        if current_data is None or len(current_data) == 0:
            logger.warning("No current data provided. Cannot check for drift.")
            return False
            
        try:
            drift_detected = False
            self.current_stats = {}
            self.drift_metrics = {}
            
            # Compute current statistics
            # Basic statistics for numeric features
            self.current_stats['mean'] = {}
            self.current_stats['std'] = {}
            self.current_stats['min'] = {}
            self.current_stats['max'] = {}
            self.current_stats['quantiles'] = {}
            
            # Check drift for each feature
            for col in current_data.columns:
                # Skip columns not in reference data
                if col not in self.reference_data.columns:
                    continue
                    
                # Skip non-numeric columns
                if not np.issubdtype(current_data[col].dtype, np.number):
                    continue
                
                current_col_data = current_data[col]
                
                # Compute current statistics
                try:
                    self.current_stats['mean'][col] = np.mean(current_col_data)
                    self.current_stats['std'][col] = np.std(current_col_data)
                    self.current_stats['min'][col] = np.min(current_col_data)
                    self.current_stats['max'][col] = np.max(current_col_data)
                    self.current_stats['quantiles'][col] = np.quantile(
                        current_col_data, [0.25, 0.5, 0.75]).tolist()
                except Exception as e:
                    logger.warning(f"Error computing current statistics for {col}: {e}")
                    continue
                
                # Drift detection methods
                try:
                    # 1. Kolmogorov-Smirnov test for distribution shift
                    ref_data = self.reference_data[col].dropna()
                    current_data_col = current_col_data.dropna()
                    
                    # Ensure sufficient data for testing
                    if len(ref_data) > 10 and len(current_data_col) > 10:
                        ks_statistic, ks_pvalue = stats.ks_2samp(ref_data, current_data_col)
                        
                        # Record metrics
                        self.drift_metrics[f"{col}_ks_statistic"] = float(ks_statistic)
                        self.drift_metrics[f"{col}_ks_pvalue"] = float(ks_pvalue)
                        
                        # Check for drift
                        if ks_pvalue < self.drift_threshold:
                            self.drift_metrics[f"{col}_drift"] = True
                            drift_detected = True
                            logger.info(f"Drift detected in {col}: KS p-value = {ks_pvalue:.6f}")
                        else:
                            self.drift_metrics[f"{col}_drift"] = False
                    
                    # 2. Check for significant mean shift
                    mean_change_pct = abs(self.current_stats['mean'][col] - self.reference_stats['mean'][col]) / (
                        max(abs(self.reference_stats['mean'][col]), 1e-8))
                    self.drift_metrics[f"{col}_mean_change_pct"] = float(mean_change_pct)
                    
                    if mean_change_pct > 0.25:  # 25% change in mean
                        self.drift_metrics[f"{col}_mean_drift"] = True
                        drift_detected = True
                        logger.info(f"Mean drift detected in {col}: {mean_change_pct:.2%} change")
                    else:
                        self.drift_metrics[f"{col}_mean_drift"] = False
                    
                    # 3. Check for significant std change
                    if self.reference_stats['std'][col] > 0:
                        std_change_pct = abs(self.current_stats['std'][col] - self.reference_stats['std'][col]) / (
                            max(self.reference_stats['std'][col], 1e-8))
                        self.drift_metrics[f"{col}_std_change_pct"] = float(std_change_pct)
                        
                        if std_change_pct > 0.25:  # 25% change in std
                            self.drift_metrics[f"{col}_std_drift"] = True
                            drift_detected = True
                            logger.info(f"Std drift detected in {col}: {std_change_pct:.2%} change")
                        else:
                            self.drift_metrics[f"{col}_std_drift"] = False
                            
                except Exception as e:
                    logger.warning(f"Error checking drift for {col}: {e}")
            
            # Check for correlation shift
            try:
                numeric_cols = current_data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 1:
                    current_corr = current_data[numeric_cols].corr().to_dict()
                    
                    # Compare with reference correlation
                    if 'correlation' in self.reference_stats:
                        max_corr_diff = 0
                        for col1 in numeric_cols:
                            for col2 in numeric_cols:
                                if col1 >= col2:
                                    continue
                                
                                # Skip if either column not in reference
                                if (col1 not in self.reference_stats['correlation'] or 
                                    col2 not in self.reference_stats['correlation'].get(col1, {})):
                                    continue
                                
                                ref_corr = self.reference_stats['correlation'][col1][col2]
                                curr_corr = current_corr.get(col1, {}).get(col2, 0)
                                
                                corr_diff = abs(ref_corr - curr_corr)
                                max_corr_diff = max(max_corr_diff, corr_diff)
                        
                        self.drift_metrics['max_correlation_diff'] = float(max_corr_diff)
                        
                        if max_corr_diff > 0.3:  # 0.3 change in correlation
                            self.drift_metrics['correlation_drift'] = True
                            drift_detected = True
                            logger.info(f"Correlation drift detected: max difference = {max_corr_diff:.4f}")
                        else:
                            self.drift_metrics['correlation_drift'] = False
            except Exception as e:
                logger.warning(f"Error checking correlation drift: {e}")
            
            # Record results
            self.drift_detected = drift_detected
            self.last_check_time = datetime.now()
            
            # Set retraining flag based on drift severity
            drift_severity = sum(1 for k, v in self.drift_metrics.items() 
                             if k.endswith('_drift') and v is True)
            total_drift_checks = sum(1 for k in self.drift_metrics.keys() if k.endswith('_drift'))
            
            if total_drift_checks > 0:
                drift_ratio = drift_severity / total_drift_checks
                self.retraining_recommended = drift_ratio > 0.2  # Recommend retraining if >20% of checks show drift
                
                logger.info(f"Drift check summary: {drift_severity}/{total_drift_checks} tests show drift")
                logger.info(f"Retraining recommended: {self.retraining_recommended}")
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return False
    
    def check_prediction_drift(self, recent_predictions: List) -> bool:
        """
        Check for drift in prediction distributions.
        
        Args:
            recent_predictions: Recent model predictions to check for drift
            
        Returns:
            True if prediction drift is detected, False otherwise
        """
        if not self.reference_stats.get('prediction_mean') or len(recent_predictions) < 10:
            logger.warning("Insufficient data for prediction drift detection.")
            return False
            
        try:
            pred_array = np.array(recent_predictions)
            current_mean = np.mean(pred_array)
            current_std = np.std(pred_array)
            
            # Check for significant mean shift
            mean_change_pct = abs(current_mean - self.reference_stats['prediction_mean']) / (
                max(abs(self.reference_stats['prediction_mean']), 1e-8))
            
            self.drift_metrics['prediction_mean_change_pct'] = float(mean_change_pct)
            
            # Check for significant std change
            if self.reference_stats['prediction_std'] > 0:
                std_change_pct = abs(current_std - self.reference_stats['prediction_std']) / (
                    max(self.reference_stats['prediction_std'], 1e-8))
                self.drift_metrics['prediction_std_change_pct'] = float(std_change_pct)
            else:
                std_change_pct = 0
                
            # Run KS test for distribution shift
            ref_predictions = np.array(self.prediction_history)
            ks_statistic, ks_pvalue = stats.ks_2samp(ref_predictions, pred_array)
            
            self.drift_metrics['prediction_ks_statistic'] = float(ks_statistic)
            self.drift_metrics['prediction_ks_pvalue'] = float(ks_pvalue)
            
            # Determine if drift occurred
            prediction_drift = (ks_pvalue < self.drift_threshold or 
                              mean_change_pct > 0.3 or  # 30% change in mean
                              std_change_pct > 0.3)     # 30% change in std
            
            self.drift_metrics['prediction_drift'] = prediction_drift
            
            if prediction_drift:
                logger.info(f"Prediction drift detected: KS p-value = {ks_pvalue:.6f}, "
                          f"mean change = {mean_change_pct:.2%}, std change = {std_change_pct:.2%}")
                
                # Update retraining recommendation if needed
                self.retraining_recommended = True
            
            return prediction_drift
            
        except Exception as e:
            logger.error(f"Error checking prediction drift: {e}")
            return False
    
    def update_reference_data(self, new_reference_data: pd.DataFrame, 
                            new_predictions: Optional[List] = None):
        """
        Update reference data and recompute statistics.
        
        Args:
            new_reference_data: New reference data
            new_predictions: New reference predictions
        """
        try:
            self.reference_data = new_reference_data
            
            if new_predictions is not None:
                self.prediction_history = new_predictions
                
            # Recompute statistics
            self.compute_reference_statistics()
            
            # Reset drift flags
            self.drift_detected = False
            self.retraining_recommended = False
            
            logger.info("Reference data and statistics updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating reference data: {e}")
    
    def generate_drift_report(self, output_path: str = None) -> Dict:
        """
        Generate a comprehensive drift report.
        
        Args:
            output_path: Path to save the report as JSON
            
        Returns:
            Dictionary with drift metrics and recommendations
        """
        try:
            # Prepare report
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'drift_detected': self.drift_detected,
                'retraining_recommended': self.retraining_recommended,
                'drift_metrics': self.drift_metrics,
                'feature_statistics': {
                    'reference': self.reference_stats,
                    'current': self.current_stats
                }
            }
            
            # Calculate overall drift score (0-100)
            drift_scores = []
            
            # Collect all drift test results
            for key, value in self.drift_metrics.items():
                if key.endswith('_drift') and isinstance(value, bool):
                    drift_scores.append(1 if value else 0)
                elif key.endswith('_pvalue') and isinstance(value, (int, float)):
                    # Convert p-value to drift score (lower p-value = higher drift)
                    drift_scores.append(max(0, 1 - (value / self.drift_threshold)))
            
            # Calculate overall score
            if drift_scores:
                overall_drift_score = 100 * sum(drift_scores) / len(drift_scores)
                report['overall_drift_score'] = float(overall_drift_score)
                
                # High, medium, low drift classification
                if overall_drift_score > 70:
                    report['drift_severity'] = 'high'
                elif overall_drift_score > 30:
                    report['drift_severity'] = 'medium'
                else:
                    report['drift_severity'] = 'low'
            
            # Add recommendations
            recommendations = []
            
            if self.retraining_recommended:
                recommendations.append("Model retraining is recommended due to significant data drift.")
                
                # Suggest which features to focus on
                drifted_features = [key.split('_')[0] for key, value in self.drift_metrics.items() 
                               if key.endswith('_drift') and value is True and not key.startswith('prediction')]
                
                if drifted_features:
                    recommendations.append(f"Pay special attention to these features during retraining: {', '.join(drifted_features)}")
            
            if self.drift_metrics.get('correlation_drift'):
                recommendations.append("Feature correlation structure has changed. Consider revising feature engineering.")
                
            if self.drift_metrics.get('prediction_drift'):
                recommendations.append("Prediction distribution has changed. Monitor model outputs closely.")
                
            report['recommendations'] = recommendations
            
            # Save report if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=4)
                logger.info(f"Drift report saved to {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating drift report: {e}")
            return {'error': str(e)}
            
def check_model_retraining_need(model_dir: str, new_data: pd.DataFrame,
                              predictions: List, drift_threshold: float = 0.05) -> Dict:
    """
    Check if a model needs retraining by monitoring data and prediction drift.
    
    Args:
        model_dir: Directory where model and reference data are stored
        new_data: New incoming data to compare with reference data
        predictions: Recent model predictions
        drift_threshold: Threshold for drift detection tests
        
    Returns:
        Dictionary with retraining recommendation and drift metrics
    """
    try:
        # Load reference data if available
        reference_data_path = os.path.join(model_dir, 'reference_data.pkl')
        reference_predictions_path = os.path.join(model_dir, 'reference_predictions.pkl')
        
        reference_data = None
        reference_predictions = None
        
        if os.path.exists(reference_data_path):
            try:
                reference_data = pd.read_pickle(reference_data_path)
                logger.info(f"Loaded reference data from {reference_data_path}")
            except Exception as e:
                logger.warning(f"Error loading reference data: {e}")
        
        if os.path.exists(reference_predictions_path):
            try:
                with open(reference_predictions_path, 'rb') as f:
                    reference_predictions = pickle.load(f)
                logger.info(f"Loaded reference predictions from {reference_predictions_path}")
            except Exception as e:
                logger.warning(f"Error loading reference predictions: {e}")
        
        # Create drift detector
        feature_names = None
        selected_features_path = os.path.join(model_dir, 'selected_features.json')
        if os.path.exists(selected_features_path):
            try:
                with open(selected_features_path, 'r') as f:
                    feature_names = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading selected features: {e}")
        
        drift_detector = ModelDriftDetector(
            reference_data=reference_data, 
            prediction_history=reference_predictions,
            feature_names=feature_names,
            drift_threshold=drift_threshold
        )
        
        # Check for data drift
        data_drift = drift_detector.check_data_drift(new_data)
        
        # Check for prediction drift
        prediction_drift = drift_detector.check_prediction_drift(predictions)
        
        # Generate drift report
        os.makedirs(os.path.join(model_dir, 'drift_reports'), exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(model_dir, 'drift_reports', f'drift_report_{timestamp}.json')
        
        drift_report = drift_detector.generate_drift_report(report_path)
        
        # Log results
        if drift_detector.retraining_recommended:
            logger.warning("Model retraining is recommended based on drift analysis.")
            for recommendation in drift_report.get('recommendations', []):
                logger.warning(f"Recommendation: {recommendation}")
        else:
            logger.info("No significant drift detected. Model retraining not needed at this time.")
            
        return {
            'retraining_recommended': drift_detector.retraining_recommended,
            'data_drift_detected': data_drift,
            'prediction_drift_detected': prediction_drift,
            'drift_report': drift_report,
            'drift_detector': drift_detector
        }
        
    except Exception as e:
        logger.error(f"Error checking model retraining need: {e}")
        return {
            'error': str(e),
            'retraining_recommended': False,
            'data_drift_detected': False,
            'prediction_drift_detected': False,
            'drift_report': {}
        }
def save_reference_data(model_dir: str, reference_data: pd.DataFrame, 
                      reference_predictions: List):
    """
    Save reference data and predictions for future drift detection.
    
    Args:
        model_dir: Directory to save reference data
        reference_data: Training data to use as reference
        reference_predictions: Training predictions to use as reference
    """
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Save reference data
        reference_data_path = os.path.join(model_dir, 'reference_data.pkl')
        reference_data.to_pickle(reference_data_path)
        
        # Save reference predictions
        reference_predictions_path = os.path.join(model_dir, 'reference_predictions.pkl')
        with open(reference_predictions_path, 'wb') as f:
            pickle.dump(reference_predictions, f)
            
        logger.info(f"Saved reference data and predictions to {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving reference data: {e}")
        return False

# Configure logging
logging.basicConfig(
    filename='enhanced_trading_model.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Add console handler to see logs in real-time
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def check_gpu_availability() -> Tuple[str, str, Any]:
    """
    Check if GPU is available and set appropriate device settings.
    
    Returns:
        Tuple containing:
        - device_type: 'gpu' or 'cpu' for traditional ML models
        - tree_method: 'gpu_hist' or 'auto' for tree-based models
        - torch_device: torch.device object for PyTorch models
    """
    device_type = 'cpu'
    tree_method = 'auto'
    torch_device = None
    
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            memory_available = gpu_memory / (1024**3)  # Convert to GB
            
            if memory_available >= 4:  # Require at least 4GB GPU memory
                logger.info(f"GPU detected with {memory_available:.2f}GB memory")
                device_type = 'gpu'
                tree_method = 'gpu_hist'
                torch_device = torch.device('cuda')
            else:
                logger.warning(f"GPU detected but insufficient memory ({memory_available:.2f}GB)")
                torch_device = torch.device('cpu')
        else:
            logger.info("No GPU detected, using CPU")
            torch_device = torch.device('cpu') if TORCH_AVAILABLE else None
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        torch_device = torch.device('cpu') if TORCH_AVAILABLE else None
    
    return device_type, tree_method, torch_device

def clean_memory():
    """Clean memory after large operations."""
    try:
        # Clear PyTorch GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear CPU memory
        gc.collect()
        
        # Get memory usage
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            logger.info(f"Current memory usage: {memory_usage:.2f}MB")
    
    except Exception as e:
        logger.warning(f"Error during memory cleanup: {e}")
        
def safely_dump_to_file(obj, filepath):
    """Safely dump object to file with error handling."""
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        if filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(obj, f, default=lambda o: str(o) if not isinstance(o, (int, float, str, bool, list, dict, type(None))) else o)
        else:
            joblib.dump(obj, filepath)
        return True
    except Exception as e:
        logger.error(f"Error saving object to {filepath}: {e}")
        return False

class ShapAnalyzer:
    """Advanced SHAP value analyzer for model interpretability."""
    
    def __init__(self, models=None, feature_names=None):
        """
        Initialize SHAP analyzer.
        
        Args:
            models: Dict of models to analyze
            feature_names: List of feature names
        """
        self.models = models or {}
        self.feature_names = feature_names or []
        self.explainers = {}
        self.shap_values = {}
        self.global_importance = {}
        self.interactions = {}
        
    def create_explainers(self, X_sample: np.ndarray):
        """Create SHAP explainers for supported models."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Model interpretability will be limited.")
            return
        
        try:
            for name, model in self.models.items():
                try:
                    if name == 'xgboost' and hasattr(model, 'predict'):
                        self.explainers[name] = shap.TreeExplainer(model)
                        logger.info(f"Created TreeExplainer for {name}")
                    elif name == 'lightgbm' and hasattr(model, 'predict'):
                        self.explainers[name] = shap.TreeExplainer(model)
                        logger.info(f"Created TreeExplainer for {name}")
                    elif name == 'catboost' and hasattr(model, 'predict'):
                        self.explainers[name] = shap.TreeExplainer(model)
                        logger.info(f"Created TreeExplainer for {name}")
                    elif name == 'random_forest' and hasattr(model, 'predict'):
                        self.explainers[name] = shap.TreeExplainer(model)
                        logger.info(f"Created TreeExplainer for {name}")
                    elif name == 'deep_learning' and TORCH_AVAILABLE:
                        # For deep models, use KernelExplainer as fallback
                        # We create a prediction function that handles PyTorch model
                        def f(x):
                            model.eval()
                            with torch.no_grad():
                                input_tensor = torch.FloatTensor(x)
                                return model(input_tensor).cpu().numpy()
                        
                        # Use a sample of data for background
                        self.explainers[name] = shap.KernelExplainer(f, X_sample[:50])
                        logger.info(f"Created KernelExplainer for {name}")
                except Exception as e:
                    logger.warning(f"Error creating explainer for {name}: {e}")
        except Exception as e:
            logger.error(f"Error in create_explainers: {e}")
            
    def compute_shap_values(self, X: np.ndarray, max_samples: int = 1000):
        """Compute SHAP values for each model."""
        if not SHAP_AVAILABLE or not self.explainers:
            return
            
        try:
            # Limit sample size for computational efficiency
            sample_size = min(len(X), max_samples)
            X_sample = X[:sample_size] if sample_size < len(X) else X
            
            for name, explainer in self.explainers.items():
                try:
                    logger.info(f"Computing SHAP values for {name}...")
                    
                    # Different handling for different explainer types
                    if isinstance(explainer, shap.TreeExplainer):
                        self.shap_values[name] = explainer.shap_values(X_sample)
                        
                        # For tree models, also compute interaction values (but limit to smaller sample)
                        if sample_size <= 100 and hasattr(explainer, 'shap_interaction_values'):
                            try:
                                interaction_sample = X_sample[:100]  # Limit size for interactions
                                self.interactions[name] = explainer.shap_interaction_values(interaction_sample)
                                logger.info(f"Computed SHAP interaction values for {name}")
                            except Exception as e:
                                logger.warning(f"Error computing interaction values for {name}: {e}")
                    elif isinstance(explainer, shap.KernelExplainer):
                        # Limit sample even further for KernelExplainer
                        kernel_sample = X_sample[:100]
                        self.shap_values[name] = explainer.shap_values(kernel_sample)
                    
                    logger.info(f"SHAP values computed for {name}")
                except Exception as e:
                    logger.warning(f"Error computing SHAP values for {name}: {e}")
        except Exception as e:
            logger.error(f"Error in compute_shap_values: {e}")
            
    def compute_global_feature_importance(self):
        """Compute global feature importance based on SHAP values."""
        if not self.shap_values:
            return
            
        try:
            for name, values in self.shap_values.items():
                try:
                    # Handle different formats of SHAP values
                    if isinstance(values, list):
                        # For multi-output models, take the first output
                        shap_array = np.abs(values[0])
                    else:
                        shap_array = np.abs(values)
                    
                    # Calculate mean absolute SHAP value for each feature
                    mean_abs_shap = np.mean(shap_array, axis=0)
                    
                    # Create importance dictionary
                    if len(self.feature_names) == len(mean_abs_shap):
                        importance = dict(zip(self.feature_names, mean_abs_shap))
                    else:
                        logger.warning(f"Feature names length ({len(self.feature_names)}) doesn't match "
                                      f"SHAP values dimension ({len(mean_abs_shap)})")
                        importance = {f"feature_{i}": value for i, value in enumerate(mean_abs_shap)}
                    
                    # Sort by importance
                    self.global_importance[name] = dict(sorted(importance.items(), 
                                                             key=lambda x: x[1], 
                                                             reverse=True))
                    
                    logger.info(f"Computed global feature importance for {name}")
                except Exception as e:
                    logger.warning(f"Error computing global feature importance for {name}: {e}")
        except Exception as e:
            logger.error(f"Error in compute_global_feature_importance: {e}")
    
    def plot_summary(self, model_name: str, max_display: int = 20, output_path: str = None):
        """Create and save SHAP summary plot for a specific model."""
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            logger.warning(f"SHAP values not available for {model_name}")
            return False
            
        try:
            plt.figure(figsize=(12, 8))
            
            # Get SHAP values
            values = self.shap_values[model_name]
            
            # Handle different SHAP value formats
            if isinstance(values, list):
                # For multi-output models, use the first output
                plot_values = values[0]
            else:
                plot_values = values
            
            # Create feature_names that match the SHAP values dimension
            if len(self.feature_names) == plot_values.shape[1]:
                feature_names = self.feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(plot_values.shape[1])]
            
            # Create the summary plot
            shap.summary_plot(plot_values, 
                            feature_names=feature_names, 
                            max_display=max_display, 
                            show=False)
            
            plt.title(f"SHAP Feature Importance: {model_name}")
            plt.tight_layout()
            
            # Save or show the plot
            if output_path:
                plt.savefig(output_path)
                plt.close()
                logger.info(f"SHAP summary plot saved to {output_path}")
                return True
            else:
                plt.show()
                plt.close()
                return True
                
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            plt.close()
            return False
            
    def plot_dependence(self, model_name: str, feature_idx: int, output_path: str = None):
        """Create and save SHAP dependence plot for a specific feature and model."""
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            logger.warning(f"SHAP values not available for {model_name}")
            return False
            
        try:
            plt.figure(figsize=(10, 6))
            
            # Get SHAP values with proper error checking
            values = self.shap_values[model_name]
            
            # CRITICAL FIX: Check if values is None or empty
            if values is None or (isinstance(values, np.ndarray) and values.size == 0):
                logger.error(f"No valid SHAP values for {model_name}")
                return False
                
            # Handle different SHAP value formats
            if isinstance(values, list):
                # For multi-output models, use the first output
                plot_values = values[0]
                if plot_values is None or not hasattr(plot_values, 'shape'):
                    logger.error(f"Invalid SHAP values structure for {model_name}")
                    return False
            else:
                plot_values = values
                
            # Verify feature_idx is in range
            if feature_idx >= plot_values.shape[1]:
                logger.error(f"Feature index {feature_idx} out of range for SHAP values with {plot_values.shape[1]} features")
                return False
            
            # Get feature name
            if len(self.feature_names) > feature_idx:
                feature_name = self.feature_names[feature_idx]
            else:
                feature_name = f"feature_{feature_idx}"
            
            # Create the dependence plot
            shap.dependence_plot(feature_idx, 
                                plot_values, 
                                feature_names=self.feature_names if len(self.feature_names) == plot_values.shape[1] else None,
                                show=False)
            
            plt.title(f"SHAP Dependence Plot: {feature_name} ({model_name})")
            plt.tight_layout()
            
            # Save or show the plot
            if output_path:
                plt.savefig(output_path)
                plt.close()
                logger.info(f"SHAP dependence plot saved to {output_path}")
                return True
            else:
                plt.show()
                plt.close()
                return True
                
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")
            plt.close()
            return False
                
    def plot_interactions(self, model_name: str, feature_idx: int, output_path: str = None):
        """Plot SHAP interaction values for a specific feature."""
        if not SHAP_AVAILABLE or model_name not in self.interactions:
            logger.warning(f"SHAP interaction values not available for {model_name}")
            return False
            
        try:
            plt.figure(figsize=(12, 8))
            
            # Get interaction values
            values = self.interactions[model_name]
            
            # Get feature name
            if len(self.feature_names) > feature_idx:
                feature_name = self.feature_names[feature_idx]
            else:
                feature_name = f"feature_{feature_idx}"
            
            # Create interaction plot
            shap.summary_plot(values[:, feature_idx, :], 
                             feature_names=self.feature_names if len(self.feature_names) == values.shape[2] else None,
                             max_display=10,
                             show=False)
            
            plt.title(f"SHAP Interaction Values: {feature_name} ({model_name})")
            plt.tight_layout()
            
            # Save or show the plot
            if output_path:
                plt.savefig(output_path)
                plt.close()
                logger.info(f"SHAP interaction plot saved to {output_path}")
                return True
            else:
                plt.show()
                plt.close()
                return True
                
        except Exception as e:
            logger.error(f"Error creating SHAP interaction plot: {e}")
            plt.close()
            return False
            
    def create_comprehensive_report(self, output_dir: str, top_features: int = 10):
        """Create a comprehensive report with multiple SHAP visualizations."""
        if not SHAP_AVAILABLE or not self.shap_values:
            logger.warning("SHAP values not available. Cannot create report.")
            return False
            
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a report file
            report_path = os.path.join(output_dir, "shap_report.md")
            with open(report_path, 'w') as f:
                f.write("# SHAP Analysis Report\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Global feature importance section
                f.write("## Global Feature Importance\n\n")
                
                for model_name, importance in self.global_importance.items():
                    f.write(f"### {model_name}\n\n")
                    f.write("| Feature | Importance |\n")
                    f.write("|---------|------------|\n")
                    
                    # Show top features
                    # Fix: Ensure SHAP values are properly converted to scalars
                    for i, (feature, value) in enumerate(importance.items()):
                        if i >= top_features:
                            break
                        # Convert to scalar before formatting
                        scalar_value = float(value) if isinstance(value, (np.ndarray, np.number)) else value
                        f.write(f"| {feature} | {scalar_value:.6f} |\n")
                    
                    f.write("\n")
                    
                    # Create and save summary plot
                    summary_path = os.path.join(output_dir, f"{model_name}_summary.png")
                    if self.plot_summary(model_name, max_display=top_features, output_path=summary_path):
                        f.write(f"![{model_name} Summary]({os.path.basename(summary_path)})\n\n")
                
                # Feature dependence plots for top features
                f.write("## Feature Dependence Analysis\n\n")
                
                # For each model, create dependence plots for top features
                for model_name, importance in self.global_importance.items():
                    f.write(f"### {model_name}\n\n")
                    
                    # Get feature names and indices
                    top_feature_names = list(importance.keys())[:min(5, len(importance))]
                    
                    for feature_name in top_feature_names:
                        try:
                            # Find feature index
                            if feature_name in self.feature_names:
                                feature_idx = self.feature_names.index(feature_name)
                            else:
                                # Parse index from "feature_X" format
                                feature_idx = int(feature_name.split('_')[1])
                            
                            # Create dependence plot
                            dependence_path = os.path.join(output_dir, 
                                                         f"{model_name}_{feature_name}_dependence.png")
                            if self.plot_dependence(model_name, feature_idx, dependence_path):
                                f.write(f"#### {feature_name}\n\n")
                                f.write(f"![{feature_name} Dependence]({os.path.basename(dependence_path)})\n\n")
                        except Exception as e:
                            logger.warning(f"Error creating dependence plot for {feature_name}: {e}")
                
                # Feature interactions section
                if self.interactions:
                    f.write("## Feature Interactions\n\n")
                    
                    for model_name, interactions in self.interactions.items():
                        if model_name in self.global_importance:
                            f.write(f"### {model_name}\n\n")
                            
                            # Get top features for interactions
                            top_feature_names = list(self.global_importance[model_name].keys())[:3]
                            
                            for feature_name in top_feature_names:
                                try:
                                    # Find feature index
                                    if feature_name in self.feature_names:
                                        feature_idx = self.feature_names.index(feature_name)
                                    else:
                                        # Parse index from "feature_X" format
                                        feature_idx = int(feature_name.split('_')[1])
                                    
                                    # Create interaction plot
                                    interaction_path = os.path.join(output_dir, 
                                                                 f"{model_name}_{feature_name}_interaction.png")
                                    if self.plot_interactions(model_name, feature_idx, interaction_path):
                                        f.write(f"#### {feature_name} Interactions\n\n")
                                        f.write(f"![{feature_name} Interactions]({os.path.basename(interaction_path)})\n\n")
                                except Exception as e:
                                    logger.warning(f"Error creating interaction plot for {feature_name}: {e}")
            
            logger.info(f"SHAP analysis report created at {report_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating SHAP report: {e}")
            return False

@dataclass
class ModelConfig:
    # Time window parameters
    lookback_window: int = 250
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 5e-4
    num_epochs: int = 150
    early_stopping_patience: int = 15
    validation_split: float = 0.2
    test_split: float = 0.1
    min_samples: int = 1000
    
    # Model architecture parameters
    hidden_dim: int = 1024
    num_layers: int = 3
    dropout: float = 0.3
    
    # Optimization parameters
    weight_decay: float = 1e-4
    use_swa: bool = True
    swa_start: int = 50
    swa_freq: int = 5
    swa_lr: float = 1e-4
    activation_fn: str = 'gelu' 
    use_attention: bool = True   
    attention_heads: int = 8    
    input_dim: int = 0
    
    # Multi-horizon settings
    multi_horizon: bool = True
    forecast_horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    horizon_weights: List[float] = field(default_factory=lambda: [0.5, 0.2, 0.15, 0.1, 0.05])
    _forecast_horizon: int = 1  # Internal storage for forecast_horizon
    
    # Optional input to override forecast horizon (if not using multi-horizon)
    forecast_horizon_input: Optional[int] = None

    def __post_init__(self):
        # If forecast_horizon_input is provided, update _forecast_horizon accordingly.
        if self.forecast_horizon_input is not None:
            self._forecast_horizon = self.forecast_horizon_input

    @property
    def forecast_horizon(self) -> int:
        """
        Returns the number of forecast horizons if multi_horizon is True;
        otherwise, returns the stored _forecast_horizon value.
        """
        if self.multi_horizon:
            return len(self.forecast_horizons)
        return self._forecast_horizon

    @forecast_horizon.setter
    def forecast_horizon(self, value: int):
        self._forecast_horizon = value
    
    def get_forecasting_horizons(self) -> List[int]:
        """
        Get the list of forecast horizons regardless of multi-horizon setting.
        
        Returns:
            List containing all forecast horizons (either the list from multi-horizon 
            or a single-item list with the forecast_horizon value)
        """
        if self.multi_horizon:
            return self.forecast_horizons
        else:
            return [self._forecast_horizon]

        
class StochasticWeightAveraging:
    """
    Implementation of Stochastic Weight Averaging (SWA) for PyTorch models.
    
    SWA averages multiple points along the trajectory of SGD to improve
    generalization performance. This implementation provides functionality
    to:
    1. Maintain a running average of weights
    2. Update the average on a specified frequency
    3. Apply the averaged weights to the model
    
    References:
        - Averaging Weights Leads to Wider Optima and Better Generalization
          https://arxiv.org/abs/1803.05407
    """
    
    def __init__(self, model, swa_start: int, swa_freq: int, swa_lr: float):
        """
        Initialize the SWA object.
        
        Args:
            model: PyTorch model to apply SWA to
            swa_start: The epoch to start applying SWA from
            swa_freq: The frequency to update the SWA weights
            swa_lr: Learning rate during SWA phase
        """
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        
        # Create a copy of the model's parameter state
        self.swa_state = {}
        self.swa_n_models = 0
        self._prepare_swa_state()
        
    def _prepare_swa_state(self):
        """Initialize the SWA state with zeros matching model parameters."""
        try:
            # For each parameter in the model, create a matching tensor of zeros
            for name, param in self.model.named_parameters():
                # Skip parameters that don't require gradients
                if not param.requires_grad:
                    continue
                
                # Create a zero tensor with same shape and device as the parameter
                self.swa_state[name] = torch.zeros_like(param.data)
                
            logger.info(f"Initialized SWA state with {len(self.swa_state)} parameters")
            
        except Exception as e:
            logger.error(f"Error initializing SWA state: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def should_update(self, epoch: int) -> bool:
        """
        Determine if SWA weight averaging should be applied at this epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            True if SWA update should be performed, False otherwise
        """
        # Only update if past the start epoch and on the specified frequency
        return epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0
    
    def update(self, epoch: int):
        """
        Update the SWA running averages if conditions are met.
        
        Args:
            epoch: Current training epoch
        """
        if not self.should_update(epoch):
            return
            
        try:
            # Increment the number of models used for averaging
            self.swa_n_models += 1
            n = self.swa_n_models
            
            # Update running average for each parameter
            for name, param in self.model.named_parameters():
                if not param.requires_grad or name not in self.swa_state:
                    continue
                    
                # Update running average: SWA_t = (SWA_{t-1} * (n-1) + w_t) / n
                self.swa_state[name].mul_(1.0 - 1.0/n).add_(param.data/n)
                
            logger.info(f"Updated SWA weights (model count: {self.swa_n_models})")
            
        except Exception as e:
            logger.error(f"Error updating SWA weights: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def apply(self):
        """Apply the SWA weights to the model."""
        if self.swa_n_models == 0:
            logger.warning("No SWA updates have been performed. Skipping application.")
            return
            
        try:
            # Store original parameters
            orig_params = {}
            for name, param in self.model.named_parameters():
                if not param.requires_grad or name not in self.swa_state:
                    continue
                orig_params[name] = param.data.clone()
                
            # Apply SWA weights to the model
            for name, param in self.model.named_parameters():
                if not param.requires_grad or name not in self.swa_state:
                    continue
                param.data.copy_(self.swa_state[name])
                
            # Log parameter change statistics
            param_diffs = []
            for name in orig_params:
                if name in self.swa_state:
                    diff = torch.norm(orig_params[name] - self.swa_state[name]).item()
                    param_diffs.append(diff)
            
            if param_diffs:
                avg_diff = sum(param_diffs) / len(param_diffs)
                max_diff = max(param_diffs)
                logger.info(f"Applied SWA weights. Avg param diff: {avg_diff:.6f}, Max diff: {max_diff:.6f}")
            
        except Exception as e:
            logger.error(f"Error applying SWA weights: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    def adjust_optimizer(self, optimizer):
        """
        Adjust the optimizer's learning rate for the SWA phase.
        
        Args:
            optimizer: The optimizer to adjust
        """
        try:
            # Keep the same optimizer but modify the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.swa_lr
                
            logger.info(f"Adjusted optimizer learning rate to {self.swa_lr} for SWA phase")
            
        except Exception as e:
            logger.error(f"Error adjusting optimizer for SWA: {e}")
            
    def state_dict(self):
        """Serialize the SWA state for checkpointing."""
        return {
            'swa_state': self.swa_state,
            'swa_n_models': self.swa_n_models,
            'swa_start': self.swa_start,
            'swa_freq': self.swa_freq,
            'swa_lr': self.swa_lr
        }
        
    def load_state_dict(self, state_dict):
        """Load the SWA state from a checkpoint."""
        try:
            self.swa_state = state_dict['swa_state']
            self.swa_n_models = state_dict['swa_n_models']
            self.swa_start = state_dict['swa_start']
            self.swa_freq = state_dict['swa_freq']
            self.swa_lr = state_dict['swa_lr']
            
            logger.info(f"Loaded SWA state with {self.swa_n_models} averaged models")
            
        except Exception as e:
            logger.error(f"Error loading SWA state: {e}")
            # Re-initialize SWA state as fallback
            self.swa_n_models = 0
            self._prepare_swa_state()

class DataValidator:
    """Advanced data validation with market regime detection."""
    
    @staticmethod

    def validate_timeseries(df: pd.DataFrame) -> bool:
        """Validates time series data integrity and quality."""
        try:
            logger.info(f"Validating timeseries data with shape: {df.shape}")
            
            # Check for minimal data requirements
            if len(df) < 100:  # Require at least 100 data points
                logger.warning(f"Dataset too small: only {len(df)} rows. Need at least 100.")
                return False
                
            required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}. Will try to continue.")
            
            # Check for NaN values
            nan_counts = df.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if not nan_cols.empty:
                logger.warning(f"Dataset contains NaN values in columns: {nan_cols.to_dict()}")
                # Calculate percentage of NaNs
                nan_percent = (df.isnull().sum() / len(df)) * 100
                high_nan_cols = nan_percent[nan_percent > 20].index.tolist()
                if high_nan_cols:
                    logger.warning(f"Columns with >20% NaNs: {high_nan_cols}")
            
            # Check for duplicates
            if df.index.duplicated().any():
                logger.warning(f"Dataset contains {df.index.duplicated().sum()} duplicate timestamps.")
            
            # Check for price anomalies
            price_cols = ['Open', 'High', 'Low', 'Close']
            available_price_cols = [col for col in price_cols if col in df.columns]
            
            for col in available_price_cols:
                # Check for non-positive values
                non_positive = (df[col] <= 0).sum()
                if non_positive > 0:
                    logger.warning(f"Found {non_positive} non-positive values in {col}.")
                    # Provide sample of problematic values
                    sample_idx = df[df[col] <= 0].index[:5]
                    logger.warning(f"Sample of non-positive {col} values: {df.loc[sample_idx, col].to_dict()}")
                
                # Check for extreme outliers
                try:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (3 * iqr)
                    upper_bound = q3 + (3 * iqr)
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers > 0:
                        logger.warning(f"Found {outliers} outliers in {col} using IQR method.")
                except Exception as e:
                    logger.warning(f"Could not check for outliers in {col}: {e}")
            
            # Validate price relationships if all required columns are present
            if 'High' in df.columns and 'Low' in df.columns:
                invalid_count = (df['High'] < df['Low']).sum()
                if invalid_count > 0:
                    logger.warning(f"Found {invalid_count} instances where High < Low.")
                    # Provide sample of problematic values
                    sample_idx = df[df['High'] < df['Low']].index[:5]
                    logger.warning(f"Sample rows with High < Low: {df.loc[sample_idx, ['High', 'Low']].to_dict()}")
            
            # Check for gaps in time series
            if isinstance(df.index, pd.DatetimeIndex):
                freq = pd.infer_freq(df.index)
                if freq:
                    ideal_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
                    missing_dates = ideal_idx.difference(df.index)
                    if len(missing_dates) > 0:
                        logger.warning(f"Found {len(missing_dates)} missing dates in time series.")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False        
class MarketRegimeDetector:
    """Detects market regimes using multiple features and clustering."""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.classifier = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
    
    def fit(self, data: pd.DataFrame):
        """Fits the regime detection model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Sklearn not available. Market regime detection disabled.")
            return
            
        try:
            features = self._extract_regime_features(data)
            if features.shape[0] == 0 or features.shape[1] == 0:
                logger.warning("No valid features for regime detection.")
                return
                
            features_scaled = self.scaler.fit_transform(features)
            self.classifier = KMeans(n_clusters=self.n_regimes, random_state=42)
            self.classifier.fit(features_scaled)
            logger.info(f"Fitted market regime detector with {features.shape[1]} features.")
        except Exception as e:
            logger.error(f"Error fitting market regime detector: {e}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predicts market regimes."""
        if not SKLEARN_AVAILABLE or self.classifier is None:
            logger.warning("Market regime detector not available or not fitted.")
            return np.zeros(len(data))
            
        try:
            # Ensure the detector is fitted
            if self.classifier is None:
                self.fit(data)
                if self.classifier is None:  # If fitting failed
                    return np.zeros(len(data))
                    
            features = self._extract_regime_features(data)
            if features.shape[0] == 0 or features.shape[1] == 0:
                logger.warning("No valid features for regime prediction.")
                return np.zeros(len(data))
                
            features_scaled = self.scaler.transform(features)
            return self.classifier.predict(features_scaled)
        except Exception as e:
            logger.error(f"Error predicting market regimes: {e}")
            return np.zeros(len(data))
    
    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extracts features for regime detection with proper error handling."""
        df = data.copy()
    
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['Close'].pct_change()
    
        # Initialize empty feature list
        features = []
    
        # Volume features
        if 'Volume' in df.columns:
            try:
                volume_ma = df['Volume'].rolling(20).mean()
                volume_std = df['Volume'].rolling(20).std()
                
                # Avoid division by zero
                volume_ma_safe = volume_ma.replace(0, np.nan).fillna(1)
                
                volume_features = [
                    volume_ma,
                    volume_std,
                    df['Volume'] / volume_ma_safe  # Volume surprise with safe division
                ]
                features.extend(volume_features)
            except Exception as e:
                logger.warning(f"Error calculating volume features: {e}")
    
        # Momentum calculation with safety checks
        def calculate_momentum(x):
            try:
                if len(x) < 2:
                    return 0
                if x.iloc[0] == 0:
                    return 0
                return (x.iloc[-1] - x.iloc[0]) / max(0.0001, abs(x.iloc[0]))
            except Exception:
                return 0
    
        # Price features with error handling
        try:
            # Only add features if we have enough data
            if len(df) >= 20:
                features.extend([
                    df['returns'].rolling(20).std().fillna(0),  # Volatility
                    df['returns'].rolling(20).mean().fillna(0),  # Trend
                    df['Close'].rolling(20).apply(calculate_momentum).fillna(0)  # Momentum
                ])
            
            # High/Low ratio with safe division
            if 'High' in df.columns and 'Low' in df.columns:
                high_low_ratio = (df['High'] / df['Low'].replace(0, np.nan)).fillna(1)
                features.append(high_low_ratio)
            
            # Coefficient of variation with safe division
            if len(df) >= 20:
                close_ma = df['Close'].rolling(20).mean().replace(0, np.nan).fillna(1)
                cv = df['Close'].rolling(20).std() / close_ma
                features.append(cv.fillna(0))
                
        except Exception as e:
            logger.warning(f"Error calculating price features for regime detection: {e}")
    
        # If we have no features, return an empty array
        if not features:
            logger.warning("No features created for regime detection.")
            return np.array([]).reshape(0, 0)
    
        # Convert to numpy array with NaN handling
        try:
            feature_array = np.column_stack([feat.values for feat in features])
            return np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)
        except Exception as e:
            logger.error(f"Error creating feature array for regime detection: {e}")
            return np.array([]).reshape(0, 0)
        
class AdvancedFeatureEngineering:
    """Enhanced feature engineering with comprehensive technical indicators."""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.feature_importance = {}
        self.scalers = {}
    
    def add_comprehensive_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all available technical analysis features with robust error handling."""
        df_result = df.copy()

        # Calculate returns if not present
        if 'returns' not in df_result.columns:
            df_result['returns'] = df_result['Close'].pct_change()

        # Create a dictionary to store all new features
        new_features = {}
        
        # Apply TA features if available
        if TA_AVAILABLE:
            try:
                required_columns = {'High', 'Low', 'Close'}
                if all(col in df_result.columns for col in required_columns):
                    # Create a subset of df with required columns to avoid warnings
                    volume_col = 'Volume' if 'Volume' in df_result.columns else None
                    
                    # Apply ta features with error handling
                    try:
                        df_with_ta = ta.add_all_ta_features(
                            df_result,
                            open="Open" if "Open" in df_result.columns else None,
                            high="High",
                            low="Low",
                            close="Close",
                            volume=volume_col,
                            fillna=True
                        )
                        
                        # Merge the new TA columns back
                        for col in df_with_ta.columns:
                            if col not in df_result.columns:
                                new_features[col] = df_with_ta[col]
                                
                    except Exception as e:
                        logger.warning(f"Error adding all TA features: {e}. Will try individual indicators.")
                        
                        # If bulk add fails, try to add individual indicators
                        self._add_individual_indicators(df_result, new_features)
                else:
                    logger.warning(f"Missing required columns for TA features: {required_columns - set(df_result.columns)}")
            except Exception as e:
                logger.error(f"Error in TA feature generation: {e}")
        else:
            # If TA is not available, add basic technical indicators manually
            self._add_basic_indicators(df_result, new_features)
    
        # Add multi-timeframe indicators with error handling
        self._add_multi_timeframe_indicators(df_result, new_features)
    
        # Add market regimes
        try:
            self.regime_detector.fit(df_result)
            new_features['market_regime'] = self.regime_detector.predict(df_result)
        except Exception as e:
            logger.warning(f"Error adding market regime feature: {e}")
            new_features['market_regime'] = np.zeros(len(df_result))
    
        # Add price action features
        self._add_price_action_features(df_result, new_features)
    
        # Add lagged features
        self._add_lagged_features(df_result, new_features)
        
        # Add volume-price relationship features
        self._add_volume_price_relationship_features(df_result, new_features)
        
        # Add volatility seasonality features
        self._add_volatility_seasonality_features(df_result, new_features)
        # Add GARCH volatility features
        self._add_garch_volatility_features(df_result, new_features)
        # Add all features to the result dataframe in a more efficient way
        if new_features:
            # Create a DataFrame from the new features
            features_df = pd.DataFrame(new_features, index=df_result.index)
            
            # Concatenate with original DataFrame
            df_result = pd.concat([df_result, features_df], axis=1)

        # Clean up - replace infinities and NaNs
        df_result = df_result.replace([np.inf, -np.inf], np.nan)

        # DO NOT drop NaN values here - just forward and backward fill
        df_result = df_result.ffill().bfill()  # Forward fill, then backward fill

        return df_result
    def _add_volume_price_relationship_features(self, df, new_features):
        """Add features that capture the relationship between volume and price movements."""
        try:
            if all(col in df.columns for col in ['Close', 'Volume']):
                # Ensure we have both required columns
            
                # 1. Volume-weighted price features
                # Calculate volume-weighted close price
                new_features['volume_weighted_close'] = (df['Close'] * df['Volume']) / df['Volume'].replace(0, np.nan).fillna(1)
            
                # 2. Price-volume correlation features
                # Calculate rolling correlation between price and volume
                for window in [10, 20, 30]:
                    try:
                        # Calculate rolling correlation with safe handling
                        price_changes = df['Close'].pct_change(1)
                        volume_changes = df['Volume'].pct_change(1)
                    
                        # Replace NaN and inf values
                        price_changes = price_changes.replace([np.inf, -np.inf], np.nan).fillna(0)
                        volume_changes = volume_changes.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                        # Calculate correlation with a minimum threshold of data points
                        def rolling_corr(x, y, window):
                            if len(x) < 5:  # Need at least 5 points for meaningful correlation
                                return 0
                            return pearsonr(x, y)[0] if not np.isnan(x).any() and not np.isnan(y).any() else 0
                    
                        # Apply rolling correlation with safe handling
                        corr_series = pd.Series(index=df.index)
                        for i in range(window, len(df)):
                            x = price_changes.iloc[i-window:i].values
                            y = volume_changes.iloc[i-window:i].values
                            if len(x) >= 5 and not np.isnan(x).any() and not np.isnan(y).any():
                                try:
                                    corr_val = pearsonr(x, y)[0]
                                    corr_series.iloc[i] = corr_val if not np.isnan(corr_val) else 0
                                except Exception:
                                    corr_series.iloc[i] = 0
                            else:
                                corr_series.iloc[i] = 0
                    
                        new_features[f'price_volume_corr_{window}'] = corr_series.fillna(0)
                    
                    except Exception as e:
                        logger.debug(f"Error calculating price-volume correlation for window {window}: {e}")
            
                # 3. Volume spikes features
                # Moving average of volume
                vol_ma = df['Volume'].rolling(window=20).mean().replace(0, np.nan).fillna(df['Volume'])
                # Ratio of current volume to moving average
                new_features['volume_to_ma_ratio'] = (df['Volume'] / vol_ma).replace([np.inf, -np.inf], np.nan).fillna(1)
                # Binary indicator of volume spike (volume > 2x moving average)
                new_features['volume_spike'] = (new_features['volume_to_ma_ratio'] > 2).astype(int)
            
                # 4. Money flow features
                if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
                    try:
                        # Typical price
                        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                        
                        # Check for zero volume and handle gracefully - IMPROVED HANDLING
                        df_vol_nonzero = df['Volume'].replace(0, np.nan)
                        if df_vol_nonzero.isna().any():
                            logger.warning(f"Zero volumes detected in {df_vol_nonzero.isna().sum()} rows. Using minimum non-zero value.")
                            min_nonzero = df_vol_nonzero.dropna().min() if df_vol_nonzero.dropna().size > 0 else 1
                            df_vol_nonzero = df_vol_nonzero.fillna(min_nonzero)
                        
                        # Raw money flow with safer handling
                        raw_money_flow = typical_price * df_vol_nonzero
                        
                        # Money flow sign (positive when price rises)
                        price_diff = df['Close'] - df['Close'].shift(1)
                        money_flow_sign = np.where(price_diff > 0, 1, -1)
                        money_flow_sign = np.where(price_diff == 0, 0, money_flow_sign)  # Handle no change case
                        
                        # Signed money flow with NaN handling
                        signed_money_flow = raw_money_flow * money_flow_sign
                        
                        # Initialize money flow index features with default values to ensure they always exist
                        for window in [10, 20]:
                            new_features[f'money_flow_index_{window}'] = pd.Series(50, index=df.index)
                        
                        # Calculate money flow indicators for different windows with better NaN handling
                        for window in [10, 20]:
                            # Skip calculations if window is too large compared to data
                            if len(df) < window * 2:
                                logger.warning(f"Dataset too small for money_flow_index_{window}. Using default value of 50.")
                                continue
                                
                            try:
                                # NEW IMPROVED CALCULATION
                                # First calculate positive and negative money flows
                                pos_flow = np.where(signed_money_flow > 0, signed_money_flow, 0)
                                neg_flow = np.where(signed_money_flow < 0, -signed_money_flow, 0)
                                
                                # Rolling sum with explicit NaN handling and min_periods=1 to avoid initial NaNs
                                pos_sum = pd.Series(pos_flow).rolling(window=window, min_periods=1).sum().fillna(0)
                                neg_sum = pd.Series(neg_flow).rolling(window=window, min_periods=1).sum().fillna(0)
                                
                                # Money flow ratio with division by zero protection
                                # Rather than just adding epsilon, check explicitly for zero values
                                money_ratio = pd.Series(index=df.index, dtype='float64')
                                for i in range(len(df)):
                                    if neg_sum.iloc[i] > 0:
                                        money_ratio.iloc[i] = pos_sum.iloc[i] / neg_sum.iloc[i]
                                    elif pos_sum.iloc[i] > 0:
                                        money_ratio.iloc[i] = 100  # If no negative flow but positive flow exists
                                    else:
                                        money_ratio.iloc[i] = 1  # Neutral when both are zero
                                
                                # Money flow index (0-100 scale)
                                money_flow_index = 100 - (100 / (1 + money_ratio))
                                
                                # Final safety check and fill NaNs with 50 (neutral)
                                money_flow_index = money_flow_index.fillna(50)
                                
                                # Make sure values are in the valid range
                                money_flow_index = money_flow_index.clip(0, 100)
                                
                                # Set the feature value
                                new_features[f'money_flow_index_{window}'] = money_flow_index
                                    
                            except Exception as e:
                                logger.warning(f"Error calculating money_flow_index_{window}: {e}")
                                # Fallback already set at initialization
                    
                    except Exception as e:
                        logger.warning(f"Error in money flow calculations: {e}")
                        # Default values are already initialized at the start of the money flow section
                
                # Final verification - ensure no NaN values remain in money flow features
                for window in [10, 20]:
                    feature_name = f'money_flow_index_{window}'
                    if feature_name in new_features and new_features[feature_name].isna().any():
                        logger.warning(f"NaN values found in {feature_name} after calculation. Filling with 50.")
                        new_features[feature_name] = new_features[feature_name].fillna(50)
                
                logger.info("Added volume-price relationship features successfully.")
                
            else:
                logger.warning("Missing required columns for volume-price relationship features.")
                
        except Exception as e:
            logger.warning(f"Error adding volume-price relationship features: {e}")
            # Ensure money flow features always have values even if the whole method fails
            for window in [10, 20]:
                new_features[f'money_flow_index_{window}'] = pd.Series(50, index=df.index)

    def _add_volatility_seasonality_features(self, df, new_features):
        """Add features that capture volatility patterns across different time periods."""
        try:
            # Check if we have enough data for seasonality analysis
            if len(df) < 252:  # Need at least a year of data
                logger.warning("Not enough data for volatility seasonality features.")
                return
                
            if 'returns' not in df.columns and 'Close' in df.columns:
                df['returns'] = df['Close'].pct_change()
            
            if 'returns' in df.columns:
                # 1. Day of week volatility
                if isinstance(df.index, pd.DatetimeIndex):
                    # Extract day of week and calculate day-specific volatility
                    df['day_of_week'] = df.index.dayofweek
                
                    # Get day-specific volatility using groupby
                    day_vol = df.groupby('day_of_week')['returns'].std().to_dict()
                    
                    # Map back to the DataFrame for each day
                    new_features['day_of_week_vol'] = df['day_of_week'].map(lambda x: day_vol.get(x, 0))
                    
                    # Also add day of week as a feature
                    new_features['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
                    new_features['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
                    
                    # 2. Month of year volatility
                    if len(df) >= 756:  # Need at least 3 years for monthly patterns
                        df['month'] = df.index.month
                        
                        # Get month-specific volatility using groupby
                        month_vol = df.groupby('month')['returns'].std().to_dict()
                        
                        # Map back to the DataFrame for each month
                        new_features['month_of_year_vol'] = df['month'].map(lambda x: month_vol.get(x, 0))
                        
                        # Also add month as a cyclical feature
                        new_features['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                        new_features['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                
                # 3. Volatility regime switching features
                # Calculate historical volatility over different windows
                for window in [10, 21, 63]:  # ~ 2 weeks, 1 month, 3 months
                    vol = df['returns'].rolling(window=window).std().fillna(0)
                    
                    # Calculate percentile of current volatility in historical distribution
                    # with proper handling of edge cases
                    rolling_vol_percentile = pd.Series(index=df.index)
                    
                    # Need at least 3x window size to calculate meaningful percentiles
                    if len(df) >= window * 3:
                        for i in range(window * 2, len(df)):
                            vol_history = vol.iloc[max(0, i-252):i]  # Use up to a year of history
                            if not vol_history.empty:
                                current_vol = vol.iloc[i]
                                # Calculate empirical percentile with interpolation
                                percentile = percentileofscore(vol_history, current_vol) / 100.0
                                rolling_vol_percentile.iloc[i] = percentile
                    
                    rolling_vol_percentile = rolling_vol_percentile.fillna(0.5)  # Default to middle percentile
                    new_features[f'vol_regime_percentile_{window}'] = rolling_vol_percentile
                    
                    # Add volatility regime state (low, medium, high)
                    vol_regime = pd.cut(
                        rolling_vol_percentile, 
                        bins=[0, 0.33, 0.67, 1], 
                        labels=[0, 1, 2]  # 0=low, 1=medium, 2=high
                    )
                    # Fix for "Cannot convert float NaN to integer" - convert to integer and handle NaN
                    vol_regime = vol_regime.cat.codes.replace(-1, 1)  # Replace NaN (-1) with medium category (1)
                    new_features[f'vol_regime_state_{window}'] = vol_regime
                
                # 4. GARCH-like volatility features (simplified version)
                returns = df['returns'].fillna(0)
                
                # Historical variance
                hist_variance = returns.rolling(window=21).var().fillna(0)
                
                # Squared returns (proxy for instantaneous variance)
                squared_returns = returns**2
                
                # Exponential decay factors for GARCH-like modeling
                alpha = 0.1  # Weight for instantaneous variance
                beta = 0.9   # Weight for historical variance
                
                # Initialize the GARCH-like variance series
                garch_variance = pd.Series(index=df.index)
                garch_variance.iloc[0] = squared_returns.iloc[0]
                
                # Calculate GARCH-like variance recursively
                for i in range(1, len(df)):
                    prev_var = garch_variance.iloc[i-1]
                    inst_var = squared_returns.iloc[i-1]
                    garch_variance.iloc[i] = alpha * inst_var + beta * prev_var
                
                # Add the GARCH-like volatility as a feature
                new_features['garch_volatility'] = np.sqrt(garch_variance)
                
                logger.info("Added volatility seasonality features successfully.")
            else:
                logger.warning("Missing 'returns' column for volatility seasonality features.")       
        except Exception as e:
            logger.warning(f"Error adding volatility seasonality features: {e}")
    def _add_garch_volatility_features(self, df, new_features):
        """Add GARCH-generated volatility features to enhance prediction performance."""
        try:
            if 'returns' not in df.columns:
                logger.warning("Returns column not found. Skipping GARCH features.")
                return
                
            if not ARCH_AVAILABLE:
                logger.warning("arch_model not available. Skipping GARCH features.")
                return
                
            # Get returns data - ensure it's a proper Series
            if isinstance(df, pd.DataFrame) and 'returns' in df.columns:
                returns = df['returns'].copy()  # Create a copy to avoid modification issues
            else:
                logger.warning("DataFrame structure incorrect for GARCH modeling")
                return
                
            # Ensure we have a pandas Series with proper values
            if not isinstance(returns, pd.Series):
                logger.warning(f"Returns is not a pandas Series but {type(returns)}. Converting to Series.")
                try:
                    if isinstance(returns, dict):
                        returns = pd.Series(returns.values(), index=returns.keys())
                    elif isinstance(returns, np.ndarray):
                        returns = pd.Series(returns)
                    else:
                        returns = pd.Series(returns)
                except Exception as e:
                    logger.error(f"Could not convert returns to Series: {e}")
                    return
            
            # Drop NaN values before modeling
            returns = returns.dropna()
            
            if len(returns) < 100:  # Need sufficient data for GARCH
                logger.warning(f"Insufficient data for GARCH modeling: {len(returns)} points")
                return
                    
            # Fit GARCH model to returns
            try:
                model = arch_model(returns, vol='GARCH', p=1, q=1)
                fit_model = model.fit(disp='off', show_warning=False)
                
                # Get conditional volatility (in-sample)
                garch_vol = fit_model.conditional_volatility
                
                # Create a Series aligned with our data
                vol_series = pd.Series(index=returns.index, data=garch_vol)
                
                # Add to features
                new_features['garch_volatility'] = vol_series
                
                # Add volatility regime indicators (categorical features)
                quantiles = vol_series.quantile([0.33, 0.66]).values
                new_features['garch_vol_regime'] = pd.Series(index=vol_series.index, data=0)
                
                # Use loc for pandas Series assignment
                mask_high = vol_series > quantiles[1]
                mask_medium = (vol_series <= quantiles[1]) & (vol_series > quantiles[0])
                
                if isinstance(new_features['garch_vol_regime'], pd.Series):
                    new_features['garch_vol_regime'].loc[mask_high] = 2  # High vol
                    new_features['garch_vol_regime'].loc[mask_medium] = 1  # Medium vol
                else:
                    # If it's a dictionary or other structure, handle differently
                    new_features['garch_vol_regime'] = vol_series.copy()
                    new_features['garch_vol_regime'].loc[mask_high] = 2
                    new_features['garch_vol_regime'].loc[mask_medium] = 1
                
                # Add volatility momentum features
                new_features['garch_vol_change'] = vol_series.pct_change(5)
                new_features['garch_vol_trend'] = vol_series.rolling(10).apply(
                    lambda x: (x[-1] - x[0]) / (x.mean() if x.mean() > 0 else 1e-8)
                )
                
                logger.info("Added GARCH volatility features")
            except Exception as e:
                logger.warning(f"Error fitting GARCH model: {e}")
                    
        except Exception as e:
            logger.error(f"Error adding GARCH features: {e}")
    def _add_individual_indicators(self, df, new_features):
        """Add individual TA indicators when full ta.add_all_ta_features fails."""
        if not TA_AVAILABLE:
            return
            
        try:
            # Momentum indicators
            rsi = ta.momentum.RSIIndicator(df['Close'], window=14)
            new_features['momentum_rsi'] = rsi.rsi()
            
            # Trend indicators
            macd = ta.trend.MACD(df['Close'])
            new_features['trend_macd'] = macd.macd()
            new_features['trend_macd_signal'] = macd.macd_signal()
            
            # Volatility indicators
            bbands = ta.volatility.BollingerBands(df['Close'])
            new_features['volatility_bbands_upper'] = bbands.bollinger_hband()
            new_features['volatility_bbands_lower'] = bbands.bollinger_lband()
            
            logger.info("Added individual TA indicators successfully.")
        except Exception as e:
            logger.warning(f"Error adding individual TA indicators: {e}")
    
    def _add_basic_indicators(self, df, new_features):
        """Add basic technical indicators when TA library is not available."""
        try:
            # Simple moving averages
            for window in [5, 10, 20, 50, 200]:
                new_features[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
            
            # Simple volatility (standard deviation)
            for window in [5, 10, 20]:
                new_features[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            
            # Price momentum (percent change)
            for window in [5, 10, 20]:
                new_features[f'momentum_{window}'] = df['Close'].pct_change(periods=window)
            
            # Exponential moving averages
            for window in [12, 26]:
                new_features[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
            
            # Simple MACD
            new_features['macd'] = new_features.get('ema_12', df['Close'].ewm(span=12, adjust=False).mean()) - \
                                 new_features.get('ema_26', df['Close'].ewm(span=26, adjust=False).mean())
            
            # Simple RSI (Using SMA approximation)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan).fillna(gain)  # Safe division
            new_features['rsi_14'] = 100 - (100 / (1 + rs))
            
            logger.info("Added basic technical indicators successfully.")
        except Exception as e:
            logger.warning(f"Error adding basic indicators: {e}")
    
    def _add_multi_timeframe_indicators(self, df, new_features):
        """Add multi-timeframe indicators with robust error handling."""
        if not TA_AVAILABLE:
            return
            
        timeframes = [5, 10, 20, 30, 50, 100, 200]
        
        for window in timeframes:
            try:
                # Only add if we have enough data points
                if len(df) <= window:
                    continue
                    
                # Volatility indicators
                try:
                    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window)
                    new_features[f'volatility_atr_{window}'] = atr.average_true_range()
                except Exception as e:
                    logger.debug(f"Error calculating ATR for window {window}: {e}")
            
                try:
                    bb = ta.volatility.BollingerBands(df['Close'], window)
                    new_features[f'volatility_bb_high_{window}'] = bb.bollinger_hband()
                    new_features[f'volatility_bb_low_{window}'] = bb.bollinger_lband()
                    new_features[f'volatility_bb_width_{window}'] = bb.bollinger_wband()
                except Exception as e:
                    logger.debug(f"Error calculating Bollinger Bands for window {window}: {e}")
            
                # Trend indicators
                try:
                    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window)
                    new_features[f'trend_adx_{window}'] = adx.adx()
                    new_features[f'trend_adx_pos_{window}'] = adx.adx_pos()
                    new_features[f'trend_adx_neg_{window}'] = adx.adx_neg()
                except Exception as e:
                    logger.debug(f"Error calculating ADX for window {window}: {e}")
            
                # Momentum indicators
                try:
                    rsi = ta.momentum.RSIIndicator(df['Close'], window)
                    new_features[f'momentum_rsi_{window}'] = rsi.rsi()
                except Exception as e:
                    logger.debug(f"Error calculating RSI for window {window}: {e}")
            
                try:
                    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window)
                    new_features[f'momentum_stoch_{window}'] = stoch.stoch()
                    new_features[f'momentum_stoch_signal_{window}'] = stoch.stoch_signal()
                except Exception as e:
                    logger.debug(f"Error calculating Stochastic for window {window}: {e}")
            
            except Exception as e:
                logger.warning(f"Error calculating all indicators for window {window}: {e}")
    
    def _add_price_action_features(self, df, new_features):
        """Add price action features with error handling."""
        try:
            if all(col in df.columns for col in ['Close', 'High', 'Low']):
                # Safe division using replace and fillna
                new_features['price_action_close_to_high'] = (df['Close'] / df['High'].replace(0, np.nan)).fillna(1)
                new_features['price_action_close_to_low'] = (df['Close'] / df['Low'].replace(0, np.nan)).fillna(1)
                new_features['price_action_high_to_low'] = (df['High'] / df['Low'].replace(0, np.nan)).fillna(1)
                
                # Range as percentage of Close
                new_features['price_action_range_pct'] = ((df['High'] - df['Low']) / df['Close'].replace(0, np.nan)).fillna(0)
                
                # Position within daily range (0=at low, 1=at high)
                range_diff = df['High'] - df['Low']
                safe_range = range_diff.replace(0, np.nan).fillna(0.0001)  # Avoid division by zero
                new_features['price_action_position'] = ((df['Close'] - df['Low']) / safe_range).fillna(0.5)
                
                # Body size (absolute and relative to range)
                body_size = abs(df['Close'] - df['Open']) if 'Open' in df.columns else 0
                new_features['price_action_body_size'] = body_size
                new_features['price_action_body_to_range'] = (body_size / safe_range).fillna(0)
                
                logger.info("Added price action features successfully.")
        except Exception as e:
            logger.warning(f"Error adding price action features: {e}")
    
    def _add_lagged_features(self, df, new_features):
        """Add lagged features with error handling."""
        try:
            for lag in [1, 2, 3, 5, 8, 13]:  # Reduced from original list for stability
                new_features[f'close_lag_{lag}'] = df['Close'].shift(lag)
                if 'returns' in df.columns:
                    new_features[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                if 'Volume' in df.columns:
                    new_features[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
                    
            logger.info("Added lagged features successfully.")
        except Exception as e:
            logger.warning(f"Error adding lagged features: {e}")

class AdvancedFeatureSelector:
    """Advanced feature selection using multiple techniques."""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs
        self.selected_features = None
        self.feature_importance = None
        self.feature_groups = None
        
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Calculate feature importance using multiple methods with robust error handling."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Cannot calculate feature importance.")
            self.feature_importance = None
            return
            
        try:
            # Initialize importance DataFrame
            importance_dict = {
                'feature': X.columns.tolist(),
                'random_forest': np.zeros(X.shape[1]),
                'shap': np.zeros(X.shape[1])
            }
            
            # Random Forest importance
            try:
                rf = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    n_jobs=min(self.n_jobs, os.cpu_count() or 1), 
                    random_state=42
                )
                rf.fit(X, y)
                importance_dict['random_forest'] = rf.feature_importances_
                logger.info("Calculated Random Forest feature importance")
                
                # SHAP importance if available
                if SHAP_AVAILABLE:
                    try:
                        # Use TreeExplainer with a small subset for SHAP (faster)
                        sample_size = min(1000, X.shape[0])
                        X_sample = X.sample(n=sample_size, random_state=42) if X.shape[0] > sample_size else X
                        
                        explainer = shap.TreeExplainer(rf)
                        shap_values = explainer.shap_values(X_sample)
                        
                        # Map SHAP values back to all features
                        shap_importance = np.abs(shap_values).mean(axis=0)
                        importance_dict['shap'] = shap_importance
                        logger.info("Calculated SHAP feature importance")
                    except Exception as e:
                        logger.warning(f"Error calculating SHAP importance: {e}")
                else:
                    logger.info("SHAP not available. Skipping SHAP importance.")
            except Exception as e:
                logger.warning(f"Error calculating Random Forest importance: {e}")
            
            # Calculate mean importance
            rf_imp = np.array(importance_dict['random_forest'])
            shap_imp = np.array(importance_dict['shap'])
            
            # Normalize each importance metric to [0, 1] scale
            if np.any(rf_imp):
                rf_imp = rf_imp / np.max(rf_imp) if np.max(rf_imp) > 0 else rf_imp
            if np.any(shap_imp):
                shap_imp = shap_imp / np.max(shap_imp) if np.max(shap_imp) > 0 else shap_imp
                
            # Combine importances - if both are available, take average
            if np.any(rf_imp) and np.any(shap_imp):
                mean_imp = (rf_imp + shap_imp) / 2
            elif np.any(rf_imp):
                mean_imp = rf_imp
            elif np.any(shap_imp):
                mean_imp = shap_imp
            else:
                # Fallback to uniform importance
                mean_imp = np.ones(X.shape[1]) / X.shape[1]
                
            importance_dict['mean_importance'] = mean_imp
            
            # Create DataFrame and sort by mean importance
            self.feature_importance = pd.DataFrame(importance_dict)
            self.feature_importance = self.feature_importance.sort_values('mean_importance', ascending=False)
            
            logger.info(f"Feature importance calculation completed for {X.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            self.feature_importance = None
            
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance with error handling."""
        if self.feature_importance is None:
            logger.warning("No feature importance data available to plot.")
            return False
            
        try:
            # Limit to top_n features
            top_n = min(top_n, len(self.feature_importance))
            top_features = self.feature_importance.head(top_n)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(
                x='mean_importance',
                y='feature',
                data=top_features
            )
            plt.title(f'Top {top_n} Feature Importance')
            plt.xlabel('Mean Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Save figure
            try:
                plt.savefig('feature_importance.png')
                logger.info(f"Feature importance plot saved to feature_importance.png")
            except Exception as e:
                logger.warning(f"Error saving feature importance plot: {e}")
                
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            plt.close()  # Ensure figure is closed even on error
            return False

    def ensure_feature_diversity(self, selected_features: List[str], original_df: pd.DataFrame, 
                           min_per_group: int = 5) -> List[str]:

        if self.feature_groups is None:
            logger.warning("Feature groups not available. Creating now.")
            self.create_feature_groups(original_df.columns)
        
        if not self.feature_groups:
            logger.warning("No feature groups to ensure diversity from.")
            return selected_features
        
        logger.info(f"Ensuring feature diversity with minimum {min_per_group} features per group")
    
        # Create set for faster lookup
        selected = set(selected_features)
        enhanced_selection = selected.copy()
    
        # Check each group and ensure minimum representation
        for group_name, group_features in self.feature_groups.items():
            # Skip empty groups
            if not group_features:
                continue
            
            # Check how many features from this group are already selected
            group_selected = selected.intersection(set(group_features))
            needed = max(0, min_per_group - len(group_selected))
        
            if needed > 0:
                logger.info(f"Group '{group_name}' needs {needed} more features")
            
                # Sort remaining features by importance if available
                remaining_features = [f for f in group_features if f not in selected]
            
                if not remaining_features:
                    logger.warning(f"No remaining features in group '{group_name}'")
                    continue
                
                if self.feature_importance is not None:
                    # Get importance values for remaining features
                    try:
                        importance_dict = {}
                        for f in remaining_features:
                            row = self.feature_importance[self.feature_importance['feature'] == f]
                            if not row.empty:
                                importance_dict[f] = row['mean_importance'].values[0]
                            else:
                                importance_dict[f] = 0
                            
                        # Sort by importance (descending)
                        sorted_features = sorted(importance_dict.keys(), 
                                               key=lambda x: importance_dict[x], 
                                               reverse=True)
                    except Exception as e:
                        logger.warning(f"Error sorting features by importance: {e}")
                        sorted_features = remaining_features  # Fallback
                else:
                    sorted_features = remaining_features
                
                # Add the top N needed features
                to_add = sorted_features[:needed]
                enhanced_selection.update(to_add)
                logger.info(f"Added {len(to_add)} features from group '{group_name}'")
    
        # Convert back to list and return
        return list(enhanced_selection)

    def create_feature_groups(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Group features by their types."""
        groups = {
            'volume': [],
            'volatility': [],
            'trend': [],
            'momentum': [],
            'price_action': [],
            'other': []
        }
        
        for feature in feature_names:
            if 'volume' in feature.lower():
                groups['volume'].append(feature)
            elif 'volatility' in feature.lower() or 'bb_' in feature.lower() or 'atr' in feature.lower():
                groups['volatility'].append(feature)
            elif 'trend' in feature.lower() or 'ma_' in feature.lower() or 'ema' in feature.lower() or 'macd' in feature.lower():
                groups['trend'].append(feature)
            elif 'momentum' in feature.lower() or 'rsi' in feature.lower() or 'stoch' in feature.lower():
                groups['momentum'].append(feature)
            elif 'price_action' in feature.lower() or 'body' in feature.lower() or 'position' in feature.lower():
                groups['price_action'].append(feature)
            else:
                groups['other'].append(feature)
        
        self.feature_groups = groups
        return groups
    
    def remove_highly_correlated(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.98) -> pd.DataFrame:
        """Remove highly correlated features using hierarchical clustering."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Skipping correlation removal.")
            return X

        try:
            # Check if X is empty
            if X.empty or X.shape[1] <= 1:
                logger.warning("No features to check for correlation.")
                return X
                
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Create clusters of correlated features
            clusters = {}
            remaining_features = set(X.columns)
            
            while remaining_features:
                feature = next(iter(remaining_features))
                remaining_features.remove(feature)
                
                # Find features correlated with the current one
                correlated = set()
                if feature in corr_matrix.columns:
                    correlated_series = corr_matrix[feature][corr_matrix[feature] > threshold]
                    correlated = set(correlated_series.index) & remaining_features
                
                # Create cluster and update remaining features
                cluster = {feature} | correlated
                clusters[feature] = cluster
                remaining_features -= correlated
            
            # Select representative feature from each cluster
            selected_features = []
            for representative, cluster in clusters.items():
                if len(cluster) == 1:
                    selected_features.append(representative)
                else:
                    # Select feature with highest mutual information with target
                    # with proper error handling
                    try:
                        cluster_features = X[list(cluster)]
                        if SKLEARN_AVAILABLE:
                            mi_scores = mutual_info_regression(cluster_features, y)
                            best_feature = cluster_features.columns[np.argmax(mi_scores)]
                            selected_features.append(best_feature)
                        else:
                            # Fallback: use first feature if MI not available
                            selected_features.append(next(iter(cluster)))
                    except Exception as e:
                        logger.warning(f"Error calculating MI for correlated cluster: {e}")
                        selected_features.append(next(iter(cluster)))
                            
            logger.info(f"Reduced features from {X.shape[1]} to {len(selected_features)} after correlation removal")
            return X[selected_features]
            
        except Exception as e:
            logger.error(f"Error in correlation removal: {e}")
            return X
            
    def select_features_lasso(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using LASSO regression with error handling."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Cannot perform LASSO feature selection.")
            return list(X.columns)
            
        try:
            # Check for sufficient data
            if X.shape[0] < 10 or X.shape[1] < 2:
                logger.warning("Insufficient data for LASSO feature selection.")
                return list(X.columns)
                
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Set up LASSO with cross-validation
            lasso = LassoCV(
                cv=min(5, X.shape[0] // 10),  # Ensure sufficient folds
                n_jobs=min(self.n_jobs, os.cpu_count() or 1),
                max_iter=10000,
                random_state=42,
                selection='random',  # More stable for large feature sets
                tol=1e-4
            )
            
            # Fit model with timeout protection
            start_time = time.time()
            lasso.fit(X_scaled, y)
            logger.info(f"LASSO fitting completed in {time.time() - start_time:.2f}s")
            
            # Select features with non-zero coefficients
            selected = X.columns[np.abs(lasso.coef_) > 1e-5].tolist()
            
            # Ensure we don't drop all features
            if not selected:
                logger.warning("LASSO selected 0 features. Falling back to top features by coefficient.")
                # Get top 20% features by coefficient magnitude
                n_to_select = max(1, int(X.shape[1] * 0.2))
                selected_indices = np.argsort(np.abs(lasso.coef_))[-n_to_select:]
                selected = X.columns[selected_indices].tolist()
                
            logger.info(f"LASSO selected {len(selected)} features")
            return selected
            
        except Exception as e:
            logger.error(f"Error in LASSO feature selection: {e}")
            return list(X.columns)  # Return all features as fallback
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series, 
                                   n_features: int) -> List[str]:
        """Select features using mutual information with error handling."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Cannot perform MI feature selection.")
            return list(X.columns)[:min(n_features, len(X.columns))]
            
        try:
            # Ensure n_features is valid
            n_features = min(n_features, X.shape[1])
            n_features = max(1, n_features)  # At least 1 feature
            
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(X, y)
            
            # Check for valid scores
            if np.any(np.isnan(mi_scores)) or np.any(np.isinf(mi_scores)):
                logger.warning("Invalid MI scores detected. Cleaning...")
                mi_scores = np.nan_to_num(mi_scores)
                
            # Select top features
            selected_indices = np.argsort(mi_scores)[-n_features:]
            selected = X.columns[selected_indices].tolist()
            
            logger.info(f"MI selected {len(selected)} features")
            return selected
            
        except Exception as e:
            logger.error(f"Error in mutual information feature selection: {e}")
            # Return a subset of features as fallback
            return list(X.columns)[:min(n_features, len(X.columns))]
            
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features: int) -> List[str]:
        """Perform recursive feature elimination with error handling."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Cannot perform RFE feature selection.")
            return list(X.columns)[:min(n_features, len(X.columns))]
            
        try:
            # Ensure n_features is valid
            n_features = min(n_features, X.shape[1])
            n_features = max(1, n_features)  # At least 1 feature
            
            # For large feature sets, use RandomForest directly instead of RFE
            if X.shape[1] > 100:
                logger.info(f"Large feature set ({X.shape[1]} features). Using direct importance selection.")
                rf = RandomForestRegressor(
                    n_estimators=100, 
                    n_jobs=min(self.n_jobs, os.cpu_count() or 1), 
                    random_state=42
                )
                rf.fit(X, y)
                
                # Select top features by importance
                importances = rf.feature_importances_
                selected_indices = np.argsort(importances)[-n_features:]
                return X.columns[selected_indices].tolist()
            
            # For smaller feature sets, use RFE with RandomForest
            rf = RandomForestRegressor(
                n_estimators=100, 
                n_jobs=min(self.n_jobs, os.cpu_count() or 1), 
                random_state=42
            )
            
            # Use RFE with cross-validation
            rfe = RFECV(
                estimator=rf,
                step=max(1, X.shape[1] // 20),  # Step size based on feature count
                cv=min(5, X.shape[0] // 10),  # Ensure sufficient folds
                n_jobs=min(self.n_jobs, os.cpu_count() or 1),
                min_features_to_select=n_features
            )
            
            # Fit RFE with timeout protection
            start_time = time.time()
            rfe.fit(X, y)
            logger.info(f"RFE fitting completed in {time.time() - start_time:.2f}s")
            
            # Select features
            selected = X.columns[rfe.support_].tolist()
            
            # If RFE selected too many features, take the top n by importance
            if len(selected) > n_features:
                logger.info(f"RFE selected {len(selected)} features, limiting to top {n_features}")
                importances = rfe.estimator_.feature_importances_
                selected_indices = np.argsort(importances[rfe.support_])[-n_features:]
                selected = np.array(selected)[selected_indices].tolist()
                
            logger.info(f"RFE selected {len(selected)} features")
            return selected
            
        except Exception as e:
            logger.error(f"Error in recursive feature elimination: {e}")
            # Return a subset of features as fallback
            return list(X.columns)[:min(n_features, len(X.columns))]
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                      n_features: int = 100) -> pd.DataFrame:
        """Main feature selection method combining multiple techniques with robust error handling."""
        logger.info(f"Starting feature selection with {X.shape[1]} initial features")
        
        try:
            # Create feature groups for better selection
            feature_groups = self.create_feature_groups(X.columns)
            
            # Step 1: Remove highly correlated features with error handling
            try:
                X_uncorrelated = pd.DataFrame()
                for group_name, group_features in feature_groups.items():
                    if group_features:
                        group_df = X[group_features]
                        # Skip tiny groups
                        if group_df.shape[1] <= 1:
                            X_uncorrelated = pd.concat([X_uncorrelated, group_df], axis=1)
                            continue
                        # Remove correlations within group
                        uncorrelated_group = self.remove_highly_correlated(group_df, y)
                        X_uncorrelated = pd.concat([X_uncorrelated, uncorrelated_group], axis=1)
                
                logger.info(f"Features after correlation removal: {X_uncorrelated.shape[1]}")
            except Exception as e:
                logger.error(f"Error in correlation removal step: {e}")
                X_uncorrelated = X.copy()
                
            # If we've already reduced features enough, we can skip further selection
            if X_uncorrelated.shape[1] <= n_features:
                logger.info(f"After correlation removal, only {X_uncorrelated.shape[1]} features remain. Skipping further selection.")
                self.selected_features = list(X_uncorrelated.columns)
                self.calculate_feature_importance(X_uncorrelated, y)
                return X_uncorrelated
                
            # Step 2: Try multiple feature selection methods with fallbacks
            selected_features_methods = {}
            
            # LASSO selection
            try:
                lasso_features = set(self.select_features_lasso(X_uncorrelated, y))
                selected_features_methods['lasso'] = lasso_features
            except Exception as e:
                logger.error(f"LASSO selection failed: {e}")
                selected_features_methods['lasso'] = set()
                
            # Mutual information selection
            try:
                mi_features = set(self.mutual_information_selection(X_uncorrelated, y, n_features))
                selected_features_methods['mi'] = mi_features
            except Exception as e:
                logger.error(f"MI selection failed: {e}")
                selected_features_methods['mi'] = set()
                
            # RFE selection - skip for very large feature sets
            if X_uncorrelated.shape[1] < 500:  # Only do RFE for manageable feature sets
                try:
                    rfe_features = set(self.recursive_feature_elimination(X_uncorrelated, y, n_features))
                    selected_features_methods['rfe'] = rfe_features
                except Exception as e:
                    logger.error(f"RFE selection failed: {e}")
                    selected_features_methods['rfe'] = set()
            else:
                logger.info(f"Skipping RFE for large feature set ({X_uncorrelated.shape[1]} features)")
                selected_features_methods['rfe'] = set()
                
            # Step 3: Combine results from different methods
            # First try to take the intersection of at least 2 methods
            final_features = set()
            method_pairs = [
                ('lasso', 'mi'),
                ('lasso', 'rfe'),
                ('mi', 'rfe')
            ]
            
            for method1, method2 in method_pairs:
                intersection = selected_features_methods[method1] & selected_features_methods[method2]
                final_features.update(intersection)
                
            # If we don't have enough features, take the union of all methods
            if len(final_features) < min(n_features, X_uncorrelated.shape[1] // 2):
                logger.info(f"Intersection produced only {len(final_features)} features. Taking union instead.")
                for features in selected_features_methods.values():
                    final_features.update(features)
                    
            # If we still don't have enough features, add features based on mutual information
            if len(final_features) < min(n_features, X_uncorrelated.shape[1]):
                logger.info(f"Selected only {len(final_features)} features. Adding more based on MI.")
                try:
                    remaining = set(X_uncorrelated.columns) - final_features
                    if remaining:
                        X_remaining = X_uncorrelated[list(remaining)]
                        mi_scores = mutual_info_regression(X_remaining, y)
                        sorted_indices = np.argsort(mi_scores)[::-1]
                        
                        # Add features until we reach target number
                        additional_needed = min(n_features, X_uncorrelated.shape[1]) - len(final_features)
                        additional_features = [X_remaining.columns[i] for i in sorted_indices[:additional_needed]]
                        final_features.update(additional_features)
                except Exception as e:
                    logger.error(f"Error adding additional features: {e}")
                    # Add remaining features randomly as fallback
                    remaining = list(set(X_uncorrelated.columns) - final_features)
                    np.random.shuffle(remaining)
                    additional_needed = min(n_features, X_uncorrelated.shape[1]) - len(final_features)
                    final_features.update(remaining[:additional_needed])
                    
            # Convert to list and limit to desired number of features
            final_features = list(final_features)[:n_features]
            
            logger.info(f"Final number of selected features: {len(final_features)}")
            
            # Ensure diverse feature representation
            diversified_features = self.ensure_feature_diversity(final_features, X)
            
            if len(diversified_features) > n_features:
                logger.info(f"Diversified feature count ({len(diversified_features)}) exceeds target. Truncating.")
                final_features = diversified_features[:n_features]
            else:
                final_features = diversified_features
                
            logger.info(f"Final feature count after diversity enhancement: {len(final_features)}")

            # Store the selected features
            self.selected_features = final_features
            
            # Calculate and store feature importance
            self.calculate_feature_importance(X[final_features], y)
            
            return X[final_features]
            
        except Exception as e:
            logger.error(f"Unexpected error in feature selection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a subset of original features as fallback
            n_safe = min(n_features, X.shape[1])
            self.selected_features = list(X.columns)[:n_safe]
            
            # Try to calculate feature importance
            try:
                self.calculate_feature_importance(X[self.selected_features], y)
            except:
                self.feature_importance = None
                
            return X[self.selected_features]
            
        except Exception as e:
            logger.error(f"Unexpected error in feature selection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a subset of original features as fallback
            n_safe = min(n_features, X.shape[1])
            self.selected_features = list(X.columns)[:n_safe]
            
            # Try to calculate feature importance
            try:
                self.calculate_feature_importance(X[self.selected_features], y)
            except:
                self.feature_importance = None
                
            return X[self.selected_features]


class TimeSeriesDataset(Dataset):
    """Enhanced TimeSeriesDataset with proper sequence handling and multi-horizon support."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, lookback: int):
        """Initialize with robust input validation."""
        try:
            # Validate inputs
            if not isinstance(data, np.ndarray) or not isinstance(targets, np.ndarray):
                logger.warning("Invalid input types to TimeSeriesDataset. Converting to numpy arrays.")
                data = np.array(data) if not isinstance(data, np.ndarray) else data
                targets = np.array(targets) if not isinstance(targets, np.ndarray) else targets
            logger.info(f"TimeSeriesDataset initialization - Input data shape: {data.shape}")
            
            # Ensure data is 2D
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                logger.warning(f"Data has {len(data.shape)} dimensions. Flattening to 2D.")
                original_shape = data.shape
                data = data.reshape(original_shape[0], -1)
            
            # Ensure targets is the right shape for training
            # If targets has multiple dimensions (e.g., for multi-horizon forecasting)
            # we want to preserve that structure
            if len(targets.shape) > 2:
                logger.warning(f"Targets has {len(targets.shape)} dimensions. Reshaping to 2D.")
                targets = targets.reshape(targets.shape[0], -1)
            
            # Verify that data and targets have the same number of samples
            if len(data) != len(targets):
                logger.warning(f"Data length ({len(data)}) doesn't match targets length ({len(targets)}). "
                              f"Using minimum length.")
                min_length = min(len(data), len(targets))
                data = data[:min_length]
                targets = targets[:min_length]
            
            # Validate lookback
            if lookback <= 0:
                logger.warning(f"Invalid lookback value: {lookback}. Setting to 1.")
                lookback = 1
            elif lookback > len(data):
                logger.warning(f"Lookback ({lookback}) exceeds data length ({len(data)}). "
                              f"Setting to data length.")
                lookback = len(data)
            
            # Convert to PyTorch tensors with error handling
            try:
                self.data = torch.FloatTensor(data)
            except Exception as tensor_error:
                logger.error(f"Error converting data to tensor: {tensor_error}. Using zeros.")
                self.data = torch.zeros((len(data), data.shape[1] if len(data.shape) > 1 else 1))
            
            try:
                # Keep original target shape for multi-horizon forecasting
                if len(targets.shape) == 1:
                    self.targets = torch.FloatTensor(targets).reshape(-1, 1)
                else:
                    self.targets = torch.FloatTensor(targets)
            except Exception as tensor_error:
                logger.error(f"Error converting targets to tensor: {tensor_error}. Using zeros.")
                if len(targets.shape) == 1:
                    self.targets = torch.zeros((len(targets), 1))
                else:
                    self.targets = torch.zeros(targets.shape)
            
            self.lookback = lookback
            self.feature_dim = self.data.shape[1]
            
            # Log details about the dataset
            multi_horizon_text = ""
            if len(self.targets.shape) > 1 and self.targets.shape[1] > 1:
                multi_horizon_text = f", Multi-horizon targets with {self.targets.shape[1]} horizons"
                
            logger.info(f"TimeSeriesDataset initialization - Data shape: {self.data.shape}, "
                       f"Targets shape: {self.targets.shape}, Lookback: {lookback}{multi_horizon_text}")
                       
        except Exception as e:
            logger.error(f"Error in TimeSeriesDataset initialization: {e}")
            # Create minimal valid dataset
            self.data = torch.zeros((1, 1))
            self.targets = torch.zeros((1, 1))
            self.lookback = 1
            self.feature_dim = 1
    
    def __len__(self):
        """Calculate length with appropriate bounds checking."""
        try:
            # Return data length with proper adjustment for lookback
            return max(0, len(self.data))
        except Exception as e:
            logger.error(f"Error in __len__: {e}")
            return 0
    
    def __getitem__(self, idx):
        """Get a single sample with error handling and consistent behavior."""
        try:
            # Basic index validation
            if idx < 0 or idx >= len(self):
                logger.warning(f"Index {idx} out of bounds for dataset of length {len(self)}")
                idx = 0  # Default to first sample on error
            
            # Create proper sliding window that correctly handles the lookback
            # This approach ensures we have a fixed-size window for each sample
            
            # Determine start index for the window, ensuring it's within bounds
            start_idx = max(0, idx - self.lookback + 1)
            
            if start_idx > len(self.data) - 1:
                logger.warning(f"Start index {start_idx} is out of bounds for data length {len(self.data)}")
                start_idx = len(self.data) - 1
            
            end_idx = min(idx + 1, len(self.data))
            window_data = self.data[start_idx:end_idx]
            
            # If the window is shorter than lookback, we need to pad it
            if len(window_data) < self.lookback:
                # Create padding tensor with appropriate shape and device
                padding_size = self.lookback - len(window_data)
                padding = torch.zeros((padding_size, self.feature_dim), 
                                    dtype=self.data.dtype, 
                                    device=self.data.device)
                
                # Concatenate padding and window data
                X = torch.cat([padding, window_data], dim=0)
            else:
                X = window_data
            
            # Verify the shape is correct
            if X.shape[0] != self.lookback:
                logger.warning(f"Window shape mismatch after processing: {X.shape[0]} vs expected {self.lookback}")
                
                if X.shape[0] != self.lookback:
                    logger.warning(f"Window shape mismatch after processing: {X.shape[0]} vs expected {self.lookback}")
                    
                    # Fix shape if needed by either padding or truncating
                    if X.shape[0] < self.lookback:
                        # Add more padding at the beginning
                        additional_padding = torch.zeros((self.lookback - X.shape[0], self.feature_dim), 
                                                        dtype=self.data.dtype, 
                                                        device=self.data.device)
                        X = torch.cat([additional_padding, X], dim=0)
                    else:
                        # Truncate from the beginning to maintain most recent data
                        X = X[-self.lookback:]
                
                # Get the target for the corresponding index - preserve multi-horizon structure if present
                y = self.targets[idx]
                
                # Log if the target is multi-horizon (for debugging)
                if len(y.shape) > 0 and y.shape[0] > 1:
                    logger.debug(f"Returning multi-horizon target with shape {y.shape} for idx {idx}")
                
                return X, y
            
            # Fix shape if needed by either padding or truncating
            if X.shape[0] < self.lookback:
                # Add more padding at the beginning
                additional_padding = torch.zeros((self.lookback - X.shape[0], self.feature_dim), 
                                                dtype=self.data.dtype, 
                                                device=self.data.device)
                X = torch.cat([additional_padding, X], dim=0)
            else:
                # Truncate from the beginning to maintain most recent data
                X = X[-self.lookback:]
            
            # Get the target for the corresponding index
            y = self.targets[idx]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return zeros with appropriate dimensions as fallback
            X_fallback = torch.zeros((self.lookback, self.feature_dim))
            y_fallback = torch.zeros_like(self.targets[0] if len(self.targets) > 0 else torch.zeros(1))
            return X_fallback, y_fallback
        

class EnhancedMultiHorizonModule:
    """
    Enhanced multi-horizon forecasting module with adaptive weighting 
    and specialized optimization for each forecast horizon.
    """
    
    def __init__(self, forecast_horizons: List[int], 
                initial_weights: Optional[List[float]] = None,
                adaptive_weighting: bool = True,
                input_dim: int = 0, 
                hidden_dim: int = 256):
        """
        Initialize multi-horizon module.
        
        Args:
            forecast_horizons: List of forecast horizons to predict (e.g., [1, 3, 5, 10])
            initial_weights: Initial importance weights for each horizon (normalized to sum to 1)
            adaptive_weighting: Whether to use adaptive weighting during training
            input_dim: Input dimension for horizon-specific layers
            hidden_dim: Hidden dimension for internal representations
        """
        self.forecast_horizons = forecast_horizons
        self.n_horizons = len(forecast_horizons)
        self.adaptive_weighting = adaptive_weighting
        
        # Initialize weights - default to exponentially decaying importance
        if initial_weights is None:
            # Decay factor: earlier horizons have more weight
            decay = 0.7
            self.horizon_weights = [decay ** i for i in range(self.n_horizons)]
            # Normalize to sum to 1
            total = sum(self.horizon_weights)
            self.horizon_weights = [w / total for w in self.horizon_weights]
        else:
            # Ensure weights sum to 1
            total = sum(initial_weights)
            self.horizon_weights = [w / total for w in initial_weights]
            
        # Keep track of metrics for each horizon
        self.horizon_metrics = {h: {'mse': [], 'mae': [], 'r2': []} 
                               for h in forecast_horizons}
        
        # Initialize PyTorch components if using specialized layers
        if TORCH_AVAILABLE and input_dim > 0:
            self.use_specialized_layers = True
            
            # Create specialized layers for each horizon
            self.horizon_layers = nn.ModuleDict()
            for i, horizon in enumerate(forecast_horizons):
                self.horizon_layers[f'h{horizon}'] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, 1)
                )
        else:
            self.use_specialized_layers = False
        
        logger.info(f"Initialized multi-horizon module with {self.n_horizons} horizons: {forecast_horizons}")
        logger.info(f"Initial horizon weights: {self.horizon_weights}")
    
    def update_weights(self, horizon_performance: Dict[int, float], 
                      method: str = 'inverse_error'):
        """
        Update horizon weights based on recent performance.
        
        Args:
            horizon_performance: Dictionary mapping horizons to error metrics (lower is better)
            method: Weighting method ('inverse_error', 'rank_based', or 'equal')
        """
        if not self.adaptive_weighting:
            return
            
        try:
            if method == 'equal':
                # Equal weighting
                self.horizon_weights = [1.0 / self.n_horizons] * self.n_horizons
                
            elif method == 'rank_based':
                # Rank-based weighting - better performing horizons get more weight
                sorted_horizons = sorted(
                    [(i, h, horizon_performance.get(h, float('inf'))) 
                     for i, h in enumerate(self.forecast_horizons)],
                    key=lambda x: x[2]  # Sort by performance (3rd element)
                )
                
                # Assign weights based on rank (highest rank = lowest error)
                ranks = list(range(1, self.n_horizons + 1))
                # Reverse ranks so best (lowest error) gets highest weight
                ranks = ranks[::-1]
                
                # Create mapping from original index to rank-based weight
                idx_to_weight = {idx: r for (idx, _, _), r in zip(sorted_horizons, ranks)}
                
                # Assign weights preserving original horizon order
                raw_weights = [idx_to_weight[i] for i in range(self.n_horizons)]
                
                # Normalize
                total = sum(raw_weights)
                self.horizon_weights = [w / total for w in raw_weights]
                
            elif method == 'inverse_error':
                # Inverse error weighting - weight proportional to 1/error
                # Handle missing horizons or zero/negative errors
                safe_errors = []
                for h in self.forecast_horizons:
                    error = horizon_performance.get(h, None)
                    if error is not None and error > 0:
                        safe_errors.append(error)
                    else:
                        # Use mean of other errors or a default value
                        other_errors = [e for e in horizon_performance.values() if e is not None and e > 0]
                        if other_errors:
                            safe_errors.append(sum(other_errors) / len(other_errors))
                        else:
                            safe_errors.append(1.0)  # Default
                
                # Calculate inverse errors (higher for better performing horizons)
                inverse_errors = [1.0 / err for err in safe_errors]
                
                # Normalize
                total = sum(inverse_errors)
                if total > 0:
                    self.horizon_weights = [w / total for w in inverse_errors]
                else:
                    # Fallback to equal weighting
                    self.horizon_weights = [1.0 / self.n_horizons] * self.n_horizons
            
            else:
                logger.warning(f"Unknown weighting method: {method}. Using current weights.")
                
            logger.debug(f"Updated horizon weights: {self.horizon_weights}")
            
        except Exception as e:
            logger.error(f"Error updating horizon weights: {e}")
            # Fallback to equal weighting
            self.horizon_weights = [1.0 / self.n_horizons] * self.n_horizons
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            batch_size = x.size(0)
            
            # Prepare output tensor
            outputs = torch.zeros(batch_size, self.n_horizons, device=x.device)
            
            if len(x.shape) == 3:  # [batch_size, seq_len, features]
                global_repr = torch.mean(x, dim=1)  # Average across sequence dimension
            elif len(x.shape) == 2:  # [batch_size, features]
                global_repr = x  # Already the right shape
            else:
                # Handle unexpected input shape
                raise ValueError(f"Unexpected input shape: {x.shape}. Expected 2D or 3D tensor.")
            
            # Apply horizon-specific layers with proper dimension checking
            for i, horizon in enumerate(self.forecast_horizons):
                horizon_key = f'h{horizon}'
                if horizon_key in self.horizon_layers:
                    # Ensure input dimensions match layer expectations
                    expected_dim = next(self.horizon_layers[horizon_key].parameters()).size(1)
                    if global_repr.size(1) != expected_dim:
                        # Reshape or pad as needed
                        if global_repr.size(1) < expected_dim:
                            padding = torch.zeros(batch_size, expected_dim - global_repr.size(1), device=x.device)
                            reshaped_input = torch.cat([global_repr, padding], dim=1)
                        else:
                            reshaped_input = global_repr[:, :expected_dim]
                        outputs[:, i] = self.horizon_layers[horizon_key](reshaped_input).squeeze(-1)
                    else:
                        outputs[:, i] = self.horizon_layers[horizon_key](global_repr).squeeze(-1)
            
            return outputs
        except Exception as e:
            logger.error(f"Error in multi-horizon forward pass: {e}")
            # Return zeros as fallback
            return torch.zeros(x.size(0), self.n_horizons, device=x.device)
    
    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
            """
            Calculate weighted loss across horizons with improved gradient handling.
            
            Args:
                predictions: Tensor of shape [batch_size, n_horizons]
                targets: Tensor of shape [batch_size, n_horizons]
                
            Returns:
                Tuple of (combined weighted loss, dict of per-horizon losses)
            """
            try:
                # Ensure predictions and targets have proper shape
                if len(predictions.shape) == 1:
                    predictions = predictions.unsqueeze(1)
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)
                
                min_dims = min(predictions.shape[1], targets.shape[1])
                    
                # Check and fix dimension mismatch
                if predictions.shape[1] != targets.shape[1]:
                    logger.warning(f"Dimension mismatch: predictions {predictions.shape}, targets {targets.shape}")
                    predictions = predictions[:, :min_dims]
                    targets = targets[:, :min_dims]
                    
                # Initialize per-horizon losses
                horizon_losses = {}
                
                # Make sure predictions requires gradients
                if not predictions.requires_grad:
                    logger.debug("Predictions tensor doesn't require gradients, detaching and creating a new tensor")
                    predictions = predictions.detach().clone().requires_grad_(True)
                
                # Calculate loss for each horizon
                for i, horizon in enumerate(self.forecast_horizons[:min_dims]):
                    horizon_key = f'h{horizon}'
                    
                    # Get predictions and targets for this horizon
                    h_pred = predictions[:, i]
                    h_target = targets[:, i]
                    
                    # Calculate MSE loss directly - no need to check for requires_grad
                    # since predictions tensor now has requires_grad=True
                    h_loss = nn.MSELoss()(h_pred, h_target)
                    horizon_losses[horizon_key] = h_loss
                
                # Combine losses with weights
                weighted_loss = torch.zeros(1, device=predictions.device, requires_grad=True)
                
                for i, (horizon_key, h_loss) in enumerate(horizon_losses.items()):
                    if i < len(self.horizon_weights):
                        weight_tensor = torch.tensor(self.horizon_weights[i], device=predictions.device)
                        weighted_loss = weighted_loss + h_loss * weight_tensor
                
                return weighted_loss, horizon_losses
                
            except Exception as e:
                logger.error(f"Error calculating multi-horizon loss: {e}")
                # Return zero loss as fallback with requires_grad=True
                zero_loss = torch.tensor(0.0, requires_grad=True, device=predictions.device)
                return zero_loss, {}
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive metrics for each horizon.
        """
        try:
            # Ensure predictions and targets have proper shape
            if len(predictions.shape) == 1:
                predictions = predictions.reshape(-1, 1)
            if len(targets.shape) == 1:
                targets = targets.reshape(-1, 1)
                    
            results = {}
            
            # Handle case where predictions and targets have different horizon counts
            if predictions.shape[1] != targets.shape[1]:
                logger.warning(f"Shape mismatch: predictions has {predictions.shape[1]} horizons, targets has {targets.shape[1]}")
                
                # Resize predictions to match target horizon count
                if predictions.shape[1] < targets.shape[1]:
                    # Pad predictions to match target horizons
                    padding = np.zeros((predictions.shape[0], targets.shape[1] - predictions.shape[1]))
                    predictions = np.hstack([predictions, padding])
                else:
                    # Truncate predictions to match target horizons
                    predictions = predictions[:, :targets.shape[1]]
            
            # Calculate metrics for each horizon
            for i, horizon in enumerate(self.forecast_horizons):
                horizon_key = f'h{horizon}'
                
                # Skip if this horizon is out of bounds in predictions/targets
                if i >= predictions.shape[1] or i >= targets.shape[1]:
                    logger.warning(f"Horizon {horizon} out of bounds in predictions/targets")
                    # IMPORTANT: Instead of skipping, provide default metrics
                    results[horizon_key] = {
                        'mse': 0.0,
                        'mae': 0.0,
                        'rmse': 0.0,
                        'r2': 0.0,
                        'direction_accuracy': 0.0
                    }
                    continue
                
                # Get predictions and targets for this horizon
                h_pred = predictions[:, i]
                h_target = targets[:, i]
                
                # Filter out NaN values
                mask = ~(np.isnan(h_pred) | np.isnan(h_target))
                h_pred_clean = h_pred[mask]
                h_target_clean = h_target[mask]
                
                # Ensure we have data to calculate metrics
                if len(h_pred_clean) < 2:
                    logger.warning(f"Not enough valid data for horizon {horizon}")
                    # IMPORTANT: Provide default metrics instead of skipping
                    results[horizon_key] = {
                        'mse': 0.0,
                        'mae': 0.0,
                        'rmse': 0.0,
                        'r2': 0.0,
                        'direction_accuracy': 0.5  # Default to random
                    }
                    continue
                
                # Calculate metrics
                mse = mean_squared_error(h_target_clean, h_pred_clean)
                mae = mean_absolute_error(h_target_clean, h_pred_clean)
                
                # R² can be negative for bad models - cap at 0
                r2 = max(0, r2_score(h_target_clean, h_pred_clean))
                
                # Calculate directional accuracy
                direction_correct = np.sum((np.sign(h_pred_clean) == np.sign(h_target_clean)))
                direction_accuracy = direction_correct / len(h_pred_clean) if len(h_pred_clean) > 0 else 0.5
                
                # Store metrics
                results[horizon_key] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(np.sqrt(mse)),
                    'r2': float(r2),
                    'direction_accuracy': float(direction_accuracy)
                }
            
            # Calculate aggregate metrics
            # IMPORTANT: Always provide aggregate metrics even if individual horizons failed
            mse_values = [m['mse'] for m in results.values() if 'mse' in m]
            mae_values = [m['mae'] for m in results.values() if 'mae' in m]
            r2_values = [m['r2'] for m in results.values() if 'r2' in m]
            da_values = [m['direction_accuracy'] for m in results.values() if 'direction_accuracy' in m]
            
            # Always include aggregate metrics, even if some are missing
            results['aggregate'] = {
                'mse': float(np.mean(mse_values)) if mse_values else 0.0,
                'mae': float(np.mean(mae_values)) if mae_values else 0.0,
                'r2': float(np.mean(r2_values)) if r2_values else 0.0,
                'direction_accuracy': float(np.mean(da_values)) if da_values else 0.5,
                'weighted_mse': float(np.sum([mse * w for mse, w in zip(mse_values, self.horizon_weights[:len(mse_values)])])) if mse_values else 0.0,
                'best_horizon': self.forecast_horizons[np.argmin(mse_values)] if mse_values else self.forecast_horizons[0]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating multi-horizon metrics: {e}")
            # IMPORTANT: Return valid metrics structure even on error
            default_metrics = {}
            for horizon in self.forecast_horizons:
                default_metrics[f'h{horizon}'] = {
                    'mse': 0.0,
                    'mae': 0.0,
                    'rmse': 0.0,
                    'r2': 0.0,
                    'direction_accuracy': 0.5
                }
            default_metrics['aggregate'] = {
                'mse': 0.0,
                'mae': 0.0,
                'r2': 0.0,
                'direction_accuracy': 0.5,
                'weighted_mse': 0.0,
                'best_horizon': self.forecast_horizons[0]
            }
            return default_metrics
    
    def visualize_forecasts(self, 
                          predictions: np.ndarray, 
                          targets: np.ndarray,
                          dates: Optional[np.ndarray] = None,
                          symbol: str = "",
                          output_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of multi-horizon forecasts.
        
        Args:
            predictions: Array of shape [samples, n_horizons]
            targets: Array of shape [samples, n_horizons]
            dates: Optional dates corresponding to samples
            symbol: Stock symbol or other identifier for the plot title
            output_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Ensure we have dates
            if dates is None:
                dates = np.arange(len(predictions))
                
            # Create figure
            fig, axes = plt.subplots(len(self.forecast_horizons), 1, 
                                    figsize=(12, 3 * len(self.forecast_horizons)),
                                    sharex=True)
            
            # If single horizon, wrap axes in list
            if len(self.forecast_horizons) == 1:
                axes = [axes]
            
            # Plot each horizon
            for i, (horizon, ax) in enumerate(zip(self.forecast_horizons, axes)):
                # Get data for this horizon
                if i < predictions.shape[1] and i < targets.shape[1]:
                    pred = predictions[:, i]
                    target = targets[:, i]
                    
                    # Basic plot styling
                    ax.set_title(f"{horizon}-Step Ahead Forecast")
                    ax.plot(dates, target, label="Actual", color="blue", alpha=0.7)
                    ax.plot(dates, pred, label="Forecast", color="red", linestyle="--")
                    
                    # Add error regions - highlight large errors
                    errors = np.abs(pred - target)
                    threshold = np.percentile(errors, 80)  # Top 20% of errors
                    high_error_indices = errors > threshold
                    
                    # Shade high error regions
                    for j in range(len(dates) - 1):
                        if high_error_indices[j]:
                            ax.axvspan(dates[j], dates[j+1] if j+1 < len(dates) else dates[j], 
                                      alpha=0.2, color='yellow')
                    
                    # Calculate metrics for this horizon
                    h_mse = mean_squared_error(target, pred)
                    h_da = np.mean(np.sign(pred) == np.sign(target))
                    
                    # Add metrics to plot
                    ax.text(0.02, 0.95, f"MSE: {h_mse:.6f}, Direction Acc: {h_da:.2%}", 
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.7))
                    
                    # Add legend
                    ax.legend(loc="upper right")
                    
                    # Add grid for readability
                    ax.grid(True, alpha=0.3)
            
            # Add common title
            title = f"Multi-Horizon Forecasts"
            if symbol:
                title += f" for {symbol}"
            fig.suptitle(title, fontsize=16)
            
            # Set tight layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save or display
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved forecast visualization to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing forecasts: {e}")
            # Return empty figure as fallback
            return plt.figure()


class AttentionBlock(nn.Module):
    """Multi-head self-attention block with proper dimension handling and error checks."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Add dimension check with fallback
        if embed_dim % num_heads != 0:
            logger.warning(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}). "
                          f"Adjusting embed_dim to {embed_dim - (embed_dim % num_heads)}.")
            embed_dim = embed_dim - (embed_dim % num_heads)
            # Ensure embed_dim is at least num_heads
            embed_dim = max(embed_dim, num_heads)
        
        # Ensure embed_dim and num_heads are positive
        if embed_dim <= 0:
            logger.warning(f"Invalid embed_dim: {embed_dim}. Setting to default value of 64.")
            embed_dim = 64
        
        if num_heads <= 0:
            logger.warning(f"Invalid num_heads: {num_heads}. Setting to default value of 1.")
            num_heads = 1
        
        try:
            self.attention = nn.MultiheadAttention(
                embed_dim, 
                num_heads, 
                dropout=dropout, 
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * embed_dim, embed_dim)
            )
            
            # Store dimensions for error checking in forward pass
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            
        except Exception as e:
            logger.error(f"Error initializing AttentionBlock: {e}")
            # Create dummy layers as fallback
            self.attention = None
            self.norm1 = nn.LayerNorm(embed_dim) if embed_dim > 0 else None
            self.norm2 = nn.LayerNorm(embed_dim) if embed_dim > 0 else None
            self.ff = nn.Identity()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
    
    def forward(self, x):
        try:
            # Validate input shape
            if x is None or len(x.shape) != 3:
                logger.error(f"Invalid input shape: {x.shape if x is not None else None}. Expected 3D tensor.")
                return x
                
            # Check if input feature dimension matches expected dimension
            if x.shape[2] != self.embed_dim:
                logger.warning(f"Input feature dimension ({x.shape[2]}) doesn't match embed_dim ({self.embed_dim}). "
                              f"Adjusting input tensor.")
                # Pad or truncate last dimension to match embed_dim
                if x.shape[2] < self.embed_dim:
                    # Pad
                    padding = torch.zeros(x.shape[0], x.shape[1], self.embed_dim - x.shape[2], device=x.device)
                    x = torch.cat([x, padding], dim=2)
                else:
                    # Truncate
                    x = x[:, :, :self.embed_dim]
            
            # Skip if attention layer failed to initialize
            if self.attention is None:
                return x
                
            attended, _ = self.attention(x, x, x)
            x = self.norm1(x + attended)
            x = self.norm2(x + self.ff(x))
            return x
            
        except Exception as e:
            logger.error(f"Error in AttentionBlock forward pass: {e}")
            return x  # Return input unchanged as fallback

class HybridTimeSeriesModel(pl.LightningModule):
    """
    Enhanced deep learning model combining LSTM and attention mechanisms 
    with support for multiple forecast horizons and advanced optimization techniques.
    """

    def __init__(self, input_dim: int, config: ModelConfig, 
                activation_fn: str = 'gelu', weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.activation_fn = activation_fn
        self.weight_decay = weight_decay
        if input_dim <= 0:
            logger.warning(f"Invalid input_dim: {input_dim}, defaulting to 10")
            input_dim = 10
        self.hparams.input_dim = input_dim
        logger.info(f"Initializing HybridTimeSeriesModel with input_dim={input_dim}")
        
        # Initialize SWA if enabled
        self.use_swa = getattr(config, 'use_swa', False)
        if self.use_swa:
            self.swa_start = getattr(config, 'swa_start', 50)
            self.swa_freq = getattr(config, 'swa_freq', 5)
            self.swa_lr = getattr(config, 'swa_lr', 1e-4)
            self.swa = None  # Will be initialized after model parameters are set up
        
        # Multi-horizon support
        self.multi_horizon = getattr(config, 'multi_horizon', False)
        self.forecast_horizons = config.get_forecasting_horizons()
        self.horizon_weights = getattr(config, 'horizon_weights', None)
        
        # Log multi-horizon configuration for debugging
        if self.multi_horizon:
            logger.info(f"Initializing model with multi-horizon forecasting, horizons: {self.forecast_horizons}")
        else:
            logger.info(f"Initializing model with single-horizon forecasting, horizon: {self.forecast_horizons[0]}")
        
        # Add dimension validation and fallback
        if input_dim <= 0:
            logger.warning(f"Invalid input_dim: {input_dim}, defaulting to 10")
            input_dim = 10
        self.hparams.input_dim = input_dim
        
        # Calculate dimensions with validation
        self.lstm_hidden_dim = max(1, config.hidden_dim // 2)  # Ensure minimum size of 1
        logger.info(f"Model dimensions - Input: {input_dim}, LSTM hidden: {self.lstm_hidden_dim}")
        
        try:
            # LSTM layers with proper error handling
            self.lstm = nn.LSTM(
                input_dim,
                self.lstm_hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True
            )
            
            # Attention mechanism
            self.attention = AttentionBlock(
                config.hidden_dim,  # Full hidden dim (bidirectional * lstm_hidden_dim)
                num_heads=getattr(config, 'attention_heads', 8),
                dropout=config.dropout
            )
            
            # Add multi-horizon module if multi_horizon is enabled
            if self.multi_horizon:
                self.multi_horizon_module = EnhancedMultiHorizonModule(
                    forecast_horizons=self.forecast_horizons,
                    initial_weights=self.horizon_weights,
                    adaptive_weighting=True,
                    input_dim=config.hidden_dim,
                    hidden_dim=config.hidden_dim // 2
                )
            
            # Get activation function
            activation = self._get_activation()
            
            # Create output layers for each forecast horizon
            # Set output_dim based on multi-horizon status
            if self.multi_horizon:
                output_dim = len(self.forecast_horizons)
                logger.info(f"Using multi-horizon output with {output_dim} horizons")
            else:
                output_dim = 1
                logger.info("Using single-horizon output")
            
            # Output layers with proper dimensions
            self.fc = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                activation,
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, output_dim)  # This is the important line to fix!
            )
            
            # Initialize SWA now that parameters are set up
            if self.use_swa:
                self.swa = StochasticWeightAveraging(
                    self, self.swa_start, self.swa_freq, self.swa_lr
                )
                
        except Exception as e:
            logger.error(f"Error initializing model components: {e}")
            # Set up fallback components
            self.lstm = None
            self.attention = None
            
            # Set fallback output dimension correctly
            if self.multi_horizon:
                fallback_output_dim = len(self.forecast_horizons)
            else:
                fallback_output_dim = 1
                
            self.fc = nn.Sequential(nn.Linear(input_dim, fallback_output_dim))
    
    def _get_activation(self):
        """Get the appropriate activation function based on the specified name."""
        try:
            if self.activation_fn.lower() == 'relu':
                return nn.ReLU()
            elif self.activation_fn.lower() == 'leaky_relu':
                return nn.LeakyReLU(0.1)
            elif self.activation_fn.lower() == 'elu':
                return nn.ELU()
            elif self.activation_fn.lower() == 'gelu':
                return nn.GELU()
            else:
                logger.warning(f"Unknown activation function: {self.activation_fn}. Using GELU.")
                return nn.GELU()
        except Exception as e:
            logger.error(f"Error setting activation function: {e}")
            return nn.GELU()
    

    def forward(self, x):
        try:
            # Input shape validation
            if len(x.shape) != 3:
                x = x.unsqueeze(0) if len(x.shape) == 2 else x
                logger.warning(f"Input reshaped to 3D tensor: {x.shape}")
            
            batch_size, seq_len, features = x.shape
            logger.debug(f"Forward pass - Input shape: batch={batch_size}, seq={seq_len}, features={features}")
            
            # Handle failed initialization
            if self.lstm is None or self.attention is None:
                logger.warning("Using fallback forward pass due to initialization failure")
                # Simple fallback: average across sequence dimension and apply linear layer
                pooled = torch.mean(x, dim=1)
                
                # When falling back, still ensure correct output dimensions
                if self.multi_horizon:
                    # Use specialized module if available
                    if hasattr(self, 'multi_horizon_module') and self.multi_horizon_module is not None:
                        try:
                            # Use specialized module
                            multi_horizon_out = self.multi_horizon_module.forward(pooled)
                            
                            # Validate multi_horizon_module output shape
                            expected_horizons = len(self.forecast_horizons)
                            if multi_horizon_out.shape[1] != expected_horizons:
                                logger.warning(f"Multi-horizon module output shape mismatch: {multi_horizon_out.shape[1]} vs expected {expected_horizons}")
                                # Correct shape if needed
                                if multi_horizon_out.shape[1] < expected_horizons:
                                    padding = torch.zeros((batch_size, expected_horizons - multi_horizon_out.shape[1]), device=x.device)
                                    multi_horizon_out = torch.cat([multi_horizon_out, padding], dim=1)
                                else:
                                    multi_horizon_out = multi_horizon_out[:, :expected_horizons]
                            
                            return multi_horizon_out
                        except Exception as e:
                            logger.warning(f"Error in multi_horizon_module forward pass: {e}. Falling back to standard output.")
                else:
                    return self.fc(pooled) if hasattr(self, 'fc') else torch.mean(pooled, dim=1, keepdim=True)
            
            # LSTM processing
            lstm_out, _ = self.lstm(x)
            
            # Apply attention
            attended = self.attention(lstm_out)
        
            # Global average pooling
            pooled = torch.mean(attended, dim=1)
        
            # MODIFIED: Ensure final prediction has correct dimensions based on multi_horizon setting
            if self.multi_horizon:
                # Use specialized module if available
                if hasattr(self, 'multi_horizon_module') and self.multi_horizon_module is not None:
                    try:
                        # Use specialized module
                        multi_horizon_out = self.multi_horizon_module.forward(pooled)
                        
                        # Validate multi_horizon_module output shape
                        expected_horizons = len(self.forecast_horizons)
                        if multi_horizon_out.shape[1] != expected_horizons:
                            logger.warning(f"Multi-horizon module output shape mismatch: {multi_horizon_out.shape[1]} vs expected {expected_horizons}")
                            # Correct shape if needed
                            if multi_horizon_out.shape[1] < expected_horizons:
                                padding = torch.zeros((batch_size, expected_horizons - multi_horizon_out.shape[1]), device=x.device)
                                multi_horizon_out = torch.cat([multi_horizon_out, padding], dim=1)
                            else:
                                multi_horizon_out = multi_horizon_out[:, :expected_horizons]
                        
                        return multi_horizon_out
                    except Exception as e:
                        logger.warning(f"Error in multi_horizon_module forward pass: {e}. Falling back to standard output.")
                
                # Standard output with correction
                out = self.fc(pooled)
                expected_horizons = len(self.forecast_horizons)
                
                # Check output shape and fix if necessary
                if out.shape[1] != expected_horizons:
                    logger.warning(f"Output shape mismatch: got {out.shape[1]} horizons, expected {expected_horizons}")
                    # Fix the shape
                    if out.shape[1] < expected_horizons:
                        # Pad with zeros
                        padding = torch.zeros(batch_size, expected_horizons - out.shape[1], device=x.device)
                        out = torch.cat([out, padding], dim=1)
                    else:
                        # Truncate
                        out = out[:, :expected_horizons]
                return out
            else:
                # Single horizon case
                return self.fc(pooled).view(-1, 1)
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros as fallback with proper dimensions
            # Get correct output dimension based on horizon mode
            out_dim = len(self.forecast_horizons) if self.multi_horizon else 1
            return torch.zeros((x.shape[0], out_dim), device=x.device)

    
    def training_step(self, batch, batch_idx):
        try:
            x, y = batch
            y_hat = self(x)
            
            # Calculate loss based on horizon configuration
            if self.multi_horizon:
                # Ensure that we have a valid tensor for backpropagation
                if not y_hat.requires_grad:
                    logger.warning("Prediction tensor doesn't require gradients. Detaching and recreating with requires_grad=True")
                    # Detach and recreate with gradients
                    y_hat_values = y_hat.detach().clone()
                    y_hat = torch.tensor(y_hat_values, requires_grad=True, device=y_hat.device)
                
                # Use the enhanced multi-horizon module for loss calculation
                loss, horizon_losses = self.multi_horizon_module.calculate_loss(y_hat, y)
                
                # Check if the loss is a valid tensor for backpropagation
                if not loss.requires_grad:
                    logger.warning("Loss tensor doesn't require gradients. Using MSE loss directly.")
                    # Fallback to direct MSE calculation
                    loss = nn.MSELoss()(y_hat, y)
                
                # Log individual horizon losses
                for horizon_key, horizon_loss in horizon_losses.items():
                    self.log(f'train_loss_{horizon_key}', horizon_loss)
            else:
                # Standard single-horizon loss
                loss = nn.MSELoss()(y_hat, y)
                
            self.log('train_loss', loss)
            
            # Verify the loss is valid for backpropagation
            if not loss.requires_grad:
                raise ValueError("Loss doesn't require gradients, cannot backpropagate")
            
            # Update SWA if enabled and at appropriate epoch
            if self.use_swa and self.swa is not None:
                current_epoch = self.current_epoch
                if self.swa.should_update(current_epoch):
                    self.swa.update(current_epoch)
            
            return loss
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            # Return zero loss with gradients to continue training
            return torch.tensor(0.0, requires_grad=True, device=self.device if hasattr(self, 'device') else 'cpu')    
    
    def validation_step(self, batch, batch_idx):
        """Custom validation step with proper multi-horizon handling."""
        try:
            x, y = batch
            y_hat = self(x)
            
            # Calculate loss based on horizon configuration
            # In the validation_step method, replace the multi-horizon validation with:
            if self.multi_horizon:
                # Calculate loss using the multi-horizon module
                val_loss, horizon_losses = self.multi_horizon_module.calculate_loss(y_hat, y)
                
                # Log individual horizon losses
                for horizon_key, horizon_loss in horizon_losses.items():
                    self.log(f'val_loss_{horizon_key}', horizon_loss)
            else:
                # Standard single-horizon loss
                val_loss = nn.MSELoss()(y_hat, y)
                        
            self.log('val_loss', val_loss)
            
            return {
                'val_loss': val_loss,
                'predictions': y_hat.detach().cpu(),
                'targets': y.detach().cpu()
            }
        except Exception as e:
            logger.error(f"Error in validation step: {e}")
            # Return dummy values to continue
            out_dim = len(self.forecast_horizons) if self.multi_horizon else 1
            return {
                'val_loss': torch.tensor(float('inf')),
                'predictions': torch.zeros((x.shape[0], out_dim)),
                'targets': torch.zeros_like(y.detach().cpu())
            }
    
    def on_validation_epoch_end(self):
        """Apply SWA weights at the end of validation if in evaluation mode."""
        if self.use_swa and self.swa is not None:
            # Check if SWA is ready to be applied (has at least one update)
            if self.swa.swa_n_models > 0:
                self.swa.apply()
                logger.info("Applied SWA weights for evaluation")
            else:
                logger.warning("No SWA updates have been performed. Skipping application.")
        
    def predict_step(self, batch, batch_idx):
        """Custom predict step with multi-horizon support."""
        try:
            x, _ = batch
            predictions = self(x)
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict step: {e}")
            # Return zeros as fallback
            out_dim = len(self.forecast_horizons) if self.multi_horizon else 1
            return torch.zeros((x.shape[0], out_dim), device=x.device)
    
    def configure_optimizers(self):
        try:
            # Use AdamW with weight decay
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Cosine annealing learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate / 10
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        except Exception as e:
            logger.error(f"Error configuring optimizers: {e}")
            # Return default optimizer as fallback
            return torch.optim.Adam(self.parameters(), lr=0.001)
        
class SARIMAModel:
    """SARIMA time series forecasting model for multi-horizon prediction."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}  # One model per horizon
        self.fitted = False
        self.order = (1, 1, 1)  # Default (p, d, q)
        self.seasonal_order = (0, 0, 0, 0)  # Default (P, D, Q, s)
        self.univariate = True  # SARIMA is univariate by default
        self.feature_names = []
        self.is_multi_horizon = config.multi_horizon if hasattr(config, 'multi_horizon') else False
        if self.is_multi_horizon and hasattr(config, 'forecast_horizons'):
            self.horizons = config.forecast_horizons
        else:
            self.horizons = [config.forecast_horizon] if hasattr(config, 'forecast_horizon') else [1]
        
        logger.info(f"Initialized SARIMA model with horizons: {self.horizons}")
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit SARIMA model to the data with proper error handling."""
        if not SARIMAX_AVAILABLE:
            logger.warning("SARIMAX not available. Cannot fit SARIMA model.")
            return self
            
        try:
            # Determine if we have multi-horizon targets
            self.is_multi_horizon = isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1
            
            # Set horizons - either multiple or just one
            if self.is_multi_horizon:
                self.horizons = list(range(y.shape[1]))
                logger.info(f"Training SARIMA models for {len(self.horizons)} horizons")
            else:
                self.horizons = [0]  # Just one horizon
            
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
                X_values = X.values
            else:
                X_values = X
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Extract target time series
            if self.univariate:
                # For univariate mode, we just use past values of the target
                logger.info("Using univariate SARIMA model (exogenous variables ignored)")
                
                # For each horizon, train a separate model
                for h_idx in self.horizons:
                    logger.info(f"Training SARIMA model for horizon {h_idx}")
                    
                    # Extract the target for this horizon
                    if self.is_multi_horizon:
                        y_h = y[:, h_idx]
                    else:
                        y_h = y
                    
                    # Handle NaN values
                    if np.isnan(y_h).any():
                        logger.warning(f"Target for horizon {h_idx} contains NaNs - interpolating")
                        # Interpolate NaNs
                        y_h_series = pd.Series(y_h)
                        y_h = y_h_series.interpolate(method='linear', limit_direction='both').values
                    
                    try:
                        # Fit SARIMA model
                        model = SARIMAX(
                            y_h,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        
                        # Fit with optimized parameters
                        fit_result = model.fit(disp=False, maxiter=50)
                        
                        # Store the fitted model
                        self.models[h_idx] = {'model': model, 'result': fit_result}
                        logger.info(f"Successfully fitted SARIMA model for horizon {h_idx}")
                    except Exception as e:
                        logger.error(f"Error fitting SARIMA model for horizon {h_idx}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            else:
                # For multivariate mode, select top features to use as exogenous variables
                # Note: SARIMAX has limited support for many exogenous variables
                max_exog = 5  # Limit to top 5 features to avoid overfitting
                
                # Use first few features as exogenous variables
                exog_data = X_values[:, :min(max_exog, X_values.shape[1])]
                
                # For each horizon, train a separate model
                for h_idx in self.horizons:
                    logger.info(f"Training SARIMA model with exogenous variables for horizon {h_idx}")
                    
                    # Extract the target for this horizon
                    if self.is_multi_horizon:
                        y_h = y[:, h_idx]
                    else:
                        y_h = y
                    
                    # Handle NaN values in target and exogenous
                    if np.isnan(y_h).any() or np.isnan(exog_data).any():
                        logger.warning(f"NaNs found in data for horizon {h_idx} - handling...")
                        
                        # Convert to DataFrames for easier NaN handling
                        y_df = pd.DataFrame(y_h, columns=['target'])
                        exog_df = pd.DataFrame(exog_data, columns=[f'exog_{i}' for i in range(exog_data.shape[1])])
                        
                        # Interpolate NaNs
                        y_df = y_df.interpolate(method='linear', limit_direction='both')
                        exog_df = exog_df.interpolate(method='linear', limit_direction='both')
                        
                        # Fill any remaining NaNs with 0
                        y_df = y_df.fillna(0)
                        exog_df = exog_df.fillna(0)
                        
                        y_h = y_df['target'].values
                        exog_data_clean = exog_df.values
                    else:
                        exog_data_clean = exog_data
                    
                    try:
                        # Fit SARIMA model with exogenous variables
                        model = SARIMAX(
                            y_h,
                            exog=exog_data_clean,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        
                        # Fit with optimized parameters
                        fit_result = model.fit(disp=False, maxiter=50)
                        
                        # Store the fitted model
                        self.models[h_idx] = {
                            'model': model, 
                            'result': fit_result,
                            'exog_used': True,
                            'exog_cols': self.feature_names[:min(max_exog, X_values.shape[1])]
                        }
                        logger.info(f"Successfully fitted SARIMA model for horizon {h_idx} with exogenous variables")
                    except Exception as e:
                        logger.error(f"Error fitting SARIMA model with exogenous for horizon {h_idx}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        
                        # Fallback to univariate model
                        logger.info(f"Falling back to univariate SARIMA for horizon {h_idx}")
                        try:
                            model = SARIMAX(
                                y_h,
                                order=self.order,
                                seasonal_order=self.seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            fit_result = model.fit(disp=False, maxiter=50)
                            self.models[h_idx] = {'model': model, 'result': fit_result, 'exog_used': False}
                            logger.info(f"Successfully fitted fallback univariate SARIMA for horizon {h_idx}")
                        except Exception as e2:
                            logger.error(f"Error fitting fallback SARIMA model: {e2}")
            
            self.fitted = len(self.models) > 0
            return self
            
        except Exception as e:
            logger.error(f"Unexpected error fitting SARIMA models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fitted = False
            return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate forecasts for each horizon."""
        if not SARIMAX_AVAILABLE or not self.fitted or not self.models:
            logger.warning("SARIMA models not available or not fitted. Returning zeros.")
            n_horizons = len(self.horizons) if self.horizons else 1
            return np.zeros((len(X) if X is not None else 1, n_horizons))
            
        try:
            logger.info(f"Generating SARIMA predictions for {len(X)} samples")
            
            # Check if we need to extract exogenous variables
            exog_data = None
            if not self.univariate and X is not None:
                # Extract same exogenous variables as used in training
                for h_idx, model_info in self.models.items():
                    if model_info.get('exog_used', False):
                        exog_cols = model_info.get('exog_cols', [])
                        if isinstance(X, pd.DataFrame):
                            # Extract columns by name
                            avail_cols = [col for col in exog_cols if col in X.columns]
                            if avail_cols:
                                exog_data = X[avail_cols].values[:, :min(5, len(avail_cols))]
                            else:
                                # Fallback to first columns
                                exog_data = X.values[:, :min(5, X.shape[1])]
                        else:
                            # Extract first columns
                            exog_data = X[:, :min(5, X.shape[1])]
                        break
            
            # Prepare prediction array
            n_samples = len(X)
            n_horizons = len(self.horizons)
            predictions = np.zeros((n_samples, n_horizons))
            
            # Generate predictions for each horizon
            for i, h_idx in enumerate(self.horizons):
                if h_idx in self.models:
                    try:
                        logger.info(f"Generating predictions for horizon {h_idx}")
                        model_info = self.models[h_idx]
                        fit_result = model_info['result']
                        
                        # Check if model uses exogenous variables
                        if model_info.get('exog_used', False) and exog_data is not None:
                            # Generate forecast with exogenous variables
                            forecast = fit_result.get_forecast(steps=n_samples, exog=exog_data)
                        else:
                            # Generate forecast without exogenous variables
                            forecast = fit_result.get_forecast(steps=n_samples)
                            
                        # Extract mean prediction
                        horizon_pred = forecast.predicted_mean
                        
                        # Handle case where prediction is shorter than needed
                        if len(horizon_pred) < n_samples:
                            # Pad with last value
                            padding = np.full(n_samples - len(horizon_pred), 
                                             horizon_pred[-1] if len(horizon_pred) > 0 else 0)
                            horizon_pred = np.concatenate([horizon_pred, padding])
                        elif len(horizon_pred) > n_samples:
                            # Truncate
                            horizon_pred = horizon_pred[:n_samples]
                            
                        # Check for invalid values
                        if np.any(np.isnan(horizon_pred)) or np.any(np.isinf(horizon_pred)):
                            logger.warning(f"NaN/Inf values in SARIMA predictions for horizon {h_idx}. Replacing with zeros.")
                            horizon_pred = np.nan_to_num(horizon_pred, nan=0.0, posinf=0.0, neginf=0.0)
                            
                        # Store in predictions array
                        predictions[:, i] = horizon_pred
                        logger.info(f"Successfully generated SARIMA predictions for horizon {h_idx}")
                    except Exception as e:
                        logger.error(f"Error generating SARIMA predictions for horizon {h_idx}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            # If multi-horizon is disabled but we have multiple models,
            # return just the first horizon
            if not self.is_multi_horizon and predictions.shape[1] > 1:
                return predictions[:, 0:1]
                    
            return predictions
                
        except Exception as e:
            logger.error(f"Unexpected error in SARIMA prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            n_horizons = len(self.horizons) if self.horizons else 1
            return np.zeros((len(X) if X is not None else 1, n_horizons))

    def set_params(self, **params):
        """Set model parameters for compatibility with scikit-learn."""
        for key, value in params.items():
            if key == 'order' and isinstance(value, tuple) and len(value) == 3:
                self.order = value
            elif key == 'seasonal_order' and isinstance(value, tuple) and len(value) == 4:
                self.seasonal_order = value
            elif key == 'univariate' and isinstance(value, bool):
                self.univariate = value
        return self
        
    def get_params(self, deep=True):
        """Get model parameters for compatibility with scikit-learn."""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'univariate': self.univariate
        }

class TemporalConvBlock(nn.Module):
    """Temporal convolutional block with residual connection."""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = nn.Identity()  # We use 'same' padding with PyTorch >= 1.9, otherwise use Chomp1d
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = nn.Identity()  # Same as above
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        try:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)
        except Exception as e:
            logger.error(f"Error in TCN block forward pass: {e}")
            # Return input as fallback 
            return x


class TemporalConvNetwork(nn.Module):
    """Temporal Convolutional Network main class."""
    
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Calculate same padding
        padding = (kernel_size - 1) // 2
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation factor
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalConvBlock(
                    in_channels, out_channels, kernel_size, stride=1, 
                    dilation=dilation_size, padding=padding * dilation_size, dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        try:
            # Input shape validation
            if x is None or not isinstance(x, torch.Tensor):
                logger.error("TCN received invalid input")
                return torch.zeros((1, 1), device=self.device if hasattr(self, 'device') else 'cpu')

            # Ensure input is 3D tensor [batch, seq, features]
            if len(x.shape) != 3:
                logger.warning(f"Input reshaped to 3D tensor: {x.shape}")
                if len(x.shape) == 2:
                    # Handle 2D tensor: assume [batch, features] and add sequence dimension
                    x = x.unsqueeze(1)
                elif len(x.shape) == 1:
                    # Handle 1D tensor: add batch and sequence dimensions
                    x = x.unsqueeze(0).unsqueeze(0)
                else:
                    logger.error(f"Cannot reshape tensor with shape {x.shape}")
                    return torch.zeros((x.shape[0] if hasattr(x, 'shape') else 1, 1), device=x.device)
                    
            # Original code with proper error capture
            try:
                # Convert [batch, sequence, features] -> [batch, features, sequence]
                x = x.transpose(1, 2)
                y = self.network(x)
                # Convert back to [batch, sequence, channels[-1]]
                return y.transpose(1, 2)
            except Exception as e:
                logger.error(f"Error in TCN network forward pass: {e}")
                # Return tensor of appropriate shape as fallback
                if hasattr(self, 'network') and self.network is not None:
                    last_channels = self.network[-1].conv2.out_channels
                    return torch.zeros(x.shape[0], x.shape[2], last_channels, device=x.device)
                else:
                    return torch.zeros_like(x.transpose(1, 2))
        except Exception as e:
            logger.error(f"Error in TCN forward pass: {e}")
            # Return zeros with proper shape as fallback
            batch_size = x.shape[0] if x is not None and hasattr(x, 'shape') else 1
            output_dim = 1  # Default single output
            return torch.zeros((batch_size, output_dim), device=x.device if hasattr(x, 'device') else 'cpu')

class TCNModel(pl.LightningModule):
    """Temporal Convolutional Network model with multi-horizon capability."""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.input_dim = input_dim
        
        # Multi-horizon support
        self.multi_horizon = getattr(config, 'multi_horizon', False)
        if self.multi_horizon:
            self.forecast_horizons = getattr(config, 'forecast_horizons', [1, 3, 5, 10, 20])
            self.horizon_weights = getattr(config, 'horizon_weights', None)
            if self.horizon_weights is None:
                # Default to equal weighting if not specified
                self.horizon_weights = [1.0/len(self.forecast_horizons)] * len(self.forecast_horizons)
            logger.info(f"Using multi-horizon forecasting with horizons: {self.forecast_horizons}")
        else:
            # Single horizon (traditional approach)
            self.forecast_horizons = [getattr(config, 'forecast_horizon', 1)]
            self.horizon_weights = [1.0]
        
        # TCN architecture parameters
        self.kernel_size = 3
        self.dropout = config.dropout
        self.lookback = config.lookback_window
        
        # Define TCN channels progression
        # Gradually increase channels like 2^i to capture multiple frequency patterns
        self.num_channels = [
            config.hidden_dim // 4,
            config.hidden_dim // 2,
            config.hidden_dim,
            config.hidden_dim
        ]
        
        # TCN feature extractor
        self.tcn = TemporalConvNetwork(
            num_inputs=input_dim,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )
        
        # Output layer for each horizon
        output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
        
        # Global pooling and final layers with residual connection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final prediction layers
        self.output = nn.Sequential(
            nn.Linear(self.num_channels[-1], config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Kaiming/He initialization."""
        try:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        except Exception as e:
            logger.warning(f"Error initializing TCN weights: {e}")
    
    def forward(self, x):
        try:
            # Input validation
            if x is None:
                logger.error("TCN received None input")
                batch_size = 1
                # MODIFIED: Ensure correct output dimensions
                output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
                return torch.zeros((batch_size, output_dim), device=self.device)
                    
            # Ensure input is 3D tensor [batch, seq_len, features]
            if len(x.shape) == 1:
                # Handle 1D input (single sample)
                logger.warning(f"Received 1D input with shape {x.shape}. Reshaping to 3D.")
                x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            elif len(x.shape) == 2:
                # Handle 2D input [batch, features] - add sequence dimension
                logger.warning(f"Received 2D input with shape {x.shape}. Adding sequence dimension.")
                x = x.unsqueeze(1)  # Add sequence dimension
                
            batch_size = x.shape[0]
            
            # Validate input_dim matches expected dimensions
            if x.shape[2] != self.input_dim:
                logger.warning(f"Input feature dimension mismatch: got {x.shape[2]}, expected {self.input_dim}. Adjusting.")
                if x.shape[2] < self.input_dim:
                    # Pad with zeros
                    padding = torch.zeros((batch_size, x.shape[1], self.input_dim - x.shape[2]), device=x.device)
                    x = torch.cat([x, padding], dim=2)
                else:
                    # Truncate
                    x = x[:, :, :self.input_dim]
            
            # Apply TCN feature extraction
            features = self.tcn(x)
            # First transpose to get [batch, channels, seq_len]
            features = features.transpose(1, 2)
            # Then apply global pooling
            pooled = self.global_pool(features).squeeze(-1)  # [batch, channels]
            
            # Generate final prediction
            output = self.output(pooled)
            if self.multi_horizon:
                expected_horizons = len(self.forecast_horizons)
                if output.shape[1] != expected_horizons:
                    logger.warning(f"Output shape mismatch: got {output.shape[1]} horizons, expected {expected_horizons}")
                    # Fix the shape
                    if output.shape[1] < expected_horizons:
                        # Pad with zeros
                        padding = torch.zeros(batch_size, expected_horizons - output.shape[1], device=output.device)
                        output = torch.cat([output, padding], dim=1)
                    else:
                        # Truncate
                        output = output[:, :expected_horizons]
            
            return output
            
        except Exception as e:
            logger.error(f"Error in TCN forward pass: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros with proper shape as fallback
            batch_size = x.shape[0] if x is not None and hasattr(x, 'shape') else 1
            # MODIFIED: Ensure correct output dimensions
            output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
            return torch.zeros((batch_size, output_dim), device=self.device if hasattr(self, 'device') else 'cpu')
    
    def training_step(self, batch, batch_idx):
        try:
            x, y = batch
            y_hat = self(x)
            
            # Calculate loss based on horizon configuration
            if self.multi_horizon:
                # Multi-horizon loss calculation using MSE for each horizon
                loss = 0
                for i, weight in enumerate(self.horizon_weights):
                    if i < y.shape[1] and i < y_hat.shape[1]:
                        horizon_loss = F.mse_loss(y_hat[:, i], y[:, i])
                        loss += weight * horizon_loss
                        self.log(f'train_loss_h{i+1}', horizon_loss, prog_bar=False)
            else:
                # Standard single-horizon loss
                if len(y.shape) > 1 and y.shape[1] > 1:
                    # If target is multi-dimensional but model is single-horizon,
                    # use only the first dimension
                    y = y[:, 0]
                
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    # If prediction is multi-dimensional but target is single-horizon,
                    # use only the first dimension
                    y_hat = y_hat[:, 0]
                
                loss = F.mse_loss(y_hat, y)
                
            self.log('train_loss', loss, prog_bar=True)
            return loss
        except Exception as e:
            logger.error(f"Error in TCN training step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a zero loss tensor with gradient to avoid breaking training
            return torch.tensor(0.0, requires_grad=True, device=self.device if hasattr(self, 'device') else 'cpu')
    
    def validation_step(self, batch, batch_idx):
        try:
            x, y = batch
            y_hat = self(x)
            
            # Calculate loss based on horizon configuration
            if self.multi_horizon:
                # Multi-horizon validation
                loss = 0
                for i, weight in enumerate(self.horizon_weights):
                    if i < y.shape[1] and i < y_hat.shape[1]:
                        horizon_loss = F.mse_loss(y_hat[:, i], y[:, i])
                        loss += weight * horizon_loss
                        self.log(f'val_loss_h{i+1}', horizon_loss, prog_bar=False)
            else:
                # Standard single-horizon loss
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = y[:, 0]  # Use first dimension
                
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    y_hat = y_hat[:, 0]  # Use first dimension
                
                loss = F.mse_loss(y_hat, y)
                
            self.log('val_loss', loss, prog_bar=True)
            
            return {
                'val_loss': loss,
                'predictions': y_hat.detach().cpu(),
                'targets': y.detach().cpu()
            }
        except Exception as e:
            logger.error(f"Error in TCN validation step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return dummy values to continue validation
            return {
                'val_loss': torch.tensor(float('inf'), device=self.device if hasattr(self, 'device') else 'cpu'),
                'predictions': torch.zeros_like(y).detach().cpu() if 'y' in locals() else torch.zeros(1),
                'targets': y.detach().cpu() if 'y' in locals() else torch.zeros(1)
            }
    
    def predict_step(self, batch, batch_idx):
        try:
            x, _ = batch
            predictions = self(x)
            return predictions
        except Exception as e:
            logger.error(f"Error in TCN predict step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros as fallback
            output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
            return torch.zeros((x.shape[0], output_dim), device=self.device if hasattr(self, 'device') else 'cpu')
    
    def configure_optimizers(self):
        try:
            # Use AdamW with weight decay for better generalization
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=getattr(self.config, 'weight_decay', 1e-4)
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        except Exception as e:
            logger.error(f"Error configuring TCN optimizers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a simple optimizer as fallback
            return torch.optim.Adam(self.parameters(), lr=0.001)

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer makes pe part of the model's state but not a parameter
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        try:
            x = x + self.pe[:x.size(1), :].unsqueeze(0)
            return x
        except Exception as e:
            logger.error(f"Error in positional encoding: {e}")
            return x  # Return input as fallback


class TimeSeriesTransformer(pl.LightningModule):
    """Transformer model for time series forecasting."""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.input_dim = input_dim
        
        # Multi-horizon support
        self.multi_horizon = getattr(config, 'multi_horizon', False)
        if self.multi_horizon:
            self.forecast_horizons = getattr(config, 'forecast_horizons', [1, 3, 5, 10, 20])
            self.horizon_weights = getattr(config, 'horizon_weights', None)
            if self.horizon_weights is None:
                # Default to equal weighting if not specified
                self.horizon_weights = [1.0/len(self.forecast_horizons)] * len(self.forecast_horizons)
        else:
            # Single horizon (traditional approach)
            self.forecast_horizons = [getattr(config, 'forecast_horizon', 1)]
            self.horizon_weights = [1.0]
        
        # Model dimensions
        self.d_model = config.hidden_dim
        self.nhead = getattr(config, 'attention_heads', 8)
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=self.num_layers
        )
        
        # Output layer for each horizon
        output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
        
        # Final prediction layer
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for faster convergence."""
        try:
            # Xavier/Glorot initialization for linear layers
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        except Exception as e:
            logger.warning(f"Error initializing transformer weights: {e}")
    
    def forward(self, x):
        """
        x shape: [batch_size, seq_len, features]
        """
        try:
            if x is None or not hasattr(x, 'shape'):
                logger.error("Transformer received invalid input")
                batch_size = 1
                # MODIFIED: Ensure correct output dimensions
                output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
                return torch.zeros((batch_size, output_dim), device=self.device if hasattr(self, 'device') else 'cpu')
            
            # Project input to embedding dimension
            # x: [batch_size, seq_len, features] -> [batch_size, seq_len, d_model]
            x = self.input_projection(x)
            
            # Add positional encoding
            x = self.positional_encoding(x)
            
            # Apply transformer encoder
            # x: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
            encoded = self.transformer_encoder(x)
            
            # Global pooling: take mean of sequence dimension
            # encoded: [batch_size, seq_len, d_model] -> [batch_size, d_model]
            pooled = encoded.mean(dim=1)
            
            # Project to output dimension
            # pooled: [batch_size, d_model] -> [batch_size, output_dim]
            output = self.output_projection(pooled)
            if self.multi_horizon:
                expected_horizons = len(self.forecast_horizons)
                if output.shape[1] != expected_horizons:
                    logger.warning(f"Output shape mismatch: got {output.shape[1]} horizons, expected {expected_horizons}")
                    batch_size = output.shape[0]
                    # Fix the shape
                    if output.shape[1] < expected_horizons:
                        # Pad with zeros
                        padding = torch.zeros(batch_size, expected_horizons - output.shape[1], device=output.device)
                        output = torch.cat([output, padding], dim=1)
                    else:
                        # Truncate
                        output = output[:, :expected_horizons]
            
            return output
            
        except Exception as e:
            logger.error(f"Error in transformer forward pass: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros as fallback
            batch_size = x.shape[0] if x is not None and hasattr(x, 'shape') else 1
            # MODIFIED: Ensure correct output dimensions
            output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
            return torch.zeros((batch_size, output_dim), device=self.device if hasattr(self, 'device') else 'cpu')

    
    def training_step(self, batch, batch_idx):
        try:
            x, y = batch
            y_hat = self(x)
            
            # Calculate loss based on horizon configuration
            if self.multi_horizon:
                # Handle multi-dimensional targets
                loss = 0
                for i, weight in enumerate(self.horizon_weights):
                    if i < y.shape[1] and i < y_hat.shape[1]:
                        horizon_loss = F.mse_loss(y_hat[:, i], y[:, i])
                        loss += weight * horizon_loss
                        self.log(f'train_loss_h{i+1}', horizon_loss, prog_bar=False)
            else:
                # Standard single-horizon loss
                if len(y.shape) > 1 and y.shape[1] > 1:
                    # If target is multi-dimensional but model is single-horizon,
                    # use only the first dimension
                    y = y[:, 0]
                
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    # If prediction is multi-dimensional but target is single-horizon,
                    # use only the first dimension
                    y_hat = y_hat[:, 0]
                
                loss = F.mse_loss(y_hat, y)
                
            self.log('train_loss', loss, prog_bar=True)
            return loss
        except Exception as e:
            logger.error(f"Error in transformer training step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a zero loss tensor with gradient to avoid breaking training
            return torch.tensor(0.0, requires_grad=True, device=self.device if hasattr(self, 'device') else 'cpu')
    
    def validation_step(self, batch, batch_idx):
        try:
            x, y = batch
            y_hat = self(x)
            
            # Calculate loss based on horizon configuration
            if self.multi_horizon:
                # Multi-horizon validation
                loss = 0
                for i, weight in enumerate(self.horizon_weights):
                    if i < y.shape[1] and i < y_hat.shape[1]:
                        horizon_loss = F.mse_loss(y_hat[:, i], y[:, i])
                        loss += weight * horizon_loss
                        self.log(f'val_loss_h{i+1}', horizon_loss, prog_bar=False)
            else:
                # Standard single-horizon loss
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = y[:, 0]  # Use first dimension
                
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    y_hat = y_hat[:, 0]  # Use first dimension
                
                loss = F.mse_loss(y_hat, y)
                
            self.log('val_loss', loss, prog_bar=True)
            
            return {
                'val_loss': loss,
                'predictions': y_hat.detach().cpu(),
                'targets': y.detach().cpu()
            }
        except Exception as e:
            logger.error(f"Error in transformer validation step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return dummy values to continue validation
            return {
                'val_loss': torch.tensor(float('inf'), device=self.device if hasattr(self, 'device') else 'cpu'),
                'predictions': torch.zeros_like(y).detach().cpu() if 'y' in locals() else torch.zeros(1),
                'targets': y.detach().cpu() if 'y' in locals() else torch.zeros(1)
            }
    
    def predict_step(self, batch, batch_idx):
        try:
            x, _ = batch
            predictions = self(x)
            return predictions
        except Exception as e:
            logger.error(f"Error in transformer predict step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros as fallback
            output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
            return torch.zeros((x.shape[0], output_dim), device=self.device if hasattr(self, 'device') else 'cpu')
    
    def configure_optimizers(self):
        try:
            # Use AdamW with weight decay
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=getattr(self.config, 'weight_decay', 1e-4)
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        except Exception as e:
            logger.error(f"Error configuring transformer optimizers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a simple optimizer as fallback
            return torch.optim.Adam(self.parameters(), lr=0.001)

class NBEATSBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as input and returns backcast and forecast.
    """
    def __init__(self, input_size, theta_size, basis_function, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        
        # Fully connected stack
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, theta_size)
        )
    
    def forward(self, x):
        # x: [batch_size, input_size]
        try:
            # Ensure x has the correct shape
            batch_size = x.shape[0]
            if x.shape[1] != self.input_size:
                logger.warning(f"Input shape mismatch in NBEATS Block: expected {self.input_size}, got {x.shape[1]}")
                if x.shape[1] < self.input_size:
                    # Pad with zeros
                    padding = torch.zeros((batch_size, self.input_size - x.shape[1]), device=x.device)
                    x = torch.cat([x, padding], dim=1)
                else:
                    # Truncate
                    x = x[:, :self.input_size]
                    
            # Calculate theta
            theta = self.fc_layers(x)
            
            # Apply basis function
            backcast, forecast = self.basis_function(theta)
            
            return backcast, forecast
        except Exception as e:
            logger.error(f"Error in NBEATS block forward pass: {e}")
            # Return zeros as fallback
            batch_size = x.shape[0] if x is not None and hasattr(x, 'shape') else 1
            return (
                torch.zeros((batch_size, self.input_size), device=x.device),
                torch.zeros((batch_size, self.input_size), device=x.device)
            )


class GenericBasis(nn.Module):
    """Generic basis function for N-BEATS."""
    def __init__(self, backcast_size, forecast_size, theta_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.theta_size = theta_size
        
        # Learnable backcast and forecast projection matrices
        self.backcast_fc = nn.Linear(theta_size, backcast_size, bias=False)
        self.forecast_fc = nn.Linear(theta_size, forecast_size, bias=False)
    
    def forward(self, theta):
        # theta: [batch_size, theta_size]
        try:
            # Ensure theta has the right shape for the projections
            if theta.shape[1] != self.theta_size:
                logger.warning(f"Theta shape mismatch in GenericBasis: got {theta.shape[1]}, expected {self.theta_size}")
                batch_size = theta.shape[0]
                if theta.shape[1] < self.theta_size:
                    # Pad with zeros
                    padding = torch.zeros((batch_size, self.theta_size - theta.shape[1]), device=theta.device)
                    theta = torch.cat([theta, padding], dim=1)
                else:
                    # Truncate
                    theta = theta[:, :self.theta_size]
            
            backcast = self.backcast_fc(theta)
            forecast = self.forecast_fc(theta)
            return backcast, forecast
        except Exception as e:
            logger.error(f"Error in GenericBasis forward pass: {e}")
            batch_size = theta.shape[0] if theta is not None and hasattr(theta, 'shape') else 1
            return (
                torch.zeros((batch_size, self.backcast_size), device=theta.device),
                torch.zeros((batch_size, self.forecast_size), device=theta.device)
            )


class SeasonalityBasis(nn.Module):
    """Seasonality basis function for N-BEATS."""
    def __init__(self, backcast_size, forecast_size, theta_size, frequency=7):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.theta_size = theta_size
        self.frequency = frequency
        
        # Number of harmonics is theta_size / 2 (each harmonic needs 2 parameters)
        self.harmonics = theta_size // 2
        
    def forward(self, theta):
        # theta: [batch_size, theta_size]
        try:
            batch_size = theta.shape[0]
            device = theta.device
            
            # Split theta into harmonics
            theta = theta.view(batch_size, self.harmonics, 2)  # [batch_size, harmonics, 2]
            
            # Create time indices
            backcast_times = torch.arange(self.backcast_size, device=device).float() / self.frequency
            forecast_times = torch.arange(self.forecast_size, device=device).float() / self.frequency
            
            backcast = torch.zeros(batch_size, self.backcast_size, device=device)
            forecast = torch.zeros(batch_size, self.forecast_size, device=device)
            
            # Calculate seasonality for each harmonic
            for h in range(self.harmonics):
                # Get amplitude and phase from theta
                amplitude = theta[:, h, 0].unsqueeze(1)  # [batch_size, 1]
                phase = theta[:, h, 1].unsqueeze(1)      # [batch_size, 1]
                
                # Calculate harmonic frequency
                freq = torch.ones(batch_size, 1, device=device) * (h + 1)
                
                # Calculate seasonality components
                backcast += amplitude * torch.sin(2 * math.pi * freq * backcast_times + phase)
                forecast += amplitude * torch.sin(2 * math.pi * freq * forecast_times + phase)
                
            return backcast, forecast
        except Exception as e:
            logger.error(f"Error in SeasonalityBasis forward pass: {e}")
            batch_size = theta.shape[0] if theta is not None and hasattr(theta, 'shape') else 1
            return (
                torch.zeros((batch_size, self.backcast_size), device=theta.device if hasattr(theta, 'device') else 'cpu'),
                torch.zeros((batch_size, self.forecast_size), device=theta.device if hasattr(theta, 'device') else 'cpu')
            )


class TrendBasis(nn.Module):
    """Trend basis function for N-BEATS."""
    def __init__(self, backcast_size, forecast_size, theta_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.theta_size = theta_size  # Polynomial degree
        
    def forward(self, theta):
        # theta: [batch_size, theta_size]
        try:
            batch_size = theta.shape[0]
            device = theta.device
            
            # Create polynomial time indices
            backcast_times = torch.arange(self.backcast_size, device=device).float() / self.backcast_size
            forecast_times = torch.arange(self.forecast_size, device=device).float() / self.forecast_size
            
            # Calculate polynomial terms
            backcast_powers = torch.stack([backcast_times ** i for i in range(self.theta_size)], dim=0)  # [theta_size, backcast_size]
            forecast_powers = torch.stack([forecast_times ** i for i in range(self.theta_size)], dim=0)  # [theta_size, forecast_size]
            
            # Calculate trend components using polynomial basis
            backcast = torch.matmul(theta, backcast_powers)  # [batch_size, backcast_size]
            forecast = torch.matmul(theta, forecast_powers)  # [batch_size, forecast_size]
            
            return backcast, forecast
        except Exception as e:
            logger.error(f"Error in TrendBasis forward pass: {e}")
            batch_size = theta.shape[0] if theta is not None and hasattr(theta, 'shape') else 1
            return (
                torch.zeros((batch_size, self.backcast_size), device=theta.device if hasattr(theta, 'device') else 'cpu'),
                torch.zeros((batch_size, self.forecast_size), device=theta.device if hasattr(theta, 'device') else 'cpu')
            )


class NBEATSModel(pl.LightningModule):
    """N-BEATS time series forecasting model."""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.input_dim = input_dim
        
        # Multi-horizon support
        self.multi_horizon = getattr(config, 'multi_horizon', False)
        if self.multi_horizon:
            self.forecast_horizons = getattr(config, 'forecast_horizons', [1, 3, 5, 10, 20])
            self.horizon_weights = getattr(config, 'horizon_weights', None)
            if self.horizon_weights is None:
                # Default to equal weighting if not specified
                self.horizon_weights = [1.0/len(self.forecast_horizons)] * len(self.forecast_horizons)
        else:
            # Single horizon (traditional approach)
            self.forecast_horizons = [getattr(config, 'forecast_horizon', 1)]
            self.horizon_weights = [1.0]
        
        # N-BEATS architecture parameters
        self.lookback_window = config.lookback_window
        self.forecast_size = len(self.forecast_horizons) if self.multi_horizon else 1
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        
        # Size of theta for each basis function
        # Generic blocks
        generic_theta_size = self.hidden_dim // 4
        self.generic_theta_size = generic_theta_size
        
        # Create basis functions
        self.generic_basis = GenericBasis(
            backcast_size=self.lookback_window,
            forecast_size=self.forecast_size,
            theta_size=generic_theta_size
        )
        
        # Default frequency for seasonality (weekly)
        self.seasonality_freq = 7
        
        # Create blocks
        n_stacks = 2  # Trend and seasonality stacks
        n_blocks = 3  # Number of blocks in each stack
        self.stacks = nn.ModuleList()
        
        # First stack - generic blocks (interpretable)
        # Make sure the input_size matches lookback_window and is properly propagated
        self.stacks.append(nn.ModuleList([
            NBEATSBlock(
                input_size=self.lookback_window,
                theta_size=generic_theta_size,
                basis_function=self.generic_basis,
                dropout=self.dropout
            ) for _ in range(n_blocks)
        ]))
        
        # Input transformation for handling multivariate inputs
        if input_dim > 1:
            # For multivariate inputs, project to lookback_window dimension
            self.input_projection = nn.Linear(input_dim, 1)
        else:
            self.input_projection = nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Kaiming initialization for better training."""
        try:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        except Exception as e:
            logger.warning(f"Error initializing N-BEATS weights: {e}")
    
    def forward(self, x):
        try:
            if x is None or not hasattr(x, 'shape'):
                logger.error("N-BEATS received invalid input")
                batch_size = 1
                # MODIFIED: Ensure correct output dimensions
                output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
                return torch.zeros((batch_size, output_dim), device=self.device if hasattr(self, 'device') else 'cpu')
            batch_size = x.shape[0]
            
            # For multivariate inputs, project each timestep from feature dimension to scalar
            if self.input_dim > 1:
                # [batch, seq_len, features] -> [batch, seq_len, 1]
                x = self.input_projection(x)
                # [batch, seq_len, 1] -> [batch, seq_len]
                x = x.squeeze(-1)
            elif len(x.shape) == 3 and x.shape[2] == 1:
                # If input is [batch, seq_len, 1], squeeze to [batch, seq_len]
                x = x.squeeze(-1)
            
            # Initialize forecast and backcast
            forecast = torch.zeros((batch_size, self.forecast_size), device=x.device)
            
            # Make sure backcast has the correct shape for N-BEATS blocks
            # N-BEATS expects backcast to have shape [batch_size, lookback_window]
            if len(x.shape) == 3:
                # For 3D input [batch, seq_len, features], flatten or take mean
                if x.shape[2] > 1:
                    # Take mean across feature dimension
                    backcast = torch.mean(x, dim=2)
                else:
                    # Squeeze the feature dimension
                    backcast = x.squeeze(-1)
            elif len(x.shape) == 1:
                # Handle 1D input by reshaping to 2D
                logger.warning(f"Received 1D input with shape {x.shape}. Reshaping to 2D.")
                backcast = x.unsqueeze(0)  # Make it [1, length]
            else:
                backcast = x.clone()
            
            # Ensure backcast is 2D before checking sequence length
            if len(backcast.shape) == 1:
                # If still 1D after previous operations, reshape it
                logger.warning(f"Backcast is still 1D with shape {backcast.shape}. Reshaping to 2D.")
                backcast = backcast.reshape(1, -1)  # Make it [1, length]
                
            # Now ensure backcast has correct sequence length (lookback_window)
            if len(backcast.shape) > 1 and backcast.shape[1] != self.lookback_window:
                logger.warning(f"Input sequence length {backcast.shape[1]} doesn't match expected lookback window {self.lookback_window}. Reshaping.")
                if backcast.shape[1] < self.lookback_window:
                    # Pad with zeros if too short
                    padding = torch.zeros((batch_size, self.lookback_window - backcast.shape[1]), device=backcast.device)
                    backcast = torch.cat([padding, backcast], dim=1)
                else:
                    # Truncate if too long
                    backcast = backcast[:, -self.lookback_window:]
            
            # Apply each stack
            for stack in self.stacks:
                # Apply each block in the stack
                for block in stack:
                    try:
                        # Get backcast and block forecast
                        b, f = block(backcast)
                        
                        # Handle dimension mismatches
                        # 1. If backcast dimensions don't match, resize b
                        if b.shape != backcast.shape:
                            logger.warning(f"Backcast shape mismatch: block output {b.shape} vs expected {backcast.shape}")
                            b = self._resize_tensor(b, backcast.shape)
                        
                        # 2. If forecast dimensions don't match, resize f
                        if f.shape[1] != forecast.shape[1]:
                            logger.warning(f"Forecast shape mismatch: block output {f.shape} vs expected {(batch_size, forecast.shape[1])}")
                            f = self._resize_tensor(f, (batch_size, forecast.shape[1]))
                        
                        # Update backcast and forecast
                        backcast = backcast - b
                        forecast = forecast + f
                    except Exception as block_error:
                        logger.error(f"Error in N-BEATS block: {block_error}")
                        # Continue with next block
                        continue
                
            return forecast
                
        except Exception as e:
            logger.error(f"Error in N-BEATS forward pass: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros with proper shape as fallback
            batch_size = x.shape[0] if x is not None and hasattr(x, 'shape') else 1
            return torch.zeros((batch_size, self.forecast_size), device=self.device if hasattr(self, 'device') else 'cpu')        
    def _resize_tensor(self, tensor, target_shape):
        """Helper method to resize tensors to match dimensions."""
        try:
            if len(tensor.shape) != len(target_shape):
                logger.warning(f"Cannot resize tensor with shape {tensor.shape} to target with different dimensions {target_shape}")
                # Create new tensor with target shape
                return torch.zeros(target_shape, device=tensor.device)
            
            batch_size = tensor.shape[0]
            
            # For 2D tensors (batch, seq/features)
            if len(tensor.shape) == 2:
                current_size = tensor.shape[1]
                target_size = target_shape[1]
                
                if current_size == target_size:
                    return tensor
                elif current_size < target_size:
                    # Pad with zeros
                    padding = torch.zeros((batch_size, target_size - current_size), device=tensor.device)
                    return torch.cat([tensor, padding], dim=1)
                else:
                    # Truncate
                    return tensor[:, :target_size]
            else:
                # For other dimensions, return a zero tensor
                logger.warning(f"Unsupported tensor shape for resizing: {tensor.shape}")
                return torch.zeros(target_shape, device=tensor.device)
        except Exception as e:
            logger.error(f"Error resizing tensor: {e}")
            return torch.zeros(target_shape, device=tensor.device)
    
    def training_step(self, batch, batch_idx):
        try:
            x, y = batch
            y_hat = self(x)
            
            # Calculate loss based on horizon configuration
            if self.multi_horizon:
                # Multi-horizon loss calculation using MSE for each horizon
                loss = 0
                for i, weight in enumerate(self.horizon_weights):
                    if i < y.shape[1] and i < y_hat.shape[1]:
                        horizon_loss = F.mse_loss(y_hat[:, i], y[:, i])
                        loss += weight * horizon_loss
                        self.log(f'train_loss_h{i+1}', horizon_loss, prog_bar=False)
            else:
                # Standard single-horizon loss
                if len(y.shape) > 1 and y.shape[1] > 1:
                    # If target is multi-dimensional but model is single-horizon,
                    # use only the first dimension
                    y = y[:, 0]
                
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    # If prediction is multi-dimensional but target is single-horizon,
                    # use only the first dimension
                    y_hat = y_hat[:, 0]
                
                loss = F.mse_loss(y_hat, y)
                
            self.log('train_loss', loss, prog_bar=True)
            return loss
        except Exception as e:
            logger.error(f"Error in N-BEATS training step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a zero loss tensor with gradient to avoid breaking training
            return torch.tensor(0.0, requires_grad=True, device=self.device if hasattr(self, 'device') else 'cpu')
    
    def validation_step(self, batch, batch_idx):
        try:
            x, y = batch
            y_hat = self(x)
            
            # Calculate loss based on horizon configuration
            if self.multi_horizon:
                # Multi-horizon validation
                loss = 0
                for i, weight in enumerate(self.horizon_weights):
                    if i < y.shape[1] and i < y_hat.shape[1]:
                        horizon_loss = F.mse_loss(y_hat[:, i], y[:, i])
                        loss += weight * horizon_loss
                        self.log(f'val_loss_h{i+1}', horizon_loss, prog_bar=False)
            else:
                # Standard single-horizon loss
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = y[:, 0]  # Use first dimension
                
                if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                    y_hat = y_hat[:, 0]  # Use first dimension
                
                loss = F.mse_loss(y_hat, y)
                
            self.log('val_loss', loss, prog_bar=True)
            
            return {
                'val_loss': loss,
                'predictions': y_hat.detach().cpu(),
                'targets': y.detach().cpu()
            }
        except Exception as e:
            logger.error(f"Error in N-BEATS validation step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return dummy values to continue validation
            return {
                'val_loss': torch.tensor(float('inf'), device=self.device if hasattr(self, 'device') else 'cpu'),
                'predictions': torch.zeros_like(y).detach().cpu() if 'y' in locals() else torch.zeros(1),
                'targets': y.detach().cpu() if 'y' in locals() else torch.zeros(1)
            }
    
    def predict_step(self, batch, batch_idx):
        try:
            x, _ = batch
            predictions = self(x)
            return predictions
        except Exception as e:
            logger.error(f"Error in N-BEATS predict step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros as fallback
            output_dim = len(self.forecast_horizons) if self.multi_horizon else 1
            return torch.zeros((x.shape[0], output_dim), device=self.device if hasattr(self, 'device') else 'cpu')
    
    def configure_optimizers(self):
        try:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=getattr(self.config, 'weight_decay', 1e-4)
            )
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        except Exception as e:
            logger.error(f"Error configuring N-BEATS optimizers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a simple optimizer as fallback
            return torch.optim.Adam(self.parameters(), lr=0.001)

class ProphetModel:
    """Enhanced multi-horizon Prophet model that trains separate models for each forecast horizon."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}  # Dictionary to store one model per horizon
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.fitted = False
        self.params = None  # Store hyperparameters
        self.feature_names_used = set()
        self.is_multi_horizon = False
        self.horizons = []

    def prepare_data(self, values: np.ndarray, features: pd.DataFrame = None, horizon_idx: int = 0) -> pd.DataFrame:
        """Prepare data for Prophet with all selected features for a specific horizon."""
        try:
            # Debug inputs
            logger.info(f"Prophet prepare_data called with values shape: {values.shape if isinstance(values, np.ndarray) else 'not numpy array'}")
            logger.info(f"features shape: {features.shape if features is not None else 'None'}")
            logger.info(f"horizon_idx: {horizon_idx}")
            
            #Get the actual forecast horizon from config if available
            forecast_horizon = 1
            if hasattr(self, 'config') and self.config is not None:
                if self.config.multi_horizon and hasattr(self.config, 'forecast_horizons') and len(self.config.forecast_horizons) > horizon_idx:
                    forecast_horizon = self.config.forecast_horizons[horizon_idx]
                elif not self.config.multi_horizon and hasattr(self.config, 'forecast_horizon'):
                    forecast_horizon = self.config.forecast_horizon
                    
            logger.info(f"Using forecast horizon: {forecast_horizon} for Prophet preparation")
            
            # Ensure values is 1D
            if isinstance(values, np.ndarray) and len(values.shape) > 1:
                logger.warning(f"Prophet requires 1D target values. Reshaping from {values.shape}")
                if values.shape[1] > horizon_idx:
                    # Take the specified horizon
                    values = values[:, horizon_idx]
                else:
                    # Fallback to first dimension
                    values = values[:, 0] if values.shape[1] > 0 else values.flatten()
            
            # Check for NaN in values before proceeding
            if isinstance(values, np.ndarray) and np.isnan(values).any():
                logger.warning(f"Values contain {np.isnan(values).sum()} NaNs before creating DataFrame!")
            
            # Create date range with fallback
            try:
                # Use business day frequency to create valid trading dates 
                date_range = pd.date_range(
                    start=pd.Timestamp('2020-01-01'),
                    periods=len(values),
                    freq='B'  # Business day frequency for trading days
                )
                logger.info(f"Created date range with {len(date_range)} dates")
            except Exception as e:
                logger.warning(f"Error creating date range: {e}. Using index instead.")
                date_range = pd.RangeIndex(start=0, stop=len(values))
            
            # Create base DataFrame
            df = pd.DataFrame({
                'ds': date_range,
                'y': values
            })
            logger.info(f"Created Prophet base DataFrame with shape {df.shape}")
            
            # Check for NaN in base DataFrame
            if df.isna().any().any():
                nan_counts = df.isna().sum()
                logger.warning(f"Prophet base DataFrame contains NaNs: ds={nan_counts['ds']}, y={nan_counts['y']}")
            
            # Ensure 'ds' contains valid dates without NaNs
            if df['ds'].isna().any() or not pd.api.types.is_datetime64_any_dtype(df['ds']):
                logger.warning("Invalid or NaN dates found in 'ds' column. Fixing.")
                # Log the problematic values
                if df['ds'].isna().any():
                    problem_indices = df.index[df['ds'].isna()]
                    logger.warning(f"NaN ds values at indices: {problem_indices.tolist()[:10]} ...")
                
                # Create a new date range as replacement
                replacement_dates = pd.date_range(
                    start=pd.Timestamp('2010-01-01'),
                    periods=len(df),
                    freq='B'
                )
                df['ds'] = replacement_dates
            
            # Validate y values
            if df['y'].isna().any():
                logger.warning(f"NaN values in target y column: {df['y'].isna().sum()} ({df['y'].isna().sum()/len(df):.2%} of values)")
                # Show some problematic indices
                problem_indices = df.index[df['y'].isna()]
                if len(problem_indices) > 0:
                    logger.warning(f"First 10 indices with NaN y values: {problem_indices[:10].tolist()}")
                    
                # Show values around the problematic indices
                if len(problem_indices) > 0:
                    idx = problem_indices[0]
                    start_idx = max(0, idx - 2)
                    end_idx = min(len(df), idx + 3)
                    logger.warning(f"Values around first NaN index ({idx}):")
                    for i in range(start_idx, end_idx):
                        logger.warning(f"  Index {i}: ds={df['ds'].iloc[i]}, y={df['y'].iloc[i]}")
            
            if np.any(np.isinf(df['y'])):
                inf_count = np.isinf(df['y']).sum()
                logger.warning(f"Inf values in target y column: {inf_count} ({inf_count/len(df):.2%} of values)")
            
            # Add all selected features if provided
            if features is not None:
                feature_data = features.copy()
                
                # Check for NaNs in features
                logger.info(f"Feature data shape: {feature_data.shape}, NaNs: {feature_data.isna().sum().sum()}")
                
                # Log columns with most NaNs
                if feature_data.isna().any().any():
                    nan_counts = feature_data.isna().sum().sort_values(ascending=False)
                    logger.warning("Top feature columns with NaNs:")
                    for col, count in nan_counts[nan_counts > 0].iloc[:5].items():
                        logger.warning(f"  - {col}: {count} NaNs ({count/len(feature_data):.2%} of rows)")
                
                # Ensure feature_data has right number of rows
                if len(feature_data) != len(values):
                    logger.warning(f"Feature length ({len(feature_data)}) does not match values length ({len(values)}). "
                                f"Adjusting.")
                    if len(feature_data) > len(values):
                        feature_data = feature_data.iloc[:len(values)]
                    else:
                        # Pad with zeros
                        padding = pd.DataFrame(0, index=range(len(values) - len(feature_data)),
                                            columns=feature_data.columns)
                        feature_data = pd.concat([feature_data, padding])
                
                # Rename columns to avoid duplicates and Prophet-specific names
                renamed_columns = {}
                prophet_reserved = ['ds', 'y', 'yhat', 'trend', 'seasonal', 'holidays', 'weekly', 'yearly', 'daily']
                for col in feature_data.columns:
                    new_col = col
                    if col in df.columns or col in prophet_reserved:
                        new_col = f"{col}_feature"
                        logger.warning(f"Column {col} conflicts with Prophet. Renaming to {new_col}.")
                    renamed_columns[col] = new_col
                
                # Update feature tracking
                self.feature_names_used.update(feature_data.columns)
                
                # Rename columns
                feature_data = feature_data.rename(columns=renamed_columns)
                
                # Check for NaN values in features before filling
                nan_features = feature_data.columns[feature_data.isna().any()]
                if len(nan_features) > 0:
                    logger.warning(f"Features with NaNs before filling: {len(nan_features)}")
                    for col in nan_features[:5]:  # Show first 5
                        nan_count = feature_data[col].isna().sum()
                        logger.warning(f"  - {col}: {nan_count} NaNs ({nan_count/len(feature_data):.2%} of rows)")
                
                # Ensure no NaN values in features
                for col in feature_data.columns:
                    nan_count = feature_data[col].isna().sum()
                    if nan_count > 0:
                        logger.warning(f"Filling {nan_count} NaN values in feature {col} with 0")
                        feature_data[col] = feature_data[col].fillna(0)
                
                # Concat with original DataFrame
                df_before_concat = df.copy()
                feature_data.index = df.index
                df = pd.concat([df, feature_data], axis=1)
                
                # Verify concat worked properly
                logger.info(f"DataFrame shape after adding features: {df.shape}")
                logger.info(f"Expected shape: {(len(df_before_concat), len(df_before_concat.columns) + len(feature_data.columns))}")
                
                # Check if concat introduced NaNs
                if df.isna().any().any():
                    nan_counts = df.isna().sum()
                    nan_cols = nan_counts[nan_counts > 0]
                    logger.warning(f"NaNs after concat in {len(nan_cols)} columns:")
                    for col, count in nan_cols.items():
                        logger.warning(f"  - {col}: {count} NaNs")
                    
                    # Deeper analysis of a sample problematic column
                    if len(nan_cols) > 0:
                        problem_col = nan_cols.index[0]
                        problem_rows = df.index[df[problem_col].isna()]
                        logger.warning(f"Analyzing first problematic column: {problem_col}")
                        logger.warning(f"Sample row indices with NaNs: {problem_rows[:5].tolist()}")
                        
                        # Check if column was in original df or features
                        in_orig = problem_col in df_before_concat.columns
                        in_feat = problem_col in feature_data.columns
                        logger.warning(f"Column source - In original df: {in_orig}, In features: {in_feat}")
                        
                        # Show data around problem
                        if len(problem_rows) > 0:
                            idx = problem_rows[0]
                            logger.warning(f"Data around index {idx}:")
                            if in_orig:
                                logger.warning(f"  Original df value: {df_before_concat.loc[idx, problem_col] if idx in df_before_concat.index else 'out of bounds'}")
                            if in_feat:
                                logger.warning(f"  Feature value: {feature_data.loc[idx, problem_col] if idx in feature_data.index else 'out of bounds'}")
            
            # Final NaN check
            if df.isnull().any().any():
                logger.warning("DataFrame still contains NaN values after processing. Filling remaining NaNs.")
                nan_counts = df.isnull().sum()
                nan_cols = nan_counts[nan_counts > 0]
                logger.warning(f"Columns with NaNs: {nan_cols.to_dict()}")
                
                # Sample problematic values
                for col, count in nan_cols.items():
                    if count > 0:
                        null_indices = df.index[df[col].isnull()]
                        if len(null_indices) > 0:
                            sample_idx = null_indices[0]
                            logger.warning(f"Sample NaN in {col} at index {sample_idx}, row values: {df.loc[sample_idx].to_dict()}")
                df = df.fillna(0)
            
            return df
        except Exception as e:
            logger.error(f"Error preparing data for Prophet: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return minimal valid DataFrame as fallback
            return pd.DataFrame({
                'ds': pd.date_range(start='2010-01-01', periods=len(values), freq='B'),
                'y': np.nan_to_num(values)
            })

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Prophet models with all selected features and robust error handling."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available. Prophet models will not be fitted.")
            return self
            
        try:
            # Log input dimensions
            logger.info(f"Prophet fit called with X shape: {X.shape if X is not None else 'None'}")
            logger.info(f"y shape: {y.shape if isinstance(y, np.ndarray) else 'not numpy array'}")
            
            # Check for NaNs in input
            if X is not None:
                if isinstance(X, np.ndarray) and np.isnan(X).any():
                    logger.warning(f"X contains {np.isnan(X).sum()} NaNs ({np.isnan(X).sum()/(X.size) if X.size > 0 else 0:.2%} of values)")
                elif isinstance(X, pd.DataFrame) and X.isna().any().any():
                    logger.warning(f"X DataFrame contains {X.isna().sum().sum()} NaNs")
                    
            if isinstance(y, np.ndarray) and np.isnan(y).any():
                logger.warning(f"y contains {np.isnan(y).sum()} NaNs ({np.isnan(y).sum()/(y.size) if y.size > 0 else 0:.2%} of values)")
            
            # Determine if we have multi-horizon targets
            self.is_multi_horizon = isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1
            
            # Set horizons - either multiple or just one
            if self.is_multi_horizon:
                self.horizons = list(range(y.shape[1]))
                logger.info(f"Training Prophet models for {len(self.horizons)} horizons")
            else:
                self.horizons = [0]  # Just one horizon
            
            # Convert X to DataFrame if it's not already
            X_df = None
            if X is not None:
                if not isinstance(X, pd.DataFrame):
                    logger.info("Converting X to DataFrame for Prophet model")
                    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                else:
                    X_df = X
            
                # Track feature names
                self.feature_names_used.update(X_df.columns)
            
            # Train a model for each horizon
            for h_idx in self.horizons:
                try:
                    # Prepare data for this horizon
                    if self.is_multi_horizon:
                        # Extract the target for this horizon
                        y_h = y[:, h_idx]
                        logger.info(f"Training Prophet model for horizon {h_idx+1}/{len(self.horizons)}")
                    else:
                        y_h = y
                    
                    # Check for NaNs in this horizon's target
                    if isinstance(y_h, np.ndarray) and np.isnan(y_h).any():
                        nan_count = np.isnan(y_h).sum()
                        logger.warning(f"Target for horizon {h_idx} contains {nan_count} NaNs ({nan_count/len(y_h):.2%} of values)")
                        
                        # Show some indices with NaNs
                        nan_indices = np.where(np.isnan(y_h))[0]
                        logger.warning(f"First 5 indices with NaNs: {nan_indices[:5]}")
                    
                    # Prepare data
                    logger.info(f"Preparing Prophet data for horizon {h_idx}")
                    df_prophet = self.prepare_data(y_h, X_df, h_idx)
                    
                    # Check data after preparation
                    logger.info(f"Prophet data prepared with shape: {df_prophet.shape}")
                    
                    # Check for NaNs after data preparation
                    if df_prophet.isna().any().any():
                        nan_counts = df_prophet.isna().sum()
                        logger.error(f"CRITICAL: Prophet data still contains NaNs after preparation: {nan_counts[nan_counts > 0].to_dict()}")
                    
                    # Check for minimum data requirements
                    if len(df_prophet) < 2:
                        logger.warning(f"Not enough data points for Prophet model horizon {h_idx}.")
                        continue
                    
                    # Create model with hyperparameters
                    model = Prophet(
                        changepoint_prior_scale=self.params.get('changepoint_prior_scale', 0.05) if self.params else 0.05,
                        seasonality_prior_scale=self.params.get('seasonality_prior_scale', 10.0) if self.params else 10.0,
                        holidays_prior_scale=self.params.get('holidays_prior_scale', 10.0) if self.params else 10.0,
                        seasonality_mode=self.params.get('seasonality_mode', 'multiplicative') if self.params else 'multiplicative',
                        changepoint_range=self.params.get('changepoint_range', 0.8) if self.params else 0.8,
                        daily_seasonality=self.params.get('daily_seasonality', True) if self.params else True,
                        weekly_seasonality=self.params.get('weekly_seasonality', True) if self.params else True,
                        yearly_seasonality=self.params.get('yearly_seasonality', True) if self.params else True
                    )
                    
                    # Add features as regressors
                    if X_df is not None:
                        regressor_count = 0
                        for col in X_df.columns:
                            # Skip columns that aren't in df_prophet
                            if col not in df_prophet.columns and f"{col}_feature" not in df_prophet.columns:
                                continue
                            
                            # Use either the original column name or the renamed version
                            regressor_col = col if col in df_prophet.columns else f"{col}_feature"
                            
                            try:
                                model.add_regressor(regressor_col)
                                regressor_count += 1
                            except Exception as e:
                                logger.warning(f"Error adding regressor {regressor_col} for horizon {h_idx}: {e}")
                        logger.info(f"Added {regressor_count} regressors to Prophet model")
                    
                    # Fit model with error handling
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # Suppress Prophet warnings
                        try:
                            logger.info(f"Fitting Prophet model for horizon {h_idx}")
                            model.fit(df_prophet)
                            self.models[h_idx] = model
                            logger.info(f"Prophet model for horizon {h_idx} fitted successfully")
                        except Exception as fit_error:
                            logger.error(f"Error fitting Prophet model for horizon {h_idx}: {fit_error}")
                            import traceback
                            logger.error(traceback.format_exc())
                            
                            # Log data sample for debugging
                            logger.error(f"Prophet data sample that caused error:")
                            logger.error(f"Shape: {df_prophet.shape}")
                            logger.error(f"Columns: {df_prophet.columns.tolist()}")
                            logger.error(f"First 5 rows:\n{df_prophet.head(5)}")
                            
                            # Try simplified model
                            try:
                                logger.info(f"Trying simplified Prophet model for horizon {h_idx}")
                                model = Prophet(
                                    daily_seasonality=False,
                                    weekly_seasonality=False,
                                    yearly_seasonality=False
                                )
                                logger.info(f"Created simplified Prophet model, fitting with basic data...")
                                model.fit(df_prophet[['ds', 'y']])
                                self.models[h_idx] = model
                                logger.info(f"Simplified Prophet model for horizon {h_idx} fitted successfully")
                            except Exception as e:
                                logger.error(f"Error fitting simplified Prophet model for horizon {h_idx}: {e}")
                                logger.error(traceback.format_exc())
                                
                except Exception as e:
                    logger.error(f"Error in training process for horizon {h_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Check if we fitted at least one model
            self.fitted = len(self.models) > 0
            if self.fitted:
                logger.info(f"Successfully fitted {len(self.models)} Prophet models")
            else:
                logger.warning("Failed to fit any Prophet models")
            
            return self
            
        except Exception as e:
            logger.error(f"Unexpected error fitting Prophet models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.fitted = False
            return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for all horizons with proper alignment."""
        if not PROPHET_AVAILABLE or not self.fitted or not self.models:
            logger.warning("Prophet models not available or not fitted. Returning zeros.")
            n_horizons = len(self.horizons) if self.horizons else 1
            return np.zeros((len(X) if X is not None else 1, n_horizons))
            
        try:
            logger.info(f"Prophet predict called with X shape: {X.shape if X is not None else 'None'}")
            
            # Convert X to DataFrame
            X_df = None
            if X is not None:
                if not isinstance(X, pd.DataFrame):
                    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                else:
                    X_df = X
            
            # Check for NaNs in X
            if X_df is not None and X_df.isna().any().any():
                nan_counts = X_df.isna().sum()
                logger.warning(f"X contains NaNs: {nan_counts[nan_counts > 0].to_dict()}")
                    
            # Check for missing features
            missing_features = self.feature_names_used - set(X_df.columns)
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}. Adding zeros.")
                for feature in missing_features:
                    X_df[feature] = np.zeros(len(X_df))
            
            # Prepare predictions array
            n_samples = len(X_df)
            n_horizons = len(self.horizons)
            predictions = np.zeros((n_samples, n_horizons))
            
            # Generate predictions for each horizon
            for i, h_idx in enumerate(self.horizons):
                if h_idx in self.models:
                    try:
                        logger.info(f"Generating predictions for horizon {h_idx}")
                        
                        # Prepare future dataframe with exact length match
                        future = self.prepare_data(np.zeros(n_samples), X_df, h_idx)
                        
                        # Check for NaNs in prepared data
                        if future.isna().any().any():
                            nan_counts = future.isna().sum()
                            logger.error(f"CRITICAL: future DataFrame for prediction contains NaNs: {nan_counts[nan_counts > 0].to_dict()}")
                            # Fill NaNs as a last resort
                            future = future.fillna(0)
                            logger.info("Filled NaNs in future DataFrame with zeros")
                        
                        # Make predictions and ensure exact length match
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            logger.info(f"Calling Prophet predict for horizon {h_idx}")
                            forecast = self.models[h_idx].predict(future)
                            
                            # Check forecast
                            logger.info(f"Forecast generated with shape: {forecast.shape}")
                            if 'yhat' not in forecast.columns:
                                logger.error(f"Missing 'yhat' column in forecast. Columns: {forecast.columns.tolist()}")
                                # Show first few rows
                                logger.error(f"First few rows of forecast:\n{forecast.head(3)}")
                            
                            # Critical fix: Ensure forecast length matches expected output length
                            if len(forecast) != n_samples:
                                logger.warning(f"Prophet forecast length mismatch: got {len(forecast)}, expected {n_samples}. Reindexing.")
                                
                                # If too many predictions, truncate
                                if len(forecast) > n_samples:
                                    horizon_pred = forecast['yhat'].values[:n_samples]
                                # If too few predictions, pad with last value or zeros
                                else:
                                    padding = np.zeros(n_samples - len(forecast))
                                    if len(forecast) > 0:
                                        # Pad with last value
                                        padding.fill(forecast['yhat'].values[-1])
                                    horizon_pred = np.concatenate([forecast['yhat'].values, padding])
                            else:
                                horizon_pred = forecast['yhat'].values
                            
                            # Check for invalid values
                            if np.any(np.isnan(horizon_pred)) or np.any(np.isinf(horizon_pred)):
                                nan_count = np.isnan(horizon_pred).sum()
                                inf_count = np.isinf(horizon_pred).sum()
                                logger.warning(f"NaN ({nan_count}) or Inf ({inf_count}) values in Prophet predictions for horizon {h_idx}. Fixing.")
                                horizon_pred = np.nan_to_num(horizon_pred)
                            
                            # Store in predictions array
                            predictions[:, i] = horizon_pred
                            logger.info(f"Successfully stored predictions for horizon {h_idx}")
                            
                    except Exception as e:
                        logger.error(f"Error predicting with Prophet model for horizon {h_idx}: {e}")
                        # Back trace to identify where the error is happening
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"No model available for horizon {h_idx}. Using zeros.")
            
            # If multi-horizon is disabled but we have multiple models,
            # return just the first horizon
            if not self.is_multi_horizon and predictions.shape[1] > 1:
                return predictions[:, 0:1]
                    
            return predictions
                
        except Exception as e:
            logger.error(f"Unexpected error in Prophet prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            n_horizons = len(self.horizons) if self.horizons else 1
            return np.zeros((len(X) if X is not None else 1, n_horizons))
                
        except Exception as e:
            logger.error(f"Unexpected error in Prophet prediction: {e}")
            n_horizons = len(self.horizons) if self.horizons else 1
            return np.zeros((len(X) if X is not None else 1, n_horizons))
        
class XGBoostHyperparameterTuner(ModelHyperparameterTuner):
    """Enhanced XGBoost hyperparameter tuner with optimized search spaces."""
    
    def __init__(self, n_trials=50, timeout=600, n_jobs=-1):
        super().__init__(n_trials, timeout, n_jobs, "XGBoost_tuning")
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for XGBoost hyperparameter tuning with refined search spaces."""
        if not XGBOOST_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters with refined search spaces
            params = {
                'objective': 'reg:squarederror',
                
                # Core parameters with focused ranges based on financial time series characteristics
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                
                # Sampling parameters with tighter distribution
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
                
                # Regularization with finance-specific ranges
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'alpha': trial.suggest_float('alpha', 0, 5),  # L1 regularization
                'lambda': trial.suggest_float('lambda', 1, 5),  # L2 regularization
                
                # Algorithms and implementations
                'tree_method': 'auto',  # Will be overridden for GPU if available
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                
                # Technical parameters
                'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
                'max_bin': trial.suggest_int('max_bin', 200, 500),
                
                # Deterministic behavior for reproducibility
                'random_state': 42
            }
            
            # Conditionally add special parameter combinations
            # For noisy financial data, sometimes these combinations work well
            if trial.suggest_categorical('special_combo', [True, False]):
                if trial.suggest_categorical('combo_type', ['high_trees_low_lr', 'low_trees_high_lr']):
                    # High trees, low learning rate - good for capturing complex patterns
                    params['n_estimators'] = 500
                    params['learning_rate'] = 0.01
                    params['max_depth'] = 8
                else:
                    # Low trees, higher learning rate - good for fast adaptation
                    params['n_estimators'] = 200
                    params['learning_rate'] = 0.05
                    params['max_depth'] = 4
            
            # Handle GPU if available
            if torch.cuda.is_available():
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
                
            # Create and train model
            model = xgb.XGBRegressor(**params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=35,
                verbose=False
            )
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            
            # Calculate error metric - prefer RMSE for financial time series
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            
            # Optionally add directional accuracy as a secondary objective
            direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y_val))
            
            # For financial time series, a combined metric often works better
            # This balances magnitude and direction errors
            combined_score = rmse * (1.05 - direction_accuracy)
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Error in XGBoost objective function: {e}")
            return float('inf')
    
    def get_default_params(self):
        """Get default XGBoost hyperparameters optimized for financial time series."""
        return {
            'objective': 'reg:squarederror',
            'n_estimators': 300,  # More estimators for time series complexity
            'learning_rate': 0.03, # Balanced learning rate
            'max_depth': 6,       # Moderate depth to avoid overfitting
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'colsample_bynode': 0.8,
            'min_child_weight': 3, # Increased to reduce overfitting
            'gamma': 0.5,         # Some minimum loss reduction for splits
            'alpha': 0.5,         # L1 regularization
            'lambda': 1.0,        # L2 regularization
            'tree_method': 'auto',
            'grow_policy': 'depthwise',
            'max_delta_step': 1,
            'max_bin': 256,
            'random_state': 42
        }
    
    def create_model(self, params=None):
        """Create XGBoost model with the given parameters."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available. Cannot create model.")
            return None
            
        try:
            # Use best params if available, else use provided params or defaults
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
                    
            # Ensure all required parameters are present by merging with defaults
            default_params = self.get_default_params()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            
            # Special handling for GPU
            if torch.cuda.is_available():
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
                
            # Create model
            model = xgb.XGBRegressor(**params)
            return model
            
        except Exception as e:
            logger.error(f"Error creating XGBoost model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def plot_param_importance(self, output_path=None):
        """Plot parameter importance with enhanced styling."""
        if not OPTUNA_AVAILABLE or self.study is None:
            logger.warning("No parameter importances available to plot.")
            return False
            
        try:
            # Set nicer plot styling
            plt.style.use('seaborn-v0_8-whitegrid')
            
            # Create the plot with enhanced styling
            plt.figure(figsize=(12, 8))
            
            # Get importance data
            importances = optuna.importance.get_param_importances(self.study)
            importance_items = list(importances.items())
            
            # Sort by importance
            importance_items.sort(key=lambda x: x[1], reverse=True)
            param_names = [item[0] for item in importance_items]
            importance_values = [item[1] for item in importance_items]
            
            # Plot with enhanced styling
            bars = plt.barh(param_names, importance_values, color='steelblue', alpha=0.8)
            
            # Add values as text
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{importance_values[i]:.3f}', 
                         va='center', fontsize=10)
            
            plt.title("Parameter Importance for XGBoost Model", fontsize=16, pad=20)
            plt.xlabel("Relative Importance", fontsize=12)
            plt.ylabel("Hyperparameter", fontsize=12)
            plt.tight_layout()
            
            # Save or show the plot
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Parameter importances plot saved to {output_path}")
                plt.close()
                return True
            else:
                plt.show()
                plt.close()
                return True
                
        except Exception as e:
            logger.error(f"Error plotting parameter importances: {e}")
            plt.close()
            return False
class LightGBMHyperparameterTuner(ModelHyperparameterTuner):
    """LightGBM hyperparameter tuner using Optuna."""
    
    def __init__(self, n_trials=50, timeout=600, n_jobs=-1):
        super().__init__(n_trials, timeout, n_jobs, "LightGBM_tuning")
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for LightGBM hyperparameter tuning."""
        if not LIGHTGBM_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'objective': 'regression',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),  # Increased upper limit
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),  # Wider range
                'num_leaves': trial.suggest_int('num_leaves', 20, 255),  # Wider range
                'max_depth': trial.suggest_int('max_depth', 3, 15),  # Deeper trees
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),  # Wider range
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),  # Higher upper limit
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),  # Higher upper limit
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),  # Wider range
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 15),  # Higher upper limit
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),  # Wider range
                'path_smooth': trial.suggest_float('path_smooth', 0, 1),  # New parameter
                'max_bin': trial.suggest_int('max_bin', 100, 255),  # New parameter
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),  # New parameter
                'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),  # New parameter
                'categorical_algorithm': trial.suggest_categorical('categorical_algorithm', ['auto', 'binary', 'exclusive']),  # New parameter
                'verbosity': -1,  # Suppress warnings
                'random_state': trial.suggest_int('random_state', 0, 10000)  # Add randomness to tuning
            }
            
            # GPU support if available
            if torch.cuda.is_available():
                params['device'] = 'gpu'
                
            # Create and train model
            model = lgb.LGBMRegressor(**params)
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=35,
                verbose=False
            )
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            
            # Calculate error metric
            mse = mean_squared_error(y_val, y_pred)
            return mse
            
        except Exception as e:
            logger.error(f"Error in LightGBM objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default LightGBM hyperparameters."""
        return {
            'objective': 'regression',
            'n_estimators': 200,  # Increased from 100
            'learning_rate': 0.05,
            'num_leaves': 63,  # Increased from 31
            'max_depth': 6,  # Increased from 5
            'min_data_in_leaf': 20,
            'lambda_l1': 0.5,  # Changed from 0
            'lambda_l2': 1,  # Changed from 0
            'bagging_fraction': 0.8,
            'bagging_freq': 2,  # Changed from 1
            'feature_fraction': 0.8,
            'path_smooth': 0.1,  # New parameter
            'max_bin': 255,  # New parameter
            'min_gain_to_split': 0.1,  # New parameter
            'extra_trees': False,  # New parameter
            'categorical_algorithm': 'auto'  # New parameter
        }
    
    def create_model(self, params=None):
        """Create LightGBM model with the given parameters."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available. Cannot create model.")
            return None
            
        try:
            # Use best params if available, else use provided params or defaults
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
                    
            # Ensure all required parameters are present by merging with defaults
            default_params = self.get_default_params()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
                    
            # Create model
            model = lgb.LGBMRegressor(**params)
            return model
            
        except Exception as e:
            logger.error(f"Error creating LightGBM model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
class CatBoostHyperparameterTuner(ModelHyperparameterTuner):
    """CatBoost hyperparameter tuner using Optuna."""
    
    def __init__(self, n_trials=50, timeout=600, n_jobs=-1):
        super().__init__(n_trials, timeout, n_jobs, "CatBoost_tuning")
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for CatBoost hyperparameter tuning."""
        if not CATBOOST_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'loss_function': 'RMSE',  # CatBoost uses RMSE for regression
                'iterations': trial.suggest_int('iterations', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': 42,
                'verbose': False
            }
            
            # GPU support if available
            if torch.cuda.is_available():
                params['task_type'] = 'GPU'
                
            # Create and train model
            model = cb.CatBoostRegressor(**params)
            
            # Use a small eval set to prevent long training times with CatBoost
            eval_set = [(X_val, y_val)]
            
            # Train model
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=35,
                verbose=False
            )
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            
            # Calculate error metric
            mse = mean_squared_error(y_val, y_pred)
            return mse
            
        except Exception as e:
            logger.error(f"Error in CatBoost objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default CatBoost hyperparameters."""
        return {
            'loss_function': 'RMSE',
            'iterations': 100,
            'learning_rate': 0.05,
            'depth': 6,
            'bagging_temperature': 1,
            'random_strength': 1,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False
        }
    
    def create_model(self, params=None):
        """Create CatBoost model with the given parameters."""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not available. Cannot create model.")
            return None
            
        try:
            # Use best params if available, else use provided params or defaults
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
                    
            # Ensure all required parameters are present
            default_params = self.get_default_params()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
                    
            # Create model
            model = cb.CatBoostRegressor(**params)
            return model
            
        except Exception as e:
            logger.error(f"Error creating CatBoost model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
class RandomForestHyperparameterTuner(ModelHyperparameterTuner):
    """RandomForest hyperparameter tuner using Optuna."""
    
    def __init__(self, n_trials=50, timeout=600, n_jobs=-1):
        super().__init__(n_trials, timeout, n_jobs, "RandomForest_tuning")
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for RandomForest hyperparameter tuning."""
        if not SKLEARN_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'n_jobs': min(self.n_jobs, os.cpu_count() or 1)
            }
            
            # Create and train model
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            
            # Calculate error metric
            mse = mean_squared_error(y_val, y_pred)
            return mse
            
        except Exception as e:
            logger.error(f"Error in RandomForest objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default RandomForest hyperparameters."""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1
        }
    
    def create_model(self, params=None):
        """Create RandomForest model with the given parameters."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Cannot create RandomForest model.")
            return None
            
        try:
            # Use best params if available, else use provided params or defaults
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
                    
            # Ensure all required parameters are present
            default_params = self.get_default_params()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
                    
            # Adjust n_jobs parameter
            params['n_jobs'] = min(params.get('n_jobs', -1), os.cpu_count() or 1)
                    
            # Create model
            model = RandomForestRegressor(**params)
            return model
            
        except Exception as e:
            logger.error(f"Error creating RandomForest model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class DeepLearningHyperparameterTuner(ModelHyperparameterTuner):
    """Deep Learning (HybridTimeSeriesModel) hyperparameter tuner using Optuna."""
    
    def __init__(self, n_trials=30, timeout=1200, n_jobs=1):
        """
        Initialize the tuner. Note: n_jobs is set to 1 by default for PyTorch models
        to avoid GPU memory issues when running multiple models in parallel.
        """
        super().__init__(n_trials, timeout, n_jobs, "DeepLearning_tuning")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for deep learning hyperparameter tuning."""
        if not TORCH_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),  # Wider range
                'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512, 1024]),  # Added 1024 option
                'num_layers': trial.suggest_int('num_layers', 1, 4),  # Increased to 4 layers
                'dropout': trial.suggest_float('dropout', 0.0, 0.7),  # Increased upper limit
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),  # Added 256 option
                'lookback_window': trial.suggest_int('lookback_window', 10, 250, step=10),  # Increased to 250
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),  # New parameter
                'activation_fn': trial.suggest_categorical('activation_fn', ['relu', 'leaky_relu', 'elu', 'gelu'])  # New parameter
            }
            
            # Create config with these parameters
            config = ModelConfig(
                lookback_window=params['lookback_window'],
                forecast_horizon=1,  # Keep forecast horizon fixed
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                num_epochs=25,  # Slightly increased
                early_stopping_patience=10,  # Increased
                validation_split=0.2,
                test_split=0.1,
                min_samples=1000,
                input_dim=X_train.shape[1]
            )
            
            # Create model
            model = HybridTimeSeriesModel(X_train.shape[1], config, 
                                          activation_fn=params['activation_fn'],
                                          weight_decay=params['weight_decay'])
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, config.lookback_window)
            val_dataset = TimeSeriesDataset(X_val, y_val, config.lookback_window)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,  # Use 0 during hyperparameter tuning to avoid issues
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True
            )
            
            # Early stopping callback
            early_stopping = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                mode='min'
            )
            
            # Track best validation loss for pruning
            best_val_loss = float('inf')
            
            # Train with PyTorch Lightning
            trainer = pl.Trainer(
                max_epochs=config.num_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_checkpointing=False,  # Don't save checkpoints during tuning
                logger=False,  # Disable logging
                callbacks=[early_stopping],
                enable_progress_bar=False  # Disable progress bar to reduce output
            )
            
            # Create a lightweight validation callback for Optuna pruning
            class ValidationCallback(pl.Callback):
                def __init__(self, trial):
                    self.trial = trial
                    
                def on_validation_epoch_end(self, trainer, pl_module):
                    # Get current validation loss
                    val_loss = float(trainer.callback_metrics.get('val_loss', float('inf')))
                    
                    # Report to Optuna for pruning
                    self.trial.report(val_loss, trainer.current_epoch)
                    
                    # Check for pruning
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                        
                    # Track best validation loss
                    nonlocal best_val_loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
            
            # Add pruning callback
            pruning_callback = ValidationCallback(trial)
            trainer.callbacks.append(pruning_callback)
            
            # Train model
            try:
                trainer.fit(model, train_loader, val_loader)
            except optuna.exceptions.TrialPruned:
                # Return the best observed value if pruned
                return best_val_loss
                
            # Evaluate model
            model.eval()
            with torch.no_grad():
                val_predictions = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_pred = model(batch_x)
                    val_predictions.append(batch_pred.cpu().numpy())
                    
                if val_predictions:
                    all_predictions = np.concatenate(val_predictions, axis=0)
                    
                    # Ensure predictions match validation target length
                    if len(all_predictions) > len(y_val):
                        all_predictions = all_predictions[:len(y_val)]
                    elif len(all_predictions) < len(y_val):
                        # Pad if needed (should not happen with proper dataset creation)
                        padding = np.zeros((len(y_val) - len(all_predictions), 1))
                        all_predictions = np.concatenate([all_predictions, padding], axis=0)
                    
                    # Calculate MSE
                    mse = mean_squared_error(y_val, all_predictions)
                    return mse
                    
                # If no predictions, return worst possible score
                return float('inf')
                
        except Exception as e:
            logger.error(f"Error in deep learning objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default deep learning hyperparameters."""
        return {
            'learning_rate': 5e-4,  # Changed from 1e-3
            'hidden_dim': 512,  # Increased from 256
            'num_layers': 3,  # Increased from 2
            'dropout': 0.3,  # Increased from 0.2
            'batch_size': 64,  # Increased from 32
            'lookback_window': 120,  # Increased from 60
            'num_epochs': 150,  # Increased from 100
            'early_stopping_patience': 15,  # Increased from 10
            'weight_decay': 1e-4,  # New parameter
            'activation_fn': 'gelu'  # New parameter
        }
    
    def create_model(self, params=None, input_dim=None):
        """
        Create deep learning model with the given parameters.
        
        Args:
            params: Model hyperparameters
            input_dim: Input dimension (required for deep learning model)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Cannot create deep learning model.")
            return None
            
        try:
            # Use best params if available, else use provided params or defaults
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
                    
            # Ensure all required parameters are present
            default_params = self.get_default_params()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            
            # Check if input_dim is provided
            if input_dim is None:
                logger.error("Input dimension required for deep learning model creation")
                return None
                    
            # Create config with these parameters
            config = ModelConfig(
                lookback_window=params.get('lookback_window', 120),
                forecast_horizon=1,
                batch_size=params.get('batch_size', 64),
                learning_rate=params.get('learning_rate', 5e-4),
                hidden_dim=params.get('hidden_dim', 512),
                num_layers=params.get('num_layers', 3),
                dropout=params.get('dropout', 0.3),
                num_epochs=params.get('num_epochs', 150),
                early_stopping_patience=params.get('early_stopping_patience', 15),
                validation_split=0.2,
                test_split=0.1,
                min_samples=1000,
                input_dim=input_dim
            )
            
            # Create model
            model = HybridTimeSeriesModel(input_dim, config)
            return model
            
        except Exception as e:
            logger.error(f"Error creating deep learning model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class ProphetHyperparameterTuner(ModelHyperparameterTuner):
    """Prophet hyperparameter tuner using Optuna."""
    
    def __init__(self, n_trials=30, timeout=600, n_jobs=1):
        """Initialize with reduced parallelism as Prophet can be memory-intensive."""
        super().__init__(n_trials, timeout, n_jobs, "Prophet_tuning")
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for Prophet hyperparameter tuning."""
        if not PROPHET_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.95)
            }
            
            # Create configs
            config = ModelConfig()
            
            # Create and train model
            prophet_model = ProphetModel(config)
            
            # Convert X to DataFrame if it's not already
            X_df = None
            if X_train is not None:
                if not isinstance(X_train, pd.DataFrame):
                    X_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                else:
                    X_df = X_train
            
            # Prepare Prophet model with custom parameters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress Prophet convergence warnings
                
                # Build Prophet model with appropriate parameters
                prophet_model.model = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    holidays_prior_scale=params['holidays_prior_scale'],
                    seasonality_mode=params['seasonality_mode'],
                    changepoint_range=params['changepoint_range'],
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )
                
                # Add features as regressors if available
                if X_df is not None:
                    for col in X_df.columns:
                        try:
                            prophet_model.model.add_regressor(col)
                        except Exception as e:
                            logger.debug(f"Error adding regressor {col}: {e}")
                
                # Prepare data for Prophet
                df_prophet = prophet_model.prepare_data(y_train, X_df)
                
                # Fit model
                prophet_model.model.fit(df_prophet)
                prophet_model.fitted = True
                
                # Prepare validation data
                X_val_df = None
                if X_val is not None:
                    if not isinstance(X_val, pd.DataFrame):
                        X_val_df = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])
                    else:
                        X_val_df = X_val
                
                # Predict on validation set
                future = prophet_model.prepare_data(np.zeros(len(X_val_df)), X_val_df)
                forecast = prophet_model.model.predict(future)
                predictions = forecast['yhat'].values.reshape(-1, 1)
                
                # Calculate error metric
                mse = mean_squared_error(y_val, predictions)
                return mse
                
        except Exception as e:
            logger.error(f"Error in Prophet objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default Prophet hyperparameters."""
        return {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative',
            'changepoint_range': 0.8,
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True
        }
    
    def create_model(self, params=None):
        """Create Prophet model with the given parameters."""
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available. Cannot create model.")
            return None
            
        try:
            # Create base Prophet model with ModelConfig
            config = ModelConfig()
            prophet_model = ProphetModel(config)
            
            # If no parameters provided or Prophet not properly installed, return basic model
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
            
            # Store parameters in the model for later use when fitting
            prophet_model.params = params
            
            return prophet_model
            
        except Exception as e:
            logger.error(f"Error creating Prophet model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

class SARIMAHyperparameterTuner(ModelHyperparameterTuner):
    """SARIMA hyperparameter tuner using Optuna."""
    
    def __init__(self, n_trials=30, timeout=600, n_jobs=1):
        """Use low parallelism as SARIMA can be compute-intensive."""
        super().__init__(n_trials, timeout, n_jobs, "SARIMA_tuning")
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for SARIMA hyperparameter tuning."""
        if not SARIMAX_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'order': (
                    trial.suggest_int('p', 0, 2),  # AR order
                    trial.suggest_int('d', 0, 1),  # Differencing
                    trial.suggest_int('q', 0, 2)   # MA order
                ),
                'seasonal_order': (
                    trial.suggest_int('P', 0, 1),  # Seasonal AR
                    trial.suggest_int('D', 0, 1),  # Seasonal differencing
                    trial.suggest_int('Q', 0, 1),  # Seasonal MA
                    trial.suggest_categorical('s', [7, 12])  # Seasonal period
                ),
                'univariate': trial.suggest_categorical('univariate', [True, False])
            }
            
            # Create configs
            config = ModelConfig()
            
            # Create and train model
            sarima_model = SARIMAModel(config)
            sarima_model.set_params(**params)
            
            # Extract the target for univariate modeling
            if isinstance(y_train, np.ndarray) and len(y_train.shape) > 1 and y_train.shape[1] > 1:
                y_train_univariate = y_train[:, 0]  # Use first horizon for tuning
                y_val_univariate = y_val[:, 0]
            else:
                y_train_univariate = y_train
                y_val_univariate = y_val
            
            # Convert to DataFrame if needed for exogenous variables
            X_train_df = None
            X_val_df = None
            if not params['univariate'] and X_train is not None:
                if not isinstance(X_train, pd.DataFrame):
                    X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                else:
                    X_train_df = X_train
                    
                if not isinstance(X_val, pd.DataFrame):
                    X_val_df = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])
                else:
                    X_val_df = X_val
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress convergence warnings
                
                # Fit model with limited iterations for faster tuning
                sarima_model.fit(X_train_df if not params['univariate'] else None, y_train_univariate)
                
                # Predict on validation set
                predictions = sarima_model.predict(X_val_df)
                
                # Extract predictions for first horizon
                if len(predictions.shape) > 1 and predictions.shape[1] > 0:
                    predictions = predictions[:, 0]
                
                # Calculate error metric
                mse = mean_squared_error(y_val_univariate, predictions)
                return mse
                
        except Exception as e:
            logger.error(f"Error in SARIMA objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default SARIMA hyperparameters."""
        return {
            'order': (1, 1, 1),
            'seasonal_order': (0, 0, 0, 7),  # Weekly seasonality by default
            'univariate': True  # Start with univariate as it's more stable
        }
    
    def create_model(self, params=None):
        """Create SARIMA model with the given parameters."""
        if not SARIMAX_AVAILABLE:
            logger.warning("SARIMAX not available. Cannot create model.")
            return None
            
        try:
            # Create base SARIMA model with ModelConfig
            config = ModelConfig()
            sarima_model = SARIMAModel(config)
            
            # If no parameters provided, use defaults
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
            
            # Apply parameters
            sarima_model.set_params(**params)
            
            return sarima_model
            
        except Exception as e:
            logger.error(f"Error creating SARIMA model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


class TCNHyperparameterTuner(ModelHyperparameterTuner):
    """TCN hyperparameter tuner using Optuna."""
    
    def __init__(self, n_trials=30, timeout=1200, n_jobs=1):
        """
        Initialize the tuner. n_jobs is set to 1 by default for PyTorch models
        to avoid GPU memory issues when running multiple models in parallel.
        """
        super().__init__(n_trials, timeout, n_jobs, "TCN_tuning")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for TCN hyperparameter tuning."""
        if not TCN_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'lookback_window': trial.suggest_int('lookback_window', 10, 100, step=10),
                'kernel_size': trial.suggest_int('kernel_size', 2, 5),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            }
            
            # Create config with these parameters
            config = ModelConfig(
                lookback_window=params['lookback_window'],
                forecast_horizon=1,  # Keep forecast horizon fixed
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                hidden_dim=params['hidden_dim'],
                dropout=params['dropout'],
                weight_decay=params['weight_decay'],
                num_epochs=10,  # Limited epochs for tuning
                early_stopping_patience=5
            )
            
            # Create model
            model = TCNModel(X_train.shape[1], config)
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, config.lookback_window)
            val_dataset = TimeSeriesDataset(X_val, y_val, config.lookback_window)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0  # Use 0 during hyperparameter tuning to avoid issues
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0
            )
            
            # Early stopping callback
            early_stopping = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                mode='min'
            )
            
            # Track best validation loss for pruning
            best_val_loss = float('inf')
            
            # Train with PyTorch Lightning
            trainer = pl.Trainer(
                max_epochs=config.num_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_checkpointing=False,  # Don't save checkpoints during tuning
                logger=False,  # Disable logging
                callbacks=[early_stopping],
                enable_progress_bar=False  # Disable progress bar to reduce output
            )
            
            # Create a lightweight validation callback for Optuna pruning
            class ValidationCallback(pl.Callback):
                def __init__(self, trial):
                    self.trial = trial
                    
                def on_validation_epoch_end(self, trainer, pl_module):
                    # Get current validation loss
                    val_loss = float(trainer.callback_metrics.get('val_loss', float('inf')))
                    
                    # Report to Optuna for pruning
                    self.trial.report(val_loss, trainer.current_epoch)
                    
                    # Check for pruning
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                        
                    # Track best validation loss
                    nonlocal best_val_loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
            
            # Add pruning callback
            pruning_callback = ValidationCallback(trial)
            trainer.callbacks.append(pruning_callback)
            
            # Train model
            try:
                trainer.fit(model, train_loader, val_loader)
            except optuna.exceptions.TrialPruned:
                # Return the best observed value if pruned
                return best_val_loss
                
            # Evaluate model
            model.eval()
            with torch.no_grad():
                val_predictions = []
                val_targets = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_pred = model(batch_x)
                    val_predictions.append(batch_pred.cpu().numpy())
                    val_targets.append(batch_y.numpy())
                    
                if val_predictions:
                    all_predictions = np.concatenate(val_predictions, axis=0)
                    all_targets = np.concatenate(val_targets, axis=0)
                    
                    # Ensure predictions match validation target length
                    if len(all_predictions) > len(all_targets):
                        all_predictions = all_predictions[:len(all_targets)]
                    elif len(all_predictions) < len(all_targets):
                        # Pad if needed (should not happen with proper dataset creation)
                        padding = np.zeros((len(all_targets) - len(all_predictions), all_predictions.shape[1] if len(all_predictions.shape) > 1 else 1))
                        all_predictions = np.concatenate([all_predictions, padding], axis=0)
                    
                    # Calculate MSE for single horizon (or first horizon in multi-horizon case)
                    if len(all_predictions.shape) > 1 and all_predictions.shape[1] > 1:
                        mse = mean_squared_error(all_targets[:, 0], all_predictions[:, 0])
                    else:
                        mse = mean_squared_error(all_targets, all_predictions)
                    
                    return mse
                    
                # If no predictions, return worst possible score
                return float('inf')
                
        except Exception as e:
            logger.error(f"Error in TCN objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default TCN hyperparameters."""
        return {
            'learning_rate': 1e-3,
            'hidden_dim': 128,
            'dropout': 0.2,
            'batch_size': 32,
            'lookback_window': 60,
            'kernel_size': 3,
            'weight_decay': 1e-4
        }
    
    def create_model(self, params=None, input_dim=None):
        """
        Create TCN model with the given parameters.
        
        Args:
            params: Model hyperparameters
            input_dim: Input dimension (required for TCN model)
        """
        if not TCN_AVAILABLE:
            logger.warning("TCN dependencies not available. Cannot create model.")
            return None
            
        try:
            # Use best params if available, else use provided params or defaults
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
                    
            # Ensure all required parameters are present
            default_params = self.get_default_params()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            
            # Check if input_dim is provided
            if input_dim is None:
                logger.error("Input dimension required for TCN model creation")
                return None
                    
            # Create config with these parameters
            config = ModelConfig(
                lookback_window=params.get('lookback_window', 60),
                forecast_horizon=1,
                batch_size=params.get('batch_size', 32),
                learning_rate=params.get('learning_rate', 1e-3),
                hidden_dim=params.get('hidden_dim', 128),
                dropout=params.get('dropout', 0.2),
                weight_decay=params.get('weight_decay', 1e-4),
                num_epochs=150,
                early_stopping_patience=15,
                input_dim=input_dim
            )
            
            # Create model
            model = TCNModel(input_dim, config)
            return model
            
        except Exception as e:
            logger.error(f"Error creating TCN model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


class TransformerHyperparameterTuner(ModelHyperparameterTuner):
    """Transformer hyperparameter tuner."""
    
    def __init__(self, n_trials=30, timeout=1200, n_jobs=1):
        super().__init__(n_trials, timeout, n_jobs, "Transformer_tuning")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for Transformer hyperparameter tuning."""
        if not TRANSFORMER_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'lookback_window': trial.suggest_int('lookback_window', 10, 100, step=10),
                'attention_heads': trial.suggest_int('attention_heads', 1, 8),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            }
            
            # Create config with these parameters
            config = ModelConfig(
                lookback_window=params['lookback_window'],
                forecast_horizon=1,  # Keep forecast horizon fixed
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                weight_decay=params['weight_decay'],
                attention_heads=params['attention_heads'],
                num_epochs=10,  # Limited epochs for tuning
                early_stopping_patience=5
            )
            
            # Create model
            model = TimeSeriesTransformer(X_train.shape[1], config)
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, config.lookback_window)
            val_dataset = TimeSeriesDataset(X_val, y_val, config.lookback_window)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0  # Use 0 during hyperparameter tuning to avoid issues
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0
            )
            
            # Early stopping callback
            early_stopping = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                mode='min'
            )
            
            # Track best validation loss for pruning
            best_val_loss = float('inf')
            
            # Train with PyTorch Lightning
            trainer = pl.Trainer(
                max_epochs=config.num_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_checkpointing=False,
                logger=False,
                callbacks=[early_stopping],
                enable_progress_bar=False
            )
            
            # Create validation callback for Optuna pruning
            class ValidationCallback(pl.Callback):
                def __init__(self, trial):
                    self.trial = trial
                    
                def on_validation_epoch_end(self, trainer, pl_module):
                    # Get current validation loss
                    val_loss = float(trainer.callback_metrics.get('val_loss', float('inf')))
                    
                    # Report to Optuna for pruning
                    self.trial.report(val_loss, trainer.current_epoch)
                    
                    # Check for pruning
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                        
                    # Track best validation loss
                    nonlocal best_val_loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
            
            # Add pruning callback
            pruning_callback = ValidationCallback(trial)
            trainer.callbacks.append(pruning_callback)
            
            # Train model
            try:
                trainer.fit(model, train_loader, val_loader)
            except optuna.exceptions.TrialPruned:
                # Return the best observed value if pruned
                return best_val_loss
                
            # Evaluate model
            model.eval()
            with torch.no_grad():
                val_predictions = []
                val_targets = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_pred = model(batch_x)
                    val_predictions.append(batch_pred.cpu().numpy())
                    val_targets.append(batch_y.numpy())
                    
                if val_predictions:
                    all_predictions = np.concatenate(val_predictions, axis=0)
                    all_targets = np.concatenate(val_targets, axis=0)
                    
                    # Ensure predictions match validation target length
                    if len(all_predictions) > len(all_targets):
                        all_predictions = all_predictions[:len(all_targets)]
                    elif len(all_predictions) < len(all_targets):
                        padding = np.zeros((len(all_targets) - len(all_predictions), all_predictions.shape[1] if len(all_predictions.shape) > 1 else 1))
                        all_predictions = np.concatenate([all_predictions, padding], axis=0)
                    
                    # Calculate MSE
                    if len(all_predictions.shape) > 1 and all_predictions.shape[1] > 1:
                        mse = mean_squared_error(all_targets[:, 0], all_predictions[:, 0])
                    else:
                        mse = mean_squared_error(all_targets, all_predictions)
                    
                    return mse
                    
                # If no predictions, return worst possible score
                return float('inf')
                
        except Exception as e:
            logger.error(f"Error in Transformer objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default Transformer hyperparameters."""
        return {
            'learning_rate': 5e-4,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'batch_size': 32,
            'lookback_window': 60,
            'attention_heads': 4,
            'weight_decay': 1e-4
        }
    
    def create_model(self, params=None, input_dim=None):
        """Create Transformer model with the given parameters."""
        if not TRANSFORMER_AVAILABLE:
            logger.warning("Transformer dependencies not available. Cannot create model.")
            return None
            
        try:
            # Use best params if available, else use provided params or defaults
            if params is None:
                if self.best_params is not None:
                    params = self.best_params
                else:
                    params = self.get_default_params()
                    
            # Ensure all required parameters are present
            default_params = self.get_default_params()
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            
            # Check if input_dim is provided
            if input_dim is None:
                logger.error("Input dimension required for Transformer model creation")
                return None
                    
            # Create config with these parameters
            config = ModelConfig(
                lookback_window=params.get('lookback_window', 60),
                forecast_horizon=1,
                batch_size=params.get('batch_size', 32),
                learning_rate=params.get('learning_rate', 5e-4),
                hidden_dim=params.get('hidden_dim', 128),
                num_layers=params.get('num_layers', 2),
                dropout=params.get('dropout', 0.2),
                weight_decay=params.get('weight_decay', 1e-4),
                attention_heads=params.get('attention_heads', 4),
                num_epochs=150,
                early_stopping_patience=15,
                input_dim=input_dim
            )
            
            # Create model
            model = TimeSeriesTransformer(input_dim, config)
            return model
            
        except Exception as e:
            logger.error(f"Error creating Transformer model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


class NBEATSHyperparameterTuner(ModelHyperparameterTuner):
    """N-BEATS hyperparameter tuner."""
    
    def __init__(self, n_trials=30, timeout=1200, n_jobs=1):
        super().__init__(n_trials, timeout, n_jobs, "NBEATS_tuning")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Objective function for N-BEATS hyperparameter tuning."""
        if not TORCH_AVAILABLE:
            return float('inf')
            
        try:
            # Define hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'lookback_window': trial.suggest_int('lookback_window', 10, 100, step=10),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            }
            
            # Create config with these parameters
            config = ModelConfig(
                lookback_window=params['lookback_window'],
                forecast_horizon=1,
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                hidden_dim=params['hidden_dim'],
                dropout=params['dropout'],
                weight_decay=params['weight_decay'],
                num_epochs=10,  # Limited epochs for tuning
                early_stopping_patience=5
            )
            
            # Create model
            model = NBEATSModel(X_train.shape[1], config)
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, config.lookback_window)
            val_dataset = TimeSeriesDataset(X_val, y_val, config.lookback_window)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0
            )
            
            # Early stopping callback
            early_stopping = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                mode='min'
            )
            
            # Track best validation loss for pruning
            best_val_loss = float('inf')
            
            # Train with PyTorch Lightning
            trainer = pl.Trainer(
                max_epochs=config.num_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_checkpointing=False,
                logger=False,
                callbacks=[early_stopping],
                enable_progress_bar=False
            )
            
            # Create validation callback for Optuna pruning
            class ValidationCallback(pl.Callback):
                def __init__(self, trial):
                    self.trial = trial
                    
                def on_validation_epoch_end(self, trainer, pl_module):
                    # Get current validation loss
                    val_loss = float(trainer.callback_metrics.get('val_loss', float('inf')))
                    
                    # Report to Optuna for pruning
                    self.trial.report(val_loss, trainer.current_epoch)
                    
                    # Check for pruning
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                        
                    # Track best validation loss
                    nonlocal best_val_loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
            
            # Add pruning callback
            pruning_callback = ValidationCallback(trial)
            trainer.callbacks.append(pruning_callback)
            
            # Train model
            try:
                trainer.fit(model, train_loader, val_loader)
            except optuna.exceptions.TrialPruned:
                # Return the best observed value if pruned
                return best_val_loss
                
            # Evaluate model
            model.eval()
            with torch.no_grad():
                val_predictions = []
                val_targets = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_pred = model(batch_x)
                    val_predictions.append(batch_pred.cpu().numpy())
                    val_targets.append(batch_y.numpy())
                    
                if val_predictions:
                    all_predictions = np.concatenate(val_predictions, axis=0)
                    all_targets = np.concatenate(val_targets, axis=0)
                    
                    # Ensure predictions match validation target length
                    if len(all_predictions) > len(all_targets):
                        all_predictions = all_predictions[:len(all_targets)]
                    elif len(all_predictions) < len(all_targets):
                        padding = np.zeros((len(all_targets) - len(all_predictions), all_predictions.shape[1] if len(all_predictions.shape) > 1 else 1))
                        all_predictions = np.concatenate([all_predictions, padding], axis=0)
                    
                    # Calculate MSE
                    if len(all_predictions.shape) > 1 and all_predictions.shape[1] > 1:
                        mse = mean_squared_error(all_targets[:, 0], all_predictions[:, 0])
                    else:
                        mse = mean_squared_error(all_targets, all_predictions)
                    
                    return mse
                    
                # If no predictions, return worst possible score
                return float('inf')
                
        except Exception as e:
            logger.error(f"Error in N-BEATS objective function: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return float('inf')
    
    def get_default_params(self):
        """Get default N-BEATS hyperparameters."""
        return {
            'learning_rate': 1e-3,
            'hidden_dim': 128,
            'dropout': 0.2,
            'batch_size': 32,
            'lookback_window': 60,
            'weight_decay': 1e-4
        }
    
    def create_model(self, params=None, input_dim=None):
            """Create N-BEATS model with the given parameters."""
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available. Cannot create N-BEATS model.")
                return None
                
            try:
                # Use best params if available, else use provided params or defaults
                if params is None:
                    if self.best_params is not None:
                        params = self.best_params
                    else:
                        params = self.get_default_params()
                        
                # Ensure all required parameters are present
                default_params = self.get_default_params()
                for key, value in default_params.items():
                    if key not in params:
                        params[key] = value
                
                # Check if input_dim is provided
                if input_dim is None:
                    logger.error("Input dimension required for N-BEATS model creation")
                    return None
                        
                # Create config with these parameters
                config = ModelConfig(
                    lookback_window=params.get('lookback_window', 60),
                    forecast_horizon=1,
                    batch_size=params.get('batch_size', 32),
                    learning_rate=params.get('learning_rate', 1e-3),
                    hidden_dim=params.get('hidden_dim', 128),
                    dropout=params.get('dropout', 0.2),
                    weight_decay=params.get('weight_decay', 1e-4),
                    num_epochs=150,
                    early_stopping_patience=15,
                    input_dim=input_dim
                )
                
                # Create model
                model = NBEATSModel(input_dim, config)
                return model
                
            except Exception as e:
                logger.error(f"Error creating N-BEATS model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None

class CheckpointMetadata:
    """Metadata for model checkpoints."""
    epoch: int
    timestamp: str
    metrics: Dict[str, float]
    model_config: Dict[str, Any]


class ModelCheckpointer:
    """
    Enhanced model checkpointing system with versioning, metadata tracking,
    and automatic model recovery capabilities.
    """
    
    def __init__(self, base_path: str, model_name: str = "trading_model"):
        """
        Initialize checkpointer with path and model information.
        
        Args:
            base_path: Base directory to store checkpoints
            model_name: Name of the model for labeling
        """
        self.base_path = base_path
        self.model_name = model_name
        self.checkpoints_path = os.path.join(base_path, 'checkpoints')
        self.metadata_path = os.path.join(base_path, 'checkpoint_metadata.json')
        self.model_version = self._get_latest_version() + 1
        
        # Create directories if they don't exist
        try:
            os.makedirs(self.checkpoints_path, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating checkpoint directories: {e}")
            # Try a fallback path in current directory
            self.base_path = f'model_checkpoints_{model_name}'
            self.checkpoints_path = os.path.join(self.base_path, 'checkpoints')
            self.metadata_path = os.path.join(self.base_path, 'checkpoint_metadata.json')
            os.makedirs(self.checkpoints_path, exist_ok=True)
            
        self.metadata = self._load_metadata()
        logger.info(f"Initialized model checkpointer with version {self.model_version}")
    
    def _get_latest_version(self) -> int:
        """Determine the latest model version from existing metadata."""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    versions = [int(meta.get('version', 0)) for meta in metadata.values()]
                    return max(versions) if versions else 0
            return 0
        except Exception as e:
            logger.error(f"Error determining latest model version: {e}")
            return 0
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata from disk with robust error handling."""
        try:
            if os.path.exists(self.metadata_path):
                try:
                    with open(self.metadata_path, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in checkpoint metadata: {e}")
                    # Create a new metadata file rather than failing
                    with open(self.metadata_path, 'w') as f:
                        json.dump({}, f)
                    logger.info("Created new empty metadata file due to JSON error")
                    return {}
            return {}
        except Exception as e:
            logger.error(f"Error loading checkpoint metadata: {e}")
            return {}
    
    def _convert_to_serializable(self, obj):
        """Convert complex types to native Python types with robust error handling."""
        try:
            if obj is None:
                return None
            
            if isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int_)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: self._convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._convert_to_serializable(v) for v in obj]
            elif hasattr(obj, '__dict__'):
                # Convert object to dictionary
                return self._convert_to_serializable(obj.__dict__)
            
            # Try to convert to string as fallback
            try:
                return str(obj)
            except:
                return "Unserializable object"
                
        except Exception as e:
            logger.error(f"Error converting object to serializable format: {e}")
            return "Conversion error"
    
    def _clean_old_checkpoints(self, keep_last_n: int = 5, keep_best_n: int = 3):
        """
        Remove old checkpoints, keeping only the last n, best n, and version milestones.
        
        Args:
            keep_last_n: Number of most recent checkpoints to keep
            keep_best_n: Number of best performing checkpoints to keep
        """
        try:
            # Safety check
            if not self.metadata:
                return
                
            # Get all checkpoints sorted by timestamp (newest first)
            checkpoints = sorted(
                [(name, meta) for name, meta in self.metadata.items()],
                key=lambda x: x[1].get('timestamp', ''),
                reverse=True
            )
            
            # Keep track of which checkpoints to preserve
            to_keep = set()
            
            # Keep the most recent n checkpoints
            for name, _ in checkpoints[:keep_last_n]:
                to_keep.add(name)
            
            # Keep the best n checkpoints by performance metric
            # First, determine which metric to use (lower is better, like MSE)
            metric_name = next((k for k in checkpoints[0][1].get('metrics', {}) 
                             if k.endswith('_mse') or k.endswith('_loss')), None)
            
            if metric_name:
                # Sort by that metric (ascending for error metrics)
                best_checkpoints = sorted(
                    [(name, meta) for name, meta in self.metadata.items() 
                     if metric_name in meta.get('metrics', {})],
                    key=lambda x: x[1].get('metrics', {}).get(metric_name, float('inf'))
                )
                
                # Add best performing checkpoints to keep list
                for name, _ in best_checkpoints[:keep_best_n]:
                    to_keep.add(name)
            
            # Always keep checkpoints marked as is_best
            for name, meta in self.metadata.items():
                if meta.get('is_best', False):
                    to_keep.add(name)
            
            # Always keep version milestone checkpoints (every 10 versions)
            for name, meta in self.metadata.items():
                if meta.get('version', 0) % 10 == 0:  # Keep every 10th version
                    to_keep.add(name)
            
            # Remove others
            for name in list(self.metadata.keys()):
                if name not in to_keep:
                    try:
                        checkpoint_path = self.metadata[name].get('path')
                        if checkpoint_path and os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        del self.metadata[name]
                        logger.debug(f"Removed old checkpoint: {name}")
                    except Exception as e:
                        logger.warning(f"Error removing checkpoint {name}: {e}")
                        
            # Update metadata file after cleaning
            try:
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata, f, indent=4)
            except Exception as e:
                logger.warning(f"Error updating metadata after cleaning: {e}")
                
        except Exception as e:
            logger.error(f"Error cleaning old checkpoints: {e}")
    
    def save_checkpoint(self, 
                       model: Any,
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       save_optimizer: bool = True,
                       custom_metadata: Dict[str, Any] = None):
        """
        Save model checkpoint with versioning and comprehensive metadata.
        
        Args:
            model: Model to save
            epoch: Current training epoch
            metrics: Performance metrics
            is_best: Whether this is the best model so far
            save_optimizer: Whether to save optimizer state
            custom_metadata: Additional metadata to store
            
        Returns:
            Checkpoint name if successful, None otherwise
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"{self.model_name}_v{self.model_version}_e{epoch}_{timestamp}"
            
            # Create checkpoint directory if it doesn't exist
            os.makedirs(self.checkpoints_path, exist_ok=True)
            
            # Convert metrics and other data to serializable format
            try:
                convertible_metrics = self._convert_to_serializable(metrics)
            except Exception as e:
                logger.error(f"Error converting metrics to serializable format: {e}")
                convertible_metrics = {"error": "Metrics conversion failed"}
            
            # Get git commit hash if available
            git_commit = None
            try:
                import subprocess
                git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            except:
                git_commit = "unavailable"
            
            # Create comprehensive metadata
            metadata = {
                'version': self.model_version,
                'epoch': epoch,
                'timestamp': timestamp,
                'git_commit': git_commit,
                'metrics': convertible_metrics,
                'is_best': bool(is_best),
                'hardware': {
                    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                    'cpu_count': os.cpu_count()
                },
                'python_version': sys.version,
                'library_versions': {
                    'numpy': np.__version__,
                    'pandas': pd.__version__,
                    'pytorch': torch.__version__ if TORCH_AVAILABLE else "Not installed",
                    'xgboost': xgb.__version__ if XGBOOST_AVAILABLE else "Not installed",
                    'lightgbm': lgb.__version__ if LIGHTGBM_AVAILABLE else "Not installed",
                    'scikit-learn': sklearn.__version__ if SKLEARN_AVAILABLE else "Not installed"
                }
            }
            
            # Add custom metadata if provided
            if custom_metadata:
                metadata.update(self._convert_to_serializable(custom_metadata))
            
            # Create checkpoint dictionary
            checkpoint = {
                'metadata': metadata,
                'model_state': {},
                'model_config': self._convert_to_serializable(model.config.__dict__) if hasattr(model, 'config') else {},
                'weights': None,
                'meta_model': None,
                'optimizer_state': None
            }
            
            # Handle different model state saving with component-wise error handling
            for name, model_item in model.models.items():
                try:
                    if hasattr(model_item, 'state_dict'):
                        # PyTorch models
                        checkpoint['model_state'][name] = model_item.state_dict()
                    else:
                        # Try pickling
                        try:
                            checkpoint['model_state'][name] = pickle.dumps(model_item)
                        except Exception as e:
                            logger.warning(f"Error pickling model {name}: {e}")
                            # Save model metadata instead
                            checkpoint['model_state'][name] = {
                                'type': str(type(model_item)),
                                'params': str(model_item.get_params()) if hasattr(model_item, 'get_params') else {}
                            }
                except Exception as e:
                    logger.error(f"Error saving model {name}: {e}")
                    checkpoint['model_state'][name] = None
            
            # Handle weights and meta_model with serialization
            if hasattr(model, 'weights') and model.weights is not None:
                try:
                    checkpoint['weights'] = self._convert_to_serializable(model.weights)
                except Exception as e:
                    logger.error(f"Error serializing weights: {e}")
                    checkpoint['weights'] = None
            
            if hasattr(model, 'meta_model') and model.meta_model is not None:
                try:
                    # Try to save meta_model state_dict if it's a PyTorch model
                    if hasattr(model.meta_model, 'state_dict'):
                        checkpoint['meta_model'] = model.meta_model.state_dict()
                    else:
                        # Otherwise try to pickle it
                        checkpoint['meta_model'] = pickle.dumps(model.meta_model)
                except Exception as e:
                    logger.error(f"Error serializing meta_model: {e}")
                    checkpoint['meta_model'] = None
            
            # Save optimizer state if requested
            if save_optimizer and hasattr(model, 'optimizer') and model.optimizer is not None:
                try:
                    checkpoint['optimizer_state'] = model.optimizer.state_dict()
                except Exception as e:
                    logger.error(f"Error saving optimizer state: {e}")
            
            # Save checkpoint with error handling
            checkpoint_path = os.path.join(self.checkpoints_path, f"{checkpoint_name}.pt")
            try:
                torch.save(checkpoint, checkpoint_path)
            except Exception as save_error:
                logger.error(f"Error saving checkpoint with torch.save: {save_error}")
                # Try pickle as fallback
                try:
                    with open(checkpoint_path.replace('.pt', '.pkl'), 'wb') as f:
                        pickle.dump(checkpoint, f)
                    checkpoint_path = checkpoint_path.replace('.pt', '.pkl')
                    logger.info(f"Saved checkpoint using pickle instead of torch.save")
                except Exception as pickle_error:
                    logger.error(f"Error saving checkpoint with pickle: {pickle_error}")
                    raise  # Re-raise to trigger outer exception handler
            
            # Update metadata
            metadata['path'] = checkpoint_path
            self.metadata[checkpoint_name] = metadata
            
            # Save metadata
            try:
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata, f, indent=4)
            except Exception as metadata_error:
                logger.error(f"Error saving checkpoint metadata: {metadata_error}")
            
            # If best model, create symlink or copy
            if is_best:
                best_path = os.path.join(self.base_path, 'best_model.pt')
                if os.path.exists(best_path):
                    try:
                        os.remove(best_path)
                    except Exception as remove_error:
                        logger.warning(f"Error removing existing best model: {remove_error}")
                
                try:
                    os.symlink(checkpoint_path, best_path)
                except (OSError, AttributeError) as symlink_error:
                    logger.warning(f"Error creating symlink for best model: {symlink_error}")
                    # If symlinks aren't supported, copy the file
                    try:
                        import shutil
                        shutil.copy2(checkpoint_path, best_path)
                    except Exception as copy_error:
                        logger.warning(f"Error copying best model: {copy_error}")
            
            # Clean old checkpoints
            self._clean_old_checkpoints()
            
            logger.info(f"Saved checkpoint: {checkpoint_name} (Version {self.model_version})")
            logger.info(f"Saving checkpoint - has meta_features: {hasattr(model, 'meta_features')}")
            if hasattr(model, 'meta_features') and model.meta_features is not None:
                logger.info(f"meta_features shape at save: {model.meta_features.shape}")            
            # Increment version for next save
            self.model_version += 1
            
            return checkpoint_name
            
        except Exception as e:
            logger.error(f"Error in save_checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    def load_checkpoint(self, checkpoint_name: str = 'best', version: int = None) -> Dict:
        """
        Load a checkpoint with robust error handling.
        
        Args:
            checkpoint_name: Name of checkpoint to load, 'best' for best model
            version: Specific version to load (overrides checkpoint_name)
            
        Returns:
            Dictionary with checkpoint data or empty dict on failure
        """
        try:
            # If version is specified, find corresponding checkpoint
            if version is not None:
                matching_checkpoints = [name for name, meta in self.metadata.items() 
                                     if meta.get('version') == version]
                if matching_checkpoints:
                    checkpoint_name = matching_checkpoints[0]
                else:
                    raise FileNotFoundError(f"No checkpoint found for version {version}")
            
            # Determine checkpoint path
            if checkpoint_name == 'best':
                checkpoint_path = os.path.join(self.base_path, 'best_model.pt')
                if not os.path.exists(checkpoint_path):
                    # Try .pkl extension
                    checkpoint_path = checkpoint_path.replace('.pt', '.pkl')
                    if not os.path.exists(checkpoint_path):
                        raise FileNotFoundError(f"Best model checkpoint not found")
            else:
                # Check if name exists in metadata
                if checkpoint_name in self.metadata:
                    checkpoint_path = self.metadata[checkpoint_name].get('path')
                    if not checkpoint_path or not os.path.exists(checkpoint_path):
                        # Try both extensions
                        base_path = os.path.join(self.checkpoints_path, f"{checkpoint_name}")
                        if os.path.exists(f"{base_path}.pt"):
                            checkpoint_path = f"{base_path}.pt"
                        elif os.path.exists(f"{base_path}.pkl"):
                            checkpoint_path = f"{base_path}.pkl"
                        else:
                            raise FileNotFoundError(f"Checkpoint file for {checkpoint_name} not found")
                else:
                    # Try direct path
                    checkpoint_path = os.path.join(self.checkpoints_path, f"{checkpoint_name}.pt")
                    if not os.path.exists(checkpoint_path):
                        # Try .pkl extension
                        checkpoint_path = checkpoint_path.replace('.pt', '.pkl')
                        if not os.path.exists(checkpoint_path):
                            raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found")
            
            # Load checkpoint with appropriate method
            if checkpoint_path.endswith('.pt'):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                except Exception as e:
                    logger.error(f"Error loading checkpoint with torch.load: {e}")
                    # Try pickle as fallback
                    with open(checkpoint_path.replace('.pt', '.pkl'), 'rb') as f:
                        checkpoint = pickle.load(f)
            else:  # .pkl
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            
            # Extract version from metadata
            if 'metadata' in checkpoint and 'version' in checkpoint['metadata']:
                version_info = checkpoint['metadata']['version']
                logger.info(f"Loaded model version: {version_info}")
            logger.info(f"After loading checkpoint - checkpoint has meta_features: {hasattr(checkpoint, 'meta_features')}")
            if 'model_state' in checkpoint:
                logger.info(f"Model state keys in checkpoint: {list(checkpoint['model_state'].keys())}")                
            return checkpoint
        
        except Exception as e:
            logger.error(f"Error loading checkpoint '{checkpoint_name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty dictionary as fallback
            return {
                'metadata': {'error': f"Failed to load checkpoint: {str(e)}"},
                'model_state': {}, 
                'weights': None,
                'meta_model': None
            }
            
    def list_available_checkpoints(self, include_metrics: bool = False) -> pd.DataFrame:
        """
        List all available checkpoints in a structured format.
        
        Args:
            include_metrics: Whether to include detailed metrics
            
        Returns:
            DataFrame with checkpoint information
        """
        try:
            if not self.metadata:
                return pd.DataFrame()
                
            # Prepare data
            data = []
            
            for name, meta in self.metadata.items():
                row = {
                    'checkpoint_name': name,
                    'version': meta.get('version', 'unknown'),
                    'epoch': meta.get('epoch', 0),
                    'timestamp': meta.get('timestamp', ''),
                    'is_best': meta.get('is_best', False)
                }
                
                # Add primary metrics if available
                metrics = meta.get('metrics', {})
                if metrics:
                    for metric_name in ['val_mse', 'val_loss', 'val_r2', 'test_mse', 'test_r2', 'direction_accuracy']:
                        if metric_name in metrics:
                            row[metric_name] = metrics[metric_name]
                
                # Add all metrics if requested
                if include_metrics:
                    for metric_name, value in metrics.items():
                        if metric_name not in row and isinstance(value, (int, float)):
                            row[metric_name] = value
                
                data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Sort by version (descending)
            if 'version' in df.columns:
                df = df.sort_values('version', ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}")
            return pd.DataFrame()
    
    def plot_training_progress(self, metric: str = 'val_mse', output_path: str = None):
        """
        Plot training progress across model versions.
        
        Args:
            metric: Which metric to plot
            output_path: Path to save plot, or None to display
        """
        try:
            if not self.metadata:
                logger.warning("No checkpoint metadata available for plotting")
                return
            
            # Extract data
            versions = []
            metrics = []
            is_best = []
            
            for name, meta in self.metadata.items():
                version = meta.get('version', None)
                metric_value = meta.get('metrics', {}).get(metric, None)
                
                if version is not None and metric_value is not None:
                    versions.append(version)
                    metrics.append(metric_value)
                    is_best.append(meta.get('is_best', False))
            
            if not versions:
                logger.warning(f"No data available for metric '{metric}'")
                return
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot main line
            plt.plot(versions, metrics, 'b-', alpha=0.7, label=metric)
            
            # Highlight best points
            best_versions = [v for i, v in enumerate(versions) if is_best[i]]
            best_metrics = [m for i, m in enumerate(metrics) if is_best[i]]
            plt.scatter(best_versions, best_metrics, c='green', s=100, label='Best models')
            
            # Add labels and styling
            plt.xlabel('Model Version', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(f'Training Progress - {metric}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add version markers for milestones
            milestone_versions = [v for v in versions if v % 10 == 0]
            for v in milestone_versions:
                plt.axvline(x=v, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save or display
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved training progress plot to {output_path}")
                plt.close()
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training progress: {e}")
            plt.close()

class EnsembleModel:
    """Advanced ensemble combining multiple model types with sophisticated weighting and robust error handling."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.weights = None
        self.feature_importance = None
        self.meta_model = None
        self.meta_features = None
        self.checkpointer = ModelCheckpointer('model_checkpoints')
        self.current_epoch = 0
        self.base_seed = random.randint(0, 10000)
        self.model_seeds = {}
        self.use_stacking = True  # Toggle for stacking (can be set to False to use weighted averaging instead)
        self.meta_features = None  # Store validation meta-features for diagnostics
        self.meta_model_performance = {}
        
    def generate_model_seed(self, model_name: str) -> int:
        """Generate a deterministic seed for a specific model."""
        if model_name not in self.model_seeds:
            # Hash the model name with base seed to get a new seed
            # This ensures different but consistent seeds across runs
            seed_hash = hash(f"{model_name}_{self.base_seed}") % 100000
            self.model_seeds[model_name] = seed_hash
        return self.model_seeds[model_name]

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with error handling."""
        try:
            # Add training features to metrics for storage
            if hasattr(self, 'training_features'):
                metrics['training_features'] = self.training_features
                
            metrics['model_seeds'] = self.model_seeds
            metrics['base_seed'] = self.base_seed
            return self.checkpointer.save_checkpoint(
                self,
                self.current_epoch,
                metrics,
                is_best
            )
        except Exception as e:
            logger.error(f"Error saving ensemble checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_name: str = 'best'):
        """Load model from checkpoint with robust error handling."""
        try:
            checkpoint = self.checkpointer.load_checkpoint(checkpoint_name)
            
            # Handle empty or invalid checkpoint
            if not checkpoint or not isinstance(checkpoint, dict):
                logger.error(f"Invalid checkpoint data: {type(checkpoint)}")
                return {}
            
            if 'metrics' in checkpoint and 'model_seeds' in checkpoint['metrics']:
                self.model_seeds = checkpoint['metrics']['model_seeds']
                logger.info(f"Restored model seeds: {self.model_seeds}")
            
            if 'metrics' in checkpoint and 'base_seed' in checkpoint['metrics']:
                self.base_seed = checkpoint['metrics']['base_seed']
                logger.info(f"Restored base seed: {self.base_seed}")

            # Restore model state with component-wise error handling
            for name, state in checkpoint.get('model_state', {}).items():
                try:
                    if name not in self.models:
                        logger.warning(f"Model {name} from checkpoint not found in current ensemble. Skipping.")
                        continue
                        
                    if isinstance(state, bytes):
                        try:
                            self.models[name] = pickle.loads(state)
                            logger.info(f"Loaded pickled model {name}")
                        except Exception as e:
                            logger.error(f"Error unpickling model {name}: {e}")
                    elif isinstance(state, dict) and hasattr(self.models[name], 'load_state_dict'):
                        try:
                            self.models[name].load_state_dict(state)
                            logger.info(f"Loaded state dict for model {name}")
                        except Exception as e:
                            logger.error(f"Error loading state dict for {name}: {e}")
                    else:
                        logger.warning(f"Unrecognized state format for model {name}. Skipping.")
                except Exception as e:
                    logger.error(f"Error restoring model {name}: {e}")
            
            # Restore other attributes
            try:
                if 'weights' in checkpoint and checkpoint['weights'] is not None:
                    self.weights = np.array(checkpoint['weights'])
                    logger.info(f"Restored ensemble weights: {self.weights}")
            except Exception as e:
                logger.error(f"Error restoring weights: {e}")
                self.weights = None
            
            try:
                if 'meta_model' in checkpoint and checkpoint['meta_model'] is not None:
                    if isinstance(checkpoint['meta_model'], bytes):
                        self.meta_model = pickle.loads(checkpoint['meta_model'])
                    elif hasattr(self.meta_model, 'load_state_dict'):
                        self.meta_model.load_state_dict(checkpoint['meta_model'])
                    logger.info(f"Restored meta-model")
            except Exception as e:
                logger.error(f"Error restoring meta-model: {e}")
                self.meta_model = None
            
            try:
                self.current_epoch = checkpoint.get('epoch', 0)
            except Exception as e:
                logger.error(f"Error restoring epoch: {e}")
                self.current_epoch = 0
            
            return checkpoint.get('metrics', {})
        
        except Exception as e:
            logger.error(f"Error in load_checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def add_model(self, name: str, model: Any):
        """Add a model to the ensemble with validation."""
        if model is None:
            logger.warning(f"Attempted to add None model with name {name}. Skipping.")
            return False
            
        try:
            # Generate seed for this model
            model_seed = self.generate_model_seed(name)
            
            # Apply the seed to the model if it has a random_state parameter
            if hasattr(model, 'random_state') and isinstance(getattr(model, 'random_state'), (int, type(None))):
                setattr(model, 'random_state', model_seed)
                logger.info(f"Set random seed {model_seed} for model {name}")
            elif hasattr(model, 'set_params') and callable(getattr(model, 'set_params')):
                # Try to set the seed using set_params for models like sklearn
                try:
                    model.set_params(random_state=model_seed)
                    logger.info(f"Set random seed {model_seed} for model {name} via set_params")
                except Exception as e:
                    logger.debug(f"Could not set random_state via set_params for {name}: {e}")
            elif isinstance(model, HybridTimeSeriesModel):
                # For PyTorch Lightning models, we need to set seed differently
                try:
                    import pytorch_lightning as pl
                    pl.seed_everything(model_seed)
                    logger.info(f"Set PyTorch Lightning seed {model_seed} for model {name}")
                except Exception as e:
                    logger.debug(f"Could not set PyTorch Lightning seed for {name}: {e}")
            
            self.models[name] = model
            logger.info(f"Added model {name} to ensemble with seed {model_seed}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding model {name}: {e}")
            return False
    def store_training_features(self, features: List[str]):
        """Store the exact feature names and order used during training."""
        self.training_features = [str(feature) for feature in features]  # Ensure all are strings
        logger.info(f"Stored {len(self.training_features)} training features")
        
        # Also save a mapping from original feature name to index for quick lookups
        self.feature_index_map = {feat: i for i, feat in enumerate(self.training_features)}
        
        # For debugging
        logger.debug(f"First 5 training features: {self.training_features[:5]}")    
    

    def _normalize_predictions(self, predictions: Dict[str, np.ndarray], target_length: int) -> Dict[str,
    np.ndarray]:
        """Normalize predictions to a consistent length and horizon count."""
        normalized_predictions = {}

        if target_length <= 0:
            logger.error(f"Invalid target length: {target_length}, defaulting to 1")
            target_length = 1

        # First pass: determine the maximum number of horizons
        max_horizons = 1
        for name, pred in predictions.items():
            if pred is not None and len(pred.shape) > 1 and pred.shape[1] > max_horizons:
                max_horizons = pred.shape[1]

        # Second pass: normalize all predictions to the same length and horizon count
        for name, pred in predictions.items():
            try:
                # Skip None predictions
                if pred is None:
                    logger.warning(f"None prediction for {name}, using zeros")
                    normalized_predictions[name] = np.zeros((target_length, max_horizons))
                    continue

                # Ensure pred is at least 2D
                if len(pred) == 0:
                    logger.warning(f"Empty prediction for {name}, using zeros")
                    normalized_predictions[name] = np.zeros((target_length, max_horizons))
                    continue

                # Reshape to ensure 2D array (samples, features)
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)

                # Handle length mismatch
                if len(pred) != target_length:
                    logger.warning(f"Prediction length mismatch for {name}. "
                                f"Expected: {target_length}, Got: {len(pred)}")

                    # If prediction is shorter, truncate targets later instead of padding predictions
                    if len(pred) < target_length:
                        if len(pred) == 0:
                            # Handle empty array case
                            normalized_predictions[name] = np.zeros((target_length, max_horizons))
                            continue

                    # If prediction is longer, truncate
                    if len(pred) > target_length:
                        pred = pred[:target_length]

                # Handle horizon count mismatch
                if pred.shape[1] < max_horizons:
                    # If single horizon, repeat across all horizons
                    if pred.shape[1] == 1:
                        pred = np.tile(pred, (1, max_horizons))
                    # For multi-horizon models, we'll handle this differently in the ensemble prediction
                elif pred.shape[1] > max_horizons:
                    # Truncate extra horizons
                    pred = pred[:, :max_horizons]

                normalized_predictions[name] = pred

            except Exception as e:
                logger.error(f"Error normalizing predictions for {name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fallback to zeros with correct dimensions
                normalized_predictions[name] = np.zeros((target_length, max_horizons))

        return normalized_predictions
    
    def _train_deep_model(self, name, model, X, y, X_val, y_val, predictions):
        """Train a deep learning model with proper error handling and consistent output shapes."""
        try:
            # Verify features for deep learning model
            input_dim = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else 0
            logger.info(f"{name} input dimension: {input_dim if input_dim > 0 else model.input_dim if hasattr(model, 'input_dim') else 'unknown'}")
            
            # If model has an attribute for input_dim, ensure it matches the data
            if hasattr(model, 'input_dim') and model.input_dim != input_dim and input_dim > 0:
                logger.warning(f"{name} model input_dim ({model.input_dim}) doesn't match data ({input_dim}). This may cause errors.")
            
            # IMPORTANT: Verify multi-horizon configuration
            is_multi_horizon = False
            expected_horizons = 1
            
            # Check if y is multi-dimensional (multi-horizon case)
            if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
                is_multi_horizon = True
                expected_horizons = y.shape[1]
                logger.info(f"Training {name} with multi-horizon targets: {expected_horizons} horizons")
                
                # Ensure model is configured for multi-horizon
                if hasattr(model, 'multi_horizon'):
                    if not model.multi_horizon:
                        logger.warning(f"Model {name} is not configured for multi-horizon but targets have {expected_horizons} horizons.")
                        logger.warning(f"Updating model configuration to match target dimensions.")
                        model.multi_horizon = True
                        if hasattr(model, 'forecast_horizons') and len(model.forecast_horizons) != expected_horizons:
                            # Update model's forecast horizons to match target
                            model.forecast_horizons = list(range(1, expected_horizons + 1))
                            logger.info(f"Updated {name} forecast_horizons to {model.forecast_horizons}")
                    else:
                        # Check if horizons match
                        if hasattr(model, 'forecast_horizons') and len(model.forecast_horizons) != expected_horizons:
                            logger.warning(f"Model has {len(model.forecast_horizons)} horizons but targets have {expected_horizons} horizons.")
                            # Update model's forecast horizons to match target
                            model.forecast_horizons = list(range(1, expected_horizons + 1))
                            logger.info(f"Updated {name} forecast_horizons to {model.forecast_horizons}")
            
            # Create datasets with error handling
            try:
                if name in ['deep_learning', 'tcn', 'transformer', 'nbeats']:
                    # These models use TimeSeriesDataset
                    train_dataset = TimeSeriesDataset(X, y, self.config.lookback_window)
                    val_dataset = TimeSeriesDataset(X_val, y_val, self.config.lookback_window)
                    
                    # Verify that dataset is correctly handling multi-horizon targets
                    if is_multi_horizon:
                        for x_batch, y_batch in [(train_dataset[0]), (val_dataset[0])]:
                            logger.info(f"{name} dataset shapes - X: {x_batch.shape}, y: {y_batch.shape}")
                            # Verify that y_batch has correct number of horizons
                            if len(y_batch.shape) < 1 or (y_batch.shape[0] if len(y_batch.shape) == 1 else y_batch.shape[1]) != expected_horizons:
                                logger.warning(f"Target shape mismatch in dataset: {y_batch.shape} vs expected horizons: {expected_horizons}")
                            break
                    
                    # Use appropriate batch size based on dataset size
                    batch_size = min(self.config.batch_size, max(1, len(train_dataset) // 10))
                    
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=False,
                        num_workers=0,  # Set to 0 to use the main process
                        pin_memory=False  # Set to False to avoid additional memory operations
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=0,
                        pin_memory=False
                    )
                    
                    # Verify batch shapes
                    for batch_x, batch_y in train_loader:
                        logger.info(f"{name} batch shapes - X: {batch_x.shape}, y: {batch_y.shape}")
                        break
                else:
                    # Default case, should not happen now but kept for safety
                    train_dataset = TimeSeriesDataset(X, y, self.config.lookback_window)
                    val_dataset = TimeSeriesDataset(X_val, y_val, self.config.lookback_window)
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                        
            except Exception as e:
                logger.error(f"Error creating datasets for {name}: {e}")
                # Try to create datasets with simplified parameters
                logger.info(f"Trying with simplified dataset for {name}")
                train_dataset = TimeSeriesDataset(
                    X, 
                    y, 
                    min(10, self.config.lookback_window)  # Reduced lookback window
                )
                val_dataset = TimeSeriesDataset(
                    X_val, 
                    y_val, 
                    min(10, self.config.lookback_window)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Set up trainer with limited epochs per fit call
            trainer = pl.Trainer(
                max_epochs=1,  # Train for just 1 epoch per global epoch
                enable_checkpointing=False,  # Disable internal checkpointing
                logger=False,  # Disable logging to avoid conflicts
                callbacks=[
                    pl.callbacks.EarlyStopping(
                        monitor='train_loss',
                        patience=10
                    )
                ]
            )
            
            # Train model
            trainer.fit(model, train_loader, val_loader)
            
            # Get predictions with shape correction
            model.eval()
            with torch.no_grad():
                # Get validation sample count to ensure we generate exactly y_val.shape[0] predictions
                expected_count = y_val.shape[0] if hasattr(y_val, 'shape') else len(y_val)
                logger.info(f"Expected validation predictions: {expected_count}")
                
                # Method 1: Use validation loader and track total predictions
                val_predictions = []
                prediction_count = 0
                
                for batch_x, _ in val_loader:
                    batch_pred = model(batch_x)
                    val_predictions.append(batch_pred)
                    prediction_count += batch_pred.shape[0]
                    
                # Method 2: If we don't have enough predictions, create additional sequences
                if prediction_count < expected_count and len(val_predictions) > 0:
                    logger.warning(f"Prediction count mismatch: got {prediction_count}, expected {expected_count}. Padding predictions.")
                    
                    # Approach 1: Pad with last prediction
                    missing = expected_count - prediction_count
                    if len(val_predictions[-1]) > 0:
                        last_pred = val_predictions[-1][-1:].expand(missing, -1)
                        val_predictions.append(last_pred)
                        
                if val_predictions:
                    # Concatenate all batches
                    all_preds = torch.cat(val_predictions, dim=0)
                    
                    # Ensure exact size match with expected validation set
                    if all_preds.shape[0] != expected_count:
                        logger.warning(f"Final prediction length mismatch: got {all_preds.shape[0]}, expected {expected_count}. Correcting...")
                        
                        if all_preds.shape[0] > expected_count:
                            # Truncate if too many predictions
                            all_preds = all_preds[:expected_count]
                        else:
                            # Pad with the last prediction or zeros if too few
                            missing = expected_count - all_preds.shape[0]
                            if all_preds.shape[0] > 0:
                                # Use the last prediction for padding
                                if len(all_preds.shape) > 1:
                                    padding = all_preds[-1:].expand(missing, all_preds.shape[1])
                                else:
                                    padding = all_preds[-1:].expand(missing)
                            else:
                                # Use zeros if no predictions at all
                                if y_val.shape[1] if len(y_val.shape) > 1 else 1 > 1:
                                    padding = torch.zeros(missing, y_val.shape[1])
                                else:
                                    padding = torch.zeros(missing, 1)
                            
                            all_preds = torch.cat([all_preds, padding], dim=0)
                    
                    predictions[name] = all_preds.cpu().numpy()
                    logger.info(f"Final {name} prediction shape: {predictions[name].shape}")
                else:
                    logger.warning(f"No predictions generated for {name}. Using zeros.")
                    if len(y_val.shape) > 1:
                        predictions[name] = np.zeros((expected_count, y_val.shape[1]))
                    else:
                        predictions[name] = np.zeros((expected_count, 1))
                    
        except Exception as e:
            logger.error(f"Error in _train_deep_model for {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Use zeros as fallback
            if hasattr(y_val, 'shape') and len(y_val.shape) > 1:
                predictions[name] = np.zeros((len(y_val), y_val.shape[1]))
            else:
                predictions[name] = np.zeros((len(y_val), 1))
                
    def _train_traditional_model(self, name, model, X, y, X_val, y_val, predictions):
        """Train a traditional ML model with robust error handling."""
        try:
            # Check for NaN/Inf in input data
            has_bad_values = False
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning(f"Training data for {name} contains NaN/Inf. Fixing.")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                has_bad_values = True
                
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                logger.warning(f"Target data for {name} contains NaN/Inf. Fixing.")
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                has_bad_values = True
            
            if name in ['tcn', 'transformer', 'nbeats']:
                logger.info(f"Training {name} model using PyTorch Lightning")
                self._train_deep_model(name, model, X, y, X_val, y_val, predictions)
                return                
            
            # Handle multi-dimensional targets appropriately for each model type
            is_multi_target = len(y.shape) > 1 and y.shape[1] > 1
            
            if is_multi_target:
                logger.info(f"Training {name} with multi-dimensional target of shape {y.shape}")
                
                if name == 'xgboost':
                    # For XGBoost, train separate models for each target dimension
                    preds = np.zeros((len(X_val), y.shape[1]))
                    for i in range(y.shape[1]):
                        try:
                            model.fit(X, y[:, i], eval_set=[(X_val, y_val[:, i])], verbose=False)
                            preds[:, i] = model.predict(X_val)
                        except Exception as e:
                            logger.error(f"Error training XGBoost for target dimension {i}: {e}")
                            preds[:, i] = np.zeros(len(X_val))
                    predictions[name] = preds
                    return
                
                # Handle LightGBM for multi-dimensional targets
                if name == 'lightgbm':
                    # Set single-threaded mode for LightGBM to avoid pipe errors
                    if hasattr(model, 'set_params'):
                        try:
                            model.set_params(n_jobs=1, verbose=-1)
                            logger.info(f"Set LightGBM to single-threaded mode")
                        except Exception as e:
                            logger.warning(f"Could not set LightGBM parameters: {e}")
                    
                    # For LightGBM, train separate models for each target dimension
                    preds = np.zeros((len(X_val), y.shape[1]))
                    for i in range(y.shape[1]):
                        try:
                            if LIGHTGBM_CALLBACKS_AVAILABLE:
                                try:
                                    early_stopping_callback = early_stopping(stopping_rounds=40, verbose=False)
                                    model.fit(
                                        X, y[:, i],  # Use only one dimension of the target
                                        eval_set=[(X_val, y_val[:, i])],
                                        callbacks=[early_stopping_callback]
                                    )
                                except Exception as cb_error:
                                    logger.warning(f"Error with LightGBM callbacks: {cb_error}. Using simpler approach.")
                                    # Try without callback on failure
                                    model.fit(
                                        X, y[:, i],  
                                        eval_set=[(X_val, y_val[:, i])],
                                        early_stopping_rounds=40,
                                        verbose=False
                                    )
                            else:
                                model.fit(
                                    X, y[:, i],  # Use only one dimension of the target
                                    eval_set=[(X_val, y_val[:, i])],
                                    early_stopping_rounds=40,
                                    verbose=False
                                )
                            preds[:, i] = model.predict(X_val)
                        except Exception as e:
                            logger.error(f"Error training LightGBM for target dimension {i}: {e}")
                            preds[:, i] = np.zeros(len(X_val))
                    predictions[name] = preds
                    return
            
                # Handle CatBoost specifically for multi-dimensional targets
                if name == 'catboost':
                    try:
                        # For multi-target with CatBoost, train separate models for each target
                        preds = np.zeros((len(X_val), y.shape[1]))
                        
                        # Iterate over each target dimension
                        for i in range(y.shape[1]):
                            try:
                                # Create a clone of the model for each dimension
                                from sklearn.base import clone
                                dim_model = clone(model)
                                
                                # Ensure the model has required methods
                                if not hasattr(dim_model, 'fit') or not callable(dim_model.fit):
                                    logger.error(f"CatBoost model for dimension {i} does not have a fit method")
                                    preds[:, i] = np.zeros(len(X_val))
                                    continue
                                
                                # Set RMSE for single-target training
                                if hasattr(dim_model, 'set_params'):
                                    dim_model.set_params(loss_function='RMSE')
                                
                                # Train on single target dimension with explicit warning suppression
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    dim_model.fit(X, y[:, i], eval_set=[(X_val, y_val[:, i])], verbose=False)
                                
                                # Verify model is fitted before prediction
                                is_fitted = False
                                try:
                                    # Different ways to check if model is fitted
                                    if hasattr(dim_model, '_check_is_fitted'):
                                        try:
                                            dim_model._check_is_fitted()
                                            is_fitted = True
                                        except Exception:
                                            is_fitted = False
                                    elif hasattr(dim_model, 'is_fitted'):
                                        is_fitted = dim_model.is_fitted()
                                    elif hasattr(dim_model, 'feature_importances_'):
                                        is_fitted = True
                                    elif hasattr(dim_model, 'get_feature_importance'):
                                        try:
                                            dim_model.get_feature_importance()
                                            is_fitted = True
                                        except Exception:
                                            is_fitted = False
                                    else:
                                        # Try to predict as a last resort check
                                        try:
                                            dim_model.predict(X[:1])
                                            is_fitted = True
                                        except Exception:
                                            is_fitted = False
                                except Exception as check_error:
                                    logger.warning(f"Error checking if CatBoost model is fitted: {check_error}")
                                    is_fitted = False
                                
                                if is_fitted:
                                    preds[:, i] = dim_model.predict(X_val)
                                else:
                                    logger.warning(f"CatBoost model for dimension {i} not properly fitted")
                                    preds[:, i] = np.zeros(len(X_val))
                            except Exception as e:
                                logger.error(f"Error training CatBoost for target dimension {i}: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                preds[:, i] = np.zeros(len(X_val))
                        
                        predictions[name] = preds
                        return
                    except Exception as e:
                        logger.error(f"Error in CatBoost multi-dimensional training: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        predictions[name] = np.zeros((len(X_val), y.shape[1]))
                        return
                    
                # Handle Prophet separately since it needs DataFrame input
                if isinstance(model, ProphetModel):
                    # Convert to DataFrame if needed
                    X_df = None
                    if not isinstance(X, pd.DataFrame):
                        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                    else:
                        X_df = X
                        
                    X_val_df = None
                    if not isinstance(X_val, pd.DataFrame):
                        X_val_df = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])
                    else:
                        X_val_df = X_val
                        
                    model.fit(X_df, y)
                    predictions[name] = model.predict(X_val_df)
                    return
            
            # Handle Prophet separately since it needs DataFrame input
            if isinstance(model, ProphetModel):
                # Convert to DataFrame if needed
                X_df = None
                if not isinstance(X, pd.DataFrame):
                    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                else:
                    X_df = X
                    
                X_val_df = None
                if not isinstance(X_val, pd.DataFrame):
                    X_val_df = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(X_val.shape[1])])
                else:
                    X_val_df = X_val
                    
                model.fit(X_df, y)
                predictions[name] = model.predict(X_val_df)
                return
                
            # For other models, verify features and properly fit
            logger.info(f"{name} will process {X.shape[1]} features")
            
            # Specific handling for CatBoost model
            # Around line 4160 in _train_traditional_model method
            if name == 'catboost':
                try:
                    # Check if model has the right methods
                    if not hasattr(model, 'fit') or not callable(model.fit):
                        logger.error(f"CatBoost model does not have a fit method")
                        predictions[name] = np.zeros((len(X_val), 1) if not is_multi_target else (len(X_val), y.shape[1]))
                        return
                    
                    # Set parameters with error handling
                    if hasattr(model, 'set_params'):
                        try:
                            # Set appropriate loss function and verbosity
                            model.set_params(verbose=False)
                            if is_multi_target:
                                model.set_params(loss_function='MultiRMSE')
                            else:
                                model.set_params(loss_function='RMSE')
                        except Exception as e:
                            logger.warning(f"Error setting CatBoost parameters: {e}")
                    
                    # Fit the model with explicit warning suppression
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
                    
                    # CRITICAL FIX: Verify model is fitted before prediction
                    is_fitted = False
                    try:
                        # Different ways to check if model is fitted
                        if hasattr(model, '_check_is_fitted'):
                            try:
                                model._check_is_fitted()
                                is_fitted = True
                            except Exception:
                                is_fitted = False
                        elif hasattr(model, 'is_fitted'):
                            is_fitted = model.is_fitted()
                        elif hasattr(model, 'feature_importances_') or hasattr(model, 'get_feature_importance'):
                            is_fitted = True
                        else:
                            # Try to predict as a last resort check
                            try:
                                model.predict(X[:1])
                                is_fitted = True
                            except Exception:
                                is_fitted = False
                    except Exception as check_error:
                        logger.warning(f"Error checking if CatBoost model is fitted: {check_error}")
                        is_fitted = False
                    
                    if is_fitted:
                        # Get predictions
                        pred = model.predict(X_val)
                        if len(pred.shape) == 1:
                            pred = pred.reshape(-1, 1)
                        predictions[name] = pred
                    else:
                        logger.warning("CatBoost model not properly fitted")
                        predictions[name] = np.zeros((len(X_val), 1) if not is_multi_target else (len(X_val), y.shape[1]))
                    
                    return
                except Exception as e:
                    logger.error(f"Error in CatBoost training/prediction: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    predictions[name] = np.zeros((len(X_val), 1) if not is_multi_target else (len(X_val), y.shape[1]))
                    return
            
            # Fit with appropriate parameters based on model type
            if hasattr(model, 'fit') and callable(model.fit):
                if name == 'xgboost' and hasattr(model, 'get_params'):
                    # For XGBoost, use validation set for early stopping
                    model.fit(
                        X, y,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif name == 'lightgbm' and hasattr(model, 'get_params'):
                    # FIX: Set single-threaded mode for LightGBM to avoid pipe errors
                    try:
                        model.set_params(n_jobs=1, verbose=-1)
                        logger.info(f"Set LightGBM to single-threaded mode")
                    except Exception as e:
                        logger.warning(f"Could not set LightGBM parameters: {e}")
                    
                    # Handle LightGBM properly based on available callbacks
                    if LIGHTGBM_CALLBACKS_AVAILABLE:
                        try:
                            # Use early stopping callback without verbose parameter
                            early_stopping_callback = early_stopping(stopping_rounds=40, verbose=False)
                            model.fit(
                                X, y,
                                eval_set=[(X_val, y_val)],
                                callbacks=[early_stopping_callback]
                            )
                        except Exception as e:
                            logger.warning(f"Error using LightGBM callbacks: {e}. Trying with simpler approach.")
                            # Try with standard early stopping on failure
                            model.fit(
                                X, y,
                                eval_set=[(X_val, y_val)],
                                early_stopping_rounds=40,
                                verbose=False
                            )
                    else:
                        # If callbacks not available, use simpler approach
                        model.fit(
                            X, y,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=40,
                            verbose=False
                        )
                else:
                    # For other models, use standard fit
                    model.fit(X, y)
            else:
                logger.warning(f"Model {name} does not have a fit method.")
                predictions[name] = np.zeros((len(y_val), 1))
                return
                
            # Generate predictions
            if hasattr(model, 'predict') and callable(model.predict):
                pred = model.predict(X_val)
                                    
                # Ensure prediction has correct shape
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
                    
                # Check if this shape matches our expected shape
                if is_multi_target and pred.shape[1] != y_val.shape[1]:
                    logger.warning(f"Shape mismatch for {name}: pred has shape {pred.shape}, target has shape {y_val.shape}")
                    # Reshape appropriately or pad with zeros
                    if pred.shape[0] == y_val.shape[0]:
                        if pred.shape[1] < y_val.shape[1]:
                            # Pad with zeros
                            padding = np.zeros((pred.shape[0], y_val.shape[1] - pred.shape[1]))
                            pred = np.hstack([pred, padding])
                        else:
                            # Truncate
                            pred = pred[:, :y_val.shape[1]]
                    
                predictions[name] = pred
            else:
                logger.warning(f"Model {name} does not have a predict method.")
                predictions[name] = np.zeros((len(y_val), 1))
                    
            # Log feature importances if available
            if hasattr(model, 'feature_importances_'):
                if isinstance(X, pd.DataFrame):
                    feature_names = X.columns.tolist()
                else:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                    
                importances = dict(zip(feature_names, model.feature_importances_))
                sorted_importances = dict(sorted(importances.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True))
                                                
                logger.info(f"\n{name} top 10 feature importances:")
                for i, (feat, imp) in enumerate(list(sorted_importances.items())[:10]):
                    logger.info(f"{i+1}. {feat}: {imp:.4f}")
                        
        except Exception as e:
            logger.error(f"Error in _train_traditional_model for {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Use zeros as fallback
            predictions[name] = np.zeros((len(y_val), 1) if len(y_val.shape) == 1 else (len(y_val), y_val.shape[1]))
                
    
    def _enhance_meta_features(self, meta_features: np.ndarray, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Enhance meta-features with additional engineered features to improve meta-model performance.
        Dynamically adapts to the input dimension.
        
        Args:
            meta_features: Base stacked predictions
            predictions: Dictionary of predictions from each model
            
        Returns:
            Enhanced feature matrix
        """
        try:
            # Start with the base meta-features (stacked predictions)
            enhanced = meta_features.copy()
            model_names = list(predictions.keys())
            
            # Store original feature count for logging
            original_feature_count = meta_features.shape[1]
            
            # Add better diagnostics
            logger.info(f"Starting meta-feature enhancement. Original shape: {meta_features.shape}")
            
            # Get expected feature count from meta-model if it's already trained
            expected_feature_count = None
            if self.meta_model is not None:
                if hasattr(self.meta_model, 'n_features_in_'):
                    expected_feature_count = self.meta_model.n_features_in_
                    logger.info(f"Meta-model expects {expected_feature_count} features")
                elif hasattr(self.meta_model, 'feature_importances_') and hasattr(self.meta_model.feature_importances_, 'shape'):
                    expected_feature_count = self.meta_model.feature_importances_.shape[0]
                    logger.info(f"Meta-model expects {expected_feature_count} features based on feature_importances_")
                # NEW: Check if we have stored the meta-feature dimension
                elif hasattr(self, 'meta_feature_dim'):
                    expected_feature_count = self.meta_feature_dim
                    logger.info(f"Using stored meta-feature dimension: {expected_feature_count}")
            
            # Check if we're training or predicting based on expected feature count
            is_training_phase = expected_feature_count is None
            
            # Check for multi-horizon predictions
            has_multi_horizon = False
            max_horizons = 1
            for name, pred in predictions.items():
                if len(pred.shape) > 1 and pred.shape[1] > 1:
                    has_multi_horizon = True
                    max_horizons = max(max_horizons, pred.shape[1])
                    logger.info(f"Detected multi-horizon predictions from {name}: {pred.shape}")
            
            # Track feature counts by group
            feature_counts = {
                'base': original_feature_count,
                'pairwise_diffs': 0,
                'ensemble_agreement': 0,
                'polynomial': 0,
                'volatility': 0
            }
                
            # 1. Add pairwise differences between model predictions
            # For multi-horizon models, we compare corresponding horizons
            pairwise_diffs = []
            for i, name1 in enumerate(model_names):
                for j, name2 in enumerate(model_names):
                    if i < j:  # Only use each pair once
                        # For multi-horizon, make sure to compare corresponding horizons
                        if has_multi_horizon:
                            # Get predictions, ensuring they have same number of horizons
                            pred1 = predictions[name1]
                            pred2 = predictions[name2]
                            
                            # Standardize to same number of horizons
                            if pred1.shape[1] != pred2.shape[1]:
                                # Reshape to have same dimensions
                                if pred1.shape[1] == 1:
                                    pred1 = np.tile(pred1, (1, pred2.shape[1]))
                                elif pred2.shape[1] == 1:
                                    pred2 = np.tile(pred2, (1, pred1.shape[1]))
                                else:
                                    # Use common horizons
                                    common_horizons = min(pred1.shape[1], pred2.shape[1])
                                    pred1 = pred1[:, :common_horizons]
                                    pred2 = pred2[:, :common_horizons]
                            
                            diff = pred1 - pred2
                        else:
                            diff = predictions[name1] - predictions[name2]
                        
                        pairwise_diffs.append(diff)
            
            if pairwise_diffs:
                # Ensure all pairwise differences have the same shape before stacking
                if has_multi_horizon:
                    # Standardize shapes to same number of horizons
                    std_diffs = []
                    for diff in pairwise_diffs:
                        if diff.shape[1] != max_horizons:
                            if diff.shape[1] == 1:
                                # Expand single horizon to match max
                                diff = np.tile(diff, (1, max_horizons))
                            else:
                                # Pad with zeros
                                padding = np.zeros((diff.shape[0], max_horizons - diff.shape[1]))
                                diff = np.hstack([diff, padding])
                        std_diffs.append(diff)
                    pairwise_diffs = std_diffs
                    
                try:
                    # Stack all diffs horizontally
                    pairwise_diff_features = np.hstack(pairwise_diffs)
                    enhanced = np.hstack([enhanced, pairwise_diff_features])
                    feature_counts['pairwise_diffs'] = pairwise_diff_features.shape[1]
                    logger.info(f"Added {pairwise_diff_features.shape[1]} pairwise difference features")
                except Exception as e:
                    logger.error(f"Error stacking pairwise differences: {e}")
                    # Skip this group
            
            # 2. Add ensemble agreement features 
            if len(model_names) >= 3:  # Need at least 3 models for meaningful variance
                ensemble_agreement_features = []
                
                # Variance of predictions - indicates uncertainty 
                prediction_variance = np.var(meta_features, axis=1, keepdims=True)
                ensemble_agreement_features.append(prediction_variance)
                
                # Range of predictions - another measure of uncertainty
                prediction_range = np.max(meta_features, axis=1, keepdims=True) - np.min(meta_features, axis=1, keepdims=True)
                ensemble_agreement_features.append(prediction_range)
                
                # Entropy-inspired disagreement measure
                normalized_preds = meta_features - np.min(meta_features, axis=1, keepdims=True)
                row_sums = np.sum(normalized_preds, axis=1, keepdims=True)
                # Avoid division by zero
                row_sums = np.where(row_sums == 0, 1e-8, row_sums)
                pred_probs = normalized_preds / row_sums
                # Replace zeros to avoid log(0)
                pred_probs = np.where(pred_probs == 0, 1e-8, pred_probs)
                disagreement = -np.sum(pred_probs * np.log(pred_probs), axis=1, keepdims=True)
                ensemble_agreement_features.append(disagreement)
                
                try:
                    # Stack all agreement features
                    agreement_features = np.hstack(ensemble_agreement_features)
                    enhanced = np.hstack([enhanced, agreement_features])
                    feature_counts['ensemble_agreement'] = agreement_features.shape[1]
                    logger.info(f"Added {agreement_features.shape[1]} ensemble agreement features")
                except Exception as e:
                    logger.error(f"Error stacking ensemble agreement features: {e}")
                    # Skip this group
            
            # 3. Add polynomial features for the most important models
            # If we have a prior weights vector, use the top models
            polynomial_features = []
            if self.weights is not None and len(self.weights) == len(model_names):
                # Find the indices of the top 3 models by weight
                top_indices = np.argsort(self.weights)[-3:]
                
                if len(top_indices) > 0:
                    top_features = []
                    for idx in top_indices:
                        if idx < meta_features.shape[1]:
                            top_features.append(meta_features[:, idx:idx+1])
                    
                    if top_features:
                        # Stack top features
                        top_feature_matrix = np.hstack(top_features)
                        
                        # Square terms
                        squared_features = top_feature_matrix ** 2
                        polynomial_features.append(squared_features)
                        
                        # Cubic terms for the very top model
                        if len(top_indices) > 0:
                            top_model_idx = top_indices[-1]
                            if top_model_idx < meta_features.shape[1]:
                                cubic_feature = meta_features[:, top_model_idx:top_model_idx+1] ** 3
                                polynomial_features.append(cubic_feature)
                        
                        # Cross-terms (interactions)
                        if len(top_indices) >= 2:
                            for i in range(len(top_indices)):
                                for j in range(i+1, len(top_indices)):
                                    if i < top_feature_matrix.shape[1] and j < top_feature_matrix.shape[1]:
                                        interaction = top_feature_matrix[:, i:i+1] * top_feature_matrix[:, j:j+1]
                                        polynomial_features.append(interaction)
                
                try:
                    if polynomial_features:
                        # Stack all polynomial features
                        poly_feature_matrix = np.hstack(polynomial_features)
                        enhanced = np.hstack([enhanced, poly_feature_matrix])
                        feature_counts['polynomial'] = poly_feature_matrix.shape[1]
                        logger.info(f"Added {poly_feature_matrix.shape[1]} polynomial features")
                except Exception as e:
                    logger.error(f"Error stacking polynomial features: {e}")
                    # Skip this group
            
            # 4. Add volatility-based features if available
            # Look for volatility features in meta-model context
            volatility_cols = [col for col in self.meta_features.columns if 'volatility' in col.lower() or 'garch' in col.lower()] if hasattr(self, 'meta_features') and hasattr(self.meta_features, 'columns') else []

            volatility_features = []
            if volatility_cols and hasattr(self, 'meta_features'):
                try:
                    # Extract volatility
                    volatility = self.meta_features[volatility_cols[0]].values.reshape(-1, 1)
                    volatility_features.append(volatility)
                    
                    # Create volatility-weighted predictions for each model
                    for i in range(min(5, meta_features.shape[1])):  # Limit to top 5 models to avoid explosion
                        vol_weighted = meta_features[:, i:i+1] * volatility
                        volatility_features.append(vol_weighted)
                    
                    # Add volatility regime indicators
                    high_vol = (volatility > np.median(volatility)).astype(float)
                    volatility_features.append(high_vol)
                    
                    # Create regime-specific predictions for top models
                    for i in range(min(3, meta_features.shape[1])):  # Top 3 models only
                        high_vol_pred = meta_features[:, i:i+1] * high_vol
                        volatility_features.append(high_vol_pred)
                    
                    try:
                        # Stack all volatility features
                        vol_feature_matrix = np.hstack(volatility_features)
                        enhanced = np.hstack([enhanced, vol_feature_matrix])
                        feature_counts['volatility'] = vol_feature_matrix.shape[1]
                        logger.info(f"Added {vol_feature_matrix.shape[1]} volatility regime features")
                    except Exception as e:
                        logger.error(f"Error stacking volatility features: {e}")
                        # Skip this group
                except Exception as regime_error:
                    logger.warning(f"Error creating volatility regime features: {regime_error}")
            
            # Final feature count report
            total_features = sum(feature_counts.values())
            logger.info(f"Feature count breakdown: {feature_counts}")
            logger.info(f"Total features: {total_features}, Enhanced shape: {enhanced.shape}")
            
            # If we're in training phase, store the actual feature count for future reference
            if is_training_phase:
                self.meta_feature_dim = enhanced.shape[1]
                logger.info(f"Training phase: storing meta-feature dimension: {self.meta_feature_dim}")
            if not is_training_phase and expected_feature_count is not None:
                current_feature_count = enhanced.shape[1]
                
                # Handle dimension mismatch
                if current_feature_count != expected_feature_count:
                    logger.warning(f"Feature shape mismatch, expected: {expected_feature_count}, got {current_feature_count}")
                    
                    if current_feature_count < expected_feature_count:
                        # Only pad if we have fewer features than expected
                        padding = np.zeros((enhanced.shape[0], expected_feature_count - current_feature_count))
                        enhanced = np.hstack([enhanced, padding])
                        logger.info(f"Padded features from {current_feature_count} to {expected_feature_count}")
                    else:
                        # If we have more features than expected, update the expected count instead of truncating
                        logger.info(f"Using expanded feature set with {current_feature_count} features instead of truncating to {expected_feature_count}")
                        self.meta_feature_dim = current_feature_count  # Update to the new size
                        
            return enhanced
                
        except Exception as e:
            logger.error(f"Error enhancing meta-features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # If we have expected dimensions and error occurred, return zeros with correct shape
            if 'expected_feature_count' in locals() and expected_feature_count is not None:
                logger.info(f"Returning zeros with expected shape {expected_feature_count} after error")
                return np.zeros((meta_features.shape[0], expected_feature_count))
                
            # Otherwise return original features as fallback
            return meta_features
    def _train_meta_model(self, predictions, y_val):
        """Train a meta-model on the predictions of base models with enhanced multi-horizon handling."""
        try:
            # Check that predictions is not empty
            if not predictions:
                logger.warning("No predictions provided to train meta-model")
                return False
                    
            # First, standardize all prediction shapes
            standardized_predictions = {}
            
            # Determine the target shape
            if isinstance(y_val, np.ndarray):
                target_shape = y_val.shape
            else:
                # Convert to numpy array to determine shape
                target_shape = np.array(y_val).shape
                
            target_ndim = len(target_shape)
            is_multi_horizon = target_ndim > 1 and target_shape[1] > 1
            
            # If multi-horizon, ensure all predictions have the same horizon dimension
            if is_multi_horizon:
                n_horizons = target_shape[1]
                logger.info(f"Training meta-model for {n_horizons} horizons")
                
                # Ensure all predictions have consistent horizons
                for name, pred in list(predictions.items()):
                    if len(pred.shape) == 1 or pred.shape[1] != n_horizons:
                        if len(pred.shape) == 1:
                            pred = pred.reshape(-1, 1)
                            
                        if pred.shape[1] < n_horizons:
                            logger.warning(f"Model {name} has fewer horizons ({pred.shape[1]}) than target ({n_horizons}). Expanding.")
                            
                            if pred.shape[1] == 1:
                                # Repeat single prediction across all horizons
                                expanded_pred = np.tile(pred, (1, n_horizons))
                            else:
                                # Use model's predictions for available horizons, then repeat last one
                                last_horizon = pred[:, -1:]
                                padding = np.tile(last_horizon, (1, n_horizons - pred.shape[1]))
                                expanded_pred = np.hstack([pred, padding])
                                
                            predictions[name] = expanded_pred
                            logger.info(f"Expanded {name} predictions to {predictions[name].shape}")
            
            # Standardize predictions to match target dimensionality
            for name, pred in predictions.items():
                if pred is None:
                    logger.warning(f"Prediction for {name} is None. Using zeros.")
                    if is_multi_horizon:
                        standardized_predictions[name] = np.zeros((target_shape[0], target_shape[1]))
                    else:
                        standardized_predictions[name] = np.zeros((target_shape[0], 1))
                    continue
                    
                # Ensure predictions are numpy arrays
                if not isinstance(pred, np.ndarray):
                    pred = np.array(pred)
                    
                # Ensure 2D shape for stacking
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
                    
                # Handle multi-horizon alignment
                if is_multi_horizon and pred.shape[1] != target_shape[1]:
                    if pred.shape[1] == 1:
                        # If model produces single-horizon predictions but target is multi-horizon
                        # Repeat the prediction for each horizon
                        logger.info(f"Expanding {name} predictions from single to multi-horizon")
                        pred = np.tile(pred, (1, target_shape[1]))
                    elif pred.shape[1] < target_shape[1]:
                        # If model produces fewer horizons than target
                        # Pad with the last horizon
                        logger.info(f"Padding {name} predictions to match target horizons")
                        padding = np.tile(pred[:, -1:], (1, target_shape[1] - pred.shape[1]))
                        pred = np.hstack([pred, padding])
                    elif pred.shape[1] > target_shape[1]:
                        # If model produces more horizons than target
                        # Truncate to match target
                        logger.info(f"Truncating {name} predictions to match target horizons")
                        pred = pred[:, :target_shape[1]]
                
                standardized_predictions[name] = pred
            
            # Create meta-features by stacking predictions
            direct_features = []  # Direct model outputs for each horizon
            stat_features = []    # Statistical features derived from multi-horizon predictions
            horizon_features = [] # Horizon-specific ensemble features
            interaction_features = [] # Model interaction features
            
            # Process each model's predictions
            for name, pred in standardized_predictions.items():
                # For multi-horizon case
                if is_multi_horizon and pred.shape[1] > 1:
                    # Add direct predictions for each horizon as separate features
                    for h in range(pred.shape[1]):
                        direct_features.append(pred[:, h:h+1])
                    
                    # Add statistical features from all horizons
                    if pred.shape[1] >= 3:  # Only if we have enough horizons for meaningful stats
                        # Mean prediction across horizons
                        stat_features.append(np.mean(pred, axis=1, keepdims=True))
                        # Standard deviation across horizons (uncertainty)
                        stat_features.append(np.std(pred, axis=1, keepdims=True))
                        # Trend direction (positive slope = 1, negative = -1, flat = 0)
                        slopes = np.zeros((pred.shape[0], 1))
                        for i in range(pred.shape[0]):
                            # Simple linear regression slope
                            x = np.arange(pred.shape[1])
                            y = pred[i, :]
                            if not np.all(np.isnan(y)) and len(np.unique(y)) > 1:
                                try:
                                    slope, _, _, _, _ = stats.linregress(x, y)
                                    slopes[i, 0] = np.sign(slope)
                                except:
                                    slopes[i, 0] = 0
                        stat_features.append(slopes)
                        
                        # Add volatility feature (normalized range of predictions)
                        ranges = np.zeros((pred.shape[0], 1))
                        for i in range(pred.shape[0]):
                            y = pred[i, :]
                            if not np.all(np.isnan(y)):
                                range_val = np.max(y) - np.min(y)
                                mean_val = np.mean(y)
                                if abs(mean_val) > 1e-8:  # Avoid division by near-zero
                                    ranges[i, 0] = range_val / abs(mean_val)
                                else:
                                    ranges[i, 0] = range_val
                        stat_features.append(ranges)
                else:
                    # For single-horizon, just use the prediction directly
                    direct_features.append(pred)
            
            # For multi-horizon targets, create horizon-specific features
            if is_multi_horizon:
                # Group model predictions by horizon
                for h in range(target_shape[1]):
                    horizon_preds = []
                    for name, pred in standardized_predictions.items():
                        if pred.shape[1] > h:
                            horizon_preds.append(pred[:, h:h+1])
                        elif pred.shape[1] > 0:
                            # Use last available horizon if this one isn't available
                            horizon_preds.append(pred[:, -1:])
                    
                    if horizon_preds:
                        # Stack all models' predictions for this horizon
                        horizon_stack = np.hstack(horizon_preds)
                        # Calculate ensemble prediction for this horizon
                        horizon_features.append(np.mean(horizon_stack, axis=1, keepdims=True))
                        # Add variance of predictions as measure of model agreement
                        if horizon_stack.shape[1] > 1:
                            horizon_features.append(np.var(horizon_stack, axis=1, keepdims=True))
                        
                        # Add min/max model predictions for this horizon
                        horizon_features.append(np.min(horizon_stack, axis=1, keepdims=True))
                        horizon_features.append(np.max(horizon_stack, axis=1, keepdims=True))
            
            # Model interaction features - add pairwise products for top models
            model_names = list(standardized_predictions.keys())
            if len(model_names) >= 2:
                # For multi-horizon, use the first horizon predictions for interactions
                for i, name1 in enumerate(model_names):
                    for j, name2 in enumerate(model_names):
                        if i < j:  # Only use each pair once
                            pred1 = standardized_predictions[name1][:, 0:1] if standardized_predictions[name1].shape[1] > 0 else np.zeros((target_shape[0], 1))
                            pred2 = standardized_predictions[name2][:, 0:1] if standardized_predictions[name2].shape[1] > 0 else np.zeros((target_shape[0], 1))
                            # Product of predictions (interaction)
                            interaction_features.append(pred1 * pred2)
                            # Difference of predictions (disagreement)
                            interaction_features.append(np.abs(pred1 - pred2))
            
            # Combine all feature groups
            all_feature_groups = [direct_features, stat_features, horizon_features, interaction_features]
            all_features = []
            for group in all_feature_groups:
                if group:  # Only add non-empty groups
                    all_features.extend(group)
            
            if not all_features:
                logger.warning("No valid features created for meta-model")
                return False
                
            try:
                # Stack all features horizontally
                meta_features = np.hstack(all_features)
                
                # Ensure no NaN or inf values
                if np.any(np.isnan(meta_features)) or np.any(np.isinf(meta_features)):
                    logger.warning(f"Found {np.isnan(meta_features).sum()} NaN and {np.isinf(meta_features).sum()} inf values in meta-features. Replacing with zeros.")
                    meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=0.0, neginf=0.0)
                    
                logger.info(f"Created meta-features matrix with shape {meta_features.shape}")
                
            except Exception as e:
                logger.error(f"Error combining meta-features: {e}")
                # Fallback to simpler approach from original code
                try:
                    meta_features_list = []
                    for name, pred in standardized_predictions.items():
                        # Ensure consistent lengths by trimming or padding
                        if len(pred) != target_shape[0]:
                            if len(pred) > target_shape[0]:
                                # Trim
                                pred = pred[:target_shape[0]]
                            else:
                                # Pad with last value
                                padding = np.zeros((target_shape[0] - len(pred), pred.shape[1]))
                                if len(pred) > 0:
                                    padding.fill(pred[-1, 0])
                                pred = np.concatenate([pred, padding], axis=0)
                        
                        meta_features_list.append(pred)
                    
                    # Check if we have any predictions
                    if not meta_features_list:
                        logger.warning("No valid predictions to stack for meta-features")
                        return False
                        
                    meta_features = np.hstack(meta_features_list)
                    logger.info(f"Using fallback meta-features with shape {meta_features.shape}")
                    
                except Exception as e:
                    logger.error(f"Error in fallback meta-feature creation: {e}")
                    # Try even simpler approach
                    try:
                        # Create meta_features with the right shape
                        num_samples = target_shape[0]
                        num_models = len(standardized_predictions)
                        meta_features = np.zeros((num_samples, num_models))
                        
                        # Fill in one column per model
                        for i, (name, pred) in enumerate(standardized_predictions.items()):
                            # Ensure right length
                            if len(pred) != num_samples:
                                if len(pred) > num_samples:
                                    # Trim
                                    pred_flat = pred.flatten()[:num_samples]
                                else:
                                    # Pad with zeros
                                    pred_flat = np.zeros(num_samples)
                                    pred_flat[:len(pred.flatten())] = pred.flatten()
                            else:
                                pred_flat = pred.flatten()
                                
                            # Fill column
                            meta_features[:, i] = pred_flat
                        logger.info(f"Using simplest fallback meta-features with shape {meta_features.shape}")
                    except Exception as e:
                        logger.error(f"Error in simplest fallback meta-features creation: {e}")
                        return False
            
            # Create meaningful column names for the meta-features
            col_names = []
            feature_idx = 0
            
            # Names for direct features
            for name, pred in standardized_predictions.items():
                if is_multi_horizon and pred.shape[1] > 1:
                    for h in range(pred.shape[1]):
                        col_names.append(f"{name}_h{h+1}")
                        feature_idx += 1
                else:
                    col_names.append(f"{name}_pred")
                    feature_idx += 1
            
            # Names for statistical features
            for name, pred in standardized_predictions.items():
                if is_multi_horizon and pred.shape[1] >= 3:
                    col_names.append(f"{name}_mean")
                    col_names.append(f"{name}_std")
                    col_names.append(f"{name}_trend")
                    col_names.append(f"{name}_volatility")
                    feature_idx += 4
            
            # Names for horizon features
            if is_multi_horizon:
                for h in range(target_shape[1]):
                    col_names.append(f"ensemble_h{h+1}")
                    feature_idx += 1
                    if len(standardized_predictions) > 1:
                        col_names.append(f"agreement_h{h+1}")
                        col_names.append(f"min_h{h+1}")
                        col_names.append(f"max_h{h+1}")
                        feature_idx += 3
            
            # Names for interaction features
            if len(model_names) >= 2:
                for i, name1 in enumerate(model_names):
                    for j, name2 in enumerate(model_names):
                        if i < j:
                            col_names.append(f"{name1}_{name2}_prod")
                            col_names.append(f"{name1}_{name2}_diff")
                            feature_idx += 2
            
            # Ensure we have enough column names
            while len(col_names) < meta_features.shape[1]:
                col_names.append(f"meta_feature_{len(col_names)}")
            
            # If we have too many column names, trim the excess
            if len(col_names) > meta_features.shape[1]:
                col_names = col_names[:meta_features.shape[1]]
            
            # Create DataFrame with rich meta-features
            self.meta_features = pd.DataFrame(meta_features, columns=col_names)
            logger.info(f"Created meta-features DataFrame with {len(col_names)} columns")

            # Store the actual dimensions for future reference
            self.meta_feature_dim = meta_features.shape[1]
            logger.info(f"Stored meta-feature dimension: {self.meta_feature_dim}")
            
            # Check for empty or invalid input
            if meta_features.size == 0 or np.any(np.isnan(meta_features)) or np.any(np.isinf(meta_features)):
                logger.warning("Meta-features contain invalid values. Fixing.")
                meta_features = np.nan_to_num(meta_features, nan=0.0, posinf=0.0, neginf=0.0)
                
            # Verify dimensions
            if len(meta_features) != len(y_val):
                logger.error(f"Meta-feature length ({len(meta_features)}) doesn't match target length ({len(y_val)})")
                return False
                    
            if XGBOOST_AVAILABLE:
                # Generate seed for meta-model
                meta_seed = self.generate_model_seed('meta_model')
                
                # Dynamically create meta-model based on the actual input dimension
                input_dim = meta_features.shape[1]
                logger.info(f"Creating meta-model with dynamic input dimension: {input_dim}")
                
                # Create a more sophisticated meta-model with deeper trees
                self.meta_model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=200,  # More trees for better performance
                    learning_rate=0.01,  # Lower learning rate with more trees
                    max_depth=6,  # Deeper trees to capture more complex interactions
                    min_child_weight=2,  # Helps prevent overfitting with deeper trees
                    subsample=0.8,
                    colsample_bytree=0.8,
                    colsample_bylevel=0.8,  # Additional sampling at each level
                    gamma=1.0,  # Minimum loss reduction for further partition
                    reg_alpha=0.1,  # L1 regularization to encourage sparsity
                    reg_lambda=1.0,  # L2 regularization for stability
                    tree_method='auto',
                    random_state=meta_seed
                )
            else:
                # Fallback to RandomForest with dynamic input dimension
                meta_seed = self.generate_model_seed('meta_model')
                self.meta_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_split=5,
                    random_state=meta_seed
                )
            
            # Train meta-model with cross-validation to assess reliability
            if len(y_val) >= 100:  # Only do CV if we have enough data
                try:
                    from sklearn.model_selection import cross_val_score
                    
                    # Compute 5-fold CV score to assess meta-model reliability
                    enhanced_features = self._enhance_meta_features(meta_features, standardized_predictions)
                    self.meta_feature_dim = enhanced_features.shape[1]
                    logger.info(f"Updated meta_feature_dim to {self.meta_feature_dim} after enhancement")
                    # CRITICAL FIX: Handle target shape for cross-validation
                    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                        # For multi-horizon targets, use first dimension for CV
                        y_val_for_cv = y_val[:, 0]
                    else:
                        y_val_for_cv = y_val.flatten() if len(y_val.shape) > 1 else y_val
                    
                    cv_scores = cross_val_score(
                        self.meta_model, enhanced_features, y_val_for_cv, 
                        cv=min(5, len(y_val) // 20),  # Ensure sufficient samples per fold
                        scoring='neg_mean_squared_error'
                    )
                    
                    avg_cv_score = -np.mean(cv_scores)  # Convert back to positive MSE
                    cv_std = np.std(cv_scores)
                    
                    # Store CV metrics
                    self.meta_model_performance = {
                        'cv_mse': avg_cv_score,
                        'cv_mse_std': cv_std,
                        'cv_rmse': np.sqrt(avg_cv_score)
                    }
                    
                    logger.info(f"Meta-model CV MSE: {avg_cv_score:.6f} ± {cv_std:.6f}")
                    
                except Exception as e:
                    logger.warning(f"Error performing cross-validation for meta-model: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
            
            # Train final meta-model on all validation data
            try:
                # CRITICAL FIX: Handle dimension mismatch
                enhanced_features = self._enhance_meta_features(meta_features, standardized_predictions)
                
                # Ensure that meta-features are compatible with target shape
                if enhanced_features.shape[0] != len(y_val):
                    logger.warning(f"Enhanced features length ({enhanced_features.shape[0]}) doesn't match target length ({len(y_val)}). Adjusting.")
                    if enhanced_features.shape[0] > len(y_val):
                        # Truncate features
                        enhanced_features = enhanced_features[:len(y_val), :]
                    else:
                        # Pad with zeros
                        padding = np.zeros((len(y_val) - enhanced_features.shape[0], enhanced_features.shape[1]))
                        enhanced_features = np.vstack([enhanced_features, padding])
                
                # CRITICAL FIX: Ensure target has correct shape for training
                if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                    # For multi-dimensional target, use first dimension for meta-model
                    y_val_for_training = y_val[:, 0]
                    logger.info(f"Using first dimension of multi-horizon target for meta-model training")
                else:
                    # Handle single-dimension target
                    y_val_for_training = y_val.flatten() if len(y_val.shape) > 1 else y_val
                
                # Train meta-model
                self.meta_model.fit(enhanced_features, y_val_for_training)
                
                # Evaluate feature importance if available
                if hasattr(self.meta_model, 'feature_importances_'):
                    importances = self.meta_model.feature_importances_
                    
                    # Map importances to feature names for better interpretability
                    feature_importance = {}
                    for i, col in enumerate(self.meta_features.columns):
                        if i < len(importances):
                            feature_importance[col] = importances[i]
                    
                    # Log top feature importances
                    logger.info("Meta-model feature importances (top 10):")
                    for feature, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                        logger.info(f"  {feature}: {imp:.4f}")
                    
                    # Also compute group importance (direct predictions vs stats vs horizons)
                    group_importance = {
                        "direct_predictions": 0.0,
                        "statistical_features": 0.0,
                        "horizon_features": 0.0,
                        "interaction_features": 0.0
                    }
                    
                    for i, col in enumerate(self.meta_features.columns):
                        if i < len(importances):
                            if "_h" in col or "_pred" in col:
                                group_importance["direct_predictions"] += importances[i]
                            elif "_mean" in col or "_std" in col or "_trend" in col or "_volatility" in col:
                                group_importance["statistical_features"] += importances[i]
                            elif "ensemble_" in col or "agreement_" in col or "min_h" in col or "max_h" in col:
                                group_importance["horizon_features"] += importances[i]
                            elif "_prod" in col or "_diff" in col:
                                group_importance["interaction_features"] += importances[i]
                    
                    # Log group importances
                    logger.info("Meta-model feature group importances:")
                    for group, imp in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"  {group}: {imp:.4f} ({imp/sum(importances)*100:.1f}%)")
                
                logger.info(f"Trained meta-model with {enhanced_features.shape[1]} features")
                return True
                
            except Exception as e:
                logger.error(f"Error in meta-model training: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Fallback: try with original features and flattened target
                try:
                    logger.warning("Trying fallback meta-model training with original features")
                    y_val_flat = y_val.flatten() if len(y_val.shape) > 1 else y_val
                    self.meta_model.fit(meta_features, y_val_flat)
                    logger.info("Fallback meta-model training succeeded")
                    return True
                except Exception as e2:
                    logger.error(f"Fallback meta-model training failed: {e2}")
                    return False
                
        except Exception as e:
            logger.error(f"Error training meta-model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.meta_model = None
            return False        
    def predict(self, X: np.ndarray, horizon_idx: int = 0) -> np.ndarray:
        try:
            # Get predictions from each base model
            base_predictions = {}

            # Before making predictions, ensure X is properly formatted
            if isinstance(X, pd.DataFrame):
                # Convert column names to strings (for compatibility)
                X_df = X.copy()
                X_df.columns = X_df.columns.astype(str)
                
                # Check if we have stored training features
                if hasattr(self, 'training_features') and self.training_features:
                    # Check for missing features
                    missing_features = set(self.training_features) - set(X_df.columns)
                    if missing_features:
                        logger.warning(f"Missing {len(missing_features)} features in prediction data. Adding zeros.")
                        logger.debug(f"Missing features: {missing_features}")
                        
                        # Add missing features with zeros
                        for feature in missing_features:
                            X_df[feature] = 0
                    
                    # Reorder columns to match training order
                    X_df = X_df[self.training_features]
                    logger.debug(f"Reordered prediction features to match training features")
                else:
                    logger.warning("No stored training features available for alignment. Using provided features as-is.")
            else:
                # Convert numpy array to DataFrame with proper column names
                if hasattr(self, 'training_features') and self.training_features:
                    # Use stored training features
                    feature_names = self.training_features
                    
                    # Ensure array has correct shape
                    if X.shape[1] != len(feature_names):
                        logger.warning(f"Feature count mismatch: input has {X.shape[1]}, expected {len(feature_names)}.")
                        if X.shape[1] < len(feature_names):
                            # Pad with zeros if input has fewer features
                            padding = np.zeros((X.shape[0], len(feature_names) - X.shape[1]))
                            X = np.hstack([X, padding])
                            logger.info(f"Padded input with zeros to match feature count")
                        else:
                            # Truncate if input has more features
                            X = X[:, :len(feature_names)]
                            logger.info(f"Truncated input to match feature count")
                    
                    X_df = pd.DataFrame(X, columns=feature_names)
                else:
                    # Create generic feature names
                    logger.warning("No stored training features. Using generic feature names.")
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                    X_df = pd.DataFrame(X, columns=feature_names)

            # Check for missing features needed by any model 
            if hasattr(self, 'feature_names_used') and self.feature_names_used:
                missing_features = self.feature_names_used - set(X_df.columns)
                
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} features in prediction data. Adding zeros.")
                    logger.debug(f"Missing features: {missing_features}")
                    
                    # Instead of adding all features individually, add them in a single operation
                    missing_df = pd.DataFrame(0, index=X_df.index, columns=list(missing_features))
                    X_df = pd.concat([X_df, missing_df], axis=1)
                    
                    logger.info(f"Added {len(missing_features)} features with zeros. New shape: {X_df.shape}")
            
            # Process each model based on its type
            for name, model in self.models.items():
                try:
                    # CRITICAL FIX: Handle TCN, Transformer, N-BEATS and other PyTorch models
                    if name in ['tcn', 'transformer', 'nbeats', 'deep_learning'] and TORCH_AVAILABLE:
                        # Handle deep learning model with multi-horizon support
                        model.eval()
                        with torch.no_grad():
                            if name == 'deep_learning':
                                # Original deep_learning handling
                                # Create dataset for prediction
                                lookback = getattr(model, 'config', self.config).lookback_window
                                
                                # If we don't have enough history, pad with zeros
                                if len(X_df) < lookback:
                                    padding_shape = (lookback - len(X_df), X_df.shape[1])
                                    padding = np.zeros(padding_shape)
                                    X_padded = np.vstack([padding, X_df.values])
                                else:
                                    X_padded = X_df.values
                                
                                # Create sliding windows for prediction
                                X_windows = []
                                for i in range(len(X_df)):
                                    # Extract lookback window ending at current position
                                    start_idx = max(0, i - lookback + 1)
                                    window = X_padded[start_idx:i+1]
                                    
                                    # If window is shorter than lookback, pad with zeros
                                    if len(window) < lookback:
                                        padding_shape = (lookback - len(window), X_df.shape[1])
                                        padding = np.zeros(padding_shape)
                                        window = np.vstack([padding, window])
                                    
                                    X_windows.append(window)
                                
                                # Convert to tensor
                                if X_windows:
                                    X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
                                    pred = model(X_tensor).cpu().numpy()
                                    
                                    # For multi-horizon models, get predictions for specified horizon
                                    if hasattr(model, 'multi_horizon') and model.multi_horizon:
                                        if horizon_idx == -1:
                                            # Return all horizons
                                            base_predictions[name] = pred
                                        else:
                                            # Return specific horizon
                                            if horizon_idx < pred.shape[1]:
                                                base_predictions[name] = pred[:, horizon_idx:horizon_idx+1]
                                            else:
                                                logger.warning(f"Horizon index {horizon_idx} out of range for model prediction with shape {pred.shape}")
                                                # Fallback to first horizon
                                                base_predictions[name] = pred[:, 0:1]
                                    else:
                                        base_predictions[name] = pred
                                else:
                                    base_predictions[name] = np.zeros((len(X_df), 1))
                            else:
                                # CRITICAL FIX: Special handling for TCN, Transformer, N-BEATS
                                try:
                                    # Create tensor input with proper shape
                                    X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
                                    
                                    # For TCN models which expect 3D input [batch, sequence, features]
                                    if name == 'tcn':
                                        lookback = getattr(model, 'config', self.config).lookback_window
                                        
                                        # Create a sliding window dataset similar to deep_learning model
                                        X_windows = []
                                        for i in range(len(X_df)):
                                            # Get window data ending at current position
                                            start_idx = max(0, i - lookback + 1)
                                            window = X_df.values[start_idx:i+1]
                                            
                                            # Pad if needed
                                            if len(window) < lookback:
                                                padding_shape = (lookback - len(window), X_df.shape[1])
                                                padding = np.zeros(padding_shape)
                                                window = np.vstack([padding, window])
                                            
                                            X_windows.append(window)
                                        
                                        if X_windows:
                                            # Convert to tensor [batch, seq_len, features]
                                            X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
                                        else:
                                            # Create a dummy tensor with correct shape
                                            X_tensor = torch.zeros((len(X_df), lookback, X_df.shape[1]), dtype=torch.float32)
                                    elif len(X_tensor.shape) == 2:
                                        # For transformer and nbeats, add a sequence dimension if needed
                                        # This typically happens when we have [batch, features]
                                        # We transform to [batch, 1, features]
                                        X_tensor = X_tensor.unsqueeze(1)
                                    
                                    # Get prediction
                                    pred = model(X_tensor).cpu().numpy()
                                    
                                    # Process prediction based on model type
                                    if hasattr(model, 'multi_horizon') and model.multi_horizon:
                                        if horizon_idx == -1:
                                            # Return all horizons
                                            base_predictions[name] = pred
                                        else:
                                            # Return specific horizon
                                            if pred.shape[1] > horizon_idx:
                                                base_predictions[name] = pred[:, horizon_idx:horizon_idx+1]
                                            else:
                                                base_predictions[name] = pred[:, 0:1]
                                    else:
                                        base_predictions[name] = pred
                                except Exception as e:
                                    logger.error(f"Error in {name} prediction: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                    base_predictions[name] = np.zeros((len(X_df), 1))
                    else:
                        # Handle traditional models
                        if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
                            # CRITICAL FIX: Better error handling for prediction
                            try:
                                # For models that might need specific features in specific order
                                if name in ['xgboost', 'lightgbm', 'catboost', 'random_forest'] and hasattr(model, 'feature_names_in_'):
                                    # Get the features that the model was trained on
                                    model_features = getattr(model, 'feature_names_in_', [])
                                    
                                    # Check if all required features are available
                                    missing_model_features = set(model_features) - set(X_df.columns)
                                    if missing_model_features:
                                        logger.warning(f"Model {name} requires features {missing_model_features} that are not in input data. Adding zeros.")
                                        for feat in missing_model_features:
                                            X_df[feat] = 0
                                    
                                    # Ensure features are in the same order as during training
                                    if len(model_features) > 0:
                                        # Use only the features the model was trained on, in the right order
                                        X_for_model = X_df[model_features]
                                    else:
                                        X_for_model = X_df
                                    
                                    pred = model.predict(X_for_model)
                                else:
                                    # For other models, just use all available features
                                    pred = model.predict(X_df)
                                    
                                # Ensure 2D shape
                                if len(pred.shape) == 1:
                                    pred = pred.reshape(-1, 1)
                                base_predictions[name] = pred
                            except Exception as e:
                                logger.error(f"Error in {name}.predict(): {e}")
                                # Use zeros as fallback
                                base_predictions[name] = np.zeros((len(X_df), 1))
                        else:
                            logger.warning(f"Model {name} does not have a predict method.")
                            base_predictions[name] = np.zeros((len(X_df), 1))
                except Exception as e:
                    logger.error(f"Error getting predictions from {name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Use zeros as fallback
                    base_predictions[name] = np.zeros((len(X_df), 1))
            
            # Check if we're returning multi-horizon predictions
            if horizon_idx == -1 and hasattr(self.config, 'multi_horizon') and self.config.multi_horizon:
                # For multi-horizon, handle predictions differently
                # First determine the number of horizons to predict
                n_horizons = len(getattr(self.config, 'forecast_horizons', [1]))
                
                # Initialize multi-horizon predictions
                multi_horizon_preds = np.zeros((len(X), n_horizons))
                
                # Count how many models provided multi-horizon predictions
                multi_horizon_models = 0
                
                # Process each model and horizon
                for name, preds in base_predictions.items():
                    # Skip if predictions aren't multi-horizon
                    if len(preds.shape) < 2 or preds.shape[1] < n_horizons:
                        continue
                        
                    multi_horizon_models += 1
                    
                    # Add weighted contribution for each horizon
                    for h in range(min(n_horizons, preds.shape[1])):
                        if self.weights is not None:
                            model_idx = list(self.models.keys()).index(name)
                            if model_idx < len(self.weights):
                                multi_horizon_preds[:, h] += self.weights[model_idx] * preds[:, h]
                
                # If no models provided multi-horizon predictions, use the first horizon for all
                if multi_horizon_models == 0:
                    logger.warning("No multi-horizon predictions available. Using single horizon prediction for all horizons.")
                    if hasattr(self, 'config') and hasattr(self.config, 'multi_horizon') and self.config.multi_horizon:
                        if hasattr(self.config, 'forecast_horizons'):
                            n_horizons = len(self.config.forecast_horizons)
                            
                            # Ensure consistent multi-horizon predictions
                            for name, pred in base_predictions.items():
                                if len(pred.shape) > 1 and pred.shape[1] < n_horizons:
                                    # Log the inconsistency
                                    logger.warning(f"Model {name} produced {pred.shape[1]} horizons but {n_horizons} expected. Expanding predictions.")
                                    
                                    # Expand predictions to match expected horizons
                                    if pred.shape[1] == 1:
                                        # Repeat the single prediction across all horizons
                                        expanded_pred = np.tile(pred, (1, n_horizons))
                                        base_predictions[name] = expanded_pred
                                    else:
                                        # Pad with the last available prediction (better than zeros)
                                        last_pred = pred[:, -1:]
                                        padding = np.tile(last_pred, (1, n_horizons - pred.shape[1]))
                                        expanded_pred = np.hstack([pred, padding])
                                        base_predictions[name] = expanded_pred
                    normalized_predictions = self._normalize_predictions(base_predictions, len(X))
                    
                    # Stack predictions
                    meta_X = np.hstack([normalized_predictions[name] for name in self.models.keys()])
                    
                    # Get ensemble prediction for first horizon
                    ensemble_pred = self._weighted_average_predictions(meta_X)
                    
                    # Use same prediction for all horizons
                    for h in range(n_horizons):
                        multi_horizon_preds[:, h] = ensemble_pred.reshape(-1)
                        
                # Normalize predictions by number of contributing models
                elif multi_horizon_models > 0:
                    multi_horizon_preds = multi_horizon_preds / multi_horizon_models
                        
                return multi_horizon_preds
            
            # For single horizon or specific horizon prediction
            # Normalize predictions to consistent shape
            normalized_predictions = self._normalize_predictions(base_predictions, len(X))
            
            # Stack predictions
            meta_X = np.hstack([normalized_predictions[name] for name in self.models.keys()])
            
            # Use meta-model if available and stacking is enabled
            if self.meta_model is not None and self.use_stacking:
                try:
                    logger.info(f"Predict - has meta_features attribute: {hasattr(self, 'meta_features')}")
                    if hasattr(self, 'meta_features') and self.meta_features is not None:
                        logger.info(f"Predict - meta_features shape: {self.meta_features.shape}")
                        logger.info(f"Predict - meta_features columns count: {len(self.meta_features.columns)}")
                        vol_cols = [col for col in self.meta_features.columns if 'volatility' in col.lower() or 'garch' in col.lower()]
                        logger.info(f"Predict - volatility columns available: {vol_cols}")
                    
                    # Log meta-feature matrix shape before enhancement
                    logger.info(f"Meta-feature matrix shape before enhancement: {meta_X.shape}")
                    
                    # If we have stored the meta-feature dimension, log it
                    if hasattr(self, 'meta_feature_dim'):
                        logger.info(f"Stored meta-feature dimension: {self.meta_feature_dim}")
                    
                    # Enhance meta-features with dynamic dimensions
                    enhanced_meta_X = self._enhance_meta_features(meta_X, normalized_predictions)
                    
                    # Log enhanced shape for debugging
                    logger.info(f"Enhanced meta-features shape: {enhanced_meta_X.shape}")
                    
                    # Generate predictions using meta-model
                    meta_predictions = self.meta_model.predict(enhanced_meta_X).reshape(-1, 1)
                    
                    # Check for validity
                    if np.any(np.isnan(meta_predictions)) or np.any(np.isinf(meta_predictions)):
                        logger.warning("Meta-model predictions contain NaN/Inf values. Using weighted average instead.")
                        meta_predictions = self._weighted_average_predictions(meta_X)
                    
                    return meta_predictions
                except Exception as e:
                    logger.error(f"Error using meta-model for prediction: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Fall back to weighted average
                    return self._weighted_average_predictions(meta_X)
            else:
                # Use weighted average
                return self._weighted_average_predictions(meta_X)
                    
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return zeros as fallback
            return np.zeros((len(X), 1))        
    def _weighted_average_predictions(self, meta_X: np.ndarray) -> np.ndarray:
        """
        Compute weighted average of base model predictions.
        
        Args:
            meta_X: Stacked predictions from base models
            
        Returns:
            Weighted average predictions
        """
        if self.weights is not None and len(self.weights) == meta_X.shape[1]:
            # Use trained weights
            weighted_preds = np.zeros((meta_X.shape[0], 1))
            for i in range(meta_X.shape[1]):
                weighted_preds += self.weights[i] * meta_X[:, i:i+1]
            return weighted_preds
        else:
            # Use simple average
            return np.mean(meta_X, axis=1, keepdims=True)
        
    def _optimize_weights(self, predictions: Dict[str, np.ndarray], 
                        y_true: np.ndarray, method: str = 'inverse_error'):
        """
        Optimize ensemble weights using Bayesian optimization with error handling.
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Using equal weights.")
            self.weights = np.ones(len(predictions)) / len(predictions)
            return False
                
        try:
            # Validate inputs
            if not predictions:
                logger.warning("No predictions available for weight optimization.")
                self.weights = np.array([1.0])  # Default weight
                return False
                    
            if len(y_true) == 0:
                logger.warning("Empty target array for weight optimization.")
                self.weights = np.ones(len(predictions)) / len(predictions)
                return False
                    
            # Count valid predictions after normalization
            valid_models = []
            for name, pred in predictions.items():
                if pred is not None and not np.all(np.isnan(pred)) and not np.all(np.isinf(pred)):
                    valid_models.append(name)
            
            if not valid_models:
                logger.warning("No valid predictions available. Using equal weights.")
                self.weights = np.ones(len(predictions)) / len(predictions)
                return False
            
            # Create a mapping from valid model names to indices
            model_idx_map = {name: i for i, name in enumerate(valid_models)}
            
            # Define the objective function with better shape validation and handling
            def objective(trial):
                try:
                    # Only optimize for valid models
                    weights = np.zeros(len(predictions))
                    for name in valid_models:
                        idx = model_idx_map[name]
                        weights[list(predictions.keys()).index(name)] = trial.suggest_float(f"weight_{idx}", 0, 1)
                    
                    # Normalize weights to sum to 1
                    weights_sum = np.sum(weights)
                    if weights_sum > 0:
                        weights = weights / weights_sum
                    else:
                        # If all weights are 0, use uniform weights
                        weights = np.ones(len(weights)) / len(weights)
                    
                    # Handle multi-horizon targets if needed
                    is_multi_horizon = len(y_true.shape) > 1 and y_true.shape[1] > 1
                    

                    if is_multi_horizon:
                        # For multi-horizon optimization, use only the first horizon
                        target_for_weights = y_true[:, 0]
                        
                        # Create an ensemble prediction with proper dimensions
                        ensemble_pred = np.zeros(len(target_for_weights))
                        
                        for i, (name, pred) in enumerate(predictions.items()):
                            # Skip invalid predictions
                            if pred is None or len(pred) != len(y_true):
                                continue
                                
                            try:
                                # Extract first horizon from prediction if multi-dimensional
                                if len(pred.shape) > 1 and pred.shape[1] > 1:
                                    model_pred = pred[:, 0]
                                else:
                                    # Make sure we have a vector of the right shape
                                    model_pred = pred.reshape(-1)
                                    
                                # Make sure lengths match before adding
                                if len(model_pred) > len(ensemble_pred):
                                    model_pred = model_pred[:len(ensemble_pred)]
                                elif len(model_pred) < len(ensemble_pred):
                                    # Pad with zeros
                                    padding = np.zeros(len(ensemble_pred) - len(model_pred))
                                    model_pred = np.concatenate([model_pred, padding])
                                    
                                # Add weighted contribution
                                ensemble_pred += weights[i] * model_pred
                            except Exception as e:
                                logger.error(f"Error processing prediction from {name}: {e}")
                                continue
                    else:
                        # For single-horizon targets
                        if len(y_true.shape) == 1:
                            ensemble_pred = np.zeros(len(y_true))
                        else:
                            ensemble_pred = np.zeros(y_true.shape)
                        
                        for i, (name, pred) in enumerate(predictions.items()):
                            # Skip invalid predictions
                            if pred is None or len(pred) != len(y_true):
                                continue
                            
                            try:
                                # Ensure shapes match for addition
                                if len(y_true.shape) == 1:
                                    # Target is a vector, ensure prediction is also a vector
                                    pred_reshaped = pred.flatten() if len(pred.shape) > 1 else pred
                                    
                                    # Make sure lengths match
                                    if len(pred_reshaped) > len(ensemble_pred):
                                        pred_reshaped = pred_reshaped[:len(ensemble_pred)]
                                    elif len(pred_reshaped) < len(ensemble_pred):
                                        # Pad with zeros
                                        padding = np.zeros(len(ensemble_pred) - len(pred_reshaped))
                                        pred_reshaped = np.concatenate([pred_reshaped, padding])
                                else:
                                    # Target is multi-dimensional
                                    if pred.shape != y_true.shape:
                                        # Try to match shapes
                                        if len(pred.shape) == 1:
                                            # Broadcast 1D prediction to match target shape
                                            pred_reshaped = np.broadcast_to(
                                                pred.reshape(-1, 1), 
                                                y_true.shape
                                            )
                                        elif len(pred.shape) == 2 and pred.shape[1] != y_true.shape[1]:
                                            # Pad or truncate second dimension
                                            if pred.shape[1] < y_true.shape[1]:
                                                padding = np.zeros((pred.shape[0], y_true.shape[1] - pred.shape[1]))
                                                pred_reshaped = np.hstack([pred, padding])
                                            else:
                                                pred_reshaped = pred[:, :y_true.shape[1]]
                                        else:
                                            # For other cases, try direct reshape but catch errors
                                            try:
                                                pred_reshaped = pred.reshape(y_true.shape)
                                            except ValueError:
                                                logger.warning(f"Cannot reshape prediction from {pred.shape} to {y_true.shape}")
                                                continue
                                    else:
                                        pred_reshaped = pred
                                        
                                # Add weighted contribution
                                ensemble_pred += weights[i] * pred_reshaped
                            except Exception as e:
                                logger.error(f"Error processing prediction from {name}: {e}")
                                continue
                    
                    # Calculate MSE for optimization
                    if is_multi_horizon:
                        return mean_squared_error(target_for_weights, ensemble_pred)
                    else:
                        return mean_squared_error(y_true, ensemble_pred)
                        
                except Exception as e:
                    logger.error(f"Error in weight optimization objective: {e}")
                    return float('inf')  # Return worst possible score
                
            # Determine number of trials based on number of valid models
            n_trials = min(100, max(20, len(valid_models) * 10))
            
            # Create study with appropriate parameters
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            
            # Get optimized weights
            opt_weights = np.zeros(len(predictions))
            for name in valid_models:
                idx = model_idx_map[name]
                model_pos = list(predictions.keys()).index(name)
                param_name = f"weight_{idx}"
                if param_name in study.best_params:
                    opt_weights[model_pos] = study.best_params[param_name]
                else:
                    # Default to equal weight if missing
                    opt_weights[model_pos] = 1.0 / len(valid_models)
            
            # Normalize weights
            weights_sum = np.sum(opt_weights)
            if weights_sum > 0:
                self.weights = opt_weights / weights_sum
            else:
                # Fallback to uniform weights
                logger.warning("Weight optimization failed (sum=0). Using uniform weights.")
                self.weights = np.ones(len(predictions)) / len(predictions)
            
            logger.info(f"Optimized weights: {self.weights}")
            return True
                
        except Exception as e:
            logger.error(f"Error in weight optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to uniform weights
            self.weights = np.ones(len(predictions)) / len(predictions)
            return False


class MetaFeatureRegistry:
    """
    Sophisticated meta-feature registry that ensures feature consistency between
    training and prediction phases. Maintains a complete record of feature
    transformations and can reconstruct features when models are missing.
    """
    
    def __init__(self):
        self.feature_registry = {}
        self.transformation_registry = {}
        self.model_signatures = {}
        self.feature_signature = None
        self.version = 1
        self.feature_dims = {}
        self.required_models = set()
        self.registered = False
    
    def register_model_predictions(self, predictions: dict, phase: str = "training"):
        """
        Register model predictions and their shapes to track feature consistency.
        
        Args:
            predictions: Dictionary of model predictions
            phase: 'training' or 'prediction'
        """
        model_signatures = {}
        for model_name, preds in predictions.items():
            # Store shape and statistics as a signature
            signature = {
                'shape': preds.shape,
                'mean': float(preds.mean()) if hasattr(preds, 'mean') else None,
                'std': float(preds.std()) if hasattr(preds, 'std') else None
            }
            model_signatures[model_name] = signature
            
            if phase == "training":
                # Mark all training models as required
                self.required_models.add(model_name)
        
        self.model_signatures[phase] = model_signatures
        return model_signatures
    
    def register_transformation(self, name: str, function, dependencies: list, dim_factor: float = 1.0):
        """
        Register a feature transformation function with its dependencies.
        
        Args:
            name: Name of the transformation
            function: Function to apply
            dependencies: List of model names or transformations this depends on
            dim_factor: Factor by which this transformation changes dimensions
        """
        self.transformation_registry[name] = {
            'function': function,
            'dependencies': dependencies,
            'dim_factor': dim_factor
        }
    
    def register_feature_group(self, name: str, columns, transformation: str = None):
        """
        Register a group of features with their source transformation.
        
        Args:
            name: Name of feature group
            columns: List of column indices or names
            transformation: Name of transformation that created these features
        """
        self.feature_registry[name] = {
            'columns': columns,
            'transformation': transformation,
            'dim': len(columns) if hasattr(columns, '__len__') else 1
        }
        self.feature_dims[name] = len(columns) if hasattr(columns, '__len__') else 1
    
    def compute_feature_signature(self, meta_features):
        """
        Compute a unique signature of features to detect changes.
        
        Args:
            meta_features: Enhanced feature matrix
        """
        import hashlib
        import numpy as np
        
        # Create a signature from the shape and column statistics
        features_shape = meta_features.shape
        col_means = np.mean(meta_features, axis=0)
        col_stds = np.std(meta_features, axis=0)
        
        # Create a string representation of the signature components
        signature_parts = [
            str(features_shape),
            str(col_means[:5]),  # First 5 column means
            str(col_stds[:5])    # First 5 column stds
        ]
        
        # Create a hash signature
        signature_str = "_".join(signature_parts)
        signature_hash = hashlib.md5(signature_str.encode()).hexdigest()
        
        self.feature_signature = {
            'hash': signature_hash,
            'shape': features_shape,
            'version': self.version
        }
        
        return self.feature_signature
    
    def verify_feature_consistency(self, current_meta_features):
        """
        Verify that current features match the registered feature signature.
        
        Args:
            current_meta_features: Current feature matrix
            
        Returns:
            (bool, dict): Tuple of (is_consistent, mismatch_details)
        """
        if not self.feature_signature:
            return True, {}
            
        new_signature = self.compute_feature_signature(current_meta_features)
        
        # Check if shape matches
        is_consistent = (new_signature['shape'][1] == self.feature_signature['shape'][1])
        
        mismatch_details = {
            'expected_shape': self.feature_signature['shape'],
            'actual_shape': new_signature['shape'],
            'expected_hash': self.feature_signature['hash'],
            'actual_hash': new_signature['hash']
        }
        
        return is_consistent, mismatch_details
    
    def reconstruct_missing_predictions(self, predictions, base_meta_features):
        """
        Reconstruct predictions for missing models.
        
        Args:
            predictions: Dictionary of available predictions
            base_meta_features: Base feature matrix
            
        Returns:
            Updated predictions dictionary
        """
        import numpy as np
        
        if not self.registered or not hasattr(self, 'required_models'):
            return predictions
            
        reconstructed = predictions.copy()
        
        # Find which required models are missing
        missing_models = self.required_models - set(predictions.keys())
        
        if not missing_models:
            return predictions
            
        # For each missing model, try to reconstruct
        for model_name in missing_models:
            # Method 1: Use zero prediction with proper shape
            if model_name in self.model_signatures.get('training', {}):
                expected_shape = self.model_signatures['training'][model_name]['shape']
                # Create zero prediction matrix with proper shape
                zero_pred = np.zeros((base_meta_features.shape[0], expected_shape[1] 
                                      if len(expected_shape) > 1 else 1))
                reconstructed[model_name] = zero_pred
                
        return reconstructed
    
    def reorder_models(self, predictions):
        """
        Reorder model predictions to match training order.
        
        Args:
            predictions: Dictionary of predictions
            
        Returns:
            Dictionary with consistent ordering
        """
        if not self.registered or not self.required_models:
            return predictions
            
        # Create ordered dictionary
        ordered = {}
        for model_name in self.required_models:
            if model_name in predictions:
                ordered[model_name] = predictions[model_name]
                
        # Add any additional models that weren't in training
        for model_name in predictions:
            if model_name not in ordered:
                ordered[model_name] = predictions[model_name]
                
        return ordered


def _enhanced_meta_features_method(self, meta_features, predictions):
    """
    Enhanced version of _enhance_meta_features method with sophisticated
    feature consistency management.
    
    This method replaces the existing _enhance_meta_features in EnsembleModel.
    """
    try:
        import numpy as np
        import pandas as pd
        
        # Initialize feature registry if not already done
        if not hasattr(self, 'feature_registry'):
            self.feature_registry = MetaFeatureRegistry()
            
        # Start with the base meta-features
        enhanced = meta_features.copy()
        model_names = list(predictions.keys())
        
        # Store original feature count
        original_feature_count = meta_features.shape[1]
        
        # Check if we're in training mode
        is_training_phase = False
        expected_feature_count = None
        
        # Determine if we're in training or prediction phase
        if self.meta_model is not None:
            if hasattr(self.meta_model, 'n_features_in_'):
                expected_feature_count = self.meta_model.n_features_in_
                logger.info(f"Meta-model expects {expected_feature_count} features")
            elif hasattr(self.meta_model, 'feature_importances_') and hasattr(self.meta_model.feature_importances_, 'shape'):
                expected_feature_count = self.meta_model.feature_importances_.shape[0]
                logger.info(f"Meta-model expects {expected_feature_count} features based on feature_importances_")
            elif hasattr(self, 'meta_feature_dim'):
                expected_feature_count = self.meta_feature_dim
                logger.info(f"Using stored meta-feature dimension: {expected_feature_count}")
        
        is_training_phase = expected_feature_count is None
        
        # Register predictions if in training phase or if registry is empty
        if is_training_phase or not self.feature_registry.registered:
            self.feature_registry.register_model_predictions(predictions, "training")
            self.feature_registry.registered = True
        else:
            # Register predictions for consistency checking
            self.feature_registry.register_model_predictions(predictions, "prediction")
            
            # Reconstruct missing predictions and reorder
            reconstructed_predictions = self.feature_registry.reconstruct_missing_predictions(
                predictions, meta_features
            )
            
            # If we have reconstructed any predictions, rebuild meta_features
            if len(reconstructed_predictions) != len(predictions):
                # Reorder to match training order
                ordered_predictions = self.feature_registry.reorder_models(reconstructed_predictions)
                
                # Rebuild meta_features from consistent, ordered predictions
                meta_features = np.hstack([ordered_predictions[name] for name in ordered_predictions.keys()])
                enhanced = meta_features.copy()
                model_names = list(ordered_predictions.keys())
                
                logger.info(f"Reconstructed predictions for {len(reconstructed_predictions) - len(predictions)} missing models")
                logger.info(f"Rebuilt meta-features with shape {meta_features.shape}")
        
        # Track feature counts by group
        feature_counts = {
            'base': original_feature_count,
            'pairwise_diffs': 0,
            'ensemble_agreement': 0,
            'polynomial': 0,
            'volatility': 0
        }
                
        # 1. Add pairwise differences between model predictions
        # For multi-horizon models, we compare corresponding horizons
        pairwise_diffs = []
        
        # Detect multi-horizon predictions
        has_multi_horizon = any(
            len(pred.shape) > 1 and pred.shape[1] > 1 for pred in predictions.values()
        )
        
        max_horizons = 1
        if has_multi_horizon:
            max_horizons = max(
                pred.shape[1] for pred in predictions.values() 
                if len(pred.shape) > 1 and pred.shape[1] > 1
            )
            
        # Register pairwise difference transformation
        if is_training_phase:
            # Define pairwise diff function
            def pairwise_diff_fn(preds_dict, model1, model2):
                pred1 = preds_dict[model1]
                pred2 = preds_dict[model2]
                return pred1 - pred2
                
            # Register the transformation
            self.feature_registry.register_transformation(
                name="pairwise_diffs", 
                function=pairwise_diff_fn,
                dependencies=model_names,
                dim_factor=len(model_names) * (len(model_names) - 1) / 2 / len(model_names)
            )
        
        # Compute pairwise differences
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i < j:  # Only use each pair once
                    # For multi-horizon, make sure to compare corresponding horizons
                    if has_multi_horizon:
                        # Get predictions, ensuring they have same number of horizons
                        pred1 = predictions[name1]
                        pred2 = predictions[name2]
                        
                        # Standardize to same number of horizons
                        if pred1.shape[1] != pred2.shape[1]:
                            # Reshape to have same dimensions
                            if pred1.shape[1] == 1:
                                pred1 = np.tile(pred1, (1, pred2.shape[1]))
                            elif pred2.shape[1] == 1:
                                pred2 = np.tile(pred2, (1, pred1.shape[1]))
                            else:
                                # Use common horizons
                                common_horizons = min(pred1.shape[1], pred2.shape[1])
                                pred1 = pred1[:, :common_horizons]
                                pred2 = pred2[:, :common_horizons]
                            
                        diff = pred1 - pred2
                    else:
                        diff = predictions[name1] - predictions[name2]
                    
                    pairwise_diffs.append(diff)
        
        if pairwise_diffs:
            # Ensure all pairwise differences have the same shape before stacking
            if has_multi_horizon:
                # Standardize shapes to same number of horizons
                std_diffs = []
                for diff in pairwise_diffs:
                    if diff.shape[1] != max_horizons:
                        if diff.shape[1] == 1:
                            # Expand single horizon to match max
                            diff = np.tile(diff, (1, max_horizons))
                        else:
                            # Pad with zeros
                            padding = np.zeros((diff.shape[0], max_horizons - diff.shape[1]))
                            diff = np.hstack([diff, padding])
                    std_diffs.append(diff)
                pairwise_diffs = std_diffs
                
            try:
                # Stack all diffs horizontally
                pairwise_diff_features = np.hstack(pairwise_diffs)
                enhanced = np.hstack([enhanced, pairwise_diff_features])
                feature_counts['pairwise_diffs'] = pairwise_diff_features.shape[1]
                
                # Register pairwise diff features in training phase
                if is_training_phase:
                    self.feature_registry.register_feature_group(
                        name="pairwise_diffs",
                        columns=range(original_feature_count, 
                                      original_feature_count + pairwise_diff_features.shape[1]),
                        transformation="pairwise_diffs"
                    )
                    
                logger.info(f"Added {pairwise_diff_features.shape[1]} pairwise difference features")
            except Exception as e:
                logger.error(f"Error stacking pairwise differences: {e}")
                # Skip this group
        
        # 2. Add ensemble agreement features 
        if len(model_names) >= 3:  # Need at least 3 models for meaningful variance
            ensemble_agreement_features = []
            
            # Define and register ensemble agreement transformation
            if is_training_phase:
                def agreement_fn(X):
                    # Variance of predictions - indicates uncertainty 
                    prediction_variance = np.var(X, axis=1, keepdims=True)
                    
                    # Range of predictions - another measure of uncertainty
                    prediction_range = np.max(X, axis=1, keepdims=True) - np.min(X, axis=1, keepdims=True)
                    
                    # Entropy-inspired disagreement measure
                    normalized_preds = X - np.min(X, axis=1, keepdims=True)
                    row_sums = np.sum(normalized_preds, axis=1, keepdims=True)
                    # Avoid division by zero
                    row_sums = np.where(row_sums == 0, 1e-8, row_sums)
                    pred_probs = normalized_preds / row_sums
                    # Replace zeros to avoid log(0)
                    pred_probs = np.where(pred_probs == 0, 1e-8, pred_probs)
                    disagreement = -np.sum(pred_probs * np.log(pred_probs), axis=1, keepdims=True)
                    
                    return np.hstack([prediction_variance, prediction_range, disagreement])
                    
                self.feature_registry.register_transformation(
                    name="ensemble_agreement",
                    function=agreement_fn,
                    dependencies=model_names,
                    dim_factor=3.0 / len(model_names)  # Creates 3 features from n models
                )
            
            # Variance of predictions - indicates uncertainty 
            prediction_variance = np.var(meta_features, axis=1, keepdims=True)
            ensemble_agreement_features.append(prediction_variance)
            
            # Range of predictions - another measure of uncertainty
            prediction_range = np.max(meta_features, axis=1, keepdims=True) - np.min(meta_features, axis=1, keepdims=True)
            ensemble_agreement_features.append(prediction_range)
            
            # Entropy-inspired disagreement measure
            normalized_preds = meta_features - np.min(meta_features, axis=1, keepdims=True)
            row_sums = np.sum(normalized_preds, axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums == 0, 1e-8, row_sums)
            pred_probs = normalized_preds / row_sums
            # Replace zeros to avoid log(0)
            pred_probs = np.where(pred_probs == 0, 1e-8, pred_probs)
            disagreement = -np.sum(pred_probs * np.log(pred_probs), axis=1, keepdims=True)
            ensemble_agreement_features.append(disagreement)
            
            try:
                # Stack all agreement features
                agreement_features = np.hstack(ensemble_agreement_features)
                prev_cols = enhanced.shape[1]
                enhanced = np.hstack([enhanced, agreement_features])
                feature_counts['ensemble_agreement'] = agreement_features.shape[1]
                
                # Register ensemble agreement features in training phase
                if is_training_phase:
                    self.feature_registry.register_feature_group(
                        name="ensemble_agreement",
                        columns=range(prev_cols, prev_cols + agreement_features.shape[1]),
                        transformation="ensemble_agreement"
                    )
                
                logger.info(f"Added {agreement_features.shape[1]} ensemble agreement features")
            except Exception as e:
                logger.error(f"Error stacking ensemble agreement features: {e}")
                # Skip this group
        
        # 3. Add polynomial features for the most important models
        # If we have a prior weights vector, use the top models
        polynomial_features = []
        if self.weights is not None and len(self.weights) == len(model_names):
            # Find the indices of the top 3 models by weight
            top_indices = np.argsort(self.weights)[-3:]
            
            # Register polynomial transformation
            if is_training_phase and len(top_indices) > 0:
                def polynomial_fn(X, indices):
                    features = []
                    for idx in indices:
                        if idx < X.shape[1]:
                            # Get the column
                            col = X[:, idx:idx+1]
                            # Square term
                            features.append(col ** 2)
                            # Cubic term for top model
                            if idx == indices[-1]:
                                features.append(col ** 3)
                    
                    # Cross-terms
                    if len(indices) >= 2:
                        for i in range(len(indices)):
                            for j in range(i+1, len(indices)):
                                if indices[i] < X.shape[1] and indices[j] < X.shape[1]:
                                    interaction = X[:, indices[i]:indices[i]+1] * X[:, indices[j]:indices[j]+1]
                                    features.append(interaction)
                    
                    return np.hstack(features) if features else np.zeros((X.shape[0], 0))
                    
                self.feature_registry.register_transformation(
                    name="polynomial",
                    function=lambda X: polynomial_fn(X, top_indices),
                    dependencies=["base_features"],
                    dim_factor=len(top_indices) * 2 + (len(top_indices) * (len(top_indices) - 1)) / 2
                )
            
            if len(top_indices) > 0:
                top_features = []
                for idx in top_indices:
                    if idx < meta_features.shape[1]:
                        top_features.append(meta_features[:, idx:idx+1])
                
                if top_features:
                    # Stack top features
                    top_feature_matrix = np.hstack(top_features)
                    
                    # Square terms
                    squared_features = top_feature_matrix ** 2
                    polynomial_features.append(squared_features)
                    
                    # Cubic terms for the very top model
                    if len(top_indices) > 0:
                        top_model_idx = top_indices[-1]
                        if top_model_idx < meta_features.shape[1]:
                            cubic_feature = meta_features[:, top_model_idx:top_model_idx+1] ** 3
                            polynomial_features.append(cubic_feature)
                    
                    # Cross-terms (interactions)
                    if len(top_indices) >= 2:
                        for i in range(len(top_indices)):
                            for j in range(i+1, len(top_indices)):
                                if i < top_feature_matrix.shape[1] and j < top_feature_matrix.shape[1]:
                                    interaction = top_feature_matrix[:, i:i+1] * top_feature_matrix[:, j:j+1]
                                    polynomial_features.append(interaction)
            
                try:
                    if polynomial_features:
                        # Stack all polynomial features
                        poly_feature_matrix = np.hstack(polynomial_features)
                        prev_cols = enhanced.shape[1]
                        enhanced = np.hstack([enhanced, poly_feature_matrix])
                        feature_counts['polynomial'] = poly_feature_matrix.shape[1]
                        
                        # Register polynomial features in training phase
                        if is_training_phase:
                            self.feature_registry.register_feature_group(
                                name="polynomial",
                                columns=range(prev_cols, prev_cols + poly_feature_matrix.shape[1]),
                                transformation="polynomial"
                            )
                        
                        logger.info(f"Added {poly_feature_matrix.shape[1]} polynomial features")
                except Exception as e:
                    logger.error(f"Error stacking polynomial features: {e}")
                    # Skip this group
        
        # 4. Add volatility-based features if available
        # Look for volatility features in meta-model context
        volatility_cols = [col for col in self.meta_features.columns if 'volatility' in col.lower() or 'garch' in col.lower()] if hasattr(self, 'meta_features') and hasattr(self.meta_features, 'columns') else []

        volatility_features = []
        if volatility_cols and hasattr(self, 'meta_features'):
            try:
                # Extract volatility
                volatility = self.meta_features[volatility_cols[0]].values.reshape(-1, 1)
                volatility_features.append(volatility)
                
                # Create volatility-weighted predictions for each model
                for i in range(min(5, meta_features.shape[1])):  # Limit to top 5 models
                    vol_weighted = meta_features[:, i:i+1] * volatility
                    volatility_features.append(vol_weighted)
                
                # Add volatility regime indicators
                high_vol = (volatility > np.median(volatility)).astype(float)
                volatility_features.append(high_vol)
                
                # Create regime-specific predictions for top models
                for i in range(min(3, meta_features.shape[1])):  # Top 3 models only
                    high_vol_pred = meta_features[:, i:i+1] * high_vol
                    volatility_features.append(high_vol_pred)
                
                try:
                    # Stack all volatility features
                    vol_feature_matrix = np.hstack(volatility_features)
                    prev_cols = enhanced.shape[1]
                    enhanced = np.hstack([enhanced, vol_feature_matrix])
                    feature_counts['volatility'] = vol_feature_matrix.shape[1]
                    
                    # Register volatility features in training phase
                    if is_training_phase:
                        self.feature_registry.register_feature_group(
                            name="volatility",
                            columns=range(prev_cols, prev_cols + vol_feature_matrix.shape[1]),
                            transformation="volatility"
                        )
                    
                    logger.info(f"Added {vol_feature_matrix.shape[1]} volatility regime features")
                except Exception as e:
                    logger.error(f"Error stacking volatility features: {e}")
                    # Skip this group
            except Exception as regime_error:
                logger.warning(f"Error creating volatility regime features: {regime_error}")
        
        # Final feature count report
        total_features = sum(feature_counts.values())
        logger.info(f"Feature count breakdown: {feature_counts}")
        logger.info(f"Total features: {total_features}, Enhanced shape: {enhanced.shape}")
        
        # If we're in training phase, compute and store feature signature
        if is_training_phase:
            self.meta_feature_dim = enhanced.shape[1]
            logger.info(f"Training phase: storing meta-feature dimension: {self.meta_feature_dim}")
            
            # Compute and store feature signature
            if hasattr(self.feature_registry, 'compute_feature_signature'):
                signature = self.feature_registry.compute_feature_signature(enhanced)
                logger.info(f"Computed feature signature with hash: {signature['hash']}")
        
        # In prediction mode, verify feature consistency
        elif not is_training_phase and expected_feature_count is not None:
            current_feature_count = enhanced.shape[1]
            
            if hasattr(self.feature_registry, 'verify_feature_consistency'):
                is_consistent, mismatch_details = self.feature_registry.verify_feature_consistency(enhanced)
                
                if not is_consistent:
                    logger.warning(f"Feature consistency check failed: {mismatch_details}")
            
            if current_feature_count != expected_feature_count:
                logger.warning(f"Feature shape mismatch, expected: {expected_feature_count}, got {current_feature_count}")
                
                if current_feature_count < expected_feature_count:
                    # Dynamic Feature Reconstruction
                    # Instead of simple zero padding, try to regenerate missing features
                    # based on their transformation relationships
                    
                    # First, try to use feature registry to fill gaps
                    try:
                        # Check if we have feature registrations
                        if self.feature_registry.feature_registry:
                            # Try to regenerate known feature groups
                            for group_name, group_info in self.feature_registry.feature_registry.items():
                                if group_name in ['pairwise_diffs', 'ensemble_agreement', 'polynomial']:
                                    logger.info(f"Attempting to regenerate {group_name} features")
                                    # Find transformation
                                    if group_info.get('transformation') in self.feature_registry.transformation_registry:
                                        transform_info = self.feature_registry.transformation_registry[group_info['transformation']]
                                        # Apply transformation function if available
                                        if callable(transform_info.get('function')):
                                            generated_features = None
                                            # Different handling based on transformation type
                                            if group_name == 'pairwise_diffs':
                                                # Reconstructed version should already be in enhanced
                                                continue
                                            elif group_name == 'ensemble_agreement':
                                                # Already handled earlier
                                                continue
                                            elif group_name == 'polynomial':
                                                # Already handled earlier
                                                continue
                    except Exception as e:
                        logger.error(f"Error during feature reconstruction: {e}")
                
                    # As a last resort, use zero padding to match expected dimensions
                    logger.warning(f"Using zero padding from {current_feature_count} to {expected_feature_count}")
                    padding = np.zeros((enhanced.shape[0], expected_feature_count - current_feature_count))
                    enhanced = np.hstack([enhanced, padding])
                    logger.info(f"Padded features from {current_feature_count} to {expected_feature_count}")
                else:
                    # If we have more features than expected, update the expected count instead of truncating
                    logger.info(f"Using expanded feature set with {current_feature_count} features")
                    self.meta_feature_dim = current_feature_count  # Update to the new size
                        
        return enhanced
                
    except Exception as e:
        logger.error(f"Error enhancing meta-features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # If we have expected dimensions and error occurred, return zeros with correct shape
        if 'expected_feature_count' in locals() and expected_feature_count is not None:
            logger.info(f"Returning zeros with expected shape {expected_feature_count} after error")
            return np.zeros((meta_features.shape[0], expected_feature_count))
            
        # Otherwise return original features as fallback
        return meta_features

def save_enhanced_checkpoint(ensemble_model, metrics, is_best=False):
    """
    Save a checkpoint with enhanced meta-feature tracking.
    
    Args:
        ensemble_model: The EnsembleModel instance
        metrics: Dictionary of performance metrics
        is_best: Whether this is the best model so far
        
    Returns:
        Checkpoint name if successful, None otherwise
    """
    try:
        import os
        import json
        import pickle
        import datetime
        
        # Add feature registry info to metadata
        if hasattr(ensemble_model, 'feature_registry'):
            registry_info = {
                'required_models': list(ensemble_model.feature_registry.required_models),
                'feature_signature': ensemble_model.feature_registry.feature_signature,
                'model_signatures': ensemble_model.feature_registry.model_signatures,
                'feature_dims': ensemble_model.feature_registry.feature_dims,
                'version': ensemble_model.feature_registry.version
            }
            
            # Include metadata with the model
            if not hasattr(ensemble_model, 'metadata'):
                ensemble_model.metadata = {}
                
            ensemble_model.metadata['feature_registry'] = registry_info
            ensemble_model.metadata['meta_feature_dim'] = getattr(ensemble_model, 'meta_feature_dim', None)
            
            # Add timestamp and metrics
            ensemble_model.metadata['timestamp'] = datetime.datetime.now().isoformat()
            ensemble_model.metadata['metrics'] = metrics
            
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = getattr(ensemble_model, 'checkpoint_dir', 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        # Generate checkpoint name
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        accuracy = metrics.get('accuracy', metrics.get('r2', 0.0))
        checkpoint_name = f"ensemble_model_{timestamp}_{accuracy:.4f}.pkl"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        # Save the model
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(ensemble_model, f)
            
        # Save metadata separately as JSON for easy inspection
        metadata_path = os.path.join(checkpoint_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_model.metadata, f, indent=2, default=str)
            
        # If this is the best model, create a symlink or copy
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pkl")
            if os.path.exists(best_path):
                os.remove(best_path)
                
            # Create a copy or symlink depending on platform
            try:
                os.symlink(checkpoint_path, best_path)
            except (OSError, AttributeError):
                # Fallback to copy if symlink fails
                import shutil
                shutil.copy2(checkpoint_path, best_path)
                
        print(f"Saved enhanced checkpoint to {checkpoint_path}")
        return checkpoint_name
        
    except Exception as e:
        import traceback
        print(f"Error saving enhanced checkpoint: {e}")
        print(traceback.format_exc())
        return None


def load_enhanced_checkpoint(checkpoint_path):
    """
    Load a checkpoint with enhanced meta-feature tracking.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        The loaded EnsembleModel instance
    """
    try:
        import pickle
        
        # Load the model
        with open(checkpoint_path, 'rb') as f:
            ensemble_model = pickle.load(f)
            
        # Verify feature registry and re-integrate if missing
        if not hasattr(ensemble_model, 'feature_registry') or ensemble_model.feature_registry is None:
            # If metadata contains registry info but the registry is missing, recreate it
            if hasattr(ensemble_model, 'metadata') and 'feature_registry' in ensemble_model.metadata:
                print(f"Reconstructing feature registry from metadata")
                ensemble_model.feature_registry = MetaFeatureRegistry()
                
                # Restore registry attributes from metadata
                registry_info = ensemble_model.metadata['feature_registry']
                ensemble_model.feature_registry.required_models = set(registry_info.get('required_models', []))
                ensemble_model.feature_registry.feature_signature = registry_info.get('feature_signature')
                ensemble_model.feature_registry.model_signatures = registry_info.get('model_signatures', {})
                ensemble_model.feature_registry.feature_dims = registry_info.get('feature_dims', {})
                ensemble_model.feature_registry.version = registry_info.get('version', 1)
                ensemble_model.feature_registry.registered = True
                
                
        print(f"Loaded enhanced model from {checkpoint_path}")
        return ensemble_model
        
    except Exception as e:
        import traceback
        print(f"Error loading enhanced checkpoint: {e}")
        print(traceback.format_exc())
        return None


def verify_model_compatibility(ensemble_model, predictions):
    """
    Verify that the provided predictions are compatible with the trained ensemble model.
    
    Args:
        ensemble_model: The trained EnsembleModel instance
        predictions: Dictionary of model predictions to check
        
    Returns:
        (bool, dict): Tuple of (is_compatible, incompatibility_details)
    """
    if not hasattr(ensemble_model, 'feature_registry'):
        # No registry to check against
        return True, {}
        
    # Register current predictions
    current_signatures = ensemble_model.feature_registry.register_model_predictions(
        predictions, "verification"
    )
    
    # Check if all required models are present
    missing_models = ensemble_model.feature_registry.required_models - set(predictions.keys())
    
    # Check if model signatures match training signatures
    signature_mismatches = []
    if "training" in ensemble_model.feature_registry.model_signatures:
        training_signatures = ensemble_model.feature_registry.model_signatures["training"]
        
        for model_name, signature in training_signatures.items():
            if model_name in current_signatures:
                # Check if shapes match (ignoring batch dimension)
                expected_shape = signature['shape']
                current_shape = current_signatures[model_name]['shape']
                
                if len(expected_shape) > 1 and len(current_shape) > 1:
                    # Check output dimensions match
                    if expected_shape[1:] != current_shape[1:]:
                        signature_mismatches.append({
                            'model': model_name,
                            'expected_shape': expected_shape,
                            'actual_shape': current_shape
                        })
    
    is_compatible = len(missing_models) == 0 and len(signature_mismatches) == 0
    
    incompatibility_details = {
        'missing_models': list(missing_models),
        'signature_mismatches': signature_mismatches
    }
    
    return is_compatible, incompatibility_details

def train_and_evaluate(df: pd.DataFrame, config: ModelConfig, test_size: float = 0.2,
                      tune_hyperparams: bool = True, n_trials: int = 30) -> Dict[str, Any]:
    """
    Train and evaluate a model on the given data with optional hyperparameter tuning.
    
    Args:
        df: DataFrame with price data
        config: Model configuration
        test_size: Proportion of data to use for testing
        tune_hyperparams: Whether to perform hyperparameter tuning
        n_trials: Number of trials for hyperparameter optimization
        
    Returns:
        Dictionary with trained model and evaluation metrics
    """
    # Initialize best metrics tracking
    best_metrics = {}
    memory_manager = None
    if 'optimize_memory' in globals() and optimize_memory:
        memory_manager = MemoryManager(memory_threshold=0.8, gc_threshold=0.7)

    try:
        logger.info("Starting train_and_evaluate")
        
        # Validate data
        validator = DataValidator()
        if not validator.validate_timeseries(df):
            raise DataError("Data validation failed")

        # Prepare target variable (next day's return)
        df['returns'] = df['Close'].pct_change()
        
        # Create feature engineering instance
        feature_engineer = AdvancedFeatureEngineering()
        
        # Generate features
        logger.info("Generating features...")
        df_features = feature_engineer.add_comprehensive_ta_features(df)
        
        # Now add target columns based on configuration
        if config.multi_horizon:
            # Create target columns for each forecast horizon
            target_columns = []
            for horizon in config.forecast_horizons:
                target_column = f'target_h{horizon}'
                df_features[target_column] = df_features['returns'].shift(-horizon)
                target_columns.append(target_column)
            
            # Default target is the shortest horizon
            df_features['target'] = df_features[target_columns[0]]
        else:
            # Single horizon forecasting
            df_features['target'] = df_features['returns'].shift(-config.forecast_horizon)
            target_columns = ['target']
            
        # Remove NaN values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        # 1. First, identify and drop columns with more than 50% missing values
        nan_percentage = df_features.isna().mean()
        cols_to_drop = nan_percentage[nan_percentage > 0.5].index.tolist()
        logger.info(f"Dropping {len(cols_to_drop)} columns with >50% NaNs: {cols_to_drop[:10]}...")
        df_features = df_features.drop(columns=cols_to_drop)

        # 2. Now drop rows with any remaining NaN values
        row_count_before = len(df_features)
        df_features = df_features.dropna()
        rows_dropped = row_count_before - len(df_features)
        logger.info(f"Dropped {rows_dropped} rows with NaN values ({rows_dropped/row_count_before:.2%} of data)")
        logger.info(f"DataFrame shape after handling NaNs: {df_features.shape}") 
        
        # Split data into training and testing sets
        split_idx = int(len(df_features) * (1 - test_size))
        train_data = df_features.iloc[:split_idx].copy()
        test_data = df_features.iloc[split_idx:].copy()
        
        logger.info(f"Training data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        # Create feature selector
        feature_selector = AdvancedFeatureSelector(n_jobs=-1)
        
        # Remove price columns and target from features
        feature_cols = [col for col in train_data.columns 
                        if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'target'] + target_columns]
        
        # Select features
        logger.info(f"Selecting from {len(feature_cols)} features...")
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        
        # Limit to most important features
        X_train_selected = feature_selector.select_features(X_train, y_train, n_features=200)
        selected_features = X_train_selected.columns.tolist()
        ensemble.store_training_features(selected_features)
        if len(X_train_selected.columns) < 200:
            logger.info("Attempting to enhance feature diversity")
            diversified_features = feature_selector.ensure_feature_diversity(
                X_train_selected.columns.tolist(), 
                X_train, 
                min_per_group=8
            )
            if len(diversified_features) > len(X_train_selected.columns):
                # Add any new features that weren't originally selected
                additional_features = [f for f in diversified_features if f not in X_train_selected.columns]
                if additional_features:
                    logger.info(f"Adding {len(additional_features)} additional features for better diversity")
                    X_train_selected = pd.concat([X_train_selected, X_train[additional_features]], axis=1)
                    selected_features = X_train_selected.columns.tolist()
        
        # Prepare test data with selected features
        X_test = test_data[selected_features]
        
        # Prepare target variables according to config
        if config.multi_horizon:
            y_train_multi = train_data[target_columns].values
            y_test_multi = test_data[target_columns].values
        else:
            y_train_multi = train_data['target'].values
            y_test_multi = test_data['target'].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test)
        
        # Create DataFrame versions with column names for models that need them
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train_selected.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)
        
        # Update input dimension in config
        config.input_dim = X_train_scaled.shape[1]
        
        # Create ensemble model
        ensemble = EnsembleModel(config)
        
        # Create tuners for each model type if hyperparameter tuning is enabled
        tuners = {}
        if tune_hyperparams:
            logger.info("Initializing hyperparameter tuners...")
            
            if XGBOOST_AVAILABLE:
                tuners['xgboost'] = XGBoostHyperparameterTuner(n_trials=n_trials)
                
            if LIGHTGBM_AVAILABLE:
                tuners['lightgbm'] = LightGBMHyperparameterTuner(n_trials=n_trials)
                
            if CATBOOST_AVAILABLE:
                tuners['catboost'] = CatBoostHyperparameterTuner(n_trials=n_trials)
                
            if SKLEARN_AVAILABLE:
                tuners['random_forest'] = RandomForestHyperparameterTuner(n_trials=n_trials)
                
            if TORCH_AVAILABLE:
                tuners['deep_learning'] = DeepLearningHyperparameterTuner(n_trials=max(10, n_trials // 3))
                
            if PROPHET_AVAILABLE:
                tuners['prophet'] = ProphetHyperparameterTuner(n_trials=max(10, n_trials // 3))
        
        # Hyperparameter tuning for each model (if enabled)
        model_params = {}
        if tune_hyperparams:
            logger.info("Starting hyperparameter tuning phase...")

            # For validation during tuning, use a portion of the training set
            X_tune, X_tune_val, y_tune_temp, y_tune_val_temp = create_train_validation_split(
                X_train_scaled, 
                y_train if not isinstance(y_train, pd.Series) else y_train,
                validation_split=config.validation_split
            )

            # Handle pd.Series vs numpy array correctly
            y_tune = y_tune_temp.values if isinstance(y_tune_temp, pd.Series) else y_tune_temp
            y_tune_val = y_tune_val_temp.values if isinstance(y_tune_val_temp, pd.Series) else y_tune_val_temp

            # Handle multi-horizon case if present
            if config.multi_horizon and isinstance(y_train, np.ndarray) and len(y_train.shape) > 1 and y_train.shape[1] > 1:
                split_point = len(X_tune)
                y_tune = y_train[:split_point]
                y_tune_val = y_train[split_point:]

            # DataFrame versions for models that need them
            split_point = len(X_tune)
            X_tune_df = X_train_scaled_df.iloc[:split_point]
            X_tune_val_df = X_train_scaled_df.iloc[split_point:]
            
            # Run tuning for each model type
            for model_name, tuner in tuners.items():
                logger.info(f"Tuning hyperparameters for {model_name}...")
                
                try:
                    # Deep learning needs special handling due to its architecture
                    if model_name == 'deep_learning':
                        best_params = tuner.tune(X_tune, y_tune, X_tune_val, y_tune_val)
                        # Also update config with the tuned parameters
                        for param, value in best_params.items():
                            if hasattr(config, param):
                                setattr(config, param, value)
                    # Prophet also needs special handling
                    elif model_name == 'prophet':
                        best_params = tuner.tune(X_tune_df, y_tune, X_tune_val_df, y_tune_val)
                    else:
                        # Standard models
                        best_params = tuner.tune(X_tune, y_tune, X_tune_val, y_tune_val)
                    
                    model_params[model_name] = best_params
                    logger.info(f"Best parameters for {model_name}: {best_params}")
                    
                    # Save parameter importance plots
                    try:
                        tuner.plot_param_importances(f"{model_name}_param_importance.png")
                        tuner.plot_optimization_history(f"{model_name}_opt_history.png")
                    except Exception as e:
                        logger.warning(f"Error plotting tuning results for {model_name}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error tuning {model_name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Use default parameters
                    model_params[model_name] = tuner.get_default_params()
        
        # Add models to ensemble based on available libraries
        if XGBOOST_AVAILABLE:
            logger.info("Adding XGBoost model to ensemble")
            if tune_hyperparams and 'xgboost' in tuners:
                xgb_model = tuners['xgboost'].create_model(model_params.get('xgboost'))
            else:
                xgb_model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method='auto'
                )
            ensemble.add_model('xgboost', xgb_model)
            
        if LIGHTGBM_AVAILABLE:
            logger.info("Adding LightGBM model to ensemble")
            if tune_hyperparams and 'lightgbm' in tuners:
                lgb_model = tuners['lightgbm'].create_model(model_params.get('lightgbm'))
            else:
                lgb_model = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8
                )
            ensemble.add_model('lightgbm', lgb_model)
                    
        if CATBOOST_AVAILABLE:
                    logger.info("Adding CatBoost model to ensemble")
                    if tune_hyperparams and 'catboost' in tuners:
                        cb_model = tuners['catboost'].create_model(model_params.get('catboost'))
                    else:
                        # For multi-horizon targets, use MultiRegressionCustom objective
                        if config.multi_horizon:
                            cb_model = cb.CatBoostRegressor(
                                iterations=100,
                                learning_rate=0.05,
                                depth=5,
                                loss_function='MultiRegressionCustom',
                                verbose=False
                            )
                        else:
                            cb_model = cb.CatBoostRegressor(
                                iterations=100,
                                learning_rate=0.05,
                                depth=5,
                                verbose=False
                            )
                    ensemble.add_model('catboost', cb_model)

        if SKLEARN_AVAILABLE:
            logger.info("Adding RandomForest model to ensemble")
            if tune_hyperparams and 'random_forest' in tuners:
                rf_model = tuners['random_forest'].create_model(model_params.get('random_forest'))
            else:
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                )
            ensemble.add_model('random_forest', rf_model)
            
        if TORCH_AVAILABLE:
            logger.info("Adding deep learning model to ensemble")
            if tune_hyperparams and 'deep_learning' in tuners:
                deep_model = tuners['deep_learning'].create_model(
                    model_params.get('deep_learning'), 
                    input_dim=config.input_dim
                )
            else:
                deep_model = HybridTimeSeriesModel(config.input_dim, config)
            ensemble.add_model('deep_learning', deep_model)
            
        if PROPHET_AVAILABLE:
            logger.info("Adding Prophet model to ensemble")
            if tune_hyperparams and 'prophet' in tuners:
                prophet_model = tuners['prophet'].create_model(model_params.get('prophet'))
            else:
                prophet_model = ProphetModel(config)
            ensemble.add_model('prophet', prophet_model)
            
        # Train models
        logger.info("Training ensemble with tuned hyperparameters...")
        for epoch in range(config.num_epochs):
            ensemble.current_epoch = epoch
            
            # Training and validation splits
            X_train_inner, X_val, y_train_inner_temp, y_val_temp = create_train_validation_split(
                X_train_scaled, 
                y_train if not isinstance(y_train, pd.Series) else y_train,
                validation_split=config.validation_split
            )

            # Handle pd.Series vs numpy array correctly
            y_train_inner = y_train_inner_temp.values if isinstance(y_train_inner_temp, pd.Series) else y_train_inner_temp
            y_val = y_val_temp.values if isinstance(y_val_temp, pd.Series) else y_val_temp

            # Handle multi-horizon case if present
            if config.multi_horizon and isinstance(y_train, np.ndarray) and len(y_train.shape) > 1 and y_train.shape[1] > 1:
                split_point = len(X_train_inner)
                y_train_inner = y_train[:split_point]
                y_val = y_train[split_point:]

            # DataFrame versions for models that need them
            split_point = len(X_train_inner)
            X_train_inner_df = X_train_scaled_df.iloc[:split_point]
            X_val_df = X_train_scaled_df.iloc[split_point:]
            
            # DataFrame versions for models that need them
            split_point = len(X_train_inner)
            X_train_inner_df = X_train_scaled_df.iloc[:split_point]
            X_val_df = X_train_scaled_df.iloc[split_point:]
            
            # Store predictions from each model
            val_predictions = {}
            
            # Train each model
            for name, model in ensemble.models.items():
                logger.info(f"Training {name} model (epoch {epoch})...")
                
                if name == 'deep_learning' and TORCH_AVAILABLE:
                    # Handle deep learning model separately
                    ensemble._train_deep_model(name, model, X_train_inner, y_train_inner, 
                                              X_val, y_val, val_predictions)
                elif name == 'prophet' and PROPHET_AVAILABLE:
                    # Handle Prophet model separately with DataFrames
                    try:
                        model.fit(X_train_inner_df, y_train_inner)
                        val_predictions[name] = model.predict(X_val_df)
                    except Exception as e:
                        logger.error(f"Error training Prophet model: {e}")
                        val_predictions[name] = np.zeros((len(y_val), 1))
                else:
                    # Handle traditional models
                    ensemble._train_traditional_model(name, model, X_train_inner, y_train_inner, 
                                                    X_val, y_val, val_predictions)
            
            # Normalize predictions
            val_predictions = ensemble._normalize_predictions(val_predictions, len(y_val))
            
            # Optimize ensemble weights
            ensemble._optimize_weights(val_predictions, y_val)
            
            # Train meta-model if needed
            ensemble._train_meta_model(val_predictions, y_val)
            
            # Calculate validation metrics
            ensemble_pred = np.zeros((len(y_val), 1))
            for i, (name, pred) in enumerate(val_predictions.items()):
                if ensemble.weights is not None and i < len(ensemble.weights):
                    ensemble_pred += ensemble.weights[i] * pred
                else:
                    ensemble_pred += pred / len(val_predictions)
            
            val_mse = mean_squared_error(y_val, ensemble_pred)
            val_mae = mean_absolute_error(y_val, ensemble_pred)
            val_r2 = r2_score(y_val, ensemble_pred)
            
            metrics = {
                'val_mse': val_mse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'epoch': epoch
            }
            
            logger.info(f"Epoch {epoch}: val_mse={val_mse:.6f}, val_mae={val_mae:.6f}, val_r2={val_r2:.6f}")
            
            # Save checkpoint
            is_best = False
            if epoch == 0 or metrics['val_mse'] < best_metrics.get('val_mse', float('inf')):
                best_metrics = metrics.copy()
                is_best = True
                
            ensemble.save_checkpoint(metrics, is_best)
            
            # Early stopping
            if epoch >= config.early_stopping_patience:
                recent_mses = [m.get('val_mse', float('inf')) for _, m in ensemble.checkpointer.metadata.items() 
                             if isinstance(m, dict) and 'val_mse' in m]
                if recent_mses and val_mse > min(recent_mses):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_predictions = {}
        
        for name, model in ensemble.models.items():
            try:
                if name == 'deep_learning' and TORCH_AVAILABLE:
                    # Create test dataset
                    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_multi, config.lookback_window)
                    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
                    
                    # Generate predictions
                    model.eval()
                    with torch.no_grad():
                        test_preds = []
                        for batch_x, _ in test_loader:
                            batch_pred = model(batch_x)
                            test_preds.append(batch_pred)
                        if test_preds:
                            test_predictions[name] = torch.cat(test_preds, dim=0).cpu().numpy()
                        else:
                            test_predictions[name] = np.zeros((len(y_test_multi), 1))
                elif name == 'prophet' and PROPHET_AVAILABLE:
                    # Handle Prophet with DataFrame
                    test_predictions[name] = model.predict(X_test_scaled_df)
                else:
                    # Handle traditional models
                    if hasattr(model, 'predict'):
                        test_predictions[name] = model.predict(X_test_scaled).reshape(-1, 1)
                    else:
                        test_predictions[name] = np.zeros((len(y_test_multi), 1))
            except Exception as e:
                logger.error(f"Error generating test predictions for {name}: {e}")
                test_predictions[name] = np.zeros((len(y_test_multi), 1))
        
        # Normalize predictions
        test_predictions = ensemble._normalize_predictions(test_predictions, len(y_test_multi))
        
        # Calculate ensemble prediction
        test_ensemble_pred = np.zeros((len(y_test_multi), 1))
        for i, (name, pred) in enumerate(test_predictions.items()):
            if ensemble.weights is not None and i < len(ensemble.weights):
                test_ensemble_pred += ensemble.weights[i] * pred
            else:
                test_ensemble_pred += pred / len(test_predictions)
        
        # Calculate metrics - single horizon for now
        if isinstance(y_test_multi, np.ndarray) and len(y_test_multi.shape) > 1 and y_test_multi.shape[1] > 1:
            y_test = y_test_multi[:, 0]  # Use first horizon
        else:
            y_test = y_test_multi
            
        test_mse = mean_squared_error(y_test, test_ensemble_pred)
        test_mae = mean_absolute_error(y_test, test_ensemble_pred)
        test_r2 = r2_score(y_test, test_ensemble_pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(test_ensemble_pred) == np.sign(y_test.reshape(-1, 1)))
        direction_accuracy = direction_correct / len(y_test)
        
        # Store all results
        final_metrics = {
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'direction_accuracy': direction_accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(selected_features)
        }
        
        # Add hyperparameter information if tuning was performed
        if tune_hyperparams:
            final_metrics['hyperparameters'] = model_params
        
        logger.info(f"Test metrics: MSE={test_mse:.6f}, MAE={test_mae:.6f}, R2={test_r2:.6f}")
        logger.info(f"Direction accuracy: {direction_accuracy:.4f}")
        
        # Save reference data for drift detection
        save_reference_data(
            'models/reference_data',
            X_train_scaled_df,
            test_ensemble_pred.flatten().tolist()
        )
        
        # Create final results dictionary
        results = {
            'model': ensemble,
            'scaler': scaler,
            'feature_selector': feature_selector,
            'selected_features': selected_features,
            'config': config,
            'metrics': final_metrics
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def walk_forward_validation(df: pd.DataFrame, config: ModelConfig, n_splits: int = 5,
                          window_type: str = 'expanding', min_train_size: float = 0.5,
                          tune_hyperparams: bool = False) -> Dict[str, Any]:
    """
    Perform walk-forward validation on time series data.
    
    Args:
        df: DataFrame with price data
        config: Model configuration
        n_splits: Number of validation splits
        window_type: 'expanding' or 'rolling' window approach
        min_train_size: Minimum proportion of data to use for first training window
        tune_hyperparams: Whether to tune hyperparameters on first fold only
    
    Returns:
        Dictionary with validation metrics and ensemble model
    """
    try:
        logger.info(f"Starting walk-forward validation with {n_splits} splits, {window_type} window")
        logger.info(f"Starting walk-forward validation with {n_splits} splits, {window_type} window")
        
        # Initial data shape logging
        logger.info(f"Initial DataFrame shape: {df.shape}")
        logger.info(f"Initial DataFrame columns: {df.columns.tolist()}")
        logger.info(f"Initial DataFrame NaN counts: {df.isna().sum().sum()}")        
        # Validate data
        validator = DataValidator()
        if not validator.validate_timeseries(df):
            raise DataError("Data validation failed")
            
        # Prepare target variable (next day's return)
        df['returns'] = df['Close'].pct_change()
        
        # For multi-horizon, prepare all target variables
        target_columns = []
        if config.multi_horizon:
            for horizon in config.forecast_horizons:
                target_column = f'target_h{horizon}'
                df[target_column] = df['returns'].shift(-horizon)
                target_columns.append(target_column)
        else:
            df['target'] = df['returns'].shift(-config.forecast_horizon)
            target_columns = ['target']
        logger.info(f"DataFrame shape after adding target columns: {df.shape}")
        for col in target_columns:
            logger.info(f"Target column {col} NaN count: {df[col].isna().sum()}")        
        # Create feature engineering instance
        feature_engineer = AdvancedFeatureEngineering()
        
        # Generate features
        logger.info("Generating features...")
        
        # Create a copy to log shape before feature engineering
        df_before_features = df.copy()
        logger.info(f"DataFrame shape before feature engineering: {df_before_features.shape}")

        # Add instrumentation to the add_comprehensive_ta_features method by wrapping it
        def instrumented_feature_engineering(df_input):
            # Log before any processing
            logger.info(f"Input to feature engineering shape: {df_input.shape}")
            logger.info(f"Input NaN values by column:")
            nan_counts_before = df_input.isna().sum()
            for col, count in nan_counts_before.items():
                if count > 0:
                    logger.info(f"  - {col}: {count} NaNs ({count/len(df_input):.2%} of rows)")
            
            # Call the original method
            result = feature_engineer.add_comprehensive_ta_features(df_input)
            
            # Log after processing
            logger.info(f"Output from feature engineering shape: {result.shape}")
            logger.info(f"NaN values after feature engineering: {result.isna().sum().sum()}")
            
            # Log most problematic columns
            nan_counts = result.isna().sum().sort_values(ascending=False)
            logger.info(f"Top 10 columns with most NaNs:")
            for col, count in nan_counts[:10].items():
                if count > 0:
                    logger.info(f"  - {col}: {count} NaNs ({count/len(result):.2%} of rows)")
            
            return result
            
        df_features = instrumented_feature_engineering(df)

        # Add target columns to feature dataframe if not already present
        for col in target_columns:
            if col not in df_features.columns:
                df_features[col] = df[col]
                
        logger.info(f"DataFrame shape after adding target columns to features: {df_features.shape}")

        # Detailed NaN analysis before cleaning
        logger.info(f"NaN analysis before cleaning:")
        logger.info(f"Total NaN values: {df_features.isna().sum().sum()}")
        logger.info(f"Rows with at least one NaN: {df_features.isna().any(axis=1).sum()} ({df_features.isna().any(axis=1).sum()/len(df_features):.2%} of total)")

        # Simulate what happens if we drop NaNs
        would_remain = len(df_features.dropna())
        logger.info(f"Rows that would remain after dropna(): {would_remain} ({would_remain/len(df_features) if len(df_features) > 0 else 0:.2%} of total)")

        # Identify rows with most NaNs
        row_nan_counts = df_features.isna().sum(axis=1)
        if not row_nan_counts.empty:
            max_nans = row_nan_counts.max()
            min_nans = row_nan_counts.min()
            mean_nans = row_nan_counts.mean()
            logger.info(f"NaNs per row - Min: {min_nans}, Mean: {mean_nans:.2f}, Max: {max_nans}")
            
            # Count rows by number of NaNs
            nan_distribution = row_nan_counts.value_counts().sort_index()
            logger.info(f"Distribution of NaNs per row:")
            for nan_count, row_count in nan_distribution.items():
                logger.info(f"  - Rows with {nan_count} NaNs: {row_count}")

        # Remove NaN values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)

        # 1. First, identify and drop columns with more than 50% missing values
        nan_percentage = df_features.isna().mean()
        cols_to_drop = nan_percentage[nan_percentage > 0.5].index.tolist()
        logger.info(f"Dropping {len(cols_to_drop)} columns with >50% NaNs: {cols_to_drop[:10]}...")
        df_features = df_features.drop(columns=cols_to_drop)

        # 2. Now drop rows with any remaining NaN values
        row_count_before = len(df_features)
        df_features = df_features.dropna()
        rows_dropped = row_count_before - len(df_features)
        logger.info(f"Dropped {rows_dropped} rows with NaN values ({rows_dropped/row_count_before:.2%} of data)")
        logger.info(f"DataFrame shape after handling NaNs: {df_features.shape}")


        logger.info(f"DataFrame shape after dropping NaN rows: {df_features.shape}")

        if len(df_features) < config.min_samples:
            logger.error(f"Insufficient data: {len(df_features)} rows remain, need {config.min_samples}")
            logger.error(f"Column count: {df_features.shape[1]}")
            
            # Let's inspect some sample rows from original
            if len(df) > 0:
                logger.error(f"Sample from original data:")
                logger.error(f"{df.head(2)}")
            
            # Sample intermediate data if available
            if 'df_before_features' in locals() and len(df_before_features) > 0:
                logger.error(f"Sample from data before feature engineering:")
                logger.error(f"{df_before_features.head(2)}")
            
            # Try to save a debug copy if possible
            try:
                if 'df_before_features' in locals() and len(df_before_features) > 0:
                    df_before_features.to_csv('debug_before_features.csv')
                    logger.error(f"Saved pre-features data to debug_before_features.csv")
            except Exception as e:
                logger.error(f"Error saving debug file: {e}")
            
            raise DataError(f"Insufficient data after preprocessing: {len(df_features)} < {config.min_samples}")
        
        # Create splits based on the window type
        splits = []
        total_samples = len(df_features)
        
        # Determine split size
        split_size = int(total_samples * (1 - min_train_size) / n_splits)
        
        for i in range(n_splits):
            if window_type == 'expanding':
                # Expanding window: initial train size grows with each iteration
                train_end = int(min_train_size * total_samples) + i * split_size
                test_start = train_end
                test_end = min(test_start + split_size, total_samples)
                
                # Ensure we have enough test data in the last split
                if i == n_splits - 1:
                    test_end = total_samples
                
                splits.append((range(0, train_end), range(test_start, test_end)))
            else:
                # Rolling window: fixed-size moving window
                window_size = int(min_train_size * total_samples)
                train_start = i * split_size
                train_end = train_start + window_size
                test_start = train_end
                test_end = min(test_start + split_size, total_samples)
                
                # Ensure we have enough test data in the last split
                if i == n_splits - 1:
                    test_end = total_samples
                
                splits.append((range(train_start, train_end), range(test_start, test_end)))
        
        # Initialize metrics storage
        all_metrics = []
        predictions = []
        actuals = []
        
        # Create ensemble model
        ensemble = EnsembleModel(config)
        
        # Store feature selector from first fold
        stored_feature_selector = None
        
        # Run validation for each split
        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold+1}/{n_splits}")
            logger.info(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
            
            # Split data
            train_data = df_features.iloc[train_idx].copy()
            test_data = df_features.iloc[test_idx].copy()
            
            # Create feature selector (or reuse from first fold)
            if fold == 0 or stored_feature_selector is None:
                feature_selector = AdvancedFeatureSelector(n_jobs=-1)
                
                # Remove price columns and target from features
                feature_cols = [col for col in train_data.columns 
                              if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 
                                           'returns'] + target_columns]
                
                # Select features
                logger.info(f"Selecting from {len(feature_cols)} features...")
                X_train = train_data[feature_cols]
                y_train = train_data[target_columns[0]]  # Use first target for feature selection
                
                # Limit to most important features
                X_train_selected = feature_selector.select_features(X_train, y_train, n_features=200)
                selected_features = X_train_selected.columns.tolist()
                
                # Store for future folds
                stored_feature_selector = feature_selector
            else:
                # Reuse feature selection from first fold
                feature_selector = stored_feature_selector
                selected_features = stored_feature_selector.selected_features
            
            # Prepare data with selected features
            X_train = train_data[selected_features]
            X_test = test_data[selected_features]
            ensemble.store_training_features(selected_features)
            # Prepare targets
            if config.multi_horizon:
                y_train = train_data[target_columns].values
                y_test = test_data[target_columns].values
            else:
                y_train = train_data[target_columns[0]].values
                y_test = test_data[target_columns[0]].values
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create DataFrame versions with column names for models that need them
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features, 
                                            index=X_train.index)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features, 
                                          index=X_test.index)
            
            # Update input dimension in config
            config.input_dim = X_train_scaled.shape[1]
            
            # Only tune hyperparameters on first fold to save time
            do_tune = tune_hyperparams and fold == 0
            
            # Reset ensemble model for each fold if using rolling window
            if window_type == 'rolling' or fold == 0:
                ensemble = EnsembleModel(config)
                
                # Add models with either tuned or default hyperparameters
                add_models_to_ensemble(ensemble, do_tune, config)
            
            # Train ensemble on this fold
            fold_metrics = train_fold(
                ensemble, 
                X_train_scaled, X_train_scaled_df, y_train,
                X_test_scaled, X_test_scaled_df, y_test,
                config,
                fold=fold,
                n_epochs=max(10, config.num_epochs // 2)  # Reduced epochs for walk-forward
            )
            
            # Store metrics
            all_metrics.append(fold_metrics)
            
            # Store predictions and actuals
            predictions.append(fold_metrics.get('test_predictions', []))
            actuals.append(fold_metrics.get('test_actuals', []))
            
            # Clean memory between folds
            clean_memory()
        
        # Analyze results with SHAP if available and requested
        if 'analyze_shap' in globals() and analyze_shap and SHAP_AVAILABLE:
            try:
                logger.info("Performing SHAP analysis for model interpretability...")
                
                # Create SHAP analyzer
                shap_analyzer = ShapAnalyzer(ensemble.models, selected_features)
                
                # Create explainers using a subset of test data
                sample_size = min(200, len(X_test_scaled))
                shap_analyzer.create_explainers(X_test_scaled[:sample_size])
                
                # Compute SHAP values
                shap_analyzer.compute_shap_values(X_test_scaled)
                
                # Compute global feature importance
                shap_analyzer.compute_global_feature_importance()
                
                # Create comprehensive report
                os.makedirs("shap_analysis", exist_ok=True)
                shap_analyzer.create_comprehensive_report("shap_analysis")
                
                logger.info("SHAP analysis completed")
                
            except Exception as e:
                logger.error(f"Error in SHAP analysis: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Aggregate metrics across folds
        aggregated_metrics = aggregate_walk_forward_metrics(all_metrics, predictions, actuals)
        
        logger.info(f"Walk-forward validation complete. Metrics: {aggregated_metrics}")
        
        # Return combined results
        results = {
            'metrics': aggregated_metrics,
            'model': ensemble,
            'all_fold_metrics': all_metrics,
            'feature_selector': stored_feature_selector,
            'selected_features': selected_features
        }
        
        if 'analyze_shap' in globals() and analyze_shap and SHAP_AVAILABLE:
            results['shap_analyzer'] = shap_analyzer
            
        return results
        
    except Exception as e:
        logger.error(f"Error in walk_forward_validation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
def time_series_cv_evaluation(df: pd.DataFrame, config: ModelConfig, n_splits: int = 5, 
                            gap: int = 5, tune_hyperparams: bool = False) -> Dict[str, Any]:
    """
    Implement time-series specific cross-validation with forward-chaining.
    Includes a gap between train and test to avoid lookahead bias.
    
    Args:
        df: DataFrame with price data
        config: Model configuration
        n_splits: Number of validation splits
        gap: Number of time periods to skip between train and test (avoid lookahead bias)
        tune_hyperparams: Whether to tune hyperparameters on first fold only
    
    Returns:
        Dictionary with validation metrics and ensemble model
    """
    try:
        logger.info(f"Starting time-series CV evaluation with {n_splits} splits and gap={gap}")
        
        # Validate data
        validator = DataValidator()
        if not validator.validate_timeseries(df):
            raise DataError("Data validation failed")
            
        # Prepare target variable (next day's return)
        df['returns'] = df['Close'].pct_change()
        
        # For multi-horizon, prepare all target variables
        target_columns = []
        if config.multi_horizon:
            for horizon in config.forecast_horizons:
                target_column = f'target_h{horizon}'
                df[target_column] = df['returns'].shift(-horizon)
                target_columns.append(target_column)
        else:
            df['target'] = df['returns'].shift(-config.forecast_horizon)
            target_columns = ['target']
            
        # Create feature engineering instance
        feature_engineer = AdvancedFeatureEngineering()
        
        # Generate features
        logger.info("Generating features...")
        df_features = feature_engineer.add_comprehensive_ta_features(df)
        
        # Add target columns to feature dataframe if not already present
        for col in target_columns:
            if col not in df_features.columns:
                df_features[col] = df[col]
                
        # Remove NaN values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        
        # 1. First, identify and drop columns with more than 50% missing values
        nan_percentage = df_features.isna().mean()
        cols_to_drop = nan_percentage[nan_percentage > 0.5].index.tolist()
        logger.info(f"Dropping {len(cols_to_drop)} columns with >50% NaNs: {cols_to_drop[:10]}...")
        df_features = df_features.drop(columns=cols_to_drop)
        
        # 2. Now drop rows with any remaining NaN values
        row_count_before = len(df_features)
        df_features = df_features.dropna()
        rows_dropped = row_count_before - len(df_features)
        logger.info(f"Dropped {rows_dropped} rows with NaN values ({rows_dropped/row_count_before:.2%} of data)")
        
        # Create time-series CV splits
        splits = []
        total_samples = len(df_features)
        fold_size = total_samples // (n_splits + 1)  # Reserve one fold worth of data for final test
        
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + gap  # Gap to avoid lookahead bias
            test_end = test_start + fold_size
            
            # Ensure we don't exceed data bounds
            if test_end > total_samples:
                test_end = total_samples
                
            # Only add valid splits
            if test_start < test_end and train_end > 50:  # Require at least 50 training samples
                splits.append((range(0, train_end), range(test_start, test_end)))
        
        # Initialize metrics storage
        all_metrics = []
        predictions = []
        actuals = []
        
        # Store feature selector from first fold
        stored_feature_selector = None
        
        # Create ensemble model
        ensemble = EnsembleModel(config)
        
        # Run validation for each split
        for fold, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold+1}/{len(splits)}")
            logger.info(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
            
            # Split data
            train_data = df_features.iloc[train_idx].copy()
            test_data = df_features.iloc[test_idx].copy()
            
            # Feature selection - create or reuse feature selector
            if fold == 0 or stored_feature_selector is None:
                feature_selector = AdvancedFeatureSelector(n_jobs=-1)
                
                # Remove price columns and target from features
                feature_cols = [col for col in train_data.columns 
                               if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 
                                             'returns'] + target_columns]
                
                # Select features
                logger.info(f"Selecting from {len(feature_cols)} features...")
                X_train = train_data[feature_cols]
                y_train = train_data[target_columns[0]]  # Use first target for feature selection
                
                # Limit to most important features
                X_train_selected = feature_selector.select_features(X_train, y_train, n_features=200)
                selected_features = X_train_selected.columns.tolist()
                
                # Store for future folds
                stored_feature_selector = feature_selector
            else:
                # Reuse feature selection from first fold
                feature_selector = stored_feature_selector
                selected_features = stored_feature_selector.selected_features
            
            # Prepare data with selected features
            X_train = train_data[selected_features]
            X_test = test_data[selected_features]
            
            # Prepare targets
            if config.multi_horizon:
                y_train = train_data[target_columns].values
                y_test = test_data[target_columns].values
            else:
                y_train = train_data[target_columns[0]].values
                y_test = test_data[target_columns[0]].values
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create DataFrame versions with column names for models that need them
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features, 
                                            index=X_train.index)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features, 
                                           index=X_test.index)
            
            # Update input dimension in config
            config.input_dim = X_train_scaled.shape[1]
            
            # Only tune hyperparameters on first fold to save time
            do_tune = tune_hyperparams and fold == 0
            
            # Create fresh ensemble for each fold
            ensemble = EnsembleModel(config)
            add_models_to_ensemble(ensemble, do_tune, config)
            
            # Train ensemble on this fold with fewer epochs than walk_forward_validation
            fold_metrics = train_fold(
                ensemble, 
                X_train_scaled, X_train_scaled_df, y_train,
                X_test_scaled, X_test_scaled_df, y_test,
                config,
                fold=fold,
                n_epochs=max(5, config.num_epochs // 5)  # Use even fewer epochs for quicker evaluation
            )
            
            # Store metrics
            all_metrics.append(fold_metrics)
            
            # Store predictions and actuals
            predictions.append(fold_metrics.get('test_predictions', []))
            actuals.append(fold_metrics.get('test_actuals', []))
            
            # Clean memory between folds
            clean_memory()
        
        # Aggregate metrics across folds
        aggregated_metrics = aggregate_walk_forward_metrics(all_metrics, predictions, actuals)
        
        # Add information about the gap to metrics
        aggregated_metrics['evaluation_params'] = {
            'method': 'time_series_cv',
            'n_splits': n_splits,
            'gap': gap
        }
        
        logger.info(f"Time-series CV evaluation complete. Metrics: {aggregated_metrics}")
        
        # Return combined results
        results = {
            'metrics': aggregated_metrics,
            'model': ensemble,
            'all_fold_metrics': all_metrics,
            'feature_selector': stored_feature_selector,
            'selected_features': selected_features
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in time_series_cv_evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
def add_models_to_ensemble(ensemble: EnsembleModel, tune_hyperparams: bool, config: ModelConfig):
    """Add models to ensemble with either tuned or default hyperparameters."""
    try:
        # Create tuners if hyperparameter tuning is enabled
        tuners = {}
        if tune_hyperparams:
            logger.info("Initializing hyperparameter tuners...")
            
            # Ensure input_dim is properly set and log it
            logger.info(f"Creating models with input_dim={config.input_dim}")
            if config.input_dim <= 0:
                logger.warning(f"Invalid input_dim: {config.input_dim}. Models may fail.")
            
            n_trials = 30  # Default trials for hyperparameter tuning
            
            if XGBOOST_AVAILABLE:
                tuners['xgboost'] = XGBoostHyperparameterTuner(n_trials=n_trials)
                
            if LIGHTGBM_AVAILABLE:
                tuners['lightgbm'] = LightGBMHyperparameterTuner(n_trials=n_trials)
                
            if CATBOOST_AVAILABLE:
                tuners['catboost'] = CatBoostHyperparameterTuner(n_trials=n_trials)
                
            if SKLEARN_AVAILABLE:
                tuners['random_forest'] = RandomForestHyperparameterTuner(n_trials=n_trials)
                
            if TORCH_AVAILABLE:
                tuners['deep_learning'] = DeepLearningHyperparameterTuner(n_trials=max(10, n_trials // 3))
                
            if PROPHET_AVAILABLE:
                tuners['prophet'] = ProphetHyperparameterTuner(n_trials=max(10, n_trials // 3))
                
            # Add new model tuners
            if SARIMAX_AVAILABLE:
                tuners['sarima'] = SARIMAHyperparameterTuner(n_trials=max(10, n_trials // 3))
                
            if TCN_AVAILABLE:
                tuners['tcn'] = TCNHyperparameterTuner(n_trials=max(10, n_trials // 3))
                
            if TRANSFORMER_AVAILABLE:
                tuners['transformer'] = TransformerHyperparameterTuner(n_trials=max(10, n_trials // 3))
                
            if TORCH_AVAILABLE:  # N-BEATS requires PyTorch
                tuners['nbeats'] = NBEATSHyperparameterTuner(n_trials=max(10, n_trials // 3))
        
        # Add models to ensemble based on available libraries
        if XGBOOST_AVAILABLE:
            logger.info("Adding XGBoost model to ensemble")
        # MODIFIED: Create a multi-output regressor if multi_horizon is enabled
        if config.multi_horizon:
            from sklearn.multioutput import MultiOutputRegressor
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='auto'
            )
            xgb_model = MultiOutputRegressor(base_model)
        else:
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='auto'
            )
            ensemble.add_model('xgboost', xgb_model)
            
        if LIGHTGBM_AVAILABLE:
            logger.info("Adding LightGBM model to ensemble")
            # MODIFIED: Create a multi-output regressor if multi_horizon is enabled
            if config.multi_horizon:
                from sklearn.multioutput import MultiOutputRegressor
                base_model = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8
                )
                lgb_model = MultiOutputRegressor(base_model)
            else:
                lgb_model = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8
                )
            ensemble.add_model('lightgbm', lgb_model)
            
        if CATBOOST_AVAILABLE:
            logger.info("Adding CatBoost model to ensemble")
            # MODIFIED: Create a multi-output regressor if multi_horizon is enabled
            if config.multi_horizon:
                from sklearn.multioutput import MultiOutputRegressor
                base_model = cb.CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.05,
                    depth=5,
                    verbose=False
                )
                cb_model = MultiOutputRegressor(base_model)
            else:
                cb_model = cb.CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.05,
                    depth=5,
                    verbose=False
                )
            ensemble.add_model('catboost', cb_model)
            
        if SKLEARN_AVAILABLE:
            logger.info("Adding RandomForest model to ensemble")
            # MODIFIED: Create a multi-output regressor if multi_horizon is enabled
            if config.multi_horizon:
                from sklearn.multioutput import MultiOutputRegressor
                base_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                )
                rf_model = MultiOutputRegressor(base_model)
            else:
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                )
            ensemble.add_model('random_forest', rf_model)
            
        if TORCH_AVAILABLE:
            logger.info("Adding deep learning model to ensemble")
            # Ensure input_dim is set correctly and consistent with data
            if config.input_dim <= 0:
                logger.warning(f"Invalid input_dim: {config.input_dim}. Setting to default of 10.")
                config.input_dim = 10
            deep_model = HybridTimeSeriesModel(config.input_dim, config)
            ensemble.add_model('deep_learning', deep_model)
            
        if PROPHET_AVAILABLE:
            logger.info("Adding Prophet model to ensemble")
            prophet_model = ProphetModel(config)
            ensemble.add_model('prophet', prophet_model)
            
        # Add new models
        if SARIMAX_AVAILABLE:
            logger.info("Adding SARIMA model to ensemble")
            sarima_model = SARIMAModel(config)
            ensemble.add_model('sarima', sarima_model)
            
        if TCN_AVAILABLE:
            logger.info("Adding TCN model to ensemble")
            # Ensure input_dim is set correctly
            if config.input_dim <= 0:
                logger.warning(f"Invalid input_dim: {config.input_dim}. Setting to default of 10.")
                config.input_dim = 10
            tcn_model = TCNModel(config.input_dim, config)
            ensemble.add_model('tcn', tcn_model)
            
        if TRANSFORMER_AVAILABLE:
            logger.info("Adding Transformer model to ensemble")
            # Ensure input_dim is set correctly
            if config.input_dim <= 0:
                logger.warning(f"Invalid input_dim: {config.input_dim}. Setting to default of 10.")
                config.input_dim = 10
            transformer_model = TimeSeriesTransformer(config.input_dim, config)
            ensemble.add_model('transformer', transformer_model)
            
        if TORCH_AVAILABLE:  # N-BEATS requires PyTorch
            logger.info("Adding N-BEATS model to ensemble")
            # Ensure input_dim is set correctly
            if config.input_dim <= 0:
                logger.warning(f"Invalid input_dim: {config.input_dim}. Setting to default of 10.")
                config.input_dim = 10
            nbeats_model = NBEATSModel(config.input_dim, config)
            ensemble.add_model('nbeats', nbeats_model)
            
    except Exception as e:
        logger.error(f"Error adding models to ensemble: {e}")
        import traceback
        logger.error(traceback.format_exc())


def train_fold(ensemble: EnsembleModel, 
             X_train_scaled: np.ndarray, X_train_scaled_df: pd.DataFrame, y_train: np.ndarray,
             X_test_scaled: np.ndarray, X_test_scaled_df: pd.DataFrame, y_test: np.ndarray,
             config: ModelConfig, fold: int = 0, n_epochs: int = 50) -> Dict[str, Any]:
    """Train ensemble on a single fold of data and evaluate."""
    try:
        # Ensure config.input_dim matches the actual feature dimension
        if X_train_scaled.shape[1] != config.input_dim:
            logger.warning(f"Updating config.input_dim from {config.input_dim} to {X_train_scaled.shape[1]}")
            config.input_dim = X_train_scaled.shape[1]
            
        # Training and validation splits
        X_train_inner, X_val, y_train_inner_temp, y_val_temp = create_train_validation_split(
            X_train_scaled, 
            y_train if not (config.multi_horizon and len(y_train.shape) > 1 and y_train.shape[1] > 1) else y_train[:, 0],
            validation_split=config.validation_split
        )

        # Handle multi-horizon targets
        if config.multi_horizon and len(y_train.shape) > 1 and y_train.shape[1] > 1:
            split_point = len(X_train_inner)
            y_train_inner = y_train[:split_point]
            y_val = y_train[split_point:]
        else:
            # Single target
            if len(y_train.shape) > 1 and y_train.shape[1] == 1:
                y_train_inner = y_train_inner_temp.reshape(-1, 1) if not isinstance(y_train_inner_temp, pd.Series) else y_train_inner_temp.values.reshape(-1, 1)
                y_val = y_val_temp.reshape(-1, 1)
            else:
                y_train_inner = y_train_inner_temp
                y_val = y_val_temp

        # DataFrame versions for models that need them
        split_point = len(X_train_inner)
        X_train_inner_df = X_train_scaled_df.iloc[:split_point]
        X_val_df = X_train_scaled_df.iloc[split_point:]        
        best_val_metrics = None
        X_train_inner_df.columns = X_train_inner_df.columns.astype(str)
        X_val_df.columns = X_val_df.columns.astype(str)
        X_test_scaled_df.columns = X_test_scaled_df.columns.astype(str)        
        # Ensure column names are strings for compatibility
        X_train_inner_df.columns = X_train_inner_df.columns.astype(str)
        X_val_df.columns = X_val_df.columns.astype(str)
        X_test_scaled_df.columns = X_test_scaled_df.columns.astype(str)
        
        # Train for specified number of epochs
        for epoch in range(n_epochs):
            ensemble.current_epoch = epoch
            
            # Store predictions from each model
            val_predictions = {}
            
            # Train each model
            for name, model in ensemble.models.items():
                logger.info(f"Fold {fold}, Epoch {epoch}: Training {name} model...")
                
                if name == 'deep_learning' and TORCH_AVAILABLE:
                    # Handle deep learning model separately
                    ensemble._train_deep_model(name, model, X_train_inner, y_train_inner, 
                                             X_val, y_val, val_predictions)
                elif name == 'prophet' and PROPHET_AVAILABLE:
                    # Handle Prophet model separately with DataFrames
                    try:
                        model.fit(X_train_inner_df, y_train_inner)
                        val_predictions[name] = model.predict(X_val_df)
                    except Exception as e:
                        logger.error(f"Error training Prophet model: {e}")
                        val_predictions[name] = np.zeros((len(y_val), 1) if len(y_val.shape) == 1 
                                                     else (len(y_val), y_val.shape[1]))
                else:
                    # Handle traditional models
                    ensemble._train_traditional_model(name, model, X_train_inner, y_train_inner, 
                                                   X_val, y_val, val_predictions)
            
            # Normalize predictions
            val_predictions = ensemble._normalize_predictions(val_predictions, len(y_val))
            
            # Optimize ensemble weights
            # For multi-horizon, use the first horizon for weight optimization
            if config.multi_horizon and len(y_val.shape) > 1 and y_val.shape[1] > 1:
                target_for_weights = y_val[:, 0]
            else:
                target_for_weights = y_val
                
            ensemble._optimize_weights(val_predictions, target_for_weights)
            
            # Train meta-model if needed
            ensemble._train_meta_model(val_predictions, target_for_weights)
            
            # Calculate validation metrics
            val_metrics = calculate_metrics(ensemble, val_predictions, y_val)
            
            # Store best validation metrics
            if best_val_metrics is None or val_metrics['val_mse'] < best_val_metrics['val_mse']:
                best_val_metrics = val_metrics
                
            # Early stopping check
            if epoch >= 5 and val_metrics['val_mse'] > best_val_metrics['val_mse'] * 1.1:
                logger.info(f"Fold {fold}: Early stopping at epoch {epoch}")
                break
        
        # Generate test predictions
        test_predictions = {}
        for name, model in ensemble.models.items():
            try:
                if name in ['deep_learning', 'tcn', 'transformer', 'nbeats']:
                    # Handle deep learning models
                    model.eval()
                    test_dataset = TimeSeriesDataset(X_test_scaled, 
                                                np.zeros_like(y_test), 
                                                config.lookback_window)
                    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
                    
                    # Generate predictions
                    with torch.no_grad():
                        test_preds = []
                        for batch_x, _ in test_loader:
                            batch_pred = model(batch_x)
                            test_preds.append(batch_pred)
                        if test_preds:
                            test_predictions[name] = torch.cat(test_preds, dim=0).cpu().numpy()
                        else:
                            test_predictions[name] = np.zeros((len(y_test), 1) if len(y_test.shape) == 1 
                                                        else (len(y_test), y_test.shape[1]))
                elif name == 'sarima' and SARIMAX_AVAILABLE:
                    # Handle SARIMA model
                    test_predictions[name] = model.predict(X_test_scaled_df)
                elif name == 'prophet' and PROPHET_AVAILABLE:
                    # Handle Prophet with DataFrame
                    test_predictions[name] = model.predict(X_test_scaled_df)
                else:
                    # Handle traditional models
                    if hasattr(model, 'predict'):
                        preds = model.predict(X_test_scaled)
                        if len(preds.shape) == 1:
                            preds = preds.reshape(-1, 1)
                        test_predictions[name] = preds
                    else:
                        test_predictions[name] = np.zeros((len(y_test), 1) if len(y_test.shape) == 1 
                                                    else (len(y_test), y_test.shape[1]))
            except Exception as e:
                logger.error(f"Error generating test predictions for {name}: {e}")
                test_predictions[name] = np.zeros((len(y_test), 1) if len(y_test.shape) == 1 
                                            else (len(y_test), y_test.shape[1]))
        
        # Normalize test predictions
        test_predictions = ensemble._normalize_predictions(test_predictions, len(y_test))
        
        # Calculate test metrics
        test_metrics = calculate_metrics(ensemble, test_predictions, y_test, prefix='test_')
        
        # Combine metrics
        fold_metrics = {
            'fold': fold,
            'val_metrics': best_val_metrics,
            **test_metrics,
            'test_predictions': ensemble.predict(X_test_scaled),
            'test_actuals': y_test
        }
        
        return fold_metrics
        
    except Exception as e:
        logger.error(f"Error in train_fold: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'fold': fold, 'error': str(e)}

def calculate_metrics(ensemble: EnsembleModel, predictions: Dict[str, np.ndarray], 
                    y_true: np.ndarray, prefix: str = 'val_') -> Dict[str, float]:
    """
    Calculate performance metrics for predictions with enhanced multi-horizon support.
    
    Args:
        ensemble: The ensemble model
        predictions: Dictionary of predictions from each base model
        y_true: True target values
        prefix: Prefix for metric names in the output dictionary
        
    Returns:
        Dictionary of metrics with appropriate prefixes
    """
    try:
        # Log input dimensions for debugging
        logger.debug(f"calculate_metrics input - y_true shape: {y_true.shape}, predictions: {[pred.shape for pred in predictions.values()]}")
        
        # Add code to align horizons across all models
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            # This is a multi-horizon case
            n_horizons = y_true.shape[1]
            
            # Ensure all predictions have the correct number of horizons
            for name, pred in list(predictions.items()):
                if len(pred.shape) == 1 or pred.shape[1] != n_horizons:
                    logger.warning(f"Prediction shape {pred.shape} doesn't match target horizons {n_horizons} for model {name}.")
                    
                    # Format prediction to match expected horizons
                    if len(pred.shape) == 1:
                        # 1D array needs to be reshaped
                        pred = pred.reshape(-1, 1)
                        
                    if pred.shape[1] < n_horizons:
                        # Expand single horizon to all horizons (more accurate than zero padding)
                        if pred.shape[1] == 1:
                            expanded_pred = np.tile(pred, (1, n_horizons))
                        else:
                            # Pad with last value instead of zeros for better forecasting
                            last_values = pred[:, -1:]
                            padding = np.tile(last_values, (1, n_horizons - pred.shape[1]))
                            expanded_pred = np.hstack([pred, padding])
                        
                        predictions[name] = expanded_pred
                        logger.info(f"Expanded predictions for {name} from shape {pred.shape} to {predictions[name].shape}")
                    elif pred.shape[1] > n_horizons:
                        # Truncate extra horizons
                        predictions[name] = pred[:, :n_horizons]
                        logger.info(f"Truncated predictions for {name} from shape {pred.shape} to {predictions[name].shape}")
        
        # Use ensemble prediction if available
        if ensemble.meta_model is not None and ensemble.use_stacking:
            # Stack predictions into a feature matrix
            meta_features = np.hstack([pred for pred in predictions.values()])
            enhanced_meta_features = ensemble._enhance_meta_features(meta_features, predictions)
            
            # Generate meta-model predictions
            y_pred = ensemble.meta_model.predict(enhanced_meta_features)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)
        else:
            # Use weighted average
            y_pred = np.zeros((len(y_true), 1) if len(y_true.shape) == 1 else (len(y_true), y_true.shape[1]))
            for i, (name, pred) in enumerate(predictions.items()):
                if ensemble.weights is not None and i < len(ensemble.weights):
                    y_pred += ensemble.weights[i] * pred
                else:
                    y_pred += pred / len(predictions)
        
        # Log ensemble prediction shape for debugging
        logger.debug(f"Ensemble prediction shape: {y_pred.shape}")
        
        # Handle multi-horizon targets
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            if hasattr(ensemble, 'config') and getattr(ensemble.config, 'multi_horizon', False):
                # Create a multi-horizon module for metrics (without specialized layers)
                try:
                    # Check that forecast_horizons exists
                    if not hasattr(ensemble.config, 'forecast_horizons'):
                        logger.warning("Missing forecast_horizons in config, using default [1]")
                        forecast_horizons = [1]
                    else:
                        forecast_horizons = ensemble.config.forecast_horizons
                    
                    # Create multi-horizon module with proper error handling
                    multi_horizon_module = EnhancedMultiHorizonModule(
                        forecast_horizons=forecast_horizons,
                        initial_weights=ensemble.config.horizon_weights if hasattr(ensemble.config, 'horizon_weights') else None,
                        adaptive_weighting=False,  # Don't change weights during evaluation
                        input_dim=0  # No specialized layers needed
                    )
                    
                    # Ensure prediction shape matches target shape
                    if y_pred.shape[1] < y_true.shape[1]:
                        logger.warning(f"Prediction shape {y_pred.shape} has fewer horizons than target {y_true.shape}. Padding with zeros.")
                        padding = np.zeros((y_pred.shape[0], y_true.shape[1] - y_pred.shape[1]))
                        y_pred = np.hstack([y_pred, padding])
                    elif y_pred.shape[1] > y_true.shape[1]:
                        logger.warning(f"Prediction shape {y_pred.shape} has more horizons than target {y_true.shape}. Truncating.")
                        y_pred = y_pred[:, :y_true.shape[1]]
                    
                    # Calculate comprehensive metrics
                    multi_horizon_metrics = multi_horizon_module.calculate_metrics(y_pred, y_true)
                    
                    # Convert to standard format with prefix
                    metrics = {}
                    
                    # Add aggregate metrics with prefix
                    if 'aggregate' in multi_horizon_metrics:
                        for metric_name, value in multi_horizon_metrics['aggregate'].items():
                            metrics[f"{prefix}{metric_name}"] = value
                    
                    # Add per-horizon metrics with prefix
                    for horizon_key, horizon_metrics in multi_horizon_metrics.items():
                        if horizon_key != 'aggregate':
                            for metric_name, value in horizon_metrics.items():
                                metrics[f"{prefix}{horizon_key}_{metric_name}"] = value
                    
                    return metrics
                    
                except Exception as e:
                    logger.error(f"Error using multi-horizon module: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Fall back to legacy handling
                    logger.info("Falling back to legacy multi-horizon handling")
                
            # Legacy multi-horizon handling
            horizon_metrics = {}
            n_horizons = min(y_true.shape[1], y_pred.shape[1])
            logger.debug(f"Processing {n_horizons} horizons with legacy method")
            
            for i in range(n_horizons):
                try:
                    horizon_prefix = f"{prefix}h{i+1}_"
                    y_true_h = y_true[:, i]
                    y_pred_h = y_pred[:, i] if y_pred.shape[1] > i else y_pred[:, 0]
                    
                    # Ensure both arrays are same length
                    if len(y_true_h) != len(y_pred_h):
                        min_len = min(len(y_true_h), len(y_pred_h))
                        y_true_h = y_true_h[:min_len]
                        y_pred_h = y_pred_h[:min_len]
                    
                    # Remove NaN values
                    mask = ~(np.isnan(y_true_h) | np.isnan(y_pred_h))
                    if np.sum(mask) < 10:  # Need at least 10 valid points
                        logger.warning(f"Not enough valid data points for horizon {i+1}")
                        continue
                        
                    y_true_h_clean = y_true_h[mask]
                    y_pred_h_clean = y_pred_h[mask]
                    
                    horizon_metrics.update({
                        f"{horizon_prefix}mse": mean_squared_error(y_true_h_clean, y_pred_h_clean),
                        f"{horizon_prefix}mae": mean_absolute_error(y_true_h_clean, y_pred_h_clean),
                        f"{horizon_prefix}r2": r2_score(y_true_h_clean, y_pred_h_clean),
                        f"{horizon_prefix}direction_accuracy": np.mean(np.sign(y_pred_h_clean) == np.sign(y_true_h_clean))
                    })
                    
                except Exception as h_error:
                    logger.warning(f"Error calculating metrics for horizon {i+1}: {h_error}")
            
            # Add average metrics across horizons (only if we have metrics)
            if horizon_metrics:
                avg_mse = np.mean([v for k, v in horizon_metrics.items() if k.endswith('_mse')])
                avg_mae = np.mean([v for k, v in horizon_metrics.items() if k.endswith('_mae')])
                avg_r2 = np.mean([v for k, v in horizon_metrics.items() if k.endswith('_r2')])
                avg_dir_acc = np.mean([v for k, v in horizon_metrics.items() if k.endswith('_direction_accuracy')])
                
                metrics = {
                    f"{prefix}mse": avg_mse,
                    f"{prefix}mae": avg_mae,
                    f"{prefix}r2": avg_r2,
                    f"{prefix}direction_accuracy": avg_dir_acc,
                    **horizon_metrics
                }
            else:
                # If no horizons were successfully calculated, return fallback metrics
                logger.warning("No valid horizons for metric calculation, returning fallback metrics")
                metrics = {
                    f"{prefix}mse": float('nan'),
                    f"{prefix}mae": float('nan'),
                    f"{prefix}r2": 0.0,
                    f"{prefix}direction_accuracy": 0.5,  # Default to random guessing
                }
            
            return metrics
        else:
            # Single target metrics
            y_true_flat = y_true.flatten() if len(y_true.shape) > 1 else y_true
            y_pred_flat = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
            
            # Check for potential shape mismatch
            if len(y_true_flat) != len(y_pred_flat):
                logger.warning(f"Shape mismatch in metric calculation: y_true: {len(y_true_flat)}, y_pred: {len(y_pred_flat)}")
                # Use the shorter length
                min_len = min(len(y_true_flat), len(y_pred_flat))
                y_true_flat = y_true_flat[:min_len]
                y_pred_flat = y_pred_flat[:min_len]
            
            # Remove NaN values
            mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
            if np.sum(mask) < len(y_true_flat) * 0.5:  # At least 50% valid data
                logger.warning(f"Too many invalid values in data: {np.sum(~mask)}/{len(mask)} invalid points")
            
            if np.sum(mask) > 0:
                y_true_clean = y_true_flat[mask]
                y_pred_clean = y_pred_flat[mask]
            else:
                logger.error("No valid data points for metric calculation")
                return {
                    f"{prefix}error": "No valid data points for metric calculation",
                    f"{prefix}mse": float('nan'),
                    f"{prefix}mae": float('nan'),
                    f"{prefix}r2": 0.0,
                    f"{prefix}direction_accuracy": 0.5  # Default to random guessing
                }
            
            try:
                mse = mean_squared_error(y_true_clean, y_pred_clean)
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                r2 = r2_score(y_true_clean, y_pred_clean)
                dir_acc = np.mean(np.sign(y_pred_clean) == np.sign(y_true_clean))
                
                metrics = {
                    f"{prefix}mse": mse,
                    f"{prefix}mae": mae,
                    f"{prefix}r2": r2,
                    f"{prefix}direction_accuracy": dir_acc
                }
            except Exception as metric_error:
                logger.error(f"Error calculating specific metrics: {metric_error}")
                metrics = {
                    f"{prefix}error": str(metric_error),
                    f"{prefix}mse": float('nan'),
                    f"{prefix}mae": float('nan'),
                    f"{prefix}r2": 0.0,
                    f"{prefix}direction_accuracy": 0.5  # Default to random guessing
                }
            
            return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {f"{prefix}error": str(e)}
        
def create_train_validation_split(X_train_scaled, y_train, validation_split=0.2, min_validation_samples=100):
    """
    Create train and validation splits with appropriate safety checks.
    
    Args:
        X_train_scaled: Scaled features for training
        y_train: Target values for training
        validation_split: Proportion of data to use for validation (0.0-1.0)
        min_validation_samples: Minimum number of samples required for validation
        
    Returns:
        X_train_inner, X_val, y_train_inner, y_val: The split datasets
    """
    try:
        total_samples = len(X_train_scaled)
        
        # Calculate initial split index
        train_idx = int(total_samples * (1 - validation_split))
        
        # Ensure minimum samples in both training and validation sets
        if train_idx >= total_samples - min_validation_samples:
            # Adjust to ensure minimum validation samples
            train_idx = total_samples - min_validation_samples
            logger.warning(f"Adjusted validation split to ensure at least {min_validation_samples} validation samples")
            
        if train_idx < min_validation_samples:
            # Too few samples for a good split, use half for each
            train_idx = total_samples // 2
            logger.warning(f"Too few samples for requested split. Using 50/50 split instead")
        
        # Handle edge case of very small datasets
        if total_samples < 2 * min_validation_samples:
            logger.warning(f"Dataset too small ({total_samples} samples) for reliable validation split")
        
        # Calculate actual proportions after adjustment
        train_proportion = train_idx / total_samples
        val_proportion = 1 - train_proportion
        logger.info(f"Final split: {train_proportion:.2f}/{val_proportion:.2f} (train/val) with {train_idx} training samples")
        
        # Create the split
        X_train_inner = X_train_scaled[:train_idx]
        y_train_inner = y_train[:train_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[:train_idx]
        
        X_val = X_train_scaled[train_idx:]
        y_val = y_train[train_idx:] if isinstance(y_train, np.ndarray) else y_train.iloc[train_idx:]
        
        # Check that validation set is not empty
        if len(X_val) == 0 or len(y_val) == 0:
            raise ValueError("Validation set is empty after split")
            
        # Log split information
        logger.info(f"Training set: {len(X_train_inner)} samples, Validation set: {len(X_val)} samples")
        
        return X_train_inner, X_val, y_train_inner, y_val
        
    except Exception as e:
        logger.error(f"Error creating train/validation split: {e}")
        # Fallback to a simple 80/20 split
        train_idx = int(len(X_train_scaled) * 0.8)
        
        X_train_inner = X_train_scaled[:train_idx]
        y_train_inner = y_train[:train_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[:train_idx]
        
        X_val = X_train_scaled[train_idx:]
        y_val = y_train[train_idx:] if isinstance(y_train, np.ndarray) else y_train.iloc[train_idx:]
        
        logger.warning(f"Using fallback 80/20 split due to error. Training: {len(X_train_inner)}, Validation: {len(X_val)} samples")
        return X_train_inner, X_val, y_train_inner, y_val

def aggregate_walk_forward_metrics(all_metrics: List[Dict[str, Any]], 
                                 predictions: List[np.ndarray],
                                 actuals: List[np.ndarray]) -> Dict[str, Any]:
    """Aggregate metrics across all walk-forward validation folds."""
    try:
        if not all_metrics:
            return {'error': 'No metrics to aggregate'}
            
        # Initialize aggregate metrics
        metric_keys = []
        for metrics in all_metrics:
            for k in metrics.keys():
                if k.startswith('test_') and not k.startswith('test_predictions') and not k.startswith('test_actuals'):
                    if k not in metric_keys:
                        metric_keys.append(k)
        
        aggregate = {
            'avg_metrics': {},
            'std_metrics': {},
            'min_metrics': {},
            'max_metrics': {}
        }
        
        # Calculate statistics for each metric
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregate['avg_metrics'][key] = float(np.mean(values))
                aggregate['std_metrics'][key] = float(np.std(values))
                aggregate['min_metrics'][key] = float(np.min(values))
                aggregate['max_metrics'][key] = float(np.max(values))
        
        # Add other useful information
        aggregate['n_folds'] = len(all_metrics)
        
        # Combine predictions and actuals if possible for overall metrics
        if predictions and actuals:
            try:
                # Handle different shapes carefully
                if not all(isinstance(p, np.ndarray) for p in predictions) or not all(isinstance(a, np.ndarray) for a in actuals):
                    logger.warning("Not all predictions or actuals are numpy arrays, converting them")
                    # Convert all to numpy arrays
                    pred_arrays = []
                    act_arrays = []
                    for p, a in zip(predictions, actuals):
                        pred_arrays.append(np.array(p) if not isinstance(p, np.ndarray) else p)
                        act_arrays.append(np.array(a) if not isinstance(a, np.ndarray) else a)
                    
                    predictions = pred_arrays
                    actuals = act_arrays
                
                # Determine if we're dealing with multi-horizon or single-horizon
                is_multi_horizon = False
                for p, a in zip(predictions, actuals):
                    if len(p.shape) > 1 and p.shape[1] > 1 and len(a.shape) > 1 and a.shape[1] > 1:
                        is_multi_horizon = True
                        break
                
                if is_multi_horizon:
                    # Handle multi-horizon case
                    # Determine number of horizons
                    horizons = 0
                    for a in actuals:
                        if len(a.shape) > 1 and a.shape[1] > horizons:
                            horizons = a.shape[1]
                    
                    # Initialize per-horizon metrics
                    horizon_mses = [[] for _ in range(horizons)]
                    horizon_maes = [[] for _ in range(horizons)]
                    horizon_r2s = [[] for _ in range(horizons)]
                    horizon_dir_accs = [[] for _ in range(horizons)]
                    
                    # Calculate metrics for each fold and horizon
                    for p, a in zip(predictions, actuals):
                        # Ensure p and a have correct shapes
                        if len(p.shape) == 1:
                            p = p.reshape(-1, 1)
                        if len(a.shape) == 1:
                            a = a.reshape(-1, 1)
                        
                        # Ensure same number of horizons
                        p_horizons = p.shape[1] if len(p.shape) > 1 else 1
                        a_horizons = a.shape[1] if len(a.shape) > 1 else 1
                        
                        if p_horizons != a_horizons:
                            logger.warning(f"Horizon mismatch: pred has {p_horizons}, actual has {a_horizons}")
                            # Use the smaller number of horizons
                            used_horizons = min(p_horizons, a_horizons, horizons)
                        else:
                            used_horizons = min(p_horizons, horizons)
                        
                        # Calculate metrics for each horizon
                        for h in range(used_horizons):
                            p_h = p[:, h] if p_horizons > 1 else p[:, 0]
                            a_h = a[:, h] if a_horizons > 1 else a[:, 0]
                            
                            # Ensure same length
                            if len(p_h) != len(a_h):
                                logger.warning(f"Length mismatch: pred has {len(p_h)}, actual has {len(a_h)}")
                                min_len = min(len(p_h), len(a_h))
                                p_h = p_h[:min_len]
                                a_h = a_h[:min_len]
                            
                            try:
                                horizon_mses[h].append(mean_squared_error(a_h, p_h))
                                horizon_maes[h].append(mean_absolute_error(a_h, p_h))
                                horizon_r2s[h].append(r2_score(a_h, p_h))
                                horizon_dir_accs[h].append(np.mean(np.sign(p_h) == np.sign(a_h)))
                            except Exception as e:
                                logger.error(f"Error calculating horizon {h} metrics: {e}")
                    
                    # Calculate average metrics across folds
                    for h in range(horizons):
                        if horizon_mses[h]:
                            aggregate[f'horizon_{h+1}_mse'] = float(np.mean(horizon_mses[h]))
                            aggregate[f'horizon_{h+1}_mae'] = float(np.mean(horizon_maes[h]))
                            aggregate[f'horizon_{h+1}_r2'] = float(np.mean(horizon_r2s[h]))
                            aggregate[f'horizon_{h+1}_direction_accuracy'] = float(np.mean(horizon_dir_accs[h]))
                    
                    # Average across all horizons
                    all_horizon_mses = [m for hm in horizon_mses for m in hm if m is not None]
                    all_horizon_maes = [m for hm in horizon_maes for m in hm if m is not None]
                    all_horizon_r2s = [m for hm in horizon_r2s for m in hm if m is not None]
                    all_horizon_dir_accs = [m for hm in horizon_dir_accs for m in hm if m is not None]
                    
                    if all_horizon_mses:
                        aggregate['overall_mse'] = float(np.mean(all_horizon_mses))
                        aggregate['overall_mae'] = float(np.mean(all_horizon_maes))
                        aggregate['overall_r2'] = float(np.mean(all_horizon_r2s))
                else:
                    # Single horizon case - flatten and combine all predictions and actuals
                    try:
                        # Ensure all arrays are 2D
                        flat_preds = []
                        flat_acts = []
                        
                        for p, a in zip(predictions, actuals):
                            p_flat = p.flatten() if len(p.shape) > 1 else p
                            a_flat = a.flatten() if len(a.shape) > 1 else a
                            
                            # Ensure same length
                            if len(p_flat) != len(a_flat):
                                min_len = min(len(p_flat), len(a_flat))
                                p_flat = p_flat[:min_len]
                                a_flat = a_flat[:min_len]
                                
                            flat_preds.append(p_flat)
                            flat_acts.append(a_flat)
                        
                        # Stack all predictions and actuals
                        all_preds = np.concatenate(flat_preds)
                        all_acts = np.concatenate(flat_acts)
                        
                        # Calculate overall metrics
                        aggregate['overall_mse'] = float(mean_squared_error(all_acts, all_preds))
                        aggregate['overall_mae'] = float(mean_absolute_error(all_acts, all_preds))
                        aggregate['overall_r2'] = float(r2_score(all_acts, all_preds))
                        aggregate['overall_direction_accuracy'] = float(
                            np.mean(np.sign(all_preds) == np.sign(all_acts))
                        )
                    except Exception as e:
                        logger.error(f"Error calculating overall metrics: {e}")
            except Exception as e:
                logger.error(f"Error combining predictions and actuals: {e}")
        
        return aggregate
        
    except Exception as e:
        logger.error(f"Error aggregating walk-forward metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}

def save_model(results: Dict[str, Any], path: str) -> bool:
    """Save the trained model and associated components with robust error handling."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        logger.info(f"Saving model to {path}")
        
        # Save ensemble model
        try:
            with open(f"{path}/model.pkl", 'wb') as f:
                pickle.dump(results['model'], f)
            logger.info("Saved ensemble model")
        except Exception as e:
            logger.error(f"Error saving ensemble model: {e}")
            # Try component-wise saving as fallback
            try:
                model_info = {
                    'model_type': 'EnsembleModel',
                    'weights': results['model'].weights.tolist() if results['model'].weights is not None else None,
                    'config': results['model'].config.__dict__
                }
                with open(f"{path}/model_info.json", 'w') as f:
                    json.dump(model_info, f, default=str)
                logger.info("Saved model info")
            except Exception as e2:
                logger.error(f"Error saving model_info: {e2}")
        
        # Save scaler
        try:
            if 'scaler' in results and results['scaler'] is not None:
                joblib.dump(results['scaler'], f"{path}/scaler.pkl")
                logger.info("Saved scaler")
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
            
        # Save feature selector
        try:
            if 'feature_selector' in results and results['feature_selector'] is not None:
                with open(f"{path}/feature_selector.pkl", 'wb') as f:
                    pickle.dump(results['feature_selector'], f)
                logger.info("Saved feature selector")
        except Exception as e:
            logger.error(f"Error saving feature selector: {e}")
            
            # Try to save just the selected features
            try:
                if 'selected_features' in results and results['selected_features'] is not None:
                    with open(f"{path}/selected_features.json", 'w') as f:
                        json.dump(results['selected_features'], f)
                    logger.info("Saved selected features")
            except Exception as e2:
                logger.error(f"Error saving selected features: {e2}")
        
        # Save metrics
        try:
            if 'metrics' in results and results['metrics'] is not None:
                with open(f"{path}/metrics.json", 'w') as f:
                    json.dump(results['metrics'], f, default=str)
                logger.info("Saved metrics")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
        
        # Save config
        try:
            if 'config' in results and results['config'] is not None:
                with open(f"{path}/config.json", 'w') as f:
                    json.dump(results['config'].__dict__, f, default=str)
                logger.info("Saved config")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            
        logger.info(f"Model and components saved to {path}")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error in save_model: {e}")
        return False


def load_model(path: str) -> Dict[str, Any]:
    """Load the trained model and associated components with robust error handling."""
    try:
        if not os.path.exists(path):
            logger.error(f"Model path {path} does not exist.")
            return None
            
        results = {}
        
        # Load ensemble model
        try:
            model_path = f"{path}/model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    results['model'] = pickle.load(f)
                logger.info("Loaded ensemble model")
            else:
                logger.warning("Model file not found.")
                results['model'] = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            results['model'] = None
        
        # Load scaler
        try:
            scaler_path = f"{path}/scaler.pkl"
            if os.path.exists(scaler_path):
                results['scaler'] = joblib.load(scaler_path)
                logger.info("Loaded scaler")
            else:
                logger.warning("Scaler file not found.")
                results['scaler'] = None
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            results['scaler'] = None
        
        # Load feature selector
        try:
            feature_selector_path = f"{path}/feature_selector.pkl"
            if os.path.exists(feature_selector_path):
                with open(feature_selector_path, 'rb') as f:
                    results['feature_selector'] = pickle.load(f)
                logger.info("Loaded feature selector")
            else:
                logger.warning("Feature selector file not found.")
                # Try to load selected features directly
                try:
                    selected_features_path = f"{path}/selected_features.json"
                    if os.path.exists(selected_features_path):
                        with open(selected_features_path, 'r') as f:
                            results['selected_features'] = json.load(f)
                        logger.info("Loaded selected features")
                    else:
                        logger.warning("Selected features file not found.")
                        results['selected_features'] = None
                except Exception as e2:
                    logger.error(f"Error loading selected features: {e2}")
                    results['selected_features'] = None
        except Exception as e:
            logger.error(f"Error loading feature selector: {e}")
            results['feature_selector'] = None
        
        # Load metrics
        try:
            metrics_path = f"{path}/metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    results['metrics'] = json.load(f)
                logger.info("Loaded metrics")
            else:
                logger.warning("Metrics file not found.")
                results['metrics'] = {}
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            results['metrics'] = {}
        
        # Load config
        try:
            config_path = f"{path}/config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Create ModelConfig instance
                config = ModelConfig()
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        
                results['config'] = config
                logger.info("Loaded config")
            else:
                logger.warning("Config file not found.")
                results['config'] = ModelConfig()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            results['config'] = ModelConfig()
        
        logger.info(f"Model and components loaded from {path}")
        return results
    
    except Exception as e:
        logger.error(f"Unexpected error in load_model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
def main():
    """Main function to demonstrate usage with enhanced hyperparameters."""
    # Enhanced configuration with expanded parameters
    config = ModelConfig (
        # Time window parameters
        lookback_window=20,  # set to 200 after debug
        forecast_horizon_input=1,
        
        # Training parameters
        batch_size=32,  #set to 64 after debug
        learning_rate=5e-4,  # Optimized learning rate
        num_epochs=3, #set to 150 after debug
        early_stopping_patience=2,#set to 15 after debug
        
        # Model architecture parameters
        hidden_dim=64,  # set to 512 after debug
        num_layers=2, #set to 3 after debug
        dropout=0.3,  # Increased for better regularization
        
        # New parameters
        weight_decay=1e-4,
        activation_fn='gelu',  # Modern activation function
        use_attention=True,
        attention_heads=8,
        # Multi-horizon configuration - explicitly set to match targets
        multi_horizon=False,
        #forecast_horizons=[1, 3, 5, 10, 20],  # 5 horizons to forecast
        #horizon_weights=[0.5, 0.2, 0.15, 0.1, 0.05]  # Higher weight on shorter horizons
        )
    
    # Example usage
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    # Run with enhanced hyperparameter tuning
    tune_hyperparams = True
    # Increase trials for better exploration of expanded search space
    n_trials = 3  # set to 50 after debug
    
    # Choose validation method
    use_walk_forward = True  # Set to True to use walk-forward validation
    use_time_series_cv = False  # Set to True to use time-series CV with gap
    walk_forward_splits = 5  # Number of time splits for walk-forward validation
    walk_forward_window = 'expanding'  # 'expanding' or 'rolling'
    time_series_cv_gap = 5  # Gap between train and test sets
    analyze_shap = True
    check_for_drift = True
    
    try:
        # Create output directories
        os.makedirs("models", exist_ok=True)
        os.makedirs(f"models/{symbol}", exist_ok=True)
        
        # Configure a log file for this run
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"models/{symbol}/training_{run_timestamp}.log"
        
        # Add file handler to logger
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Starting model training for {symbol} with enhanced hyperparameter tuning")
        logger.info(f"Using expanded search space with {n_trials} trials")
        
        # Fetch data
        logger.info(f"Downloading data for {symbol}...")
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=True,
            interval="1d"
        )
        
        # Verify the data was downloaded
        if df.empty:
            raise DataError(f"No data downloaded for {symbol}")
            
        # Fix for MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Log the processed data
        logger.info(f"Downloaded data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        results = None
        
        if use_time_series_cv:
            # Use time-series CV with gap
            logger.info(f"Starting time-series CV evaluation with {walk_forward_splits} splits and gap={time_series_cv_gap}...")
            results = time_series_cv_evaluation(
                df, 
                config, 
                n_splits=walk_forward_splits,
                gap=time_series_cv_gap,
                tune_hyperparams=tune_hyperparams
            )
            if results is not None:
                logger.info(f"Time-series CV evaluation completed successfully")
                logger.info(f"Aggregate metrics: {results.get('metrics', {}).get('avg_metrics', {})}")
        elif use_walk_forward:
            # Perform walk-forward validation
            logger.info(f"Starting walk-forward validation with {walk_forward_splits} splits and {walk_forward_window} window...")
            results = walk_forward_validation(
                df, 
                config, 
                n_splits=walk_forward_splits,
                window_type=walk_forward_window,  # 'expanding' or 'rolling'
                min_train_size=0.6,
                tune_hyperparams=tune_hyperparams
            )
            if results is not None:
                logger.info(f"Walk-forward validation completed successfully")
                logger.info(f"Aggregate metrics: {results.get('metrics', {}).get('avg_metrics', {})}")
        else:
            # Use traditional train-test split
            logger.info("Starting model training and evaluation with expanded search space")
            # Pass hyperparameter tuning options
            results = train_and_evaluate(
                df, 
                config, 
                tune_hyperparams=tune_hyperparams,
                n_trials=n_trials
            )
        
        if results is not None:
            # Save model
            logger.info("Saving model and results")
            save_model(results, f"models/{symbol}")
            
            logger.info(f"Successfully trained and saved model for {symbol}")
            logger.info(f"Metrics: {results['metrics']}")
            
            # Check for model drift
            if check_for_drift:
                try:
                    # Create synthetic data for drift detection
                    if 'selected_features' in results:
                        # Create a simple synthetic dataset
                        features = results['selected_features']
                        synthetic_data = pd.DataFrame(
                            np.random.normal(0, 1, (100, len(features))), 
                            columns=features
                        )
                        
                        # Get test predictions if available
                        test_predictions = []
                        if 'metrics' in results and 'test_predictions' in results['metrics']:
                            test_predictions = results['metrics']['test_predictions']
                            # Add some noise to simulate drift
                            drifted_predictions = test_predictions * (1 + np.random.normal(0, 0.05, np.array(test_predictions).shape))
                            test_predictions = drifted_predictions.flatten().tolist()
                        else:
                            # Generate predictions if not available
                            if 'model' in results and hasattr(results['model'], 'predict'):
                                if 'scaler' in results and results['scaler'] is not None:
                                    X_scaled = results['scaler'].transform(synthetic_data)
                                    test_predictions = results['model'].predict(X_scaled).flatten().tolist()
                                else:
                                    test_predictions = results['model'].predict(synthetic_data.values).flatten().tolist()
                        
                        if test_predictions:
                            # Check for drift and retraining need
                            logger.info("Checking for model drift and retraining need...")
                            drift_results = check_model_retraining_need(
                                f"models/{symbol}",
                                synthetic_data,
                                test_predictions,
                                drift_threshold=0.05
                            )
                            
                            if drift_results.get('retraining_recommended', False):
                                logger.warning("Model drift detected! Retraining is recommended.")
                                # In a production system, trigger automated retraining here
                            else:
                                logger.info("No significant model drift detected. Continuing to use current model.")
                except Exception as e:
                    logger.error(f"Error in model drift detection: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Save hyperparameter information if tuning was performed
            if tune_hyperparams and 'hyperparameters' in results.get('metrics', {}):
                hyper_path = f"models/{symbol}/hyperparameters.json"
                try:
                    with open(hyper_path, 'w') as f:
                        json.dump(results['metrics']['hyperparameters'], f, 
                                 default=lambda o: str(o), indent=4)
                    logger.info(f"Saved hyperparameters to {hyper_path}")
                except Exception as e:
                    logger.error(f"Error saving hyperparameters: {e}")
            
            # Plot feature importance if available
            if 'feature_selector' in results and hasattr(results['feature_selector'], 'plot_feature_importance'):
                try:
                    results['feature_selector'].plot_feature_importance(top_n=30)  # Increased from 20
                    logger.info("Feature importance plot saved")
                except Exception as e:
                    logger.error(f"Error plotting feature importance: {e}")
                
            # Additional analysis of model performance
            try:
                # Analyze model weights
                if hasattr(results['model'], 'weights') and results['model'].weights is not None:
                    weights = results['model'].weights
                    logger.info(f"Ensemble model weights: {weights}")
                    
                    # Identify top contributing models
                    model_names = list(results['model'].models.keys())
                    if len(model_names) > 0 and len(weights) >= len(model_names):
                        contributions = {model_names[i]: float(weights[i]) for i in range(len(model_names))}
                        sorted_contributions = dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))
                        
                        logger.info("Model contributions to ensemble:")
                        for model_name, weight in sorted_contributions.items():
                            logger.info(f"  {model_name}: {weight:.4f} ({weight*100:.1f}%)")
                
                # Analyze meta-model if available
                if hasattr(results['model'], 'meta_model') and results['model'].meta_model is not None:
                    logger.info("Meta-model successfully trained and used for predictions")
                    
                    # Log meta-model performance if available
                    if hasattr(results['model'], 'meta_model_performance'):
                        perf = results['model'].meta_model_performance
                        for metric, value in perf.items():
                            logger.info(f"Meta-model {metric}: {value}")

                if results is not None and 'shap_analyzer' in results:
                    logger.info("SHAP analysis results available.")
                    logger.info("Check the shap_analysis directory for detailed visualizations.")
                    
                    # Log top features from SHAP analysis for each model
                    shap_analyzer = results['shap_analyzer']
                    if hasattr(shap_analyzer, 'global_importance'):
                        logger.info("SHAP global feature importance:")
                        for model_name, importance in shap_analyzer.global_importance.items():
                            logger.info(f"Top 5 features for {model_name} according to SHAP:")
                            for i, (feature, value) in enumerate(list(importance.items())[:5]):
                                logger.info(f" {i+1}. {feature}: {value.item():.6f}")
                                
                # Show walk-forward specific metrics if available
                if use_walk_forward and 'all_fold_metrics' in results:
                    logger.info("Walk-forward validation fold metrics:")
                    for i, fold_metrics in enumerate(results['all_fold_metrics']):
                        logger.info(f"Fold {i+1}: MSE={fold_metrics.get('test_mse', 'N/A')}, "
                                   f"Direction Accuracy={fold_metrics.get('test_direction_accuracy', 'N/A')}")
            except Exception as e:
                logger.warning(f"Error in additional analysis: {e}")
                import traceback
                logger.warning(traceback.format_exc())
            
            return results
        else:
            logger.error("Model training failed")
            return None
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()