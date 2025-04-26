import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
from optuna.visualization import plot_param_importances, plot_optimization_history
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Callable, Tuple, Union, Optional
import joblib
from sklearn.model_selection import train_test_split
import os
import datetime
from functools import partial
import warnings
import logging
import random
from scipy.stats import spearmanr
import empyrical
import copy
from collections import defaultdict
import shap
from concurrent.futures import ProcessPoolExecutor
import time
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dataclasses import dataclass, field
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#########################################
# 1. Custom Time Series Cross-Validation
#########################################

class PurgedTimeSeriesCV:
    """
    Purged and Embargo Time Series Cross-Validation
    
    Implements Combinatorial Purged Cross-Validation (CPCV) with 
    embargo periods to prevent any form of information leakage.
    
    Parameters:
    -----------
    n_splits: int
        Number of splits for cross-validation
    embargo_pct: float
        Percentage of validation window to embargo after the train set
    purge_pct: float
        Percentage of validation window to purge before the validation set
    regime_aware: bool
        Whether to ensure that CV splits account for different market regimes
    regime_column: str, optional
        Column name that indicates the regime each sample belongs to
    """
    
    def __init__(
        self, 
        n_splits=5, 
        embargo_pct=0.01, 
        purge_pct=0.01,
        regime_aware=True,
        regime_column=None
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        self.regime_aware = regime_aware
        self.regime_column = regime_column
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and validation set
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Training data
        y : pandas Series or numpy array, optional
            Target variable
        groups : pandas Series or numpy array, optional
            Groups for the samples (typically timestamps)
            
        Yields:
        -------
        train_index, test_index : tuple of arrays
            Indices for the training and validation sets
        """
        if groups is None:
            if isinstance(X, pd.DataFrame) and X.index.is_monotonic_increasing:
                groups = X.index
            else:
                groups = np.arange(len(X))
        
        if isinstance(groups, pd.DatetimeIndex) or isinstance(groups, pd.Series):
            groups = groups.values
            
        # Convert to numpy array for consistent handling
        groups = np.array(groups)
            
        # Sort unique timestamps
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        # Time-based splitting (CPCV)
        k_fold_size = n_groups // self.n_splits
        
        # For each fold as validation set
        for i in range(self.n_splits):
            # Determine validation set indices
            val_start_idx = i * k_fold_size
            val_end_idx = (i + 1) * k_fold_size if i < self.n_splits - 1 else n_groups
            
            # Get timestamps for validation set
            val_groups = unique_groups[val_start_idx:val_end_idx]
            
            # Find sample indices for validation set using searchsorted
            # This is faster than np.where(np.isin())
            val_start_time = val_groups[0]
            val_end_time = val_groups[-1]
            
            # Find indices where groups fall in the validation period
            left_idx = np.searchsorted(groups, val_start_time, side='left')
            right_idx = np.searchsorted(groups, val_end_time, side='right')
            test_indices = np.arange(left_idx, right_idx)
            
            # Determine purging and embargo periods
            time_delta = val_end_time - val_start_time
            if isinstance(time_delta, np.timedelta64):
                # For datetime types
                purge_delta = np.timedelta64(int(time_delta * self.purge_pct), 'ns')
                embargo_delta = np.timedelta64(int(time_delta * self.embargo_pct), 'ns')
            else:
                # For numeric types
                purge_delta = time_delta * self.purge_pct
                embargo_delta = time_delta * self.embargo_pct
            
            purge_start = val_start_time - purge_delta
            embargo_end = val_end_time + embargo_delta
            
            # Apply purging and embargo with searchsorted
            purge_idx = np.searchsorted(groups, purge_start, side='left')
            embargo_idx = np.searchsorted(groups, embargo_end, side='right')
            
            # Create train indices as combination of pre-purge and post-embargo periods
            pre_indices = np.arange(0, purge_idx)
            post_indices = np.arange(embargo_idx, len(groups))
            train_indices = np.concatenate([pre_indices, post_indices])
            
            yield train_indices, test_indices
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations"""
        return self.n_splits


class WalkForwardTimeSeriesCV:
    """
    Walk-Forward Time Series Cross-Validation
    
    Implements a walk-forward validation approach where each validation fold
    comes strictly after the training data, simulating real deployment.
    
    Parameters:
    -----------
    n_splits: int
        Number of splits for cross-validation
    first_train_pct: float
        Percentage of data to include in the first training split
    step_pct: float
        Percentage step between consecutive training end points
    embargo_pct: float
        Percentage of validation window to embargo after the train set
    """
    
    def __init__(
        self, 
        n_splits=5, 
        first_train_pct=0.5,
        step_pct=0.1,
        embargo_pct=0.01
    ):
        self.n_splits = n_splits
        self.first_train_pct = first_train_pct
        self.step_pct = step_pct
        self.embargo_pct = embargo_pct
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and validation set
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Training data
        y : pandas Series or numpy array, optional
            Target variable
        groups : pandas Series or numpy array, optional
            Groups for the samples (typically timestamps)
            
        Yields:
        -------
        train_index, test_index : tuple of arrays
            Indices for the training and validation sets
        """
        if groups is None:
            if isinstance(X, pd.DataFrame) and X.index.is_monotonic_increasing:
                groups = X.index
            else:
                groups = np.arange(len(X))
        
        if isinstance(groups, pd.DatetimeIndex) or isinstance(groups, pd.Series):
            groups = groups.values
            
        # Convert to numpy array for consistent handling
        groups = np.array(groups)
            
        # Sort unique timestamps
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        # Calculate split points
        first_train_end = int(n_groups * self.first_train_pct)
        step_size = int(n_groups * self.step_pct)
        
        for i in range(self.n_splits):
            if i == 0:
                train_end = first_train_end
            else:
                train_end = first_train_end + i * step_size
                
            if train_end >= n_groups - 1:
                # Not enough data for another split
                break
                
            # Validation set starts after train set plus embargo
            embargo_size = int(step_size * self.embargo_pct)
            val_start = train_end + embargo_size
            val_end = min(val_start + step_size, n_groups)
            
            if val_start >= val_end:
                # Not enough data for validation
                break
                
            # Get timestamps for each set
            train_groups = unique_groups[:train_end]
            val_groups = unique_groups[val_start:val_end]
            
            # Find sample indices
            train_indices = np.where(np.isin(groups, train_groups))[0]
            test_indices = np.where(np.isin(groups, val_groups))[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the maximum number of splitting iterations"""
        return self.n_splits

#########################################
# 2. Financial Metrics
#########################################

class FinancialMetrics:
    """
    Collection of financial and trading-specific metrics for model evaluation.
    
    Provides metrics that focus on risk-adjusted returns, drawdowns,
    tail risks, and trading costs.
    """
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252):
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
        risk_free_rate : float
            Risk-free rate
        annualization_factor : int
            Factor to annualize returns (252 for daily, 12 for monthly, etc.)
            
        Returns:
        --------
        float
            Sharpe ratio
        """
        return empyrical.sharpe_ratio(
            returns, 
            risk_free=risk_free_rate,
            annualization=annualization_factor
        )
    
    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.0, annualization_factor=252):
        """
        Calculate Sortino ratio (downside risk-adjusted return).
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
        risk_free_rate : float
            Risk-free rate
        annualization_factor : int
            Factor to annualize returns (252 for daily, 12 for monthly, etc.)
            
        Returns:
        --------
        float
            Sortino ratio
        """
        return empyrical.sortino_ratio(
            returns, 
            risk_free=risk_free_rate,
            annualization=annualization_factor
        )
    
    @staticmethod
    def max_drawdown(returns):
        """
        Calculate maximum drawdown.
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
            
        Returns:
        --------
        float
            Maximum drawdown as a positive percentage
        """
        return empyrical.max_drawdown(returns)
    
    @staticmethod
    def calmar_ratio(returns, annualization_factor=252):
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
        annualization_factor : int
            Factor to annualize returns (252 for daily, 12 for monthly, etc.)
            
        Returns:
        --------
        float
            Calmar ratio
        """
        return empyrical.calmar_ratio(returns, annualization=annualization_factor)
    
    @staticmethod
    def conditional_value_at_risk(returns, percentile=5):
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
        percentile : int
            Percentile for CVaR calculation (typically 1 or 5)
            
        Returns:
        --------
        float
            CVaR value (expected loss in worst percentile cases)
        """
        # Sort returns in ascending order (worst to best)
        sorted_returns = np.sort(returns)
        
        # Calculate the cutoff index for the given percentile
        cutoff = int(np.ceil(len(returns) * (percentile / 100.0)))
        
        # Average the worst 'cutoff' returns
        if cutoff > 0:
            cvar = np.mean(sorted_returns[:cutoff])
        else:
            cvar = np.min(sorted_returns)
            
        return cvar
    
    @staticmethod
    def annualized_volatility(returns, annualization_factor=252):
        """
        Calculate annualized volatility.
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
        annualization_factor : int
            Factor to annualize returns (252 for daily, 12 for monthly, etc.)
            
        Returns:
        --------
        float
            Annualized volatility
        """
        return empyrical.annual_volatility(returns, annualization=annualization_factor)
    
    @staticmethod
    def information_ratio(returns, benchmark_returns, annualization_factor=252):
        """
        Calculate Information Ratio (excess return / tracking error).
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
        benchmark_returns : array-like
            Array of benchmark returns
        annualization_factor : int
            Factor to annualize returns (252 for daily, 12 for monthly, etc.)
            
        Returns:
        --------
        float
            Information Ratio
        """
        return empyrical.excess_sharpe(returns, benchmark_returns, annualization=annualization_factor)
    
    @staticmethod
    def gain_to_pain_ratio(returns):
        """
        Calculate Gain to Pain ratio (sum of returns / sum of absolute losses).
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
            
        Returns:
        --------
        float
            Gain to Pain ratio
        """
        total_returns = np.sum(returns)
        total_pain = np.sum(np.abs(np.minimum(returns, 0)))
        
        if total_pain == 0:
            return np.inf if total_returns > 0 else 0
        
        return total_returns / total_pain
    
    @staticmethod
    def cost_adjusted_sharpe(returns, predicted, transaction_cost_bps=3, 
                             risk_free_rate=0.0, annualization_factor=252):
        """
        Calculate cost-adjusted Sharpe ratio, accounting for transaction costs.
        
        Parameters:
        -----------
        returns : array-like
            Array of actual returns
        predicted : array-like
            Array of model predictions (signals)
        transaction_cost_bps : float
            Transaction costs in basis points (1 bps = 0.01%)
        risk_free_rate : float
            Risk-free rate
        annualization_factor : int
            Factor to annualize returns (252 for daily, 12 for monthly, etc.)
            
        Returns:
        --------
        float
            Cost-adjusted Sharpe ratio
        """
        # Convert basis points to decimal
        cost_decimal = transaction_cost_bps / 10000
        
        # Calculate position changes (signals + or - from previous signal)
        signal_changes = np.abs(np.diff(np.concatenate([[0], predicted])))
        
        # Calculate costs as signal changes * transaction cost
        costs = signal_changes * cost_decimal
        
        # Adjust returns by subtracting costs
        adjusted_returns = returns - costs
        
        # Calculate Sharpe on adjusted returns
        return empyrical.sharpe_ratio(
            adjusted_returns, 
            risk_free=risk_free_rate,
            annualization=annualization_factor
        )
    
    @staticmethod
    def parameter_stability_score(param_sets):
        """
        Calculate stability score across multiple parameter sets.
        
        Higher stability means parameters don't vary much across CV folds.
        
        Parameters:
        -----------
        param_sets : list of dict
            List of parameter dictionaries from different folds or regimes
            
        Returns:
        --------
        float
            Stability score between 0 (unstable) and 1 (stable)
        """
        if len(param_sets) <= 1:
            return 1.0  # Only one set, so perfectly stable
            
        # Extract common numerical parameters
        common_keys = set.intersection(*[set(params.keys()) for params in param_sets])
        numerical_keys = [k for k in common_keys if isinstance(param_sets[0][k], (int, float))]
        
        if not numerical_keys:
            return 0.0  # No numerical keys to compare
            
        # Calculate coefficient of variation for each parameter
        cv_scores = []
        for key in numerical_keys:
            values = [params[key] for params in param_sets]
            mean_val = np.mean(values)
            if mean_val == 0:
                continue  # Skip parameters with zero mean to avoid division by zero
                
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val != 0 else 0
            cv_scores.append(cv)
            
        if not cv_scores:
            return 0.0  # No valid scores
            
        # Convert average CV to stability score (1 - avg_cv, bounded between 0 and 1)
        avg_cv = np.mean(cv_scores)
        stability = max(0, min(1, 1 - avg_cv))
        
        return stability
    
    @staticmethod
    def regime_consistency_score(returns_by_regime):
        """
        Calculate consistency of returns across different market regimes.
        
        Parameters:
        -----------
        returns_by_regime : dict
            Dictionary mapping regime names to arrays of returns
            
        Returns:
        --------
        float
            Consistency score between 0 (inconsistent) and 1 (consistent)
        """
        if len(returns_by_regime) <= 1:
            return 1.0  # Only one regime, so perfectly consistent
            
        # Calculate Sharpe ratio for each regime
        regime_sharpes = {
            regime: empyrical.sharpe_ratio(returns) if len(returns) > 1 else 0
            for regime, returns in returns_by_regime.items()
        }
        
        # Calculate coefficient of variation of Sharpe ratios
        sharpe_values = list(regime_sharpes.values())
        mean_sharpe = np.mean(sharpe_values)
        
        if mean_sharpe == 0:
            return 0.0  # Zero mean Sharpe, return zero consistency
            
        std_sharpe = np.std(sharpe_values)
        cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else 0
        
        # Convert CV to consistency score (1 - cv, bounded between 0 and 1)
        consistency = max(0, min(1, 1 - cv))
        
        return consistency
    
    @staticmethod
    def fit_clustering_regimes(returns, n_regimes=3, lookback=20):
        """
        Cluster market returns into distinct regimes.
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
        n_regimes : int
            Number of regimes to identify
        lookback : int
            Lookback window for feature creation
            
        Returns:
        --------
        numpy.ndarray
            Array of regime labels
        """
        if len(returns) < lookback * 2:
            raise ValueError(f"Returns array too short for lookback of {lookback}")
            
        # Create features for clustering
        features = pd.DataFrame()
        
        # Rolling volatility
        features['volatility'] = pd.Series(returns).rolling(lookback).std().fillna(method='bfill')
        
        # Rolling cumulative return
        features['cum_return'] = pd.Series(returns).rolling(lookback).sum().fillna(method='bfill')
        
        # Rolling skewness
        features['skew'] = pd.Series(returns).rolling(lookback).skew().fillna(method='bfill')
        
        # Rolling Sharpe (simplified)
        features['sharpe'] = features['cum_return'] / features['volatility']
        features['sharpe'] = features['sharpe'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Apply PCA if we have many features
        if features.shape[1] > 2:
            pca = PCA(n_components=2)
            feature_matrix = pca.fit_transform(features)
        else:
            feature_matrix = features.values
            
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(feature_matrix)
        
        return regimes

    @staticmethod
    def split_returns_by_regime(returns, regimes):
        """
        Split returns into dictionaries by regime.
        
        Parameters:
        -----------
        returns : array-like
            Array of returns
        regimes : array-like
            Array of regime labels
            
        Returns:
        --------
        dict
            Dictionary mapping regime indices to arrays of returns
        """
        unique_regimes = np.unique(regimes)
        returns_by_regime = {
            f"regime_{regime}": returns[regimes == regime]
            for regime in unique_regimes
        }
        
        return returns_by_regime

#########################################
# 3. Advanced Optuna Optimization
#########################################

class OptunaFinanceTuner:
    """
    Advanced hyperparameter optimization for financial ML models.
    
    Implements state-of-the-art Bayesian optimization with domain-specific
    enhancements for financial time series data.
    
    Parameters:
    -----------
    model_type : str
        Type of model to optimize ('lgbm', 'xgboost', 'catboost', etc.)
    objective : str or callable
        Optimization objective ('sharpe', 'sortino', 'custom', etc.)
    cv : object
        Cross-validation strategy (PurgedTimeSeriesCV or WalkForwardTimeSeriesCV)
    study_name : str, optional
        Name for the Optuna study
    n_trials : int
        Number of optimization trials
    timeout : int, optional
        Timeout in seconds
    n_jobs : int
        Number of parallel jobs
    use_warm_start : bool
        Whether to use warm-starting from previous tuning
    warm_start_path : str, optional
        Path to load previous tuning results
    use_gpu : bool
        Whether to use GPU acceleration
    """
    
    def __init__(
        self,
        model_type='lgbm',
        objective='sharpe',
        cv=None,
        study_name=None,
        n_trials=100,
        timeout=None,
        n_jobs=-1,
        use_warm_start=False,
        warm_start_path=None,
        use_gpu=False,
        storage=None,
        regime_aware=False,
        multi_objective=False,
        advanced_pruning=True,
        cost_aware=True,
        categorical_group_constraints=True,
        track_parameter_importance=True
    ):
        self.model_type = model_type.lower()
        self.objective_name = objective
        self.cv = cv if cv is not None else PurgedTimeSeriesCV()
        self.study_name = study_name or f"{model_type}_tuning_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.use_warm_start = use_warm_start
        self.warm_start_path = warm_start_path
        self.use_gpu = use_gpu
        self.storage = storage
        self.regime_aware = regime_aware
        self.multi_objective = multi_objective
        self.advanced_pruning = advanced_pruning
        self.cost_aware = cost_aware
        self.categorical_group_constraints = categorical_group_constraints
        self.track_parameter_importance = track_parameter_importance
        
        # Initialize financial metrics
        self.metrics = FinancialMetrics()
        
        # Store optimization history
        self.optimization_history = []
        self.param_importances = {}
        self.models_by_fold = {}
        self.best_params = None
        self.study = None
        self.shap_values_by_fold = {}
        self.shap_stability = None
        
        # Pre-defined parameter search spaces
        self._init_param_search_spaces()
        
        # Cached data for regime-specific optimization
        self.cached_regimes = None
        self.regime_models = {}
        self.transition_model = None
        
        # Store transfer learning state
        self.transfer_learning_params = {}
    
    def _init_param_search_spaces(self):
        """Initialize search spaces for different model types"""
        # LightGBM parameter search space
        self.lgbm_param_space = {
            'num_leaves': (15, 4095),
            'learning_rate': (0.001, 0.3, 'log'),
            'n_estimators': (50, 10000, 'log'),
            'max_depth': (3, 25),
            'min_data_in_leaf': (10, 200),
            'max_bin': (100, 1000),
            'bagging_fraction': (0.5, 1.0),
            'bagging_freq': (0, 10),
            'feature_fraction': (0.5, 1.0),
            'lambda_l1': (0, 5),
            'lambda_l2': (0, 5),
        }
        
        # XGBoost parameter search space
        self.xgb_param_space = {
            'max_depth': (3, 25),
            'learning_rate': (0.001, 0.3, 'log'),
            'n_estimators': (50, 10000, 'log'),
            'min_child_weight': (1, 20),
            'gamma': (0, 5),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'colsample_bylevel': (0.5, 1.0),
            'reg_alpha': (0, 5),
            'reg_lambda': (0, 5),
        }
        
        # CatBoost parameter search space
        self.catboost_param_space = {
            'iterations': (50, 10000, 'log'),
            'learning_rate': (0.001, 0.3, 'log'),
            'depth': (4, 16),
            'l2_leaf_reg': (1, 100, 'log'),
            'bagging_temperature': (0, 10),
            'random_strength': (0, 10),
            'border_count': (32, 255),
        }
        
        # Parameter constraints for model types
        self.lgbm_constraints = [
            {'constraint': 'num_leaves <= 2**max_depth', 'params': ['num_leaves', 'max_depth']},
            {'constraint': 'min_data_in_leaf >= 10', 'params': ['min_data_in_leaf']},
        ]
        
        self.xgb_constraints = [
            {'constraint': 'min_child_weight * subsample > 0.5', 'params': ['min_child_weight', 'subsample']},
        ]
        
        # Categorical group constraints
        # These ensure mutually exclusive parameter groups
        if self.categorical_group_constraints:
            self.lgbm_categorical_groups = [
                {
                    'name': 'regularization_strategy',
                    'options': [
                        {'name': 'l1_focused', 'params': {'lambda_l1': (0.5, 5), 'lambda_l2': (0, 0.5)}},
                        {'name': 'l2_focused', 'params': {'lambda_l1': (0, 0.5), 'lambda_l2': (0.5, 5)}},
                        {'name': 'balanced', 'params': {'lambda_l1': (0.1, 1.0), 'lambda_l2': (0.1, 1.0)}},
                                            ]
                                        },
                                        {
                                            'name': 'sampling_strategy',
                                            'options': [
                                                {'name': 'bagging_focused', 'params': {'bagging_fraction': (0.5, 0.9), 'feature_fraction': (0.8, 1.0)}},
                                                {'name': 'feature_focused', 'params': {'bagging_fraction': (0.8, 1.0), 'feature_fraction': (0.5, 0.9)}},
                                                {'name': 'balanced_sampling', 'params': {'bagging_fraction': (0.7, 0.9), 'feature_fraction': (0.7, 0.9)}},
                                            ]
                                        }
                                    ]
                                    
            self.xgb_categorical_groups = [
                {
                    'name': 'regularization_strategy',
                    'options': [
                        {'name': 'alpha_focused', 'params': {'reg_alpha': (0.5, 5), 'reg_lambda': (0, 0.5)}},
                        {'name': 'lambda_focused', 'params': {'reg_alpha': (0, 0.5), 'reg_lambda': (0.5, 5)}},
                        {'name': 'balanced', 'params': {'reg_alpha': (0.1, 1.0), 'reg_lambda': (0.1, 1.0)}},
                    ]
                },
                {
                    'name': 'sampling_strategy',
                    'options': [
                        {'name': 'row_focused', 'params': {'subsample': (0.5, 0.9), 'colsample_bytree': (0.8, 1.0)}},
                        {'name': 'col_focused', 'params': {'subsample': (0.8, 1.0), 'colsample_bytree': (0.5, 0.9)}},
                        {'name': 'balanced_sampling', 'params': {'subsample': (0.7, 0.9), 'colsample_bytree': (0.7, 0.9)}},
                    ]
                }
            ]
    
    def _get_param_space(self):
        """Get parameter search space based on model type"""
        if self.model_type == 'lgbm':
            return self.lgbm_param_space
        elif self.model_type == 'xgboost':
            return self.xgb_param_space
        elif self.model_type == 'catboost':
            return self.catboost_param_space
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _get_constraints(self):
        """Get parameter constraints based on model type"""
        if self.model_type == 'lgbm':
            return self.lgbm_constraints
        elif self.model_type == 'xgboost':
            return self.xgb_constraints
        else:
            return []
    
    def _get_categorical_groups(self):
        """Get categorical group constraints based on model type"""
        if not self.categorical_group_constraints:
            return []
            
        if self.model_type == 'lgbm':
            return self.lgbm_categorical_groups
        elif self.model_type == 'xgboost':
            return self.xgb_categorical_groups
        else:
            return []
            
    def _objective_function(self, trial, X, y, groups=None, sample_weight=None):
        """
        Objective function for Optuna optimization.
        
        Parameters:
        -----------
        trial : optuna.trial.Trial
            Optuna trial object
        X : pandas DataFrame or numpy array
            Features
        y : pandas Series or numpy array
            Target variable
        groups : pandas Series or numpy array, optional
            Groups for CV splitting (typically timestamps)
        sample_weight : pandas Series or numpy array, optional
            Sample weights
            
        Returns:
        --------
        float or list of float
            Objective value(s) to minimize/maximize
        """
        # Get parameter space for this model type
        param_space = self._get_param_space()
        constraints = self._get_constraints()
        categorical_groups = self._get_categorical_groups()
        
        # Suggest parameters from the search space
        params = {}
        
        # Handle categorical group constraints
        for group in categorical_groups:
            # Suggest which option to use from this group
            option_idx = trial.suggest_categorical(f"{group['name']}_option", 
                                                list(range(len(group['options']))))
            
            # Get the selected option
            selected_option = group['options'][option_idx]
            
            # Log which strategy was selected
            trial.set_user_attr(f"{group['name']}", selected_option['name'])
            
            # Add the parameters from this option
            for param_name, param_range in selected_option['params'].items():
                if len(param_range) == 2:
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif len(param_range) == 3 and param_range[2] == 'log':
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
                elif len(param_range) == 3:
                    params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
        
        # Suggest parameters not covered by categorical groups
        for param_name, param_range in param_space.items():
            if param_name not in params:  # Skip if already set by categorical groups
                if len(param_range) == 2:
                    if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif len(param_range) == 3 and param_range[2] == 'log':
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
        
        # Apply constraints to ensure valid parameter combinations
        for constraint in constraints:
            if 'constraint' in constraint:
                if 'num_leaves <= 2**max_depth' in constraint['constraint'] \
                        and 'num_leaves' in params and 'max_depth' in params:
                    max_leaves = 2 ** params['max_depth']          # upper bound allowed
                    params['num_leaves'] = min(params['num_leaves'], max_leaves) 
                elif 'min_data_in_leaf >= 10' in constraint['constraint'] and 'min_data_in_leaf' in params:
                    params['min_data_in_leaf'] = max(params['min_data_in_leaf'], 10)
                elif 'min_child_weight * subsample > 0.5' in constraint['constraint'] and 'min_child_weight' in params and 'subsample' in params:
                    if params['min_child_weight'] * params['subsample'] <= 0.5:
                        # Adjust subsample to satisfy constraint
                        params['subsample'] = max(0.5 / params['min_child_weight'], 0.5)
        
        # Add GPU acceleration if requested
        if self.use_gpu:
            if self.model_type == 'lgbm':
                params['device'] = 'gpu'
            elif self.model_type == 'xgboost':
                params['tree_method'] = 'gpu_hist'
            elif self.model_type == 'catboost':
                params['task_type'] = 'GPU'
        
        # Cross-validation
        cv_scores = []
        fold_models = {}
        fold_predictions = []
        fold_metrics = []
        fold_returns = []
        
        # Initialize pruning callbacks
        if self.advanced_pruning:
            if self.model_type == 'lgbm':
                pruning_callback = LightGBMPruningCallback(trial, 'Sharpe')
                params['callbacks'] = [pruning_callback]
            elif self.model_type == 'xgboost':
                pruning_callback = XGBoostPruningCallback(trial)
                params['callbacks'] = [pruning_callback]
        
        # Keep track of fold-specific data for robust evaluation
        all_test_indices = []
        all_predictions = []
        
        # Create cross-validation splits
        if isinstance(X, pd.DataFrame) and X.index.is_monotonic_increasing:
            cv_splits = list(self.cv.split(X, y, X.index))
        else:
            cv_splits = list(self.cv.split(X, y, groups))
            
        # Perform cross-validation
        for fold_idx, (train_indices, test_indices) in enumerate(cv_splits):
            # Get train/test data for this fold
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            
            # Apply sample weights if provided
            fold_sample_weight = None
            if sample_weight is not None:
                fold_sample_weight = sample_weight.iloc[train_indices]
            
            # Fit model
            model = self._fit_model(params, X_train, y_train, sample_weight=fold_sample_weight)
            
            # Make predictions
            preds = model.predict(X_test)
            
            # Track fold data
            fold_models[fold_idx] = model
            all_test_indices.extend(test_indices)
            all_predictions.extend(preds)
            
            # Calculate fold metrics
            fold_metrics.append(self._calculate_metrics(y_test, preds))
            
            # Calculate returns (for financial metrics)
            # Assuming y contains actual returns and predictions are signals
            if self.objective_name in ['sharpe', 'sortino', 'calmar', 'cvar', 'profit_factor']:
                test_returns = y_test.values
                # Simple strategy: sign of prediction * returns
                strategy_returns = np.sign(preds) * test_returns
                fold_returns.append(strategy_returns)
            
            # Calculate objective score for this fold
            fold_score = self._calculate_objective(fold_metrics[-1], fold_returns[-1] if fold_returns else None)
            cv_scores.append(fold_score)
            
            # Early stopping via pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Calculate aggregated metrics
        if self.multi_objective:
            # Multiple objectives for Pareto front optimization
            aggregated_metrics = {}
            
            # Average fold metrics
            for metric_name in fold_metrics[0].keys():
                aggregated_metrics[metric_name] = np.mean([metrics[metric_name] for metrics in fold_metrics])
            
            # Return multiple objectives
            if 'sharpe' in aggregated_metrics and 'max_drawdown' in aggregated_metrics:
                # Return both Sharpe ratio and negative max drawdown
                return [aggregated_metrics['sharpe'], -aggregated_metrics['max_drawdown']]
            else:
                # Default to Sharpe ratio if multi-objective but metrics missing
                return np.mean(cv_scores)
        else:
            # Single objective
            return np.mean(cv_scores)
    
    def _fit_model(self, params, X_train, y_train, sample_weight=None):
        """
        Fit model based on model type and parameters.
        
        Parameters:
        -----------
        params : dict
            Model parameters
        X_train : pandas DataFrame or numpy array
            Training features
        y_train : pandas Series or numpy array
            Training target
        sample_weight : pandas Series or numpy array, optional
            Sample weights
            
        Returns:
        --------
        object
            Fitted model
        """
        # Clone parameters to avoid modifying the original
        model_params = params.copy()
        
        # Remove non-model parameters
        if 'callbacks' in model_params:
            callbacks = model_params.pop('callbacks')
        else:
            callbacks = None
            
        # Set n_jobs/threads to 1 if using parallelism at the Optuna level
        if self.n_jobs > 1:
            if self.model_type == 'lgbm':
                model_params['n_jobs'] = 1
            elif self.model_type == 'xgboost':
                model_params['nthread'] = 1
            elif self.model_type == 'catboost':
                model_params['thread_count'] = 1
                
        if self.model_type == 'lgbm':
            # LightGBM model - use native API for proper pruning support
            if callbacks and self.advanced_pruning:
                # Convert to native LightGBM API for proper pruning
                train_data = lgb.Dataset(
                    X_train, 
                    label=y_train,
                    weight=sample_weight
                )
                
                # Create validation data for pruning
                # Use a small validation set from train data
                from sklearn.model_selection import train_test_split
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                
                val_data = lgb.Dataset(
                    X_val,
                    label=y_val
                )
                
                # Convert scikit-learn parameters to native parameters
                native_params = model_params.copy()
                if 'n_estimators' in native_params:
                    native_params['num_iterations'] = native_params.pop('n_estimators')
                if 'random_state' in native_params:
                    native_params['seed'] = native_params.pop('random_state')
                    
                def sharpe_metric(preds, dtrain):
                    labels = dtrain.get_label()
                    # Simple strategy: sign of prediction * returns
                    strategy_returns = np.sign(preds) * labels
                    
                    # Calculate Sharpe ratio
                    sharpe = self.metrics.sharpe_ratio(strategy_returns)
                    
                    # Return name, value, is_higher_better
                    return 'Sharpe', sharpe, True

                # Then update the train call
                gbm = lgb.train(
                    native_params,
                    train_data,
                    valid_sets=[val_data],
                    feval=sharpe_metric if self.objective_name == 'sharpe' else None,
                    callbacks=callbacks
                )
                
                # Convert back to scikit-learn for API consistency
                from lightgbm.sklearn import LGBMRegressor
                model = LGBMRegressor()
                model._Booster = gbm
                model._n_features = X_train.shape[1]
            else:
                # Use scikit-learn API when pruning not needed
                model = lgb.LGBMRegressor(**model_params)
                
                fit_params = {}
                if sample_weight is not None:
                    fit_params['sample_weight'] = sample_weight
                    
                model.fit(X_train, y_train, **fit_params)
                
        elif self.model_type == 'xgboost':
            # XGBoost model - use native API for proper pruning
            if callbacks and self.advanced_pruning:
                # Convert to native XGBoost API for proper pruning
                dtrain = xgb.DMatrix(
                    X_train, 
                    label=y_train,
                    weight=sample_weight
                )
                
                # Create validation data for pruning
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                
                dval = xgb.DMatrix(X_val, label=y_val)
                
                # Convert scikit-learn parameters to native parameters
                native_params = model_params.copy()
                if 'n_estimators' in native_params:
                    native_params['num_boost_round'] = native_params.pop('n_estimators')
                if 'random_state' in native_params:
                    native_params['seed'] = native_params.pop('random_state')
                    
                def sharpe_metric(preds, dtrain):
                    labels = dtrain.get_label()
                    # Simple strategy: sign of prediction * returns
                    strategy_returns = np.sign(preds) * labels
                    
                    # Calculate Sharpe ratio
                    sharpe = self.metrics.sharpe_ratio(strategy_returns)
                    
                    # Return metric name and value
                    return [('Sharpe', sharpe)]

                # Update the XGBoost train call
                bst = xgb.train(
                    native_params,
                    dtrain,
                    evals=[(dval, 'val')],
                    feval=sharpe_metric if self.objective_name == 'sharpe' else None,
                    callbacks=callbacks
                )
                
                # Convert back to scikit-learn for API consistency
                model = xgb.XGBRegressor()
                model._Booster = bst
            else:
                # Use scikit-learn API when pruning not needed
                model = xgb.XGBRegressor(**model_params)
                
                fit_params = {}
                if sample_weight is not None:
                    fit_params['sample_weight'] = sample_weight
                    
                model.fit(X_train, y_train, **fit_params)
                
        elif self.model_type == 'catboost':
            # CatBoost model
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**model_params)
            
            fit_params = {}
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
                
            model.fit(X_train, y_train, **fit_params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return model
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate regression metrics.
        
        Parameters:
        -----------
        y_true : pandas Series or numpy array
            True target values
        y_pred : pandas Series or numpy array
            Predicted values
            
        Returns:
        --------
        dict
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # If y contains financial returns and predictions are signals
        if hasattr(y_true, 'values'):
            y_true_values = y_true.values
        else:
            y_true_values = y_true
            
        # Directional accuracy (for classification-like tasks)
        metrics['direction_accuracy'] = np.mean(np.sign(y_pred) == np.sign(y_true_values))
        
        # Calculate rank correlation
        metrics['rank_correlation'] = spearmanr(y_true_values, y_pred)[0]
        
        return metrics
    
    def _calculate_objective(self, metrics, returns=None):
        """
        Calculate optimization objective based on metrics.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of metric names and values
        returns : array-like, optional
            Strategy returns for financial metrics
            
        Returns:
        --------
        float
            Objective value to optimize
        """
        if self.objective_name == 'mse':
            return -metrics['mse']  # Negate for maximization
        elif self.objective_name == 'rmse':
            return -metrics['rmse']  # Negate for maximization
        elif self.objective_name == 'direction':
            return metrics['direction_accuracy']
        elif self.objective_name == 'rank':
            return metrics['rank_correlation']
        elif self.objective_name == 'sharpe' and returns is not None:
            # Calculate Sharpe ratio
            return self.metrics.sharpe_ratio(returns)
        elif self.objective_name == 'sortino' and returns is not None:
            # Calculate Sortino ratio
            return self.metrics.sortino_ratio(returns)
        elif self.objective_name == 'calmar' and returns is not None:
            # Calculate Calmar ratio
            return self.metrics.calmar_ratio(returns)
        elif self.objective_name == 'cvar' and returns is not None:
            # Negate CVaR (lower is better)
            return -self.metrics.conditional_value_at_risk(returns)
        elif self.objective_name == 'custom' and hasattr(self, 'custom_objective'):
            # Custom objective function
            return self.custom_objective(metrics, returns)
        else:
            # Default to minimizing MSE
            return -metrics['mse']
    
    def _setup_optuna_study(self):
        """Set up Optuna study with appropriate configuration"""
        # Choose sampler based on dimensionality and requirements
        if self.multi_objective:
            # Multi-objective sampler
            sampler = optuna.samplers.NSGAIISampler(
                population_size=50,
                seed=42
            )
            
            # Define directions (maximize or minimize) for each objective
            directions = []
            if 'sharpe' in self.objective_name or 'sortino' in self.objective_name:
                directions.append('maximize')  # Sharpe or Sortino
            if 'drawdown' in self.objective_name:
                directions.append('minimize')  # Drawdown
                
            if not directions:
                directions = ['maximize', 'minimize']  # Default
                
            # Create multi-objective study
            if self.storage:
                study = optuna.create_study(
                    study_name=self.study_name,
                    directions=directions,
                    sampler=sampler,
                    storage=self.storage,
                    load_if_exists=True
                )
            else:
                study = optuna.create_study(
                    study_name=self.study_name,
                    directions=directions,
                    sampler=sampler
                )
                
        else:
            # Single objective sampler
            if self.model_type in ['lgbm', 'xgboost', 'catboost']:
                # TPE sampler for tree-based models
                sampler = TPESampler(
                    n_startup_trials=10,
                    multivariate=True,
                    seed=42
                )
            else:
                # CMA-ES for continuous parameter spaces
                sampler = CmaEsSampler(
                    seed=42
                )
                
            # Create study with appropriate direction
            direction = 'maximize'
            if self.objective_name in ['mse', 'rmse', 'cvar']:
                direction = 'minimize'
                
            if self.storage:
                study = optuna.create_study(
                    study_name=self.study_name,
                    direction=direction,
                    sampler=sampler,
                    storage=self.storage,
                    load_if_exists=True
                )
            else:
                study = optuna.create_study(
                    study_name=self.study_name,
                    direction=direction,
                    sampler=sampler
                )
                
        # Apply warm starting if requested
        if self.use_warm_start and self.warm_start_path:
            if os.path.exists(self.warm_start_path):
                try:
                    # Load previous tuning results
                    with open(self.warm_start_path, 'rb') as f:
                        previous_study = pickle.load(f)
                        
                    # Add previous trials to new study (adjust for volatility if applicable)
                    for trial in previous_study.trials:
                        if trial.state == optuna.trial.TrialState.COMPLETE:
                            study.enqueue_trial(trial.params)
                            
                    logger.info(f"Warm-started with {len(previous_study.trials)} previous trials")
                except Exception as e:
                    logger.warning(f"Failed to warm-start: {e}")
                    
        return study
    
    def _apply_advanced_pruning(self, study):
        """
        Apply advanced pruning strategies to the study.
        
        Parameters:
        -----------
        study : optuna.study.Study
            Optuna study object
            
        Returns:
        --------
        optuna.study.Study
            Updated study with pruning configuration
        """
        if not self.advanced_pruning:
            return study
            
        # Add pruners
        pruner = None
        
        # Median pruner for stable metrics
        if self.objective_name in ['sharpe', 'sortino', 'calmar']:
            # For financial metrics, use median pruner
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=3
            )
        else:
            # For other metrics, use Hyperband pruner
            pruner = HyperbandPruner(
                min_resource=3,
                reduction_factor=3
            )
            
        # Apply pruner to study
        study.pruner = pruner
        
        return study
    
    def _setup_regime_specific_optimization(self, X, y, groups=None):
        """
        Set up regime-specific optimization.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Features
        y : pandas Series or numpy array
            Target variable
        groups : pandas Series or numpy array, optional
            Groups for CV splitting (typically timestamps)
            
        Returns:
        --------
        dict
            Dictionary of regime-specific data
        """
        if not self.regime_aware:
            return None
            
        # Determine regimes if not provided
        if self.cached_regimes is None:
            if 'regime' in X.columns:
                # Use provided regime column
                regimes = X['regime'].values
            else:
                # Detect regimes automatically
                if hasattr(y, 'values'):
                    returns = y.values
                else:
                    returns = y
                    
                regimes = self.metrics.fit_clustering_regimes(returns)
                
            self.cached_regimes = regimes
        else:
            regimes = self.cached_regimes
            
        # Split data by regime
        unique_regimes = np.unique(regimes)
        regime_data = {}
        
        for regime in unique_regimes:
            regime_mask = regimes == regime
            regime_data[f"regime_{regime}"] = {
                'X': X[regime_mask],
                'y': y[regime_mask],
                'indices': np.where(regime_mask)[0]
            }
            
            # Add timestamp info if available
            if groups is not None:
                if hasattr(groups, 'iloc'):
                    regime_data[f"regime_{regime}"]['groups'] = groups.iloc[regime_mask]
                else:
                    regime_data[f"regime_{regime}"]['groups'] = groups[regime_mask]
                    
        return regime_data
    
    def _optimize_for_regime(self, regime_name, regime_data):
        """
        Perform optimization for a specific market regime.
        
        Parameters:
        -----------
        regime_name : str
            Name of the regime
        regime_data : dict
            Regime-specific data
            
        Returns:
        --------
        dict
            Best parameters for this regime
        """
        logger.info(f"Optimizing for regime: {regime_name}")
        
        # Create regime-specific study name
        regime_study_name = f"{self.study_name}_{regime_name}"
        
        # Setup study for this regime
        regime_study = optuna.create_study(
            study_name=regime_study_name,
            direction='maximize' if self.objective_name not in ['mse', 'rmse', 'cvar'] else 'minimize',
            sampler=TPESampler(n_startup_trials=5, multivariate=True, seed=42)
        )
        
        # Number of trials proportional to regime size
        regime_size = len(regime_data['y'])
        total_size = len(self.cached_regimes)
        regime_trials = max(10, int(self.n_trials * (regime_size / total_size)))
        
        # Define objective function for this regime
        def regime_objective(trial):
            return self._objective_function(
                trial, 
                regime_data['X'], 
                regime_data['y'], 
                regime_data.get('groups', None)
            )
            
        # Optimize
        regime_study.optimize(
            regime_objective,
            n_trials=regime_trials,
            timeout=self.timeout,
            n_jobs=min(self.n_jobs, 4)  # Limit parallelism for regimes
        )
        
        # Store best parameters
        regime_best_params = regime_study.best_params
        
        return regime_best_params
    
    def _fit_regime_transition_model(self, X, regimes):
        """
        Fit a model to predict regime transitions.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Features
        regimes : array-like
            Regime labels
            
        Returns:
        --------
        object
            Fitted transition model
        """
        # Create shifted regime labels (next regime)
        next_regimes = np.roll(regimes, -1)
        
        # Create transition indicator (1 if regime changes, 0 otherwise)
        transitions = (regimes != next_regimes).astype(int)
        
        # Fit a model to predict transitions
        transition_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        
        transition_model.fit(X, transitions)
        
        return transition_model
    
    def _extract_meta_knowledge(self):
        """
        Extract meta-knowledge from optimization results.
        
        Returns:
        --------
        dict
            Meta-knowledge for transfer learning
        """
        if not self.study:
            return {}
            
        meta_knowledge = {}
        
        # Extract parameter importance
        if len(self.study.trials) > 5:
            importance = optuna.importance.get_param_importances(self.study)
            meta_knowledge['param_importance'] = importance
            
            # Save for later use
            self.param_importances = importance
            
        # Extract parameter correlations with objective
        param_correlations = {}
        
        # Get completed trials once
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Then for each parameter, collect values and calculate correlation
        for param_name in self.best_params.keys():
            # For each parameter, create matching pairs of parameter values and objective values
            param_values = []
            matching_objective_values = []
            
            # Collect matching values for this parameter
            for trial in completed_trials:
                if param_name in trial.params:
                    param_values.append(trial.params[param_name])
                    matching_objective_values.append(trial.value)
            
            # Calculate correlation for this parameter
            if len(param_values) > 5:  # Need enough samples
                try:
                    # Fix: Use proper parameter order without keyword args
                    correlation, _ = spearmanr(param_values, matching_objective_values)
                    param_correlations[param_name] = correlation
                except Exception:
                    pass
                    
        meta_knowledge['param_correlations'] = param_correlations
        
        # Extract best parameter ranges (percentiles around best value)
        top_trials = sorted(
            completed_trials,  # Reuse completed_trials we already collected
            key=lambda t: t.value,
            reverse=not (self.objective_name in ['mse', 'rmse', 'cvar'])
        )
        
        # Take top 10% of trials
        num_top = max(1, int(len(top_trials) * 0.1))
        top_trials = top_trials[:num_top]
        
        best_ranges = {}
        if self.best_params:
            for param_name in self.best_params.keys():
                param_values = [t.params.get(param_name, np.nan) for t in top_trials 
                            if param_name in t.params]
            
                if len(param_values) > 3:
                    best_ranges[param_name] = {
                        'min': np.min(param_values),
                        'max': np.max(param_values),
                        'median': np.median(param_values)
                    }
                    
        meta_knowledge['best_ranges'] = best_ranges
        
        return meta_knowledge
    
    def _calculate_shap_stability(self, X, shap_values_by_fold):
        """
        Calculate stability of SHAP values across folds.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Features
        shap_values_by_fold : dict
            Dictionary of SHAP values by fold
            
        Returns:
        --------
        dict
            Feature stability scores
        """
        if not shap_values_by_fold or len(shap_values_by_fold) <= 1:
            return {}
            
        feature_names = X.columns
        stability_scores = {}
        
        # Calculate average absolute SHAP value per feature per fold
        avg_importance = {}
        for fold, shap_values in shap_values_by_fold.items():
            fold_importance = {}
            for i, feature in enumerate(feature_names):
                fold_importance[feature] = np.mean(np.abs(shap_values[:, i]))
                
            avg_importance[fold] = fold_importance
            
        # Calculate coefficient of variation across folds for each feature
        for feature in feature_names:
            values = [fold_imp[feature] for fold_imp in avg_importance.values()]
            mean_val = np.mean(values)
            
            if mean_val > 0:
                std_val = np.std(values)
                cv = std_val / mean_val
                
                # Convert to stability score (1 - cv, bounded between 0 and 1)
                stability = max(0, min(1, 1 - cv))
            else:
                stability = 0.0
                
            stability_scores[feature] = stability
            
        return stability_scores
    
    def optimize(self, X, y, groups=None, sample_weight=None, custom_objective=None):
        """
        Perform hyperparameter optimization.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Features
        y : pandas Series or numpy array
            Target variable
        groups : pandas Series or numpy array, optional
            Groups for CV splitting (typically timestamps)
        sample_weight : pandas Series or numpy array, optional
            Sample weights
        custom_objective : callable, optional
            Custom objective function
            
        Returns:
        --------
        dict
            Best parameters
        """
        if custom_objective:
            self.custom_objective = custom_objective
            self.objective_name = 'custom'
            
        # Set up regime-specific optimization if requested
        regime_data = None
        if self.regime_aware:
            regime_data = self._setup_regime_specific_optimization(X, y, groups)
            
        # If regime-specific optimization is enabled
        if self.regime_aware and regime_data:
            # Optimize for each regime separately
            for regime_name, data in regime_data.items():
                self.regime_models[regime_name] = self._optimize_for_regime(regime_name, data)
                
            # Fit transition model
            self.transition_model = self._fit_regime_transition_model(X, self.cached_regimes)
            
            # Use global optimization as fallback
            logger.info("Performing global optimization as fallback")
                        
        # Set up Optuna study
        self.study = self._setup_optuna_study()
        
        # Apply advanced pruning
        if self.advanced_pruning:
            self.study = self._apply_advanced_pruning(self.study)
            
        # Define the objective function with X, y, groups
        objective = lambda trial: self._objective_function(trial, X, y, groups, sample_weight)
        
        # Perform optimization
        logger.info(f"Starting optimization with {self.n_trials} trials")
        self.study.optimize(
            objective, 
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Extract best parameters
        if self.multi_objective:
            # For multi-objective, take best params from the trial with highest first objective
            pareto_front = self.study.best_trials
            if pareto_front:
                # Sort by first objective (e.g., Sharpe)
                sorted_trials = sorted(
                    pareto_front, 
                    key=lambda t: t.values[0],
                    reverse=True
                )
                self.best_params = sorted_trials[0].params
            else:
                self.best_params = {}
        else:
            # For single objective, take best params
            self.best_params = self.study.best_params
            
        # Extract meta-knowledge for transfer learning
        self.transfer_learning_params = self._extract_meta_knowledge()
        
        # Calculate SHAP stability if available
        if hasattr(self, 'shap_values_by_fold') and self.shap_values_by_fold:
            self.shap_stability = self._calculate_shap_stability(X, self.shap_values_by_fold)
            
        logger.info(f"Optimization complete. Best parameters: {self.best_params}")
        
        return self.best_params
        
    def fit_final_model(self, X, y, sample_weight=None):
        """
        Fit final model using best parameters.
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Features
        y : pandas Series or numpy array
            Target variable
        sample_weight : pandas Series or numpy array, optional
            Sample weights
            
        Returns:
        --------
        object
            Fitted model
        """
        if not self.best_params:
            raise ValueError("Must run optimize() before fit_final_model()")
            
        # Clone parameters to avoid modifying the original
        model_params = self.best_params.copy()
        
        # Remove non-model parameters
        for param in ['callbacks']:
            if param in model_params:
                model_params.pop(param)
                
        # Fit model with best parameters
        model = self._fit_model(model_params, X, y, sample_weight)
        
        return model
    

    def calculate_shap_values(self, model, X_sample):
        """
        Calculate SHAP values for feature importance.
        
        Parameters:
        -----------
        model : object
            Fitted model
        X_sample : pandas DataFrame
            Sample data for SHAP calculation
            
        Returns:
        --------
        numpy.ndarray
            SHAP values
        """
        # Limit sample size for CatBoost + GPU to avoid O(n²) performance issues
        if self.model_type == 'catboost' and self.use_gpu:
            X_sample = X_sample.sample(n=min(len(X_sample), 2000), random_state=42)
        
        # Initialize SHAP explainer based on model type
        if self.model_type == 'lgbm':
            explainer = shap.TreeExplainer(model)
        elif self.model_type == 'xgboost':
            explainer = shap.TreeExplainer(model)
        elif self.model_type == 'catboost':
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback for non-tree models
            explainer = shap.Explainer(model.predict, X_sample)
            
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values
    
    def plot_parameter_importance(self):
        """
        Plot parameter importance from the study.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Parameter importance plot
        """
        if not self.study or len(self.study.trials) < 5:
            raise ValueError("Not enough trials for parameter importance")
            
        fig = plot_param_importances(self.study)
        return fig
    
    def plot_optimization_history(self):
        """
        Plot optimization history from the study.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Optimization history plot
        """
        if not self.study:
            raise ValueError("No study available")
            
        fig = plot_optimization_history(self.study)
        return fig
    
    def save_study(self, filepath):
        """
        Save the study to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the study
            
        Returns:
        --------
        bool
            Success flag
        """
        if not self.study:
            raise ValueError("No study to save")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save study
        with open(filepath, 'wb') as f:
            pickle.dump(self.study, f)
            
        logger.info(f"Study saved to {filepath}")
        
        return True
    
    def load_study(self, filepath):
        """
        Load a study from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to load the study from
            
        Returns:
        --------
        optuna.study.Study
            Loaded study
        """
        if not os.path.exists(filepath):
            raise ValueError(f"Study file not found: {filepath}")
            
        # Load study
        with open(filepath, 'rb') as f:
            self.study = pickle.load(f)
            
        logger.info(f"Study loaded from {filepath}")
        
        # Extract best parameters
        if self.multi_objective:
            # For multi-objective, take best params from the trial with highest first objective
            pareto_front = self.study.best_trials
            if pareto_front:
                # Sort by first objective (e.g., Sharpe)
                sorted_trials = sorted(
                    pareto_front, 
                    key=lambda t: t.values[0],
                    reverse=True
                )
                self.best_params = sorted_trials[0].params
            else:
                self.best_params = {}
        else:
            # For single objective, take best params
            self.best_params = self.study.best_params
            
        return self.study
    
    def get_meta_learning_priors(self):
        """
        Get priors for meta-learning based on previous optimization.
        
        Returns:
        --------
        dict
            Priors for parameters
        """
        if not self.transfer_learning_params:
            return {}
            
        # Return best ranges as priors
        if 'best_ranges' in self.transfer_learning_params:
            return self.transfer_learning_params['best_ranges']
        else:
            return {}
    
    def evaluate_out_of_distribution(self, X_ood, y_ood, model=None):
        """
        Evaluate model on out-of-distribution data.
        
        Parameters:
        -----------
        X_ood : pandas DataFrame
            Out-of-distribution features
        y_ood : pandas Series or numpy array
            Out-of-distribution target
        model : object, optional
            Fitted model (if None, will fit a model with best params)
            
        Returns:
        --------
        dict
            OOD performance metrics
        """
        if model is None:
            if not self.best_params:
                raise ValueError("Must run optimize() before evaluate_out_of_distribution()")
                
            # Fit model with best parameters
            model = self.fit_final_model(X_ood, y_ood)
            
        # Make predictions
        y_pred = model.predict(X_ood)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_ood, y_pred)
        
        # Calculate returns if applicable
        if hasattr(y_ood, 'values'):
            y_ood_values = y_ood.values
        else:
            y_ood_values = y_ood
            
        # Simple strategy: sign of prediction * returns
        strategy_returns = np.sign(y_pred) * y_ood_values
        
        # Add financial metrics
        if len(strategy_returns) > 5:  # Need enough samples
            metrics['sharpe'] = self.metrics.sharpe_ratio(strategy_returns)
            metrics['sortino'] = self.metrics.sortino_ratio(strategy_returns)
            metrics['max_drawdown'] = self.metrics.max_drawdown(strategy_returns)
            metrics['cvar_5'] = self.metrics.conditional_value_at_risk(strategy_returns, 5)
            
        return metrics
    
    def run_sensitivity_analysis(self, X, y, param_ranges=None, n_samples=10):
        """
        Run sensitivity analysis on parameters.
        
        Parameters:
        -----------
        X : pandas DataFrame
            Features
        y : pandas Series or numpy array
            Target variable
        param_ranges : dict, optional
            Parameter ranges to test
        n_samples : int
            Number of samples to test per parameter
            
        Returns:
        --------
        dict
            Sensitivity analysis results
        """
        if not self.best_params:
            raise ValueError("Must run optimize() before sensitivity_analysis()")
            
        if param_ranges is None:
            # Default: vary each parameter by ±20%
            param_ranges = {}
            for param, value in self.best_params.items():
                if isinstance(value, (int, float)) and param != 'random_state':
                    param_ranges[param] = {
                        'min': value * 0.8,
                        'max': value * 1.2
                    }
                    
        # Initialize results
        sensitivity_results = {}
        
        # Split data for faster sensitivity analysis
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Baseline performance
        baseline_model = self.fit_final_model(X_train, y_train)
        baseline_preds = baseline_model.predict(X_test)
        
        if hasattr(y_test, 'values'):
            y_test_values = y_test.values
        else:
            y_test_values = y_test
            
        baseline_returns = np.sign(baseline_preds) * y_test_values
        baseline_sharpe = self.metrics.sharpe_ratio(baseline_returns)
        
        # Test each parameter
        for param_name, range_dict in param_ranges.items():
            param_min = range_dict['min']
            param_max = range_dict['max']
            
            # Generate parameter values
            if isinstance(self.best_params[param_name], int):
                # Integer parameter
                param_values = np.linspace(param_min, param_max, n_samples).astype(int)
            else:
                # Float parameter
                param_values = np.linspace(param_min, param_max, n_samples)
                
            # Initialize results for this parameter
            param_results = {
                'param_values': param_values.tolist(),
                'metrics': []
            }
            
            # Test each value
            for value in param_values:
                # Create modified parameters
                modified_params = self.best_params.copy()
                modified_params[param_name] = value
                
                # Remove non-model parameters
                for p in ['callbacks']:
                    if p in modified_params:
                        modified_params.pop(p)
                        
                # Fit model and evaluate
                try:
                    model = self._fit_model(modified_params, X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate returns
                    strategy_returns = np.sign(y_pred) * y_test_values
                    sharpe = self.metrics.sharpe_ratio(strategy_returns)
                    
                    # Calculate relative performance
                    rel_performance = (sharpe / baseline_sharpe if baseline_sharpe != 0 else 0)
                    
                    param_results['metrics'].append({
                        'sharpe': sharpe,
                        'relative_performance': rel_performance
                    })
                except Exception as e:
                    logger.warning(f"Error in sensitivity analysis for {param_name}={value}: {e}")
                    param_results['metrics'].append({
                        'sharpe': np.nan,
                        'relative_performance': np.nan
                    })
                    
            sensitivity_results[param_name] = param_results
            
        return sensitivity_results