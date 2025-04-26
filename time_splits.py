import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Callable, Tuple, Optional
import warnings
from joblib import Parallel, delayed
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.optimize import minimize
import scipy.stats as ss
import datetime
import logging
import itertools


# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFinancialTimeSeriesValidator:
    """
    Advanced validation framework for financial time series forecasting.
    
    Implements nested Combinatorial Purged Cross-Validation (CPCV) with 
    additional enhancements for financial time series data.
    
    Features:
    - Respects time series nature of data through proper train/test splitting
    - Purging to prevent information leakage
    - Embargo periods to mitigate temporal correlation issues
    - Multiple evaluation metrics specific to financial forecasting
    - Stationarity testing and monitoring
    - Walk-forward validation
    - Performance decay analysis
    - Backtest overfitting protection
    - Monte Carlo simulation for robustness assessment
    - Regime detection and conditional performance evaluation
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_period: int = 1,
        embargo_period: int = 1,
        eval_metrics: List[str] = None,
        significance_level: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Initialize the financial time series validation framework.
        
        Parameters:
        -----------
        n_splits : int, default=5
            Number of splits for outer cross-validation.
        n_test_splits : int, default=2
            Number of splits for the inner/nested validation.
        purge_period : int, default=1
            Number of periods to purge between training and testing sets
            to avoid information leakage.
        embargo_period : int, default=1
            Number of periods to add as embargo after test set to avoid
            information leakage in future folds.
        eval_metrics : List[str], default=None
            List of evaluation metrics to compute. If None, defaults to
            ['rmse', 'mae', 'r2', 'mda', 'sharpe'].
        significance_level : float, default=0.05
            Significance level for statistical tests.
        random_state : int, default=42
            Random state for reproducibility.
        n_jobs : int, default=-1
            Number of jobs to run in parallel (-1 means using all processors).
        verbose : bool, default=True
            Whether to print progress information.
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_period = purge_period
        self.embargo_period = embargo_period
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.significance_level = significance_level
        
        # Define default metrics if none provided
        if eval_metrics is None:
            self.eval_metrics = ['rmse', 'mae', 'r2', 'mda', 'sharpe']
        else:
            self.eval_metrics = eval_metrics
            
        # Results storage
        self.results = {}
        self.fold_performances = {}
        self.feature_importances = {}
        self.model_diagnostics = {}
        
        # Validation statistics
        self.optimal_purge_params = None
        self.stationarity_results = None
        self.performance_decay = None
        self.regime_performance = None
        self.pbo_stats = None  # Probability of backtest overfitting
        
    def _get_purged_indices(
        self, 
        timestamps: pd.Series, 
        train_indices: np.ndarray, 
        test_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply purging and embargo to prevent information leakage.
        
        Parameters:
        -----------
        timestamps : pd.Series
            Series of datetime indices for the dataset.
        train_indices : np.ndarray
            Indices for the training set.
        test_indices : np.ndarray
            Indices for the test set.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Purged train indices and test indices with embargo.
        """
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        # Get min and max timestamps for test set
        test_start = timestamps.iloc[test_indices.min()]
        test_end = timestamps.iloc[test_indices.max()]
        
        # Calculate purge period timestamps
        purge_start = test_start - pd.Timedelta(days=self.purge_period)
        
        # Apply embargo after test set
        embargo_end = test_end + pd.Timedelta(days=self.embargo_period)
        
        # Identify indices to purge from training
        to_purge = np.where(
            (timestamps >= purge_start) & (timestamps <= embargo_end)
        )[0]
        
        # Remove purged indices from training set
        clean_train = np.setdiff1d(train_indices, to_purge)
        
        return clean_train, test_indices
    def _get_adaptive_purged_indices(
        self, 
        timestamps: pd.Series, 
        returns: np.ndarray,
        train_indices: np.ndarray, 
        test_indices: np.ndarray,
        base_purge_period: int = 1,
        base_embargo_period: int = 1,
        vol_lookback: int = 22,
        vol_scale: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply adaptive purging and embargo based on market volatility.
        
        Parameters:
        -----------
        timestamps : pd.Series
            Series of datetime indices for the dataset.
        returns : np.ndarray
            Financial returns series.
        train_indices : np.ndarray
            Indices for the training set.
        test_indices : np.ndarray
            Indices for the test set.
        base_purge_period : int, default=1
            Base number of days for purging.
        base_embargo_period : int, default=1
            Base number of days for embargo.
        vol_lookback : int, default=22
            Lookback window for volatility calculation.
        vol_scale : float, default=2.0
            Scaling factor for volatility-based adjustment.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Purged train indices and test indices with adaptive embargo.
        """
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)
        
        # Get min and max timestamps for test set
        test_start = timestamps.iloc[test_indices.min()]
        test_end = timestamps.iloc[test_indices.max()]
        
        # Calculate rolling volatility
        returns_series = pd.Series(returns)
        rolling_vol = returns_series.rolling(window=vol_lookback).std().fillna(method='bfill')
        
        # Calculate average volatility
        avg_vol = rolling_vol.mean()
        
        # Calculate volatility at test start and end
        test_start_idx = test_indices.min()
        test_end_idx = test_indices.max()
        
        # Get volatility scaling factors (clip to reasonable range)
        if test_start_idx >= vol_lookback:
            start_vol_ratio = min(max(rolling_vol.iloc[test_start_idx] / avg_vol, 0.5), 3.0)
        else:
            start_vol_ratio = 1.0
            
        if test_end_idx >= vol_lookback:
            end_vol_ratio = min(max(rolling_vol.iloc[test_end_idx] / avg_vol, 0.5), 3.0)
        else:
            end_vol_ratio = 1.0
        
        # Scale purge and embargo periods by volatility
        adaptive_purge = int(base_purge_period * start_vol_ratio * vol_scale)
        adaptive_embargo = int(base_embargo_period * end_vol_ratio * vol_scale)
        
        # Enforce minimum periods
        adaptive_purge = max(adaptive_purge, base_purge_period)
        adaptive_embargo = max(adaptive_embargo, base_embargo_period)
        
        if self.verbose:
            logger.info(f"Adaptive purge: {adaptive_purge} days (base={base_purge_period})")
            logger.info(f"Adaptive embargo: {adaptive_embargo} days (base={base_embargo_period})")
        
        # Calculate purge period timestamps
        purge_start = test_start - pd.Timedelta(days=adaptive_purge)
        
        # Apply embargo after test set
        embargo_end = test_end + pd.Timedelta(days=adaptive_embargo)
        
        # Identify indices to purge from training
        to_purge = np.where(
            (timestamps >= purge_start) & (timestamps <= embargo_end)
        )[0]
        
        # Remove purged indices from training set
        clean_train = np.setdiff1d(train_indices, to_purge)
        
        return clean_train, test_indices    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate various performance metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted target values.
        returns : np.ndarray, optional
            Financial returns series (if available).
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of metric names and values.
        """
        metrics = {}
        
        # Standard regression metrics
        if 'rmse' in self.eval_metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        if 'mae' in self.eval_metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        if 'r2' in self.eval_metrics:
            metrics['r2'] = r2_score(y_true, y_pred)
            
        # Financial-specific metrics
        if 'mda' in self.eval_metrics:  # Mean Directional Accuracy
            direction_true = np.sign(np.diff(np.append([0], y_true)))
            direction_pred = np.sign(np.diff(np.append([0], y_pred)))
            metrics['mda'] = np.mean(direction_true == direction_pred)
        
        if returns is not None and 'sharpe' in self.eval_metrics:
            # Calculate Sharpe ratio based on strategy returns
            # For this example, we'll use a simple strategy based on prediction direction
            direction_pred = np.sign(y_pred)
            strategy_returns = direction_pred[:-1] * returns[1:]  # Align returns with signals
            metrics['sharpe'] = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)  # Annualized
            
            # Calculate Sortino ratio (downside risk only)
            if 'sortino' in self.eval_metrics:
                downside_returns = strategy_returns.copy()
                downside_returns[downside_returns > 0] = 0
                downside_std = np.std(downside_returns) + 1e-10
                metrics['sortino'] = np.mean(strategy_returns) / downside_std * np.sqrt(252)
        
        # Information Coefficient (if available)
        if 'ic' in self.eval_metrics:
            metrics['ic'] = np.corrcoef(y_true, y_pred)[0, 1]
            
        return metrics
    
    def _test_stationarity(self, y: np.ndarray, alpha: float = 0.05) -> Dict:
        """
        Test for stationarity in the time series.
        
        Parameters:
        -----------
        y : np.ndarray
            Time series data.
        alpha : float, default=0.05
            Significance level.
            
        Returns:
        --------
        Dict
            Results of stationarity tests.
        """
        from statsmodels.tsa.stattools import adfuller, kpss
        
        result = {}
        
        # Augmented Dickey-Fuller test (null hypothesis: unit root exists)
        adf_stat, adf_p, _, _, critical_values, _ = adfuller(y)
        result['adf_statistic'] = adf_stat
        result['adf_p_value'] = adf_p
        result['adf_critical_values'] = critical_values
        result['adf_is_stationary'] = adf_p < alpha
        
        # KPSS test (null hypothesis: series is stationary)
        try:
            kpss_stat, kpss_p, _, critical_values = kpss(y)
            result['kpss_statistic'] = kpss_stat
            result['kpss_p_value'] = kpss_p
            result['kpss_critical_values'] = critical_values
            result['kpss_is_stationary'] = kpss_p > alpha
        except:
            result['kpss_is_stationary'] = "Error in KPSS test"
            
        # Conflicting results?
        if isinstance(result['kpss_is_stationary'], bool):
            result['conflicting_results'] = result['adf_is_stationary'] != result['kpss_is_stationary']
        else:
            result['conflicting_results'] = None
            
        return result
    
    def _detect_regime(self, returns: np.ndarray, window: int = 63) -> np.ndarray:
        """
        Detect market regimes based on volatility and momentum.
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of financial returns.
        window : int, default=63
            Lookback window for regime detection (approximately 3 months).
            
        Returns:
        --------
        np.ndarray
            Array of regime labels.
        """
        returns_series = pd.Series(returns)
        
        # Calculate volatility (rolling std)
        volatility = returns_series.rolling(window=window).std()
        
        # Calculate momentum (rolling mean)
        momentum = returns_series.rolling(window=window).mean()
        
        # Initialize regimes array
        regimes = np.zeros(len(returns))
        
        # Define regimes:
        # 0: Normal (default)
        # 1: High volatility, positive momentum (bullish volatile)
        # 2: High volatility, negative momentum (bearish volatile) 
        # 3: Low volatility, positive momentum (bullish stable)
        # 4: Low volatility, negative momentum (bearish stable)
        
        # Set thresholds (could be optimized)
        vol_threshold = volatility.quantile(0.7)
        
        # Assign regimes
        high_vol_mask = volatility > vol_threshold
        pos_mom_mask = momentum > 0
        
        regimes[high_vol_mask & pos_mom_mask] = 1
        regimes[high_vol_mask & ~pos_mom_mask] = 2
        regimes[~high_vol_mask & pos_mom_mask] = 3
        regimes[~high_vol_mask & ~pos_mom_mask] = 4
        
        return regimes
    
    def _calculate_pbo(self, performances: List[float], n_trials: int = 1000) -> float:
        """
        Calculate Probability of Backtest Overfitting (PBO) using
        the method described by Bailey et al.
        
        Parameters:
        -----------
        performances : List[float]
            List of performance metrics across multiple configurations.
        n_trials : int, default=1000
            Number of bootstrap trials.
            
        Returns:
        --------
        float
            Probability of backtest overfitting.
        """
        n_configs = len(performances)
        if n_configs < 4:
            return np.nan  # Need at least 4 configurations
            
        # Convert to numpy array
        performances = np.array(performances)
        
        # Initialize counter
        count = 0
        
        # Run bootstrap trials
        for _ in range(n_trials):
            # Sample with replacement
            sample_idx = np.random.choice(n_configs, size=n_configs, replace=True)
            sample_perf = performances[sample_idx]
            
            # Find best configuration in sample
            best_idx = np.argmax(sample_perf)
            
            # Find performance of this configuration in complement
            complement_idx = np.setdiff1d(np.arange(n_configs), sample_idx)
            
            if len(complement_idx) == 0:
                continue  # Skip this trial
                
            best_config_in_complement = sample_perf[best_idx]
            
            # Find rank of this performance in complement
            rank_in_complement = ss.percentileofscore(
                performances[complement_idx], best_config_in_complement
            )
            
            # If rank is below median (50%), increment counter
            if rank_in_complement < 50:
                count += 1
                
        # Calculate PBO
        pbo = count / n_trials
        return pbo
        
    def _optimize_purge_embargo(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        timestamps: pd.Series,
        model: BaseEstimator,
        purge_range: List[int] = [0, 1, 2, 3, 5, 7, 10],
        embargo_range: List[int] = [0, 1, 2, 3, 5, 7, 10],
        metric: str = 'rmse'
    ) -> Dict:
        """
        Optimize purge and embargo periods.
        
        Parameters:
        -----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Target values.
        timestamps : pd.Series
            Series of datetime indices for the dataset.
        model : BaseEstimator
            Model to evaluate.
        purge_range : List[int], default=[0, 1, 2, 3, 5, 7, 10]
            Range of purge values to try.
        embargo_range : List[int], default=[0, 1, 2, 3, 5, 7, 10]
            Range of embargo values to try.
        metric : str, default='rmse'
            Metric to optimize for.
            
        Returns:
        --------
        Dict
            Optimal parameters and results.
        """
        best_score = float('inf') if metric in ['rmse', 'mae'] else float('-inf')
        best_params = {'purge_period': 1, 'embargo_period': 1}
        
        results = {}
        
        # Grid search over purge and embargo parameters
        for purge in purge_range:
            for embargo in embargo_range:
                # Set temporary parameters
                self.purge_period = purge
                self.embargo_period = embargo
                
                # Create CV splits
                cv_splits = self._generate_cv_splits(timestamps)
                
                # Track scores
                fold_scores = []
                
                for train_idx, test_idx in cv_splits:
                    # Apply purging and embargo
                    purged_train_idx, test_idx = self._get_purged_indices(
                        timestamps, train_idx, test_idx
                    )
                    
                    # Train model
                    model_clone = clone(model)
                    model_clone.fit(X[purged_train_idx], y[purged_train_idx])
                    
                    # Make predictions
                    y_pred = model_clone.predict(X[test_idx])
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y[test_idx], y_pred)
                    fold_scores.append(metrics[metric])
                
                # Calculate average score
                avg_score = np.mean(fold_scores)
                
                # Track result
                results[(purge, embargo)] = avg_score
                
                # Check if better than current best
                is_better = (
                    (avg_score < best_score) if metric in ['rmse', 'mae'] 
                    else (avg_score > best_score)
                )
                
                if is_better:
                    best_score = avg_score
                    best_params = {'purge_period': purge, 'embargo_period': embargo}
        
        # Reset parameters to optimal values
        self.purge_period = best_params['purge_period']
        self.embargo_period = best_params['embargo_period']
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def _generate_cv_splits(self, timestamps: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate cross-validation splits respecting time series nature.
        Memory-optimized version using views instead of copies where possible.
        
        Parameters:
        -----------
        timestamps : pd.Series
            Series of datetime indices for the dataset.
            
        Returns:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples.
        """
        n = len(timestamps)
        # Use a memory view instead of copying the indices
        indices = np.arange(n)
        
        # Calculate fold size
        fold_size = n // self.n_splits
        
        # Generate splits
        splits = []
        for i in range(self.n_splits):
            if i < self.n_splits - 1:
                test_start = i * fold_size
                test_end = (i + 1) * fold_size
            else:
                # Last fold may be larger to include all remaining data
                test_start = i * fold_size
                test_end = n
                
            # Create views into the indices array rather than copies
            test_indices = indices[test_start:test_end]
            
            # This is more efficient than np.setdiff1d for large arrays
            # Create a mask of indices to keep
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_start:test_end] = False
            train_indices = indices[train_mask]
            
            splits.append((train_indices, test_indices))
            
        return splits
    
    def _generate_nested_cv_splits(
        self, 
        train_indices: np.ndarray,
        timestamps: pd.Series
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate nested cross-validation splits for inner loop.
        
        Parameters:
        -----------
        train_indices : np.ndarray
            Training indices from outer loop.
        timestamps : pd.Series
            Series of datetime indices for the dataset.
            
        Returns:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, val_indices) tuples.
        """
        # Get timestamps for training data
        train_timestamps = timestamps.iloc[train_indices]
        
        # Sort indices by time
        sorted_indices = np.argsort(train_timestamps)
        sorted_train_indices = train_indices[sorted_indices]
        
        n = len(sorted_train_indices)
        fold_size = n // self.n_test_splits
        
        # Generate splits
        splits = []
        for i in range(self.n_test_splits):
            if i < self.n_test_splits - 1:
                val_start = i * fold_size
                val_end = (i + 1) * fold_size
            else:
                # Last fold may be larger to include all remaining data
                val_start = i * fold_size
                val_end = n
                
            val_indices = sorted_train_indices[val_start:val_end]
            inner_train_indices = np.setdiff1d(sorted_train_indices, val_indices)
            
            # Convert indices to original space
            splits.append((inner_train_indices, val_indices))
            
        return splits
    
    def _analyze_performance_decay(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        timestamps: pd.Series
    ) -> Dict:
        """
        Analyze how model performance decays over time.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True target values.
        y_pred : np.ndarray
            Predicted target values.
        timestamps : pd.Series
            Series of datetime indices.
            
        Returns:
        --------
        Dict
            Performance decay analysis results.
        """
        # Convert to DataFrame for easier manipulation
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'y_true': y_true,
            'y_pred': y_pred,
            'error': np.abs(y_true - y_pred)
        })
        
        # Sort by timestamp
        results_df = results_df.sort_values('timestamp')
        
        # Split into quantiles
        n_quantiles = 5  # Default to quintiles
        quantile_labels = [f"Q{i+1}" for i in range(n_quantiles)]
        
        results_df['quantile'] = pd.qcut(
            results_df.index, n_quantiles, labels=quantile_labels
        )
        
        # Calculate metrics by quantile
        quantile_metrics = {}
        for q in quantile_labels:
            q_data = results_df[results_df['quantile'] == q]
            
            # Calculate metrics
            q_metrics = self._calculate_metrics(
                q_data['y_true'].values, 
                q_data['y_pred'].values
            )
            
            quantile_metrics[q] = q_metrics
            
        # Calculate decay rates (linear regression)
        decay_rates = {}
        for metric in self.eval_metrics:
            if metric in ['rmse', 'mae']:
                # For error metrics, increase over time is decay
                values = [quantile_metrics[q].get(metric, np.nan) for q in quantile_labels]
                if not any(np.isnan(values)):
                    slope, _, r_value, p_value, _ = stats.linregress(
                        range(len(values)), values
                    )
                    decay_rates[metric] = {
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'significant': p_value < self.significance_level
                    }
            else:
                # For other metrics, decrease over time is decay
                values = [quantile_metrics[q].get(metric, np.nan) for q in quantile_labels]
                if not any(np.isnan(values)):
                    slope, _, r_value, p_value, _ = stats.linregress(
                        range(len(values)), values
                    )
                    decay_rates[metric] = {
                        'slope': -slope,  # Negative slope indicates decay
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'significant': p_value < self.significance_level
                    }
                
        return {
            'quantile_metrics': quantile_metrics,
            'decay_rates': decay_rates
        }
    
    def _monte_carlo_robustness(
        self, 
        model: BaseEstimator, 
        X: np.ndarray, 
        y: np.ndarray,
        n_samples: int = 100,
        sample_size: float = 0.8
    ) -> Dict:
        """
        Perform Monte Carlo simulation to assess model robustness.
        
        [rest of docstring remains the same]
        """
        n = len(y)
        sample_n = int(n * sample_size)
        
        # Define a helper function for a single Monte Carlo run
        def single_monte_carlo_run(_):
            # Sample indices
            sample_indices = np.random.choice(n, sample_n, replace=False)
            
            # Train on sample
            model_clone_mc = clone(model)
            model_clone_mc.fit(X[sample_indices], y[sample_indices])
            
            # Test on out-of-sample data with mini-embargo
            # Apply a small embargo to prevent information leakage
            # Calculate a mini-embargo (e.g., 3 points)
            mini_embargo = 3
            
            # Find valid test indices (exclude training and embargo points)
            all_indices = np.arange(n)
            potential_test_indices = np.setdiff1d(all_indices, sample_indices)
            
            # Create a mask for indices that are too close to training data
            invalid_test = []
            for train_idx in sample_indices:
                # Find indices within mini_embargo of train_idx
                embargo_range = range(max(0, train_idx - mini_embargo), 
                                    min(n, train_idx + mini_embargo + 1))
                invalid_test.extend(embargo_range)
            
            # Remove duplicates
            invalid_test = np.unique(invalid_test)
            
            # Get final test indices (exclude both training and embargo)
            oos_indices = np.setdiff1d(potential_test_indices, invalid_test)
            
            # If no valid test indices after embargo, use original approach
            if len(oos_indices) == 0:
                oos_indices = np.setdiff1d(all_indices, sample_indices)
            
            # Make predictions
            y_pred = model_clone_mc.predict(X[oos_indices])
            
            # Calculate metrics
            metrics = self._calculate_metrics(y[oos_indices], y_pred)
            return metrics
        
        # Use joblib to parallelize the Monte Carlo runs
        metrics_samples = Parallel(n_jobs=self.n_jobs)(
            delayed(single_monte_carlo_run)(_) for _ in range(n_samples)
        )
        
        # Calculate statistics
        results = {}
        for metric in self.eval_metrics:
            metric_values = [m.get(metric, np.nan) for m in metrics_samples]
            metric_values = [v for v in metric_values if not np.isnan(v)]
            
            if len(metric_values) > 0:
                results[metric] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'median': np.median(metric_values),
                    '5th_percentile': np.percentile(metric_values, 5),
                    '95th_percentile': np.percentile(metric_values, 95)
                }
                
        return results
    
    def validate(
        self, 
        model: BaseEstimator,
        X: np.ndarray, 
        y: np.ndarray,
        timestamps: pd.Series,
        returns: Optional[np.ndarray] = None,
        optimize_purge: bool = True,
        analyze_decay: bool = True,
        monte_carlo: bool = True,
        evaluate_regimes: bool = True,
        feature_importance: bool = False,
        adaptive_purging: bool = False
    ) -> Dict:
        """
        Perform comprehensive validation of the model.
        
        [rest of docstring remains the same]
        """
        if self.verbose:
            logger.info("Starting financial time series validation")
            
        # Convert inputs to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if returns is not None and isinstance(returns, pd.Series):
            returns = returns.values
            
        # Test stationarity of target variable
        if self.verbose:
            logger.info("Testing stationarity of target variable")
        self.stationarity_results = self._test_stationarity(y)
        
        # Store original purge/embargo values before optimization
        original_purge = self.purge_period
        original_embargo = self.embargo_period
        
        # Optimize purge and embargo periods if requested
        if optimize_purge and self.verbose:
            logger.info("Optimizing purge and embargo periods")
            self.optimal_purge_params = self._optimize_purge_embargo(
                X, y, timestamps, model
            )
        else:
            # Set fallback for when not optimizing to ensure consistency
            if self.verbose:
                logger.info(f"Using default purge={self.purge_period}, embargo={self.embargo_period}")
            self.optimal_purge_params = {
                'best_params': {'purge_period': self.purge_period, 'embargo_period': self.embargo_period},
                'best_score': None,
                'all_results': {}
            }
        
        # Generate CV splits
        if self.verbose:
            logger.info(f"Generating {self.n_splits} CV splits")
        cv_splits = self._generate_cv_splits(timestamps)
        
        # Track predictions, true values, and fold information for post-analysis
        all_y_true = []
        all_y_pred = []
        all_fold_indices = []
        all_timestamps = []
        
        # Track metrics per fold
        fold_metrics = []
        
        # Track feature importance if requested
        if feature_importance:
            feature_importances = []
            
        # Loop through CV splits
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            if self.verbose:
                logger.info(f"Processing fold {fold_idx+1}/{self.n_splits}")
                
            # Apply purging and embargo
            if adaptive_purging and returns is not None:
                purged_train_idx, test_idx = self._get_adaptive_purged_indices(
                    timestamps, returns, train_idx, test_idx,
                    base_purge_period=self.purge_period,
                    base_embargo_period=self.embargo_period
                )
            else:
                purged_train_idx, test_idx = self._get_purged_indices(
                    timestamps, train_idx, test_idx
                )
            
            # Train model
            model_clone = clone(model)
            model_clone.fit(X[purged_train_idx], y[purged_train_idx])
            
            # Track feature importance if available
            if feature_importance and hasattr(model_clone, 'feature_importances_'):
                feature_importances.append(model_clone.feature_importances_)
            
            # Generate predictions
            y_pred = model_clone.predict(X[test_idx])
            
            # Calculate metrics
            fold_returns = returns[test_idx] if returns is not None else None
            metrics = self._calculate_metrics(y[test_idx], y_pred, fold_returns)
            fold_metrics.append(metrics)
            
            # Store predictions, true values, and fold information
            all_y_true.extend(y[test_idx])
            all_y_pred.extend(y_pred)
            all_fold_indices.extend([fold_idx] * len(test_idx))
            all_timestamps.extend(timestamps.iloc[test_idx])
            
        # Convert to arrays for easier processing
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_fold_indices = np.array(all_fold_indices)
        all_timestamps = np.array(all_timestamps)
        
        # Calculate overall metrics
        if self.verbose:
            logger.info("Calculating overall metrics")
        overall_returns = returns if returns is not None else None
        overall_metrics = self._calculate_metrics(all_y_true, all_y_pred, overall_returns)
        
        # Analyze performance decay if requested
        if analyze_decay and self.verbose:
            logger.info("Analyzing performance decay over time")
            self.performance_decay = self._analyze_performance_decay(
                all_y_true, all_y_pred, pd.Series(all_timestamps)
            )
        # Evaluate performance across different market regimes if requested
        if evaluate_regimes and returns is not None and self.verbose:
            logger.info("Evaluating performance across market regimes")
            
            # Detect regimes
            regimes = self._detect_regime(returns)
            
            # Create DataFrame for analysis
            regime_df = pd.DataFrame({
                'y_true': all_y_true,
                'y_pred': all_y_pred,
                'fold': all_fold_indices,
                'timestamp': all_timestamps,
                'regime': regimes
            })
            
            # Calculate metrics by regime
            regime_metrics = {}
            for regime_id in np.unique(regimes):
                regime_data = regime_df[regime_df['regime'] == regime_id]
                
                if len(regime_data) > 0:
                    regime_returns = returns[regime_data.index] if returns is not None else None
                    metrics = self._calculate_metrics(
                        regime_data['y_true'].values, 
                        regime_data['y_pred'].values,
                        regime_returns
                    )
                    regime_metrics[int(regime_id)] = metrics
                    
            self.regime_performance = regime_metrics
            
        # Perform Monte Carlo simulation for robustness assessment if requested
        if monte_carlo and self.verbose:
            logger.info("Performing Monte Carlo simulation for robustness assessment")
            mc_results = self._monte_carlo_robustness(model, X, y)
            
        # Calculate Probability of Backtest Overfitting
        if self.verbose:
            logger.info("Calculating Probability of Backtest Overfitting")
            
        # Extract relevant metric for PBO calculation
        primary_metric = 'sharpe' if 'sharpe' in self.eval_metrics and returns is not None else 'rmse'
        pbo_metric_values = [metrics.get(primary_metric, np.nan) for metrics in fold_metrics]
        
        if not any(np.isnan(pbo_metric_values)):
            self.pbo_stats = self._calculate_pbo(pbo_metric_values)
            
        # Prepare results dict
        self.results = {
            'overall_metrics': overall_metrics,
            'fold_metrics': fold_metrics,
            'feature_importances': feature_importances if feature_importance else None,
            'stationarity_results': self.stationarity_results,
            'optimal_purge_params': self.optimal_purge_params,
            'performance_decay': self.performance_decay,
            'regime_performance': self.regime_performance,
            'monte_carlo_results': mc_results if monte_carlo else None,
            'pbo_stats': self.pbo_stats
        }
        
        return self.results
    
    def get_best_split(self, X: np.ndarray, y: np.ndarray, timestamps: pd.Series) -> Tuple:
        """
        Find the best train/test split based on stationarity and distribution similarity.
        
        Parameters:
        -----------
        X : np.ndarray
            Features matrix.
        y : np.ndarray
            Target variable.
        timestamps : pd.Series
            Time stamps for the data.
            
        Returns:
        --------
        Tuple
            Tuple of (train_indices, test_indices) for the best split.
        """
        if self.verbose:
            logger.info("Finding optimal train/test split")
            
        # Generate CV splits
        cv_splits = self._generate_cv_splits(timestamps)
        
        best_score = float('inf')
        best_split = None
        
        for train_idx, test_idx in cv_splits:
            # Calculate distribution similarity
            train_dist = y[train_idx]
            test_dist = y[test_idx]
            
            # KS statistic for distribution similarity (lower is more similar)
            ks_stat, _ = stats.ks_2samp(train_dist, test_dist)
            
            # Check stationarity of both train and test
            train_stationary = self._test_stationarity(train_dist)['adf_is_stationary']
            test_stationary = self._test_stationarity(test_dist)['adf_is_stationary']
            
            # Score based on KS statistic and stationarity
            # Prefer splits where both train and test are stationary
            score = ks_stat
            if not train_stationary or not test_stationary:
                score += 1.0  # Penalty for non-stationarity
                
            if score < best_score:
                best_score = score
                best_split = (train_idx, test_idx)
                
        return best_split
        
    def walk_forward_validation(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.Series,
        returns: Optional[np.ndarray] = None,
        initial_train_size: float = 0.3,
        step_size: int = 1,
        metric: str = 'rmse'
    ) -> Dict:
        """
        Perform walk-forward validation.
        
        Parameters:
        -----------
        model : BaseEstimator
            Model to validate.
        X : np.ndarray
            Features.
        y : np.ndarray
            Target values.
        timestamps : pd.Series
            Series of datetime indices for the dataset.
        returns : np.ndarray, optional
            Financial returns series (if available).
        initial_train_size : float, default=0.3
            Initial training set size as a fraction of the dataset.
        step_size : int, default=1
            Number of steps to advance the window in each iteration.
        metric : str, default='rmse'
            Metric to track.
            
        Returns:
        --------
        Dict
            Walk-forward validation results.
        """
        if self.verbose:
            logger.info("Starting walk-forward validation")
            
        # Convert inputs to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if returns is not None and isinstance(returns, pd.Series):
            returns = returns.values
            
        n = len(y)
        initial_train_end = int(n * initial_train_size)
        
        # Track predictions, actual values, and timestamps
        all_y_true = []
        all_y_pred = []
        all_timestamps = []
        
        # Track metrics over time
        step_metrics = []
        
        # Iterate over the test set
        current_idx = initial_train_end
        
        while current_idx < n:
            # Define train and test indices
            train_indices = np.arange(current_idx - initial_train_end)
            test_indices = np.arange(current_idx, min(current_idx + step_size, n))
            
            # Apply purging and embargo
            purged_train_indices, test_indices = self._get_purged_indices(
                timestamps, train_indices, test_indices
            )
            
            # Train model
            model_clone = clone(model)
            model_clone.fit(X[purged_train_indices], y[purged_train_indices])
            
            # Make predictions
            y_pred = model_clone.predict(X[test_indices])
            
            # Calculate metrics
            step_returns = returns[test_indices] if returns is not None else None
            metrics = self._calculate_metrics(y[test_indices], y_pred, step_returns)
            step_metrics.append(metrics)
            
            # Store results
            all_y_true.extend(y[test_indices])
            all_y_pred.extend(y_pred)
            all_timestamps.extend(timestamps.iloc[test_indices])
            
            # Advance the window
            current_idx += step_size
            
        # Convert to arrays
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_timestamps = np.array(all_timestamps)
        
        # Calculate overall metrics
        overall_returns = returns[initial_train_end:] if returns is not None else None
        overall_metrics = self._calculate_metrics(all_y_true, all_y_pred, overall_returns)
        
        # Calculate performance curve
        performance_curve = [metrics.get(metric, np.nan) for metrics in step_metrics]
        
        # Calculate trend in performance
        steps = np.arange(len(performance_curve))
        valid_metrics = ~np.isnan(performance_curve)
        
        if np.sum(valid_metrics) > 1:
            slope, _, r_value, p_value, _ = stats.linregress(
                steps[valid_metrics], np.array(performance_curve)[valid_metrics]
            )
            
            trend = {
                'slope': slope if metric not in ['rmse', 'mae'] else -slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'significant': p_value < self.significance_level
            }
        else:
            trend = {
                'slope': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'significant': False
            }
            
        return {
            'overall_metrics': overall_metrics,
            'step_metrics': step_metrics,
            'performance_curve': performance_curve,
            'trend': trend
        }
    
    def combinatorial_purged_cv(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.Series,
        returns: Optional[np.ndarray] = None,
        n_combinations: int = 10
    ) -> Dict:
        """
        Perform Combinatorial Purged Cross-Validation as described by 
        Marcos Lopez de Prado.
        
        Parameters:
        -----------
        model : BaseEstimator
            Model to validate.
        X : np.ndarray
            Features.
        y : np.ndarray
            Target values.
        timestamps : pd.Series
            Series of datetime indices for the dataset.
        returns : np.ndarray, optional
            Financial returns series (if available).
        n_combinations : int, default=10
            Number of random combinations to try.
            
        Returns:
        --------
        Dict
            CPCV validation results.
        """
        if self.verbose:
            logger.info("Starting Combinatorial Purged CV")
            
        # Convert inputs to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if returns is not None and isinstance(returns, pd.Series):
            returns = returns.values
            
        # Generate base CV splits
        cv_splits = self._generate_cv_splits(timestamps)
        n_splits = len(cv_splits)
        
        # Generate random combinations of K folds out of N
        if n_combinations > scipy.special.comb(n_splits, n_splits // 2):
            # If n_combinations is greater than the number of possible combinations,
            # use all combinations
            train_fold_indices = list(
                itertools.combinations(range(n_splits), n_splits // 2)
            )
            if len(train_fold_indices) > n_combinations:
                # Randomly select if still too many
                np.random.seed(self.random_state)
                train_fold_indices = [
                    train_fold_indices[i] for i in 
                    np.random.choice(len(train_fold_indices), n_combinations, replace=False)
                ]
        else:
            # Randomly select n_combinations
            train_fold_indices = []
            while len(train_fold_indices) < n_combinations:
                np.random.seed(self.random_state + len(train_fold_indices))
                comb = tuple(sorted(np.random.choice(
                    range(n_splits), n_splits // 2, replace=False
                )))
                if comb not in train_fold_indices:
                    train_fold_indices.append(comb)
                    
        # Track metrics for each combination
        combination_metrics = []
        
        # Iterate over combinations
        for comb_idx, train_folds in enumerate(train_fold_indices):
            if self.verbose:
                logger.info(f"Processing combination {comb_idx+1}/{len(train_fold_indices)}")
                
            # Get test folds
            test_folds = tuple(fold for fold in range(n_splits) if fold not in train_folds)
            
            # Combine train indices
            train_indices = np.concatenate([cv_splits[fold][0] for fold in train_folds])
            
            # Combine test indices
            test_indices = np.concatenate([cv_splits[fold][1] for fold in test_folds])
            
            # Apply purging and embargo
            purged_train_indices, test_indices = self._get_purged_indices(
                timestamps, train_indices, test_indices
            )
            
            # Train model
            model_clone = clone(model)
            model_clone.fit(X[purged_train_indices], y[purged_train_indices])
            
            # Make predictions
            y_pred = model_clone.predict(X[test_indices])
            
            # Calculate metrics
            comb_returns = returns[test_indices] if returns is not None else None
            metrics = self._calculate_metrics(y[test_indices], y_pred, comb_returns)
            combination_metrics.append(metrics)
            
        # Calculate summary statistics
        summary = {}
        for metric in self.eval_metrics:
            values = [m.get(metric, np.nan) for m in combination_metrics]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) > 0:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    '5th_percentile': np.percentile(values, 5),
                    '95th_percentile': np.percentile(values, 95),
                    'iqr': np.percentile(values, 75) - np.percentile(values, 25)
                }
                
        # Calculate Probability of Backtest Overfitting
        primary_metric = 'sharpe' if 'sharpe' in self.eval_metrics and returns is not None else 'rmse'
        pbo_metric_values = [m.get(primary_metric, np.nan) for m in combination_metrics]
        pbo_metric_values = [v for v in pbo_metric_values if not np.isnan(v)]
        
        if len(pbo_metric_values) > 3:
            pbo = self._calculate_pbo(pbo_metric_values)
        else:
            pbo = np.nan
            
        return {
            'combination_metrics': combination_metrics,
            'summary': summary,
            'pbo': pbo
        }
    
    def plot_performance_decay(self, filename: str = None) -> plt.Figure:
        """
        Plot performance decay over time.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save the plot.
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure.
        """
        if self.performance_decay is None:
            raise ValueError("Performance decay analysis has not been performed yet")
            
        # Extract data
        metrics = self.eval_metrics
        quantiles = sorted(list(self.performance_decay['quantile_metrics'].keys()))
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [
                self.performance_decay['quantile_metrics'][q].get(metric, np.nan) 
                for q in quantiles
            ]
            
            # Plot values
            ax.plot(range(len(quantiles)), values, marker='o', label=metric)
            
            # Add linear trend line if available
            if metric in self.performance_decay['decay_rates']:
                slope = self.performance_decay['decay_rates'][metric]['slope']
                intercept = values[0]  # Use first point as intercept
                trend_line = [intercept + slope * i for i in range(len(quantiles))]
                ax.plot(range(len(quantiles)), trend_line, linestyle='--', 
                        label=f"Trend (slope={slope:.4f})")
                
            # Add labels and title
            ax.set_xlabel("Time Quantile")
            ax.set_ylabel(metric.upper())
            ax.set_xticks(range(len(quantiles)))
            ax.set_xticklabels(quantiles)
            ax.set_title(f"{metric.upper()} Performance Decay")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_regime_performance(self, filename: str = None) -> plt.Figure:
        """
        Plot performance across different market regimes.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save the plot.
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure.
        """
        if self.regime_performance is None:
            raise ValueError("Regime performance analysis has not been performed yet")
            
        # Extract data
        metrics = self.eval_metrics
        regimes = sorted(list(self.regime_performance.keys()))
        
        # Define regime names
        regime_names = {
            0: "Normal",
            1: "Bullish Volatile",
            2: "Bearish Volatile",
            3: "Bullish Stable",
            4: "Bearish Stable"
        }
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [
                self.regime_performance[r].get(metric, np.nan) 
                for r in regimes
            ]
            
            # Create barplot
            bars = ax.bar(range(len(regimes)), values)
            
            # Add labels and title
            ax.set_xlabel("Market Regime")
            ax.set_ylabel(metric.upper())
            ax.set_xticks(range(len(regimes)))
            ax.set_xticklabels([regime_names.get(r, f"Regime {r}") for r in regimes])
            ax.set_title(f"{metric.upper()} by Market Regime")
            
            # Add value labels
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f"{value:.4f}",
                        ha='center', va='bottom', rotation=0
                    )
                    
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_feature_importance(
        self, 
        feature_names: List[str] = None, 
        top_n: int = 20,
        filename: str = None
    ) -> plt.Figure:
        """
        Plot feature importance across folds.
        
        Parameters:
        -----------
        feature_names : List[str], optional
            List of feature names. If None, uses generic names.
        top_n : int, default=20
            Number of top features to plot.
        filename : str, optional
            Filename to save the plot.
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure.
        """
        if self.results.get('feature_importances') is None:
            raise ValueError("Feature importance has not been computed")
            
        # Extract feature importances
        importances = np.array(self.results['feature_importances'])
        
        # Calculate mean and std
        mean_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(mean_imp))]
            
        # Create DataFrame
        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_imp,
            'Std': std_imp
        })
        
        # Sort by importance
        imp_df = imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Take top N
        imp_df = imp_df.head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot horizontal barplot
        bars = ax.barh(imp_df['Feature'], imp_df['Importance'], 
                     xerr=imp_df['Std'], alpha=0.8)
        
        # Add labels and title
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance Across Folds')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to have the highest importance at the top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        return fig
    
    def report(self, filename: str = None) -> str:
        """
        Generate a detailed validation report.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save the report.
            
        Returns:
        --------
        str
            Text report.
        """
        if not self.results:
            raise ValueError("No validation results available yet")
            
        # Create report
        report = []
        
        # Add header
        report.append("=" * 80)
        report.append("FINANCIAL TIME SERIES VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add overall metrics
        report.append("-" * 40)
        report.append("OVERALL METRICS")
        report.append("-" * 40)
        
        for metric, value in self.results['overall_metrics'].items():
            report.append(f"{metric.upper()}: {value:.6f}")
        report.append("")
        
        # Add fold metrics
        report.append("-" * 40)
        report.append("FOLD METRICS")
        report.append("-" * 40)
        
        for i, metrics in enumerate(self.results['fold_metrics']):
            report.append(f"Fold {i+1}:")
            for metric, value in metrics.items():
                report.append(f"  {metric.upper()}: {value:.6f}")
            report.append("")
            
        # Add stationarity results
        if self.results['stationarity_results']:
            report.append("-" * 40)
            report.append("STATIONARITY ANALYSIS")
            report.append("-" * 40)
            
            report.append(f"ADF Test p-value: {self.results['stationarity_results']['adf_p_value']:.6f}")
            report.append(f"ADF Test Stationary: {self.results['stationarity_results']['adf_is_stationary']}")
            
            if 'kpss_p_value' in self.results['stationarity_results']:
                report.append(f"KPSS Test p-value: {self.results['stationarity_results']['kpss_p_value']:.6f}")
                report.append(f"KPSS Test Stationary: {self.results['stationarity_results']['kpss_is_stationary']}")
                
            if 'conflicting_results' in self.results['stationarity_results']:
                report.append(f"Conflicting Results: {self.results['stationarity_results']['conflicting_results']}")
                
            report.append("")
            
        # Add purge/embargo optimization results
        if self.results['optimal_purge_params']:
            report.append("-" * 40)
            report.append("PURGE/EMBARGO OPTIMIZATION")
            report.append("-" * 40)
            
            best_params = self.results['optimal_purge_params']['best_params']
            best_score = self.results['optimal_purge_params']['best_score']
            
            report.append(f"Optimal Purge Period: {best_params['purge_period']}")
            report.append(f"Optimal Embargo Period: {best_params['embargo_period']}")
            report.append(f"Best Score: {best_score:.6f}")
            report.append("")
            
        # Add performance decay results
        if self.results['performance_decay']:
            report.append("-" * 40)
            report.append("PERFORMANCE DECAY ANALYSIS")
            report.append("-" * 40)
            
            decay_rates = self.results['performance_decay']['decay_rates']
            
            for metric, stats in decay_rates.items():
                report.append(f"{metric.upper()}:")
                report.append(f"  Decay Rate (Slope): {stats['slope']:.6f}")
                report.append(f"  R-squared: {stats['r_squared']:.6f}")
                report.append(f"  p-value: {stats['p_value']:.6f}")
                report.append(f"  Significant Decay: {stats['significant']}")
                report.append("")
                
        # Add regime performance results
        if self.results['regime_performance']:
            report.append("-" * 40)
            report.append("REGIME PERFORMANCE ANALYSIS")
            report.append("-" * 40)
            
            # Define regime names
            regime_names = {
                0: "Normal",
                1: "Bullish Volatile",
                2: "Bearish Volatile",
                3: "Bullish Stable",
                4: "Bearish Stable"
            }
            
            for regime, metrics in self.results['regime_performance'].items():
                regime_name = regime_names.get(regime, f"Regime {regime}")
                report.append(f"{regime_name}:")
                
                for metric, value in metrics.items():
                    report.append(f"  {metric.upper()}: {value:.6f}")
                    
                report.append("")
                
        # Add PBO stats
        if self.results['pbo_stats'] is not None:
            report.append("-" * 40)
            report.append("PROBABILITY OF BACKTEST OVERFITTING")
            report.append("-" * 40)
            
            report.append(f"PBO: {self.results['pbo_stats']:.6f}")
            
            if self.results['pbo_stats'] > 0.5:
                report.append("WARNING: PBO > 0.5 indicates high probability of backtest overfitting!")
            elif self.results['pbo_stats'] > 0.3:
                report.append("CAUTION: PBO > 0.3 suggests moderate risk of backtest overfitting.")
            else:
                report.append("GOOD: PBO < 0.3 suggests low risk of backtest overfitting.")
                
            report.append("")
            
        # Add Monte Carlo results
        if self.results['monte_carlo_results']:
            report.append("-" * 40)
            report.append("MONTE CARLO ROBUSTNESS ASSESSMENT")
            report.append("-" * 40)
            
            for metric, stats in self.results['monte_carlo_results'].items():
                report.append(f"{metric.upper()}:")
                report.append(f"  Mean: {stats['mean']:.6f}")
                report.append(f"  Std: {stats['std']:.6f}")
                report.append(f"  Min: {stats['min']:.6f}")
                report.append(f"  Max: {stats['max']:.6f}")
                report.append(f"  5th Percentile: {stats['5th_percentile']:.6f}")
                report.append(f"  95th Percentile: {stats['95th_percentile']:.6f}")
                report.append("")
                
        # Join report
        report_text = "\n".join(report)
        
        # Save if filename provided
        if filename:
            with open(filename, 'w') as f:
                f.write(report_text)
                
        return report_text