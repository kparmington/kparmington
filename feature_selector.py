"""
Improved Financial Features Framework

This module implements various feature engineering techniques for financial time series data,
with focus on market regimes, technical indicators, volatility modeling, and feature selection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass, field
import warnings

# Technical libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from itertools import permutations
# Add these to the existing imports
import numba
from numba import njit, prange
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
# Technical analysis libraries
import ta
import talib
import pywt
import ruptures
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import shap
from finta import TA as finta_TA
import tulipy as ti

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class FeatureConfig:
    """Configuration parameters for feature engineering"""
    # Market regime parameters
    n_regimes: int = 4
    hmm_components: int = 3
    regime_windows: List[int] = field(default_factory=lambda: [20, 50, 200])
    
    # Technical indicator parameters
    timeframes: List[int] = field(default_factory=lambda: [5, 20, 50, 200])
    
    # Volatility modeling parameters
    garch_order: Tuple[int, int, int] = (1, 1, 1)
    vol_windows: List[int] = field(default_factory=lambda: [5, 20, 60])
    
    # Time series parameters
    wavelet_family: str = 'db4'
    wavelet_levels: int = 5
    frac_diff_d: float = 0.4
    perm_entropy_window: int = 100  # Add this line
    perm_entropy_order: int = 3     # Add this line
    rolling_pca_window: int = 100   # Add this line
    coint_window: int = 120         # Add this line
    
    # Calendar parameters
    market_hours: Tuple[int, int] = (9, 16)
    
    # Feature selection parameters
    cv_folds: int = 5
    purge_overlap: int = 10
    n_estimators: int = 100
    random_state: int = 42
    max_features: int = 50
    
    # Window sizes
    standard_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    correlation_threshold: float = 0.7
    
    # Numerical stability
    epsilon: float = 1e-8


class MarketRegimeDetector:
    """
    Implements various market regime detection algorithms to identify structural
    context for feature engineering and model selection.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the market regime detector with configuration parameters.
        
        Args:
            config: Configuration parameters. If None, uses default configuration.
        """
        self.config = config or FeatureConfig()
        self.n_regimes = self.config.n_regimes
        self.hmm_components = self.config.hmm_components
        self.regime_windows = self.config.regime_windows
        self.models = {}
        self.regime_labels = {}
        self.transitions = {}
    
    def detect_kmeans_regimes(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Detect market regimes using K-means clustering on selected features.
        
        Args:
            data: DataFrame containing price/volatility data
            features: List of feature columns to use for clustering
            
        Returns:
            Array of regime labels
        """
        X = data[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.config.random_state)
        labels = kmeans.fit_predict(X_scaled)
        
        self.models['kmeans'] = kmeans
        self.regime_labels['kmeans'] = labels
        return labels
    
    def detect_hmm_regimes(self, data: pd.DataFrame, feature: str) -> np.ndarray:
        """
        Detect market regimes using Hidden Markov Models.
        
        Args:
            data: DataFrame containing price/volatility data
            feature: Feature column to use for HMM (typically returns)
            
        Returns:
            Array of regime labels
        """
        try:
            from hmmlearn import hmm
            
            # Reshape data for HMM (requires 2D array with n_samples, n_features)
            X = data[feature].values.reshape(-1, 1)
            
            # Initialize and fit HMM
            model = hmm.GaussianHMM(n_components=self.hmm_components, 
                               covariance_type="full", 
                               n_iter=1000,
                               random_state=self.config.random_state)
            model.fit(X)
            
            # Predict hidden states
            hidden_states = model.predict(X)
            
            self.models['hmm'] = model
            self.regime_labels['hmm'] = hidden_states
            
            # Calculate transition probabilities
            self.transitions['hmm'] = self._calculate_transition_probs(hidden_states)
            
            return hidden_states
        except ImportError:
            logger.warning("hmmlearn not installed. Using k-means for regime detection instead.")
            # Fallback to k-means with single feature
            return self.detect_kmeans_regimes(data, [feature])
    
    def detect_hierarchical_hmm_regimes(self, data: pd.DataFrame, 
                                  primary_feature: str,
                                  secondary_feature: str,
                                  n_primary_states: int = 2,
                                  n_secondary_states: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect hierarchical market regimes using nested HMMs.
        The first level captures major market states (bull/bear),
        while the second level captures sub-regimes within each major state.
        
        Args:
            data: DataFrame containing price/volatility data
            primary_feature: Feature column for primary regime (e.g., returns)
            secondary_feature: Feature column for secondary regime (e.g., volatility)
            n_primary_states: Number of primary regime states
            n_secondary_states: Number of secondary regime states per primary state
            
        Returns:
            Tuple of (primary_states, secondary_states) regime labels
        """
        try:
            from hmmlearn import hmm
            import joblib
            
            # Step 1: Fit primary HMM (e.g., trend regimes)
            X_primary = data[primary_feature].values.reshape(-1, 1)
            primary_model = hmm.GaussianHMM(n_components=n_primary_states, 
                                      covariance_type="full", 
                                      n_iter=1000,
                                      random_state=self.config.random_state)
            primary_model.fit(X_primary)
            primary_states = primary_model.predict(X_primary)
            
            # Step 2: For each primary state, fit a secondary HMM
            X_secondary = data[secondary_feature].values.reshape(-1, 1)
            secondary_states = np.zeros_like(primary_states)
            
            # Create separate models for each primary regime
            secondary_models = {}
            
            # Get indices for each primary state
            state_indices = {}
            for state in range(n_primary_states):
                state_indices[state] = np.where(primary_states == state)[0]
            
            # Define a function to fit secondary HMM for a specific regime
            def fit_secondary_hmm(state):
                indices = state_indices[state]
                
                if len(indices) > n_secondary_states * 5:  # Need enough data
                    # Extract data for this regime
                    X_sub = X_secondary[indices]
                    
                    # Fit secondary HMM for this regime
                    secondary_model = hmm.GaussianHMM(n_components=n_secondary_states,
                                              covariance_type="full",
                                              n_iter=1000,
                                              random_state=self.config.random_state)
                    secondary_model.fit(X_sub)
                    
                    # Predict sub-regimes
                    sub_states = secondary_model.predict(X_sub)
                    
                    # Create results tuple
                    return state, secondary_model, sub_states
                else:
                    # Not enough data
                    return state, None, np.zeros(len(indices))
            
            # Parallel processing for secondary HMMs
            try:
                results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(fit_secondary_hmm)(state) for state in range(n_primary_states)
                )
                
                # Process results
                for state, model, sub_states in results:
                    if model is not None:
                        secondary_models[state] = model
                    
                    # Assign sub-states to the original array
                    indices = state_indices[state]
                    secondary_states[indices] = sub_states + state * n_secondary_states
            except:
                # Fallback to sequential processing
                for state in range(n_primary_states):
                    state, model, sub_states = fit_secondary_hmm(state)
                    if model is not None:
                        secondary_models[state] = model
                    
                    # Assign sub-states to the original array
                    indices = state_indices[state]
                    secondary_states[indices] = sub_states + state * n_secondary_states
            
            # Store models and states
            self.models['hierarchical_hmm_primary'] = primary_model
            self.models['hierarchical_hmm_secondary'] = secondary_models
            self.regime_labels['hierarchical_hmm_primary'] = primary_states
            self.regime_labels['hierarchical_hmm_secondary'] = secondary_states
            
            # Calculate transition probabilities for primary states
            self.transitions['hierarchical_hmm_primary'] = self._calculate_transition_probs(primary_states)
            
            # Generate nested state labels (combining primary and secondary)
            nested_states = primary_states * n_secondary_states + (secondary_states % n_secondary_states)
            self.regime_labels['hierarchical_hmm_nested'] = nested_states
            
            return primary_states, secondary_states
            
        except ImportError:
            logger.warning("hmmlearn not installed. Hierarchical HMMs not available.")
            return np.zeros(len(data)), np.zeros(len(data))
    
    def detect_dtw_regimes(self, data: pd.DataFrame, 
                      feature: str,
                      pattern_length: int = 20,
                      n_clusters: int = 4,
                      step_size: int = 5) -> np.ndarray:
        """
        Detect market regimes using Dynamic Time Warping (DTW) to cluster
        similar price/indicator patterns regardless of exact timing.
        
        Args:
            data: DataFrame containing price/volatility data
            feature: Feature column to use for DTW clustering
            pattern_length: Length of pattern to consider for DTW
            n_clusters: Number of pattern clusters to identify
            step_size: Step size for overlapping patterns
            
        Returns:
            Array of regime labels based on DTW clustering
        """
        try:
            from tslearn.clustering import TimeSeriesKMeans
            from tslearn.preprocessing import TimeSeriesScalerMeanVariance
            
            # Extract the feature series
            series = data[feature].values
            
            # Create overlapping subsequences
            subsequences = []
            for i in range(0, len(series) - pattern_length + 1, step_size):
                subsequences.append(series[i:i+pattern_length])
            
            # Normalize subsequences
            scaler = TimeSeriesScalerMeanVariance()
            subsequences_scaled = scaler.fit_transform(
                np.array(subsequences).reshape(len(subsequences), pattern_length, 1)
            )
            
            # Apply DTW clustering
            dtw_km = TimeSeriesKMeans(n_clusters=n_clusters, 
                                  metric="dtw", 
                                  random_state=self.config.random_state,
                                  n_jobs=-1)
            subsequence_labels = dtw_km.fit_predict(subsequences_scaled)
            
            # Initialize regime labels array with -1 (no label)
            regime_labels = np.full(len(series), -1)
            
            # Map back to original time series
            for i, idx in enumerate(range(0, len(series) - pattern_length + 1, step_size)):
                # Assign the same label to all points in this pattern
                # For overlapping patterns, use the most recent label
                regime_labels[idx:idx+pattern_length] = subsequence_labels[i]
            
            # Fill any remaining unlabeled points using nearest valid label
            mask = regime_labels == -1
            if np.any(mask):
                valid_indices = np.where(~mask)[0]
                if len(valid_indices) > 0:
                    # For each unlabeled point, find the nearest labeled point
                    for i in np.where(mask)[0]:
                        # Find nearest valid index
                        nearest_idx = valid_indices[np.abs(valid_indices - i).argmin()]
                        regime_labels[i] = regime_labels[nearest_idx]
                else:
                    # No valid labels found, use default
                    regime_labels[mask] = 0
            
            # Store models and results
            self.models['dtw_kmeans'] = dtw_km
            self.regime_labels['dtw'] = regime_labels.astype(int)
            
            # Store pattern prototypes
            self.models['dtw_prototypes'] = dtw_km.cluster_centers_
            
            return regime_labels.astype(int)
            
        except ImportError:
            logger.warning("tslearn not installed. DTW clustering not available.")
            # Fallback to k-means with single feature
            return self.detect_kmeans_regimes(data, [feature])
    
    def detect_change_points(self, data: pd.DataFrame, feature: str, 
                           method: str = 'pelt', model: str = 'l2', 
                           min_size: int = 30) -> List[int]:
        """
        Detect change points in time series data.
        
        Args:
            data: DataFrame containing price/volatility data
            feature: Feature column to detect change points on
            method: Method for change point detection ('pelt', 'binseg', etc.)
            model: Cost model ('l1', 'l2', 'rbf', etc.)
            min_size: Minimum segment size
            
        Returns:
            List of change point indices
        """
        signal = data[feature].values
        
        # Initialize detector
        if method == 'pelt':
            algo = ruptures.Pelt(model=model, min_size=min_size)
        else:
            algo = ruptures.Binseg(model=model, min_size=min_size)
        
        # Fit and predict change points
        algo.fit(signal.reshape(-1, 1))
        change_points = algo.predict(pen=10)
        
        # Store for later use
        self.regime_labels['change_points'] = change_points
        
        # Create regime labels based on change points
        regimes = np.zeros(len(signal), dtype=int)
        current_regime = 0
        
        for i in range(len(change_points)-1):
            regimes[change_points[i]:change_points[i+1]] = current_regime
            current_regime += 1
            
        self.regime_labels['change_point_regimes'] = regimes
        
        return change_points
    
    def detect_markov_switching_regimes(self, data: pd.DataFrame, feature: str) -> np.ndarray:
        """
        Detect market regimes using Markov switching models.
        
        Args:
            data: DataFrame containing price/volatility data
            feature: Feature column to use for regime detection
            
        Returns:
            Array of regime probabilities
        """
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
            
            # Prepare data
            endog = data[feature].values
            
            # Fit Markov switching model (k_regimes=2 for simplicity)
            model = MarkovRegression(endog, k_regimes=2, trend='c', switching_variance=True)
            results = model.fit()
            
            # Get smoothed probabilities
            regime_probs = results.smoothed_marginal_probabilities
            regime_labels = np.argmax(regime_probs, axis=1)
            
            self.models['markov_switching'] = results
            self.regime_labels['markov_switching'] = regime_labels
            
            return regime_probs
        except:
            # Fallback if statsmodels markov_regression is not available
            logger.warning("MarkovRegression not available. Using simpler approach.")
            return self.detect_kmeans_regimes(data, [feature])
    
    def generate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features based on identified regimes.
        
        Args:
            data: Original DataFrame
            
        Returns:
            DataFrame with additional regime-based features
        """
        df_features = data.copy()
        
        # Add regime labels if they exist
        for regime_type, labels in self.regime_labels.items():
            if regime_type != 'change_points':  # Skip change_points as they're not per-bar
                if isinstance(labels, np.ndarray) and len(labels) == len(data):
                    df_features[f'regime_{regime_type}'] = labels
        
        # Add transition probabilities if they exist
        for regime_type, trans_probs in self.transitions.items():
            if isinstance(trans_probs, dict):
                for from_state, to_probs in trans_probs.items():
                    for to_state, prob in to_probs.items():
                        col_name = f'trans_prob_{regime_type}_{from_state}_to_{to_state}'
                        regime_idx = np.where(self.regime_labels[regime_type] == from_state)[0]
                        df_features[col_name] = 0
                        df_features.loc[regime_idx, col_name] = prob
        
        # Generate multi-timeframe regime features
        for window in self.regime_windows:
            if 'kmeans' in self.regime_labels:
                df_features[f'regime_kmeans_{window}'] = self._rolling_regime_mode(
                    self.regime_labels['kmeans'], window)
            
            if 'hmm' in self.regime_labels:
                df_features[f'regime_hmm_{window}'] = self._rolling_regime_mode(
                    self.regime_labels['hmm'], window)
        
        return df_features
    
    def _rolling_regime_mode(self, regimes: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mode of regimes over a specified window."""
        result = np.zeros_like(regimes)
        
        # Vectorized implementation using pandas rolling
        regime_series = pd.Series(regimes)
        
        # Custom mode function that handles edge cases
        def rolling_mode(x):
            if len(x) == 0:
                return 0
            mode_result = stats.mode(x, nan_policy='omit')
            return mode_result.mode[0]
        
        # Apply rolling mode
        rolling_result = regime_series.rolling(
            window=window, min_periods=1
        ).apply(rolling_mode)
        
        # Convert back to numpy array
        result = rolling_result.values
        
        return result
    
    def _calculate_transition_probs(self, states: np.ndarray) -> Dict:
        """
        Calculate transition probabilities between states.
        
        Args:
            states: Array of state labels
            
        Returns:
            Dictionary of transition probabilities
        """
        unique_states = np.unique(states)
        n_states = len(unique_states)
        
        # Count transitions
        transitions = {}
        
        # Initialize with zeros
        for s in unique_states:
            transitions[s] = {s2: 0 for s2 in unique_states}
        
        # Count state transitions
        for i in range(len(states)-1):
            from_state, to_state = states[i], states[i+1]
            transitions[from_state][to_state] += 1
        
        # Convert to probabilities
        for s in unique_states:
            total = sum(transitions[s].values())
            if total > 0:
                for s2 in unique_states:
                    transitions[s][s2] /= total
            else:
                # If no transitions from this state, assign uniform probability
                for s2 in unique_states:
                    transitions[s][s2] = 1.0 / n_states
        
        return transitions


class TechnicalIndicatorGenerator:
    """
    Generates a comprehensive set of technical indicators using TA library's
    built-in functions, with support for multiple timeframes.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the technical indicator generator.
        
        Args:
            config: Configuration parameters. If None, uses default configuration.
        """
        self.config = config or FeatureConfig()
        self.timeframes = self.config.timeframes
    
    def generate_indicators(self, data: pd.DataFrame, 
                          ohlcv_cols: Dict[str, str] = None) -> pd.DataFrame:
        """
        Generate a comprehensive set of technical indicators using TA library's
        built-in functions for efficiency.
        
        Args:
            data: DataFrame with OHLCV data
            ohlcv_cols: Dictionary mapping standard names to actual column names
                        {'open': 'Open', 'high': 'High', 'low': 'Low', 
                         'close': 'Close', 'volume': 'Volume'}
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Default column mapping if not provided
        if ohlcv_cols is None:
            ohlcv_cols = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
        
        # For each timeframe, apply all TA indicators
        for timeframe in self.timeframes:
            # Create a copy for this timeframe
            temp_df = df.copy()
            
            # Rename columns to match what TA library expects
            temp_df = temp_df.rename(columns={
                ohlcv_cols['open']: 'open',
                ohlcv_cols['high']: 'high',
                ohlcv_cols['low']: 'low',
                ohlcv_cols['close']: 'close',
                ohlcv_cols['volume']: 'volume' if ohlcv_cols['volume'] in temp_df.columns else 'volume'
            })
            
            # Check if volume column exists before applying indicators
            has_volume = 'volume' in temp_df.columns
            
            # Apply all TA indicators with the current timeframe
            temp_df = ta.add_all_ta_features(
                temp_df, 
                open="open", 
                high="high", 
                low="low", 
                close="close", 
                volume="volume" if has_volume else None,
                fillna=True,
                window=timeframe
            )
            
            # Rename columns to include timeframe
            for col in temp_df.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    df[f'{col}_{timeframe}'] = temp_df[col]
        
        # Add custom indicators not available in TA library
        self._add_custom_indicators(df, ohlcv_cols)
        
        # Add adaptive indicators that adjust based on volatility
        self._add_adaptive_indicators(df, ohlcv_cols)
        
        # Add fractal indicators
        self._add_fractal_indicators(df, ohlcv_cols['close'])
        
        # Add tail risk indicators
        self._add_tail_risk_indicators(df)
        
        # Generate consensus indicators
        self._add_consensus_indicators(df)
        
        # Add extended indicators from finta and tulipy
        self._add_extended_indicators(df, ohlcv_cols)
        
        # Create standardized versions of indicators
        self._standardize_indicators(df)
        
        return df
    
    def _add_custom_indicators(self, df: pd.DataFrame, ohlcv_cols: Dict[str, str]) -> None:
        """
        Add custom indicators not available in TA library.
        
        Args:
            df: DataFrame to add indicators to
            ohlcv_cols: Dictionary mapping standard column names
        """
        # Extract price data
        close_data = df[ohlcv_cols['close']].values
        high_data = df[ohlcv_cols['high']].values
        low_data = df[ohlcv_cols['low']].values
        open_data = df[ohlcv_cols['open']].values
        
        # Returns - avoid duplicating if they already exist
        if 'returns' not in df.columns:
            df['returns'] = df[ohlcv_cols['close']].pct_change()
        
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df[ohlcv_cols['close']]/df[ohlcv_cols['close']].shift(1))
        
        # Volatility measures
        df['high_low_range'] = df[ohlcv_cols['high']] - df[ohlcv_cols['low']]
        df['high_low_range_pct'] = df['high_low_range'] / df[ohlcv_cols['close']]
        
        # For Parkinson volatility estimate
        df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * 
                                (np.log(df[ohlcv_cols['high']]/df[ohlcv_cols['low']]))**2)
        
        # For Garman-Klass volatility estimate with protection against division by zero
        ln_hl = np.log(high_data/low_data)
        
        # Protect against zero or negative values
        valid_open = open_data > 0
        valid_close = close_data > 0
        
        ln_co = np.zeros_like(close_data)
        mask = valid_open & valid_close
        ln_co[mask] = np.log(close_data[mask]/open_data[mask])
        
        df['garman_klass_vol'] = np.sqrt(0.5 * ln_hl**2 - (2*np.log(2)-1) * ln_co**2)
        
        # Candle pattern recognition using TA-Lib (not in TA library)
        patterns = {
            'cdl_doji': talib.CDLDOJI,
            'cdl_hammer': talib.CDLHAMMER,
            'cdl_engulfing': talib.CDLENGULFING,
            'cdl_morning_star': talib.CDLMORNINGSTAR,
            'cdl_evening_star': talib.CDLEVENINGSTAR,
            'cdl_shooting_star': talib.CDLSHOOTINGSTAR,
            'cdl_hanging_man': talib.CDLHANGINGMAN,
            'cdl_harami': talib.CDLHARAMI,
            'cdl_marubozu': talib.CDLMARUBOZU
        }
        
        for pattern_name, pattern_func in patterns.items():
            df[pattern_name] = pattern_func(open_data, high_data, low_data, close_data)
    
    def _add_adaptive_indicators(self, df: pd.DataFrame, ohlcv_cols: Dict[str, str]) -> None:
        """
        Add indicators with adaptive timeframes based on volatility.
        
        Args:
            df: DataFrame to add indicators to
            ohlcv_cols: Column name mapping
        """
        # Extract required data
        close = df[ohlcv_cols['close']].values
        high = df[ohlcv_cols['high']].values
        low = df[ohlcv_cols['low']].values
        
        # Calculate ATR as volatility estimate
        atr = talib.ATR(high, low, close, timeperiod=14)
        
        # Normalize ATR by close price
        norm_atr = atr / close
        
        # Calculate volatility-adjusted timeframes
        vol_rank = pd.Series(norm_atr).rolling(50).rank(pct=True)
        
        # Volatility-adjusted SMA timeframe (shorter during low vol, longer during high vol)
        adaptive_timeframe = np.round(20 * (1 + vol_rank)).fillna(20).astype(int)
        adaptive_timeframe = np.maximum(5, np.minimum(50, adaptive_timeframe))
        
        # Apply adaptive SMA
        adaptive_sma = np.zeros_like(close)
        
        # Create Series for vectorized calculation
        close_series = pd.Series(close)
        
        # Calculate adaptive SMA
        for i in range(len(close)):
            if i < 5:  # Minimum lookback
                adaptive_sma[i] = np.nan
            else:
                lookback = min(i+1, adaptive_timeframe[i])
                # Use vectorized operation
                adaptive_sma[i] = close_series.iloc[i-lookback+1:i+1].mean()
        
        df['adaptive_sma'] = adaptive_sma
        df['adaptive_timeframe'] = adaptive_timeframe

    def _add_fractal_indicators(self, df: pd.DataFrame, close_data: np.ndarray) -> None:
        """
        Add fractal-based indicators to capture self-similarity.
        
        Args:
            df: DataFrame to add indicators to
            close_data: Close price array
        """
        # Hurst exponent (simplified version)
        def hurst_exponent(prices, lags=20):
            tau = []
            lagvec = []
            
            # Step through the different lags
            for lag in range(2, lags):
                # Construct price difference with lag
                pp = np.subtract(prices[lag:], prices[:-lag])
                # Calculate variance of difference
                lagvec.append(lag)
                tau.append(np.sqrt(np.std(pp)))
            
            # Calculate slope of log-log plot
            m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)
            hurst = m[0] * 2.0
            return hurst
        
        # Calculate Hurst exponent using rolling windows
        window = 100
        hurst_values = np.zeros_like(close_data)
        for i in range(window, len(close_data)):
            hurst_values[i] = hurst_exponent(close_data[i-window:i], lags=min(20, window//4))
        
        df['hurst_exponent'] = hurst_values
        
        # Fractal dimension indicator
        df['fractal_dimension'] = 2 - df['hurst_exponent']
        
        # Self-similarity score (higher when patterns are more fractal-like)
        df['self_similarity'] = np.where(
            (df['hurst_exponent'] > 0.5) & (df['hurst_exponent'] < 0.7),
            (df['hurst_exponent'] - 0.5) * 5, 
            np.where(df['hurst_exponent'] >= 0.7, 1.0, 0.0)
        )
    def _add_tail_risk_indicators(self, df: pd.DataFrame) -> None:
        """
        Add indicators to capture tail risk.
        
        Args:
            df: DataFrame to add indicators to
        """
        # Calculate historical VaR (Value at Risk) at different confidence levels
        returns = df['returns'].fillna(0).values
        
        # 95% VaR
        df['var_95'] = pd.Series(returns).rolling(window=252).quantile(0.05)
        
        # 99% VaR
        df['var_99'] = pd.Series(returns).rolling(window=252).quantile(0.01)
        
        # Conditional VaR (Expected Shortfall)
        def rolling_cvar(r, window=252, alpha=0.05):
            """Vectorized Conditional VaR calculation"""
            result = np.zeros_like(r)
            r_series = pd.Series(r)
            
            # Pre-compute quantile values for each window
            var_values = r_series.rolling(window).quantile(alpha)
            
            # For each window, calculate mean of values below VaR
            for i in range(window, len(r)):
                r_window = r[i-window:i]
                var = var_values.iloc[i]
                below_var = r_window[r_window <= var]
                result[i] = np.mean(below_var) if len(below_var) > 0 else var
            
            return result
        
        df['cvar_95'] = rolling_cvar(returns, window=252, alpha=0.05)
        
        # Tail risk ratio (CVaR / Volatility)
        rolling_vol = pd.Series(returns).rolling(window=252).std()
        df['tail_risk_ratio'] = df['cvar_95'] / rolling_vol
    
    def _add_consensus_indicators(self, df: pd.DataFrame) -> None:
        """
        Add consensus indicators that aggregate multiple signals.
        
        Args:
            df: DataFrame to add indicators to
        """
        # Identify indicator columns by type
        trend_indicators = [col for col in df.columns if any(x in col for x in 
                                                            ['trend_', 'adx_', 'sma_', 'ema_'])]
        
        oscillator_indicators = [col for col in df.columns if any(x in col for x in 
                                                                ['rsi_', 'stoch_', 'cci_', 'mom_'])]
        
        volume_indicators = [col for col in df.columns if any(x in col for x in 
                                                            ['volume_', 'obv_', 'mfi_', 'cmf_'])]
        
        # Trend consensus
        if trend_indicators:
            trend_scores = np.zeros(len(df))
            trend_weights = np.ones(len(trend_indicators)) / len(trend_indicators)
            
            for i, indicator in enumerate(trend_indicators):
                if 'sma_' in indicator or 'ema_' in indicator:
                    # Price above MA: positive trend, below MA: negative trend
                    if 'Close' in df.columns and indicator in df.columns:
                        trend_scores += trend_weights[i] * np.sign(df['Close'] - df[indicator])
                elif 'adx_' in indicator:
                    # ADX strength: higher values indicate stronger trend
                    if indicator in df.columns:
                        trend_scores += trend_weights[i] * (df[indicator] > 25).astype(int)
            
            # Normalize to [-1, 1]
            df['trend_consensus'] = np.clip(trend_scores, -1, 1)
        
        # Oscillator consensus
        if oscillator_indicators:
            oscillator_scores = np.zeros(len(df))
            oscillator_weights = np.ones(len(oscillator_indicators)) / len(oscillator_indicators)
            
            for i, indicator in enumerate(oscillator_indicators):
                if 'rsi_' in indicator:
                    # RSI > 70: overbought (-1), RSI < 30: oversold (+1)
                    oscillator_scores += oscillator_weights[i] * np.where(df[indicator] > 70, -1, 
                                                                       np.where(df[indicator] < 30, 1, 0))
                elif 'stoch_' in indicator:
                    # Stoch > 80: overbought (-1), Stoch < 20: oversold (+1)
                    oscillator_scores += oscillator_weights[i] * np.where(df[indicator] > 80, -1,
                                                                       np.where(df[indicator] < 20, 1, 0))
                elif 'cci_' in indicator:
                    # CCI > 100: overbought (-1), CCI < -100: oversold (+1)
                    oscillator_scores += oscillator_weights[i] * np.where(df[indicator] > 100, -1,
                                                                       np.where(df[indicator] < -100, 1, 0))
            
            # Normalize to [-1, 1]
            df['oscillator_consensus'] = np.clip(oscillator_scores, -1, 1)
        
        # Volume consensus
        if volume_indicators:
            volume_scores = np.zeros(len(df))
            volume_weights = np.ones(len(volume_indicators)) / len(volume_indicators)
            
            for i, indicator in enumerate(volume_indicators):
                if 'mfi_' in indicator:
                    # MFI > 80: overbought (-1), MFI < 20: oversold (+1)
                    volume_scores += volume_weights[i] * np.where(df[indicator] > 80, -1,
                                                               np.where(df[indicator] < 20, 1, 0))
                elif 'cmf_' in indicator:
                    # CMF > 0: positive, CMF < 0: negative
                    volume_scores += volume_weights[i] * np.sign(df[indicator])
            
            # Normalize to [-1, 1]
            df['volume_consensus'] = np.clip(volume_scores, -1, 1)
        
        # Overall consensus (combination of trend, oscillator, and volume)
        consensus_columns = ['trend_consensus', 'oscillator_consensus', 'volume_consensus']
        available_consensus = [col for col in consensus_columns if col in df.columns]
        
        if available_consensus:
            # Weighted average (trend: 50%, oscillator: 30%, volume: 20%)
            weights = {'trend_consensus': 0.5, 'oscillator_consensus': 0.3, 'volume_consensus': 0.2}
            
            # Adjust weights if some components are missing
            total_weight = sum(weights[col] for col in available_consensus)
            normalized_weights = {col: weights[col]/total_weight for col in available_consensus}
            
            df['market_consensus'] = sum(df[col] * normalized_weights[col] 
                                       for col in available_consensus)
    
    def _standardize_indicators(self, df: pd.DataFrame) -> None:
        """
        Create z-score normalized versions of indicators.
        
        Args:
            df: DataFrame to add normalized indicators to
        """
        # List of indicator categories to normalize
        indicator_prefixes = ['rsi_', 'macd_', 'stoch_', 'cci_', 'mfi_', 'willr_']
        
        for prefix in indicator_prefixes:
            # Find all columns with this prefix
            cols = [col for col in df.columns if col.startswith(prefix)]
            
            for col in cols:
                # Calculate rolling z-score with 100-day window
                df[f'zscore_{col}'] = df[col].rolling(window=100).apply(
                    lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0)
    def _add_extended_indicators(self, df: pd.DataFrame, ohlcv_cols: Dict[str, str]) -> None:
        """
        Add extended technical indicators from finta and tulipy libraries.
        
        Args:
            df: DataFrame to add indicators to
            ohlcv_cols: Dictionary mapping standard column names
        """
        # Create a DataFrame with the proper column names for finta
        finta_df = df.copy()
        finta_df.rename(columns={
            ohlcv_cols['open']: 'open',
            ohlcv_cols['high']: 'high',
            ohlcv_cols['low']: 'low',
            ohlcv_cols['close']: 'close',
            ohlcv_cols['volume'] if ohlcv_cols['volume'] in df.columns else 'volume': 'volume'
        }, inplace=True)
        
        # Add finta indicators with multiple timeframes
        for timeframe in self.timeframes:
            # Moving average / overlay indicators
            try:
                df[f'hma_{timeframe}'] = finta_TA.HMA(finta_df, timeframe)
                df[f'vidya_{timeframe}'] = finta_TA.VIDYA(finta_df, timeframe)
                df[f'vwma_{timeframe}'] = finta_TA.VWMA(finta_df, timeframe)
                df[f'wilders_{timeframe}'] = finta_TA.WILDERS(finta_df, timeframe)
                df[f'zlema_{timeframe}'] = finta_TA.ZLEMA(finta_df, timeframe)
            except Exception as e:
                warnings.warn(f"Error calculating finta MA indicators: {str(e)}")
            
            # Momentum / oscillator indicators
            try:
                df[f'ao_{timeframe}'] = finta_TA.AO(finta_df, timeframe)
                df[f'cvi_{timeframe}'] = finta_TA.CVI(finta_df, timeframe)
                df[f'dpo_{timeframe}'] = finta_TA.DPO(finta_df, timeframe)
                df[f'emv_{timeframe}'] = finta_TA.EMV(finta_df, timeframe)
                df[f'fisher_{timeframe}'] = finta_TA.FISHER(finta_df, timeframe)
                df[f'fosc_{timeframe}'] = finta_TA.FOSC(finta_df, timeframe)
                df[f'kvo_{timeframe}'] = finta_TA.KVO(finta_df, timeframe)
                df[f'marketfi_{timeframe}'] = finta_TA.MARKETFI(finta_df)  # No period param
                df[f'mass_{timeframe}'] = finta_TA.MASS(finta_df, timeframe)
                df[f'msw_{timeframe}'] = finta_TA.MSW(finta_df, timeframe)
                df[f'nvi_{timeframe}'] = finta_TA.NVI(finta_df)  # No period param
                df[f'pvi_{timeframe}'] = finta_TA.PVI(finta_df)  # No period param
                df[f'qstick_{timeframe}'] = finta_TA.QSTICK(finta_df, timeframe)
                df[f'vhf_{timeframe}'] = finta_TA.VHF(finta_df, timeframe)
                df[f'volatility_{timeframe}'] = finta_TA.VOLATILITY(finta_df, timeframe)
                df[f'vosc_{timeframe}'] = finta_TA.VOSC(finta_df, timeframe)
            except Exception as e:
                warnings.warn(f"Error calculating finta oscillator indicators: {str(e)}")
            
            # Directional movement set
            try:
                df[f'di_plus_{timeframe}'] = finta_TA.DI(finta_df, timeframe)["DI+"]
                df[f'di_minus_{timeframe}'] = finta_TA.DI(finta_df, timeframe)["DI-"]
                dm_result = finta_TA.DM(finta_df, timeframe)
                df[f'dm_plus_{timeframe}'] = dm_result["DM+"]
                df[f'dm_minus_{timeframe}'] = dm_result["DM-"]
            except Exception as e:
                warnings.warn(f"Error calculating finta directional indicators: {str(e)}")
            
            # Other price/vol indicators
            try:
                df[f'wad_{timeframe}'] = finta_TA.WAD(finta_df)  # No period param
            except Exception as e:
                warnings.warn(f"Error calculating finta price indicators: {str(e)}")
            
            # Regression / statistics
            try:
                df[f'linreg_{timeframe}'] = finta_TA.LINREG(finta_df, timeframe)
                df[f'linregintercept_{timeframe}'] = finta_TA.LINREGINTERCEPT(finta_df, timeframe)
                df[f'linregslope_{timeframe}'] = finta_TA.LINREGSLOPE(finta_df, timeframe)
            except Exception as e:
                warnings.warn(f"Error calculating finta regression indicators: {str(e)}")
            
            # Range / volatility (alias forms)
            try:
                df[f'tr_{timeframe}'] = finta_TA.TR(finta_df)  # No period param
            except Exception as e:
                warnings.warn(f"Error calculating finta range indicators: {str(e)}")
            
            # Parabolic SAR
            try:
                df[f'psar_{timeframe}'] = finta_TA.PSAR(finta_df)  # No period param
            except Exception as e:
                warnings.warn(f"Error calculating finta PSAR indicator: {str(e)}")
            
            # Price transforms
            try:
                df[f'wcprice_{timeframe}'] = finta_TA.WCPRICE(finta_df)  # No period param
            except Exception as e:
                warnings.warn(f"Error calculating finta price transforms: {str(e)}")

        # Add tulipy indicators
        # Extract OHLCV arrays for tulipy
        close_data = df[ohlcv_cols['close']].values
        high_data = df[ohlcv_cols['high']].values
        low_data = df[ohlcv_cols['low']].values
        open_data = df[ohlcv_cols['open']].values
        volume_data = df[ohlcv_cols['volume']].values if ohlcv_cols['volume'] in df.columns else None
        
        for timeframe in self.timeframes:
            # ADL - Accumulation/Distribution Line
            if volume_data is not None:
                try:
                    result = ti.adl(high_data, low_data, close_data, volume_data)
                    # Pad with NaNs to match original DataFrame length
                    pad_length = len(df) - len(result)
                    if pad_length > 0:
                        result = np.pad(result, (pad_length, 0), 'constant', constant_values=np.nan)
                    df[f'adl_{timeframe}'] = result
                except Exception as e:
                    warnings.warn(f"Error calculating tulipy ADL: {str(e)}")
            
            # CHAIKIN - Chaikin Oscillator
            if volume_data is not None:
                try:
                    result = ti.chaikin(high_data, low_data, close_data, volume_data, min(3, timeframe), timeframe)
                    pad_length = len(df) - len(result)
                    if pad_length > 0:
                        result = np.pad(result, (pad_length, 0), 'constant', constant_values=np.nan)
                    df[f'chaikin_osc_{timeframe}'] = result
                except Exception as e:
                    warnings.warn(f"Error calculating tulipy CHAIKIN: {str(e)}")
            
            # DMI - Directional Movement Index
            try:
                result = ti.dm(high_data, low_data, timeframe)
                pad_length = len(df) - len(result)
                if pad_length > 0:
                    result = np.pad(result, (pad_length, 0), 'constant', constant_values=np.nan)
                df[f'dm_tulipy_{timeframe}'] = result
            except Exception as e:
                warnings.warn(f"Error calculating tulipy DM: {str(e)}")
            
            try:
                result = ti.dx(high_data, low_data, close_data, timeframe)
                pad_length = len(df) - len(result)
                if pad_length > 0:
                    result = np.pad(result, (pad_length, 0), 'constant', constant_values=np.nan)
                df[f'dx_{timeframe}'] = result
            except Exception as e:
                warnings.warn(f"Error calculating tulipy DX: {str(e)}")
            
            # TYPPRICE - Typical Price
            try:
                result = ti.typprice(high_data, low_data, close_data)
                pad_length = len(df) - len(result)
                if pad_length > 0:
                    result = np.pad(result, (pad_length, 0), 'constant', constant_values=np.nan)
                df[f'typprice_{timeframe}'] = result
            except Exception as e:
                warnings.warn(f"Error calculating tulipy TYPPRICE: {str(e)}")
            
            # UO - Ultimate Oscillator
            try:
                result = ti.ultosc(high_data, low_data, close_data, 7, 14, timeframe)
                pad_length = len(df) - len(result)
                if pad_length > 0:
                    result = np.pad(result, (pad_length, 0), 'constant', constant_values=np.nan)
                df[f'uo_{timeframe}'] = result
            except Exception as e:
                warnings.warn(f"Error calculating tulipy UO: {str(e)}")
            
            # WILLR - Williams %R
            try:
                result = ti.willr(high_data, low_data, close_data, timeframe)
                pad_length = len(df) - len(result)
                if pad_length > 0:
                    result = np.pad(result, (pad_length, 0), 'constant', constant_values=np.nan)
                df[f'willr_tulipy_{timeframe}'] = result
            except Exception as e:
                warnings.warn(f"Error calculating tulipy WILLR: {str(e)}")
            
            # FISHER - Fisher Transform
            try:
                result = ti.fisher(high_data, low_data, timeframe)
                pad_length = len(df) - len(result)
                if pad_length > 0:
                    result = np.pad(result, (pad_length, 0), 'constant', constant_values=np.nan)
                df[f'fisher_tulipy_{timeframe}'] = result
            except Exception as e:
                warnings.warn(f"Error calculating tulipy FISHER: {str(e)}")


class VolatilityModeler:
    """
    Implements various volatility modeling approaches to capture
    risk dynamics and provide context for feature engineering.
    """
    def __init__(self, garch_order: Tuple[int, int, int] = (1, 1, 1),
                 vol_windows: List[int] = [5, 20, 60]):
        """
        Initialize the volatility modeler with configuration parameters.
        
        Args:
            garch_order: Order of GARCH model (p, q, o) where o is for asymmetric terms
            vol_windows: List of window sizes for rolling volatility calculations
        """
        self.garch_order = garch_order
        self.vol_windows = vol_windows
        self.models = {}
    
    def calculate_rolling_volatility(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate simple rolling volatility for multiple windows."""
        volatilities = {}
        
        for window in self.vol_windows:
            # Replace this:
            # vol = np.zeros_like(returns)
            # for i in range(len(returns)):
            #    if i < window:
            #        vol[i] = np.std(returns[:i+1]) if i > 0 else np.nan
            #    else:
            #        vol[i] = np.std(returns[i-window+1:i+1])
            
            # With this vectorized version:
            vol = pd.Series(returns).rolling(window=window, min_periods=1).std(ddof=0).to_numpy()
            
            volatilities[f'rolling_vol_{window}'] = vol
            # Annualized volatility (assuming daily returns)
            volatilities[f'rolling_vol_annual_{window}'] = vol * np.sqrt(252)
        
        return volatilities
    
    def calculate_parkinson_volatility(self, high: np.ndarray, low: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate Parkinson volatility estimator based on high-low range."""
        # Parkinson formula uses log(high/low)^2
        hl_ratio = np.log(high / low)
        
        # Replace manual loop with vectorized operation
        scale = 1.0 / (4.0 * np.log(2.0))
        parkinsons = np.sqrt(scale * pd.Series(hl_ratio**2).rolling(window=window, min_periods=1).mean().to_numpy())
        
        return parkinsons
    
    def calculate_garman_klass_volatility(self, open_: np.ndarray, high: np.ndarray, 
                                        low: np.ndarray, close: np.ndarray, 
                                        window: int = 20) -> np.ndarray:
        """Calculate Garman-Klass volatility estimator using OHLC data."""
        # Garman-Klass components
        hl_component = 0.5 * np.log(high / low)**2
        
        # Protect against division by zero
        valid_mask = (open_ > 0) & (close > 0)
        co_component = np.zeros_like(close)
        co_component[valid_mask] = (2 * np.log(2) - 1) * np.log(close[valid_mask] / open_[valid_mask])**2
        
        # GK formula for daily values
        gk_daily_var = hl_component - co_component
            
        # Calculate rolling GK volatility using vectorized operations
        gk_vol = np.sqrt(pd.Series(gk_daily_var).rolling(window=window, min_periods=1).mean().to_numpy())
        
        return gk_vol
        
    def fit_garch_model(self, returns: np.ndarray, variant: str = 'garch') -> None:
        """
        Fit a GARCH model to returns data.
        
        Args:
            returns: Array of return values
            variant: Type of GARCH model ('garch', 'egarch', 'tgarch', 'gjr')
            
        Returns:
            Fitted model object
        """
        try:
            from arch import arch_model
            
            # Remove NaNs from returns
            clean_returns = pd.Series(returns).fillna(0).values
            
            # Select GARCH variant
            if variant == 'egarch':
                model = arch_model(clean_returns, vol='EGARCH', p=self.garch_order[0], 
                                q=self.garch_order[1], o=self.garch_order[2])
            elif variant == 'tgarch':
                model = arch_model(clean_returns, vol='TGARCH', p=self.garch_order[0], 
                                q=self.garch_order[1], o=self.garch_order[2])
            elif variant == 'gjr':
                model = arch_model(clean_returns, vol='GARCH', p=self.garch_order[0], 
                                q=self.garch_order[1], o=self.garch_order[2])
            else:  # Default GARCH
                model = arch_model(clean_returns, vol='GARCH', p=self.garch_order[0], 
                                q=self.garch_order[1])
            
            # Fit model
            model_fit = model.fit(disp='off')
            
            # Store the model
            self.models[variant] = model_fit
            
            return model_fit
        except:
            warnings.warn(f"Error fitting {variant} model. Using rolling volatility instead.")
            return None
    
    def generate_volatility_features(self, data: pd.DataFrame, 
                                ohlcv_cols: Dict[str, str] = None) -> pd.DataFrame:
        """
        Generate comprehensive volatility features.
        
        Args:
            data: DataFrame with OHLCV data
            ohlcv_cols: Dictionary mapping standard names to actual column names
            
        Returns:
            DataFrame with additional volatility features
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Default column mapping if not provided
        if ohlcv_cols is None:
            ohlcv_cols = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
        
        # Extract required data
        close = df[ohlcv_cols['close']].values
        
        # Calculate log returns
        log_returns = np.diff(np.log(close))
        log_returns = np.insert(log_returns, 0, 0)  # Pad first value
        df['log_returns'] = log_returns
        
        # Rolling volatility for multiple windows
        volatilities = self.calculate_rolling_volatility(log_returns)
        for key, vol in volatilities.items():
            df[key] = vol
        
        # Calculate Parkinson volatility estimator
        if ohlcv_cols['high'] in df.columns and ohlcv_cols['low'] in df.columns:
            high = df[ohlcv_cols['high']].values
            low = df[ohlcv_cols['low']].values
            
            for window in self.vol_windows:
                df[f'parkinson_vol_{window}'] = self.calculate_parkinson_volatility(
                    high, low, window=window)
        
        # Calculate Garman-Klass volatility estimator
        if (ohlcv_cols['open'] in df.columns and ohlcv_cols['high'] in df.columns and
            ohlcv_cols['low'] in df.columns):
            
            open_ = df[ohlcv_cols['open']].values
            high = df[ohlcv_cols['high']].values
            low = df[ohlcv_cols['low']].values
            
            for window in self.vol_windows:
                df[f'garman_klass_vol_{window}'] = self.calculate_garman_klass_volatility(
                    open_, high, low, close, window=window)
        
        # Try to fit GARCH models
        try:
            # Basic GARCH
            garch_model = self.fit_garch_model(log_returns, variant='garch')
            if garch_model is not None:
                # Extract conditional volatility
                df['garch_vol'] = garch_model.conditional_volatility
                
                # Extract GARCH parameters
                df['garch_alpha'] = garch_model.params['alpha[1]']
                df['garch_beta'] = garch_model.params['beta[1]']
                
                # Forecast volatility for next period
                forecast = garch_model.forecast(horizon=1)
                df['garch_forecast'] = forecast.variance.values[-1, 0]
                
                # Calculate volatility of volatility
                vol_of_vol = pd.Series(garch_model.conditional_volatility).rolling(window=20).std()
                df['vol_of_vol'] = vol_of_vol
            
            # EGARCH for asymmetric volatility
            egarch_model = self.fit_garch_model(log_returns, variant='egarch')
            if egarch_model is not None:
                df['egarch_vol'] = egarch_model.conditional_volatility
                
                # Asymmetry parameter
                if 'gamma[1]' in egarch_model.params:
                    df['vol_asymmetry'] = egarch_model.params['gamma[1]']
        except:
            warnings.warn("GARCH modeling failed. Using only rolling volatility measures.")
        
        # Volatility regime features
        self._add_volatility_regime_features(df)
        
        # Volatility term structure (relationship between short and long-term volatility)
        if f'rolling_vol_{self.vol_windows[0]}' in df.columns and f'rolling_vol_{self.vol_windows[-1]}' in df.columns:
            df['vol_term_structure'] = df[f'rolling_vol_{self.vol_windows[0]}'] / df[f'rolling_vol_{self.vol_windows[-1]}']
        
        # Volatility seasonality (if timestamp data is available)
        if 'date' in df.columns or 'timestamp' in df.columns:
            self._add_volatility_seasonality(df)
        
        return df
    
    def _add_volatility_regime_features(self, df: pd.DataFrame) -> None:
        """
        Add features related to volatility regimes.
        
        Args:
            df: DataFrame to add features to
        """
        # Use rolling volatility as base
        if 'rolling_vol_20' in df.columns:
            vol_series = df['rolling_vol_20']
            
            # Calculate percentile of current volatility within its history
            df['vol_percentile'] = vol_series.rolling(window=252).rank(pct=True)
            
            # Discrete volatility regimes
            df['vol_regime'] = pd.qcut(
                df['vol_percentile'].fillna(0.5), 
                q=5, 
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
            
            # Numeric vol regime (for calculation purposes)
            df['vol_regime_numeric'] = pd.qcut(
                df['vol_percentile'].fillna(0.5), 
                q=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
            
            # Volatility acceleration (change in volatility)
            df['vol_acceleration'] = vol_series.pct_change(5)
            
            # Volatility deviation from trend
            vol_sma = vol_series.rolling(window=20).mean()
            df['vol_deviation'] = (vol_series - vol_sma) / vol_sma
    
    def _add_volatility_seasonality(self, df: pd.DataFrame) -> None:
        """
        Add features related to volatility seasonality patterns.
        
        Args:
            df: DataFrame to add features to
        """
        # Determine date column
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        
        # Ensure date is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                warnings.warn(f"Could not convert {date_col} to datetime. Skipping seasonality features.")
                return
        
        # Extract time components
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['day_of_month'] = df[date_col].dt.day
        
        # Calculate average volatility by day of week (if 'rolling_vol_20' exists)
        if 'rolling_vol_20' in df.columns:
            day_vol = df.groupby('day_of_week')['rolling_vol_20'].transform('mean')
            avg_vol = df['rolling_vol_20'].mean()
            
            # Daily volatility seasonality factor
            df['vol_day_factor'] = day_vol / avg_vol
            
            # Monthly volatility seasonality
            month_vol = df.groupby('month')['rolling_vol_20'].transform('mean')
            df['vol_month_factor'] = month_vol / avg_vol


class TimeSeriesTransformer:
    """
    Implements various time series transformations for feature engineering.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the time series transformer.
        
        Args:
            config: Configuration parameters. If None, uses default configuration.
        """
        self.config = config or FeatureConfig()
        self.wavelet_family = self.config.wavelet_family
        self.wavelet_levels = self.config.wavelet_levels
        self.frac_diff_d = self.config.frac_diff_d
    def apply_transformations(self, data: pd.DataFrame, 
                        target_col: str = 'Close') -> pd.DataFrame:
        """
        Apply various time series transformations to the data.
        
        Args:
            data: DataFrame containing the time series data
            target_col: Column to apply transformations to
            
        Returns:
            DataFrame with additional transformed features
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Get the target series
        series = df[target_col].values
        
        # Apply wavelet decomposition
        wavelet_features = self._apply_wavelet_transform(series)
        for name, values in wavelet_features.items():
            df[name] = values
        
        # Apply fractional differentiation
        df[f'frac_diff_{target_col}'] = self._fractional_differentiation(series, self.frac_diff_d)
        
        # Apply various non-linear transformations
        df[f'log_{target_col}'] = np.log(series)
        
        # Box-Cox transformation (only for positive values)
        if np.all(series > 0):
            df[f'boxcox_{target_col}'], _ = stats.boxcox(series)
        
        # Apply empirical mode decomposition (EMD)
        emd_features = self._apply_emd(series)
        for name, values in emd_features.items():
            df[name] = values
        
        # Calculate spectral entropy
        df['spectral_entropy'] = self._calculate_spectral_entropy(series)
        
        # Calculate permutation entropy
        df['perm_entropy'] = self._calculate_permutation_entropy(series)
        
        # Apply rolling PCA
        pca_features = self._apply_rolling_pca(df)
        for name, values in pca_features.items():
            df[name] = values
        
        # Time domain frequency analysis
        freq_features = self._apply_frequency_analysis(series)
        for name, values in freq_features.items():
            df[name] = values
            
        return df
    
    def _apply_wavelet_transform(self, series: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply wavelet decomposition for multi-resolution analysis.
        
        Args:
            series: Time series data array
            
        Returns:
            Dictionary of wavelet components
        """
        results = {}
        
        try:
            # Ensure the series length is appropriate for wavelet decomposition
            N = len(series)
            
            # Find the maximum level that can be computed
            max_level = pywt.dwt_max_level(N, self.wavelet_family)
            level = min(self.wavelet_levels, max_level)
            
            # Compute the wavelet decomposition
            coeffs = pywt.wavedec(series, self.wavelet_family, level=level)
            
            # Extract components (approximation + details)
            approx = coeffs[0]
            details = coeffs[1:]
            
            # Align lengths (wavelet transforms often change the length)
            # Use zero-padding for simplicity
            for i, detail in enumerate(details):
                pad_length = N - len(detail)
                if pad_length > 0:
                    details[i] = np.pad(detail, (0, pad_length), 'constant')
                else:
                    details[i] = detail[:N]
            
            pad_length = N - len(approx)
            if pad_length > 0:
                approx = np.pad(approx, (0, pad_length), 'constant')
            else:
                approx = approx[:N]
            
            # Store components
            results['wavelet_approx'] = approx
            for i, detail in enumerate(details):
                results[f'wavelet_detail_{i+1}'] = detail
                
            # Calculate wavelet energy features
            total_energy = np.sum([np.sum(np.square(d)) for d in coeffs])
            for i, coeff in enumerate(coeffs):
                energy = np.sum(np.square(coeff))
                
                # Add safeguard against division by zero
                if total_energy > self.config.epsilon:
                    results[f'wavelet_energy_level_{i}'] = np.ones(N) * (energy / total_energy)
                else:
                    results[f'wavelet_energy_level_{i}'] = np.ones(N) * (1.0 if i == 0 else 0.0)
        except Exception as e:
            warnings.warn(f"Wavelet decomposition failed: {str(e)}")
        
        return results
    
    def _fractional_differentiation(self, series: np.ndarray, d: float) -> np.ndarray:
        """
        Apply fractional differentiation to achieve stationarity while preserving memory.
        
        Args:
            series: Time series data array
            d: Fractional differentiation parameter
            
        Returns:
            Fractionally differenced series
        """
        # Generate weights
        weights = self._get_weights(d, len(series))
        
        # Use numba-accelerated function for the heavy computation
        return self._apply_weights_numba(series, weights)

    @staticmethod
    @njit(fastmath=True)
    def _get_weights(d, size):
        """Generate weights with protection against overflow for large series"""
        w = np.zeros(size)
        w[0] = 1
        
        # Set threshold for early truncation
        threshold = 1e-8
        
        # Generate weights until they become very small
        for k in range(1, size):
            w[k] = w[k-1] * (d - k + 1) / k
            
            # Early truncation when weights become very small
            if abs(w[k]) < threshold:
                # Fill remaining with zeros and exit
                w[k+1:] = 0
                break
        
        return w

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _apply_weights_numba(series, weights):
        """Apply weights using numba acceleration"""
        res = np.zeros_like(series)
        n = len(series)
        weight_len = len(weights)
        
        for i in prange(n):
            if i < weight_len:
                # For early positions, we don't have enough history
                # Use partial weights on available history
                for j in range(i+1):
                    res[i] += weights[j] * series[i-j]
            else:
                # For later positions, we can use the full weight window
                for j in range(weight_len):
                    res[i] += weights[j] * series[i-j]
        
        return res
    def _apply_emd(self, series: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply Empirical Mode Decomposition for adaptive signal analysis.
        
        Args:
            series: Time series data array
            
        Returns:
            Dictionary of IMF components
        """
        results = {}
        
        try:
            from PyEMD import EMD
            
            # Initialize EMD
            emd = EMD()
            
            # Extract IMFs
            imfs = emd(series)
            
            # Store IMFs (max 5 to avoid too many features)
            for i in range(min(5, imfs.shape[0])):
                results[f'emd_imf_{i+1}'] = imfs[i]
                
            # Calculate residual
            if imfs.shape[0] > 0:
                residual = series - np.sum(imfs, axis=0)
                results['emd_residual'] = residual
        except:
            # VMD as an alternative
            try:
                from vmdpy import VMD
                
                # VMD parameters
                alpha = 2000       # bandwidth constraint
                tau = 0            # noise-tolerance (no strict fidelity enforcement)
                K = 3              # 3 modes
                DC = 0             # no DC part imposed
                init = 1           # initialize omegas uniformly
                tol = 1e-7
                
                # Run VMD
                u, _, _ = VMD(series, alpha, tau, K, DC, init, tol)
                
                # Store VMD components
                for i in range(K):
                    results[f'vmd_mode_{i+1}'] = u[i, :]
            except:
                warnings.warn("Both EMD and VMD failed. Skipping mode decomposition features.")
        
        return results
    
    def _calculate_spectral_entropy(self, series: np.ndarray, window: int = 100) -> np.ndarray:
        """
        Calculate spectral entropy to measure predictability.
        
        Args:
            series: Time series data array
            window: Rolling window size
            
        Returns:
            Array of spectral entropy values
        """
        entropy = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < window:
                entropy[i] = np.nan
            else:
                # Get window of data
                window_data = series[i-window:i]
                
                # Calculate periodogram
                freq, power = np.fft.rfftfreq(window), np.abs(np.fft.rfft(window_data))**2
                
                # Normalize power spectrum
                power_norm = power / np.sum(power)
                
                # Calculate entropy (avoid log(0))
                power_norm = power_norm[power_norm > 0]
                entropy[i] = -np.sum(power_norm * np.log2(power_norm))
        
        return entropy
    
    def _calculate_permutation_entropy(self, series: np.ndarray, 
                                    window: int = None, 
                                    order: int = None, 
                                    delay: int = 1) -> np.ndarray:
        """
        Calculate permutation entropy to measure complexity.
        
        Args:
            series: Time series data array
            window: Rolling window size
            order: Permutation order
            delay: Delay between points
            
        Returns:
            Array of permutation entropy values
        """
        # Use config values if not explicitly provided
        window = window if window is not None else self.config.perm_entropy_window
        order = order if order is not None else self.config.perm_entropy_order
        
        # Create all possible permutation patterns
        possible_patterns = list(permutations(range(order)))
        
        # Create a lookup dictionary to avoid repeated searches
        pattern_to_idx = {p: i for i, p in enumerate(possible_patterns)}
        
        # Initialize entropy array
        entropy = np.zeros_like(series)
        
        for i in range(len(series)):
            if i < window:
                entropy[i] = np.nan
            else:
                # Get window of data
                window_data = series[i-window:i]
                
                # Count patterns
                pattern_counts = np.zeros(len(possible_patterns))
                
                for j in range(len(window_data) - (order-1)*delay):
                    # Extract consecutive points
                    pattern = window_data[j:j+(order*delay):delay]
                    
                    # Get permutation pattern
                    pattern_idx = tuple(np.argsort(pattern))
                    
                    # Use dictionary lookup instead of list index search
                    pattern_num = pattern_to_idx[pattern_idx]
                    
                    # Increment count
                    pattern_counts[pattern_num] += 1
                
                # Calculate entropy
                pattern_freqs = pattern_counts / np.sum(pattern_counts)
                pattern_freqs = pattern_freqs[pattern_freqs > 0]
                entropy[i] = -np.sum(pattern_freqs * np.log2(pattern_freqs))
        
        # Normalize by maximum entropy
        max_entropy = np.log2(np.math.factorial(order))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def _apply_rolling_pca(self, df: pd.DataFrame, 
                        window: int = 100, 
                        n_components: int = 3) -> Dict[str, np.ndarray]:
        """
        Apply rolling PCA to extract dynamic latent structures.
        
        Args:
            df: DataFrame with price features
            window: Rolling window size
            n_components: Number of PCA components to extract
            
        Returns:
            Dictionary of PCA component series
        """
        results = {}
        
        # Select numeric columns for PCA
        num_columns = df.select_dtypes(include=[np.number]).columns
        
        # Filter out columns with too many NaN values
        valid_columns = [col for col in num_columns 
                    if df[col].isna().sum() < len(df) * 0.3]
        
        if len(valid_columns) < 5:  # Need at least a few features for PCA to be meaningful
            return results
        
        # Prepare array for PCA components
        n_rows = len(df)
        pca_components = np.zeros((n_rows, n_components))
        pca_components.fill(np.nan)
        ipca = IncrementalPCA(n_components=n_components)
        # Apply rolling PCA
        for i in range(window, n_rows):
            # Extract window of data
            window_data = df[valid_columns].iloc[i-window:i].dropna(axis=1)
            
            # Only proceed if we have enough columns
            if len(window_data.columns) < 3:
                continue
                
            # Standardize
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(window_data)
            
            # Update the model with new data
            ipca.partial_fit(scaled_data)
            
            # Project the latest observation
            try:
                # Use double brackets to ensure DataFrame shape is preserved
                latest_point = scaler.transform(df[window_data.columns].iloc[[i]])
                projection = ipca.transform(latest_point)
                
                # Store components
                for j in range(min(n_components, projection.shape[1])):
                    pca_components[i, j] = projection[0, j]
            except Exception as e:
                # Skip if there's an error (like NaN values or shape mismatch)
                logger.warning(f"Error in PCA projection at index {i}: {str(e)}")
                continue
            
            # Store components
            for j in range(min(n_components, projection.shape[1])):
                pca_components[i, j] = projection[0, j]
            
            # Also store explained variance ratio for the first component
            if i == window:
                for j in range(min(n_components, len(ipca.explained_variance_ratio_))):
                    results[f'pca_exp_var_ratio_{j+1}'] = np.ones(n_rows) * ipca.explained_variance_ratio_[j]
        
        # Store PCA components
        for j in range(n_components):
            results[f'pca_component_{j+1}'] = pca_components[:, j]
        
        return results
    
    def _apply_frequency_analysis(self, series: np.ndarray, 
                            window: int = 100) -> Dict[str, np.ndarray]:
        """
        Apply time-domain frequency analysis.
        
        Args:
            series: Time series data array
            window: Rolling window size
            
        Returns:
            Dictionary of frequency domain features
        """
        results = {}
        n_rows = len(series)
        
        # Fourier transform-based metrics
        dominant_freq = np.zeros(n_rows)
        dominant_power = np.zeros(n_rows)
        spectral_density = np.zeros(n_rows)
        
        for i in range(window, n_rows):
            # Extract window of data
            window_data = series[i-window:i]
            
            # Apply FFT
            fft_result = np.fft.rfft(window_data)
            fft_freq = np.fft.rfftfreq(window)
            fft_amp = np.abs(fft_result)
            
            # Find dominant frequency
            dominant_idx = np.argmax(fft_amp[1:]) + 1  # Skip DC component
            dominant_freq[i] = fft_freq[dominant_idx]
            dominant_power[i] = fft_amp[dominant_idx]
            
            # Calculate spectral density (sum of squared amplitudes)
            spectral_density[i] = np.sum(fft_amp**2)
        
        results['dominant_freq'] = dominant_freq
        results['dominant_power'] = dominant_power
        results['spectral_density'] = spectral_density
        
        return results
class CalendarFeatureGenerator:
    """
    Generates features related to calendar effects, seasonality,
    and market-specific time patterns.
    """
    def __init__(self, known_holidays: List[str] = None, 
                market_hours: Tuple[int, int] = (9, 16)):
        """
        Initialize the calendar feature generator.
        
        Args:
            known_holidays: List of known holiday dates (YYYY-MM-DD format)
            market_hours: Tuple of (open_hour, close_hour) in 24-hour format
        """
        self.known_holidays = known_holidays if known_holidays else []
        self.market_hours = market_hours
    
    def generate_calendar_features(self, data: pd.DataFrame, 
                                date_col: str = 'date') -> pd.DataFrame:
        """
        Generate calendar and time-based features.
        
        Args:
            data: DataFrame with date information
            date_col: Name of the date column
            
        Returns:
            DataFrame with additional calendar features
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure date column is datetime type
        if date_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
            
            # Extract basic date components
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['day_of_month'] = df[date_col].dt.day
            df['month'] = df[date_col].dt.month
            df['quarter'] = df[date_col].dt.quarter
            df['year'] = df[date_col].dt.year
            df['week_of_year'] = df[date_col].dt.isocalendar().week
            
            # Create cyclical encodings for temporal features (sin/cos transformations)
            # These preserve the cyclical nature of time features
            
            # Day of week (0-6) -> cyclical
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Month (1-12) -> cyclical
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Day of month (1-31) -> cyclical
            df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
            df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
            
            # Check for holidays
            self._add_holiday_features(df, date_col)
            
            # Add market microstructure features
            if 'time' in df.columns:
                self._add_intraday_features(df, 'time')
            elif hasattr(df[date_col].dt, 'time'):
                df['time'] = df[date_col].dt.time
                self._add_intraday_features(df, 'time')
            
            # Add FOMC meeting features
            self._add_fomc_features(df, date_col)
            
            # Add quarter-end effects
            self._add_quarter_end_features(df, date_col)
            
            # Add pre/post holiday effects
            self._add_holiday_effect_features(df, date_col)
        
        return df
    
    def _add_holiday_features(self, df: pd.DataFrame, date_col: str) -> None:
        """
        Add features for holidays and days around holidays.
        
        Args:
            df: DataFrame to add features to
            date_col: Name of the date column
        """
        try:
            from pandas.tseries.holiday import USFederalHolidayCalendar
            
            # Create US holiday calendar
            cal = USFederalHolidayCalendar()
            
            # Get holidays for the date range in the data
            start_date = df[date_col].min()
            end_date = df[date_col].max()
            holidays = cal.holidays(start=start_date, end=end_date)
            
            # Mark holidays
            df['is_holiday'] = df[date_col].isin(holidays).astype(int)
            
            # Mark days before and after holidays
            df['is_day_before_holiday'] = df[date_col].isin(
                [holiday - pd.Timedelta(days=1) for holiday in holidays]).astype(int)
            
            df['is_day_after_holiday'] = df[date_col].isin(
                [holiday + pd.Timedelta(days=1) for holiday in holidays]).astype(int)
            
            # Add custom holidays if provided
            if self.known_holidays:
                custom_holidays = pd.to_datetime(self.known_holidays)
                df['is_custom_holiday'] = df[date_col].isin(custom_holidays).astype(int)
        except:
            warnings.warn("Failed to add holiday features. Using basic day-of-week only.")
    
    def _add_intraday_features(self, df: pd.DataFrame, time_col: str) -> None:
        """
        Add intraday and market microstructure features.
        
        Args:
            df: DataFrame to add features to
            time_col: Name of the time column
        """
        try:
            # Convert time to datetime or extract hour and minute
            if isinstance(df[time_col].iloc[0], str):
                df['hour'] = pd.to_datetime(df[time_col]).dt.hour
                df['minute'] = pd.to_datetime(df[time_col]).dt.minute
            elif hasattr(df[time_col].iloc[0], 'hour'):
                df['hour'] = df[time_col].apply(lambda x: x.hour)
                df['minute'] = df[time_col].apply(lambda x: x.minute)
            
            # Create normalized time of day (0 to 1)
            df['time_of_day'] = (df['hour'] * 60 + df['minute']) / (24 * 60)
            
            # Create cyclical encoding for time of day
            df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day'])
            df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day'])
            
            # Market open and close indicators
            market_open_hour, market_close_hour = self.market_hours
            
            # First 30 minutes after market open (often volatile)
            df['market_open_30min'] = ((df['hour'] == market_open_hour) & 
                                    (df['minute'] < 30)).astype(int)
            
            # Last 30 minutes before market close (often volatile)
            df['market_close_30min'] = ((df['hour'] == market_close_hour - 1) & 
                                     (df['minute'] >= 30)).astype(int)
            
            # Lunch hour (often has lower volatility)
            df['lunch_hour'] = ((df['hour'] >= 12) & (df['hour'] < 13)).astype(int)
            
            # Create U-shaped volume pattern feature (typical in many markets)
            t = df['time_of_day']
            market_open = market_open_hour / 24
            market_close = market_close_hour / 24
            market_mid = (market_open + market_close) / 2
            
            df['u_shape_pattern'] = 1 - np.minimum(
                abs(t - market_open) / abs(market_mid - market_open),
                abs(t - market_close) / abs(market_mid - market_close)
            )
            
            # If volume data exists, add volume-weighted time features
            if 'Volume' in df.columns:
                # Calculate volume percentile within the day
                if 'date' in df.columns:
                    # If we have a separate date column, use that
                    df['volume_percentile'] = df.groupby(df['date'])['Volume'].transform(
                        lambda x: x.rank(pct=True))
                elif hasattr(df[time_col], 'dt') and hasattr(df[time_col].dt, 'date'):
                    # If time_col has datetime objects with dt accessor
                    df['volume_percentile'] = df.groupby(df[time_col].dt.date)['Volume'].transform(
                        lambda x: x.rank(pct=True))
                else:
                    # We have time but no date, can't group by day
                    df['volume_percentile'] = df['Volume'].rank(pct=True)
                    warnings.warn("Could not group volume percentile by date - using overall percentile instead")
        except:
            warnings.warn("Failed to add intraday features.")
    
    def _add_fomc_features(self, df: pd.DataFrame, date_col: str) -> None:
        """
        Add features related to Federal Reserve FOMC meetings.
        
        Args:
            df: DataFrame to add features to
            date_col: Name of the date column
        """
        # List of common FOMC meeting months (typically 8 meetings per year)
        fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
        
        # Approximate FOMC meeting dates (3rd week of the month, usually Tuesday-Wednesday)
        df['fomc_month'] = df['month'].isin(fomc_months).astype(int)
        df['likely_fomc_week'] = ((df['month'].isin(fomc_months)) & 
                               (df['day_of_month'] >= 15) & 
                               (df['day_of_month'] <= 22)).astype(int)
    
    def _add_quarter_end_features(self, df: pd.DataFrame, date_col: str) -> None:
        """
        Add features related to quarter-end effects.
        
        Args:
            df: DataFrame to add features to
            date_col: Name of the date column
        """
        # Get month and day
        df['is_quarter_end'] = ((df['month'].isin([3, 6, 9, 12])) & 
                             (df['day_of_month'] >= 25)).astype(int)
        
        # Last trading day of the month
        df['last_5_days_of_month'] = (df['day_of_month'] >= 
                                  df.groupby(['year', 'month'])['day_of_month'].transform('max') - 5).astype(int)
        
        # First 3 days of the month (often show different behavior)
        df['first_3_days_of_month'] = (df['day_of_month'] <= 3).astype(int)
        
        # Month end window (potentially important for rebalancing)
        df['month_end_window'] = ((df['day_of_month'] >= 25) | 
                               (df['day_of_month'] <= 3)).astype(int)
    
    def _add_holiday_effect_features(self, df: pd.DataFrame, date_col: str) -> None:
        """
        Add features related to market anomalies around holidays.
        
        Args:
            df: DataFrame to add features to
            date_col: Name of the date column
        """
        # Tax-related seasonality (e.g., tax-loss harvesting in December)
        df['december_tax_effect'] = ((df['month'] == 12) & 
                                  (df['day_of_month'] >= 15)).astype(int)
        
        # January effect (early January often positive for small caps)
        df['january_effect'] = ((df['month'] == 1) & 
                             (df['day_of_month'] <= 15)).astype(int)
        
        # Summer months (often lower volatility)
        df['summer_months'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Halloween effect ("Sell in May and go away")
        df['winter_months'] = df['month'].isin([11, 12, 1, 2, 3, 4]).astype(int)


class CrossAssetFeatureGenerator:
    """
    Generates features based on relationships between multiple assets,
    sectors, indices, and macroeconomic indicators.
    """
    def __init__(self, reference_assets: Dict[str, pd.DataFrame] = None):
        """
        Initialize the cross-asset feature generator.
        
        Args:
            reference_assets: Dictionary of reference asset DataFrames
                             {'SPY': spy_df, 'VIX': vix_df, ...}
        """
        self.reference_assets = reference_assets if reference_assets else {}
    
    def generate_cross_asset_features(self, data: pd.DataFrame, 
                                    date_col: str = 'date',
                                    close_col: str = 'Close') -> pd.DataFrame:
        """
        Generate features based on relationships with other assets.
        
        Args:
            data: DataFrame with price data for the target asset
            date_col: Name of the date column
            close_col: Name of the close price column
            
        Returns:
            DataFrame with additional cross-asset features
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Add beta and correlation features if reference assets are provided
        if self.reference_assets:
            for asset_name, asset_df in self.reference_assets.items():
                # Ensure reference asset has datetime index
                ref_df = asset_df.copy()
                
                if date_col in ref_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(ref_df[date_col]):
                        ref_df[date_col] = pd.to_datetime(ref_df[date_col])
                    ref_df = ref_df.set_index(date_col)
                
                # Calculate relative strength
                merged_df = self._merge_and_align(df, ref_df, date_col, close_col)
                
                if merged_df is not None:
                    # Add the relative strength features
                    self._add_relative_strength_features(df, merged_df, asset_name)
                    
                    # Add correlation and beta features
                    self._add_correlation_features(df, merged_df, asset_name)
                    
                    # Add cointegration features
                    self._add_cointegration_features(df, merged_df, asset_name)
        
        # If no reference assets, or as a fallback, use sector ETFs
        if not self.reference_assets or len(self.reference_assets) == 0:
            warnings.warn("No reference assets provided. Cross-asset features will be limited.")
            
            # Add macro environment features if possible
            self._add_macro_environment_features(df)
        
        return df
    
    def _merge_and_align(self, target_df: pd.DataFrame, 
                      reference_df: pd.DataFrame,
                      date_col: str, 
                      close_col: str) -> Optional[pd.DataFrame]:
        """
        Merge and align the target and reference DataFrames.
        
        Args:
            target_df: DataFrame with target asset data
            reference_df: DataFrame with reference asset data
            date_col: Name of the date column
            close_col: Name of the close price column
            
        Returns:
            Merged DataFrame with aligned dates or None if merge fails
        """
        try:
            # Create a copy of the target with just date and close
            target_subset = target_df[[date_col, close_col]].copy()
            target_subset.rename(columns={close_col: 'target_close'}, inplace=True)
            
            # Get the close column from reference
            if close_col in reference_df.columns:
                ref_close = close_col
            elif 'Close' in reference_df.columns:
                ref_close = 'Close'
            elif 'close' in reference_df.columns:
                ref_close = 'close'
            else:
                # Try to find a numeric column if no obvious close column
                num_cols = reference_df.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    ref_close = num_cols[0]
                else:
                    return None
            
            # Create a reference subset with just date and close
            if reference_df.index.name is not None and date_col == reference_df.index.name:
                reference_df = reference_df.reset_index()
            
            reference_subset = reference_df[[date_col, ref_close]].copy()
            reference_subset.rename(columns={ref_close: 'ref_close'}, inplace=True)
            
            # Merge on date
            merged = pd.merge(target_subset, reference_subset, on=date_col, how='left')
            
            # Forward fill any missing reference data
            merged['ref_close'] = merged['ref_close'].fillna(method='ffill')
            
            return merged
        except Exception as e:
            warnings.warn(f"Failed to merge with reference asset: {str(e)}")
            return None
    
    def _add_relative_strength_features(self, df: pd.DataFrame, 
                                     merged_df: pd.DataFrame, 
                                     asset_name: str) -> None:
        """
        Add relative strength features between target and reference asset.
        
        Args:
            df: Original DataFrame to add features to
            merged_df: Merged DataFrame with target and reference prices
            asset_name: Name of the reference asset
        """
        # Calculate price ratio
        merged_df['price_ratio'] = merged_df['target_close'] / merged_df['ref_close']
        
        # Calculate returns for both series
        merged_df['target_return'] = merged_df['target_close'].pct_change()
        merged_df['ref_return'] = merged_df['ref_close'].pct_change()
        
        # Calculate relative return
        merged_df['relative_return'] = merged_df['target_return'] - merged_df['ref_return']
        
        # Calculate rolling relative strength indicators
        for window in [5, 20, 60]:
            # Relative strength (ratio of cumulative returns)
            merged_df[f'rel_strength_{window}'] = (
                (1 + merged_df['target_return']).rolling(window).apply(lambda x: np.prod(x), raw=True) /
                (1 + merged_df['ref_return']).rolling(window).apply(lambda x: np.prod(x), raw=True)
            )
            
            # Relative strength index (normalized)
            rs = merged_df[f'rel_strength_{window}']
            merged_df[f'rel_strength_norm_{window}'] = (rs - rs.rolling(window).min()) / (
                rs.rolling(window).max() - rs.rolling(window).min())
        
        # Add these features to the original DataFrame
        for col in ['price_ratio', 'relative_return'] + [
            f'rel_strength_{w}' for w in [5, 20, 60]] + [
            f'rel_strength_norm_{w}' for w in [5, 20, 60]]:
            df[f'{col}_vs_{asset_name}'] = merged_df[col].values
    
    def _add_correlation_features(self, df: pd.DataFrame, 
                               merged_df: pd.DataFrame, 
                               asset_name: str) -> None:
        """
        Add correlation and beta features between target and reference asset.
        
        Args:
            df: Original DataFrame to add features to
            merged_df: Merged DataFrame with target and reference prices
            asset_name: Name of the reference asset
        """
        # Calculate rolling correlation
        for window in [20, 60, 120]:
            merged_df[f'correlation_{window}'] = merged_df['target_return'].rolling(
                window).corr(merged_df['ref_return'])
        
        # Calculate rolling beta (regression slope)
        for window in [20, 60, 120]:
            # Function to calculate beta
            def rolling_beta(returns):
                if len(returns) < 2 or returns.isna().any():
                    return np.nan
                
                y = returns.iloc[:, 0].values  # Target returns
                x = returns.iloc[:, 1].values  # Reference returns
                
                # Add constant to x for regression
                x_with_const = np.column_stack([np.ones(len(x)), x])
                
                try:
                    # OLS regression to get beta
                    beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0][1]
                    return beta
                except:
                    return np.nan
            
            # Apply rolling beta calculation
            merged_df[f'beta_{window}'] = merged_df[['target_return', 'ref_return']].rolling(
                window).apply(rolling_beta, raw=False)
            
            # For high-vol vs. low-vol regime conditional beta
            vol = merged_df['ref_return'].rolling(window).std()
            high_vol_mask = vol > vol.rolling(window).median()
            
            # Split beta calculation for high/low vol regimes
            merged_df[f'beta_highvol_{window}'] = np.nan
            merged_df[f'beta_lowvol_{window}'] = np.nan
            
            # Calculate regime-specific betas
            for i in range(window, len(merged_df)):
                regime_window = merged_df.iloc[i-window:i]
                
                # High vol regime
                high_vol_window = regime_window[high_vol_mask.iloc[i-window:i]]
                if len(high_vol_window) >= 10:  # Ensure enough data points
                    high_vol_beta = rolling_beta(high_vol_window[['target_return', 'ref_return']])
                    merged_df.iloc[i, merged_df.columns.get_loc(f'beta_highvol_{window}')] = high_vol_beta
                
                # Low vol regime
                low_vol_window = regime_window[~high_vol_mask.iloc[i-window:i]]
                if len(low_vol_window) >= 10:  # Ensure enough data points
                    low_vol_beta = rolling_beta(low_vol_window[['target_return', 'ref_return']])
                    merged_df.iloc[i, merged_df.columns.get_loc(f'beta_lowvol_{window}')] = low_vol_beta
        
        # Add these features to the original DataFrame
        for col in [f'correlation_{w}' for w in [20, 60, 120]] + [
            f'beta_{w}' for w in [20, 60, 120]] + [
            f'beta_highvol_{w}' for w in [20, 60, 120]] + [
            f'beta_lowvol_{w}' for w in [20, 60, 120]]:
            df[f'{col}_vs_{asset_name}'] = merged_df[col].values
    
    def _add_cointegration_features(self, df: pd.DataFrame, 
                                    merged_df: pd.DataFrame, 
                                    asset_name: str) -> None:
            """
            Add cointegration-based features between target and reference asset.
            
            Args:
                df: Original DataFrame to add features to
                merged_df: Merged DataFrame with target and reference prices
                asset_name: Name of the reference asset
            """
            try:
                # Calculate rolling cointegration and spread metrics
                window = 120  # Use at least 120 days for meaningful cointegration
                
                if len(merged_df) > window:
                    # Calculate spread
                    merged_df['log_target'] = np.log(merged_df['target_close'])
                    merged_df['log_ref'] = np.log(merged_df['ref_close'])
                    
                    # Initialize arrays for results
                    coint_pvalue = np.full(len(merged_df), np.nan)
                    spread_zscore = np.full(len(merged_df), np.nan)
                    hedge_ratio = np.full(len(merged_df), np.nan)
                    
                    # Rolling calculation
                    for i in range(window, len(merged_df)):
                        # Get window of log prices
                        window_data = merged_df.iloc[i-window:i]
                        y = window_data['log_target'].values
                        x = window_data['log_ref'].values
                        
                        try:
                            # Calculate hedge ratio using regression
                            x_with_const = np.column_stack([np.ones(len(x)), x])
                            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                            
                            # Hedge ratio is the beta for x
                            hedge_ratio[i] = beta[1]
                            
                            # Calculate residuals (spread)
                            spread = y - (beta[0] + beta[1] * x)
                            
                            # Check for cointegration using ADF test
                            adf_result = adfuller(spread, maxlag=1, regression='c')
                            coint_pvalue[i] = adf_result[1]
                            
                            # Calculate z-score of current spread
                            current_spread = y[-1] - (beta[0] + beta[1] * x[-1])
                            spread_mean = np.mean(spread)
                            spread_std = np.std(spread)
                            
                            if spread_std > 0:
                                spread_zscore[i] = (current_spread - spread_mean) / spread_std
                        except:
                            pass
                    
                    # Add to merged DataFrame
                    merged_df['coint_pvalue'] = coint_pvalue
                    merged_df['spread_zscore'] = spread_zscore
                    merged_df['hedge_ratio'] = hedge_ratio
                    
                    # Add to original DataFrame
                    df[f'coint_pvalue_vs_{asset_name}'] = merged_df['coint_pvalue'].values
                    df[f'spread_zscore_vs_{asset_name}'] = merged_df['spread_zscore'].values
                    df[f'hedge_ratio_vs_{asset_name}'] = merged_df['hedge_ratio'].values
            except Exception as e:
                warnings.warn(f"Failed to calculate cointegration features: {str(e)}")
        
    def _add_macro_environment_features(self, df: pd.DataFrame) -> None:
        """
        Add features representing the macro environment.
        
        Args:
            df: DataFrame to add features to
        """
        # As a placeholder, we'll create synthetic features
        # In a real implementation, these would come from actual market data
        
        # Generate synthetic interest rate and yield curve features
        if 'date' in df.columns and 'year' in df.columns:
            # Number of business days in the year for each date
            business_days = np.minimum(252, np.maximum(230, 
                                                    (df['year'] - 2000) * 2 + 230))
            
            # Create synthetic interest rate cycles (simplified)
            rate_cycle = np.sin(2 * np.pi * np.arange(len(df)) / business_days / 5) * 2 + 3
            
            # Create synthetic yield curve slope
            yield_curve = np.sin(2 * np.pi * np.arange(len(df)) / business_days / 3)
            
            # Add to DataFrame
            df['interest_rate_proxy'] = rate_cycle
            df['yield_curve_slope_proxy'] = yield_curve
            
            # Create synthetic monetary policy cycle indicator
            df['tightening_cycle'] = (rate_cycle > np.roll(rate_cycle, 10)).astype(int)
            df['easing_cycle'] = (rate_cycle < np.roll(rate_cycle, 10)).astype(int)
        elif 'date' in df.columns:
            # Extract year if not already present
            df['year'] = pd.to_datetime(df['date']).dt.year
            
            # Number of business days in the year for each date
            business_days = np.minimum(252, np.maximum(230, 
                                                    (df['year'] - 2000) * 2 + 230))
            
            # Create synthetic interest rate cycles (simplified)
            rate_cycle = np.sin(2 * np.pi * np.arange(len(df)) / business_days / 5) * 2 + 3
            
            # Create synthetic yield curve slope
            yield_curve = np.sin(2 * np.pi * np.arange(len(df)) / business_days / 3)
            
            # Add to DataFrame
            df['interest_rate_proxy'] = rate_cycle
            df['yield_curve_slope_proxy'] = yield_curve
            
            # Create synthetic monetary policy cycle indicator
            df['tightening_cycle'] = (rate_cycle > np.roll(rate_cycle, 10)).astype(int)
            df['easing_cycle'] = (rate_cycle < np.roll(rate_cycle, 10)).astype(int)
        
        # Create synthetic market regime indicators
        if len(df) > 100:
            # Use the first numeric column for synthetic features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                first_col = numeric_cols[0]
                
                df['bull_market_proxy'] = np.where(
                    df[first_col].rolling(100).mean() > df[first_col].rolling(200).mean(), 
                    1, 0)
                
                # Create synthetic volatility regime
                vol = df[first_col].pct_change().rolling(20).std() * np.sqrt(252)
                df['high_vol_regime'] = np.where(vol > vol.rolling(100).mean(), 1, 0)
            
            # Create synthetic correlation regime (when correlations are high across assets)
            df['high_correlation_regime'] = np.sin(2 * np.pi * np.arange(len(df)) / 500)**2
            df['high_correlation_regime'] = np.where(df['high_correlation_regime'] > 0.7, 1, 0)



class FeatureSelector:
    """
    Implements feature selection methods with bias prevention.
    """
    def __init__(self, cv_folds: int = 5, 
               purge_overlap: int = 10,
               n_estimators: int = 100,
               random_state: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            cv_folds: Number of cross-validation folds
            purge_overlap: Number of samples to purge around test set
            n_estimators: Number of estimators for ensemble methods
            random_state: Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.purge_overlap = purge_overlap
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_importance = {}
        self.selected_features = {}
    
    def select_features(self, data: pd.DataFrame, 
                      target_col: str,
                      feature_names: List[str] = None,
                      methods: List[str] = None,
                      max_features: int = 50) -> Dict[str, List[str]]:
        """
        Select features using multiple methods with purging.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of the target column
            feature_names: List of feature column names to consider
            methods: List of feature selection methods to use
            max_features: Maximum number of features to select
            
        Returns:
            Dictionary of selected features by method
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # If no feature names provided, use all numeric columns
        if feature_names is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Exclude the target column
            if target_col in numeric_cols:
                feature_names = numeric_cols.difference([target_col]).tolist()
            else:
                feature_names = list(numeric_cols)
        
        # Default methods
        if methods is None:
            methods = ['rf_importance', 'mutual_info', 'lasso']
        
        # Filter out features with too many NaNs
        valid_features = []
        for feature in feature_names:
            if feature in df.columns:
                nan_ratio = df[feature].isna().mean()
                if nan_ratio < 0.3:  # Less than 30% NaNs
                    valid_features.append(feature)
        
        # Prepare data
        X = df[valid_features].fillna(0)  # Simple imputation for selection
        y = df[target_col]
        
        # Apply purging and feature selection
        for method in methods:
            selected = self._apply_method_with_purging(X, y, method, max_features)
            self.selected_features[method] = selected
        
        # Create ensemble selection using all methods
        all_selected = set()
        for method, features in self.selected_features.items():
            all_selected.update(features)
        
        # Rank features by frequency of selection and importance
        feature_scores = {}
        for feature in all_selected:
            # Count how many methods selected this feature
            count = sum(1 for method_features in self.selected_features.values() 
                       if feature in method_features)
            
            # Average the normalized importance across methods
            importance_sum = 0
            methods_with_importance = 0
            
            for method, importance_dict in self.feature_importance.items():
                if feature in importance_dict:
                    importance_sum += importance_dict[feature]
                    methods_with_importance += 1
            
            avg_importance = importance_sum / methods_with_importance if methods_with_importance > 0 else 0
            
            # Final score is a combination of selection frequency and importance
            feature_scores[feature] = 0.5 * (count / len(methods)) + 0.5 * avg_importance
        
        # Select top features based on score
        ensemble_features = sorted(feature_scores.keys(), 
                                 key=lambda x: feature_scores[x], 
                                 reverse=True)[:max_features]
        
        self.selected_features['ensemble'] = ensemble_features
        
        return self.selected_features
    
    def _apply_method_with_purging(self, X: pd.DataFrame, y: pd.Series, 
                                method: str, max_features: int) -> List[str]:
        """
        Apply a feature selection method with proper purging to prevent leakage.
        
        Args:
            X: Feature DataFrame
            y: Target series
            method: Selection method name
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        # Create importance dictionary for this method
        self.feature_importance[method] = {}
        
        if method == 'rf_importance':
            selected = self._select_with_random_forest(X, y, max_features)
        elif method == 'mutual_info':
            selected = self._select_with_mutual_info(X, y, max_features)
        elif method == 'lasso':
            selected = self._select_with_lasso(X, y, max_features)
        else:
            warnings.warn(f"Unknown method: {method}. Using random forest importance.")
            selected = self._select_with_random_forest(X, y, max_features)
        
        return selected
    
    def _create_purged_cv_folds(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds with purging to prevent leakage.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            List of train/test indices for each fold
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Create basic fold indices
        fold_size = n_samples // self.cv_folds
        folds = []
        
        for fold in range(self.cv_folds):
            # Test indices for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.cv_folds - 1 else n_samples
            test_indices = indices[test_start:test_end]
            
            # Purging: remove overlap from train set
            purge_before = max(0, test_start - self.purge_overlap)
            purge_after = min(n_samples, test_end + self.purge_overlap)
            
            # Train indices with purging
            train_indices = np.concatenate([
                indices[:purge_before],
                indices[purge_after:]
            ])
            
            folds.append((train_indices, test_indices))
        
        return folds
    
    def _select_with_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                                max_features: int) -> List[str]:
        """
        Select features using Random Forest importance with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        # Initialize importance accumulator
        feature_importances = np.zeros(X.shape[1])
        
        # Create CV folds with purging
        cv_folds = self._create_purged_cv_folds(X)
        
        # Run RF on each fold
        for train_idx, test_idx in cv_folds:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=self.n_estimators, 
                                     random_state=self.random_state,
                                     n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # Accumulate feature importance
            feature_importances += rf.feature_importances_
        
        # Average importance across folds
        feature_importances /= self.cv_folds
        
        # Create feature importance dictionary
        importance_sum = np.sum(feature_importances)
        if importance_sum > 0:  # Avoid division by zero
            normalized_importances = feature_importances / importance_sum
        else:
            normalized_importances = feature_importances
        
        for idx, feature in enumerate(X.columns):
            self.feature_importance['rf_importance'][feature] = normalized_importances[idx]
        
        # Select top features
        top_indices = np.argsort(feature_importances)[-max_features:]
        selected_features = [X.columns[i] for i in top_indices]
        
        return selected_features
    
    def _select_with_mutual_info(self, X: pd.DataFrame, y: pd.Series, 
                              max_features: int) -> List[str]:
        """
        Select features using Mutual Information with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        # Initialize importance accumulator
        mi_scores = np.zeros(X.shape[1])
        
        # Create CV folds with purging
        cv_folds = self._create_purged_cv_folds(X)
        
        # Calculate MI on each fold
        for train_idx, _ in cv_folds:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            
            # Calculate mutual information
            fold_mi = mutual_info_regression(X_train, y_train, random_state=self.random_state)
            
            # Accumulate scores
            mi_scores += fold_mi
        
        # Average scores across folds
        mi_scores /= self.cv_folds
        
        # Create feature importance dictionary
        importance_sum = np.sum(mi_scores)
        if importance_sum > 0:  # Avoid division by zero
            normalized_scores = mi_scores / importance_sum
        else:
            normalized_scores = mi_scores
        
        for idx, feature in enumerate(X.columns):
            self.feature_importance['mutual_info'][feature] = normalized_scores[idx]
        
        # Select top features
        top_indices = np.argsort(mi_scores)[-max_features:]
        selected_features = [X.columns[i] for i in top_indices]
        
        return selected_features
    
    def _select_with_lasso(self, X: pd.DataFrame, y: pd.Series, 
                        max_features: int) -> List[str]:
        """
        Select features using LASSO regularization with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target series
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        # Initialize importance accumulator
        coef_scores = np.zeros(X.shape[1])
        
        # Create CV folds with purging
        cv_folds = self._create_purged_cv_folds(X)
        
        # Run LASSO on each fold
        for train_idx, _ in cv_folds:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            
            # Scale features for LASSO
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Find optimal alpha using validation
            alphas = np.logspace(-5, 2, 20)
            best_alpha = 0.1  # Default
            best_score = -np.inf
            
            for alpha in alphas:
                lasso = Lasso(alpha=alpha, random_state=self.random_state)
                lasso.fit(X_train_scaled, y_train)
                
                # Score on training data (just for selection, not for actual model evaluation)
                # For actual model evaluation, we'd use a separate validation set
                score = -np.mean((y_train - lasso.predict(X_train_scaled)) ** 2)
                
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            
            # Train final LASSO with best alpha
            lasso = Lasso(alpha=best_alpha, random_state=self.random_state)
            lasso.fit(X_train_scaled, y_train)
            
            # Accumulate absolute coefficient values
            coef_scores += np.abs(lasso.coef_)
        
        # Average coefficients across folds
        coef_scores /= self.cv_folds
        
        # Create feature importance dictionary
        importance_sum = np.sum(coef_scores)
        if importance_sum > 0:  # Avoid division by zero
            normalized_scores = coef_scores / importance_sum
        else:
            normalized_scores = coef_scores
        
        for idx, feature in enumerate(X.columns):
            self.feature_importance['lasso'][feature] = normalized_scores[idx]
        
        # Select top features
        top_indices = np.argsort(coef_scores)[-max_features:]
        selected_features = [X.columns[i] for i in top_indices]
        
        return selected_features
    
    def get_feature_importance(self, method: str = None) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            method: Specific method to get importance for, or None for all
            
        Returns:
            DataFrame with feature importance scores
        """
        if method and method in self.feature_importance:
            # Return importance for specific method
            return pd.DataFrame({
                'feature': list(self.feature_importance[method].keys()),
                'importance': list(self.feature_importance[method].values())
            }).sort_values('importance', ascending=False)
        else:
            # Combine importance from all methods
            all_features = set()
            for method_importances in self.feature_importance.values():
                all_features.update(method_importances.keys())
            
            # Create DataFrame with all methods
            result = pd.DataFrame({'feature': list(all_features)})
            
            for method in self.feature_importance:
                result[f'{method}_importance'] = result['feature'].map(
                    lambda x: self.feature_importance[method].get(x, 0))
            
            # Add average importance
            importance_columns = [col for col in result.columns if '_importance' in col]
            result['avg_importance'] = result[importance_columns].mean(axis=1)
            
            # Sort by average importance
            result = result.sort_values('avg_importance', ascending=False)
            
            return result
    
    def calculate_shap_values(self, X: pd.DataFrame, y: pd.Series, 
                           method: str = 'rf', n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate SHAP values for feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target series
            method: Model type to use ('rf' or 'gbm')
            n_samples: Number of samples to use for SHAP calculation
            
        Returns:
            Dictionary of SHAP values by feature
        """
        try:
            # Limit number of samples for efficiency
            if len(X) > n_samples:
                # Stratified sampling to ensure diverse samples
                bins = pd.qcut(y, 5, duplicates='drop')
                X_sample = X.groupby(bins, observed=False).apply(
                    lambda x: x.sample(min(len(x), n_samples // 5))).reset_index(drop=True)
                y_sample = y.loc[X_sample.index]
            else:
                X_sample = X
                y_sample = y
            
            # Train model based on method
            if method == 'gbm':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            
            model.fit(X_sample, y_sample)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Create SHAP importance dictionary
            shap_importance = {}
            for i, col in enumerate(X_sample.columns):
                shap_importance[col] = np.abs(shap_values[:, i]).mean()
            
            # Normalize importances
            total_importance = sum(shap_importance.values())
            for col in shap_importance:
                shap_importance[col] /= total_importance
            
            return shap_importance
        
        except Exception as e:
            warnings.warn(f"SHAP calculation failed: {str(e)}")
            return {}
    
    def diversify_features(self, features: List[str], 
                         X: pd.DataFrame, 
                         max_features: int = 50,
                         correlation_threshold: float = 0.7) -> List[str]:
        """
        Diversify feature set by removing highly correlated features.
        
        Args:
            features: List of feature names
            X: Feature DataFrame
            max_features: Maximum number of features to keep
            correlation_threshold: Threshold for correlation
            
        Returns:
            List of diversified features
        """
        # If fewer features than max, return all
        if len(features) <= max_features:
            return features
        
        # Calculate correlation matrix
        X_features = X[features]
        corr_matrix = X_features.corr().abs()
        
        # Initialize list of features to keep
        keep_features = []
        
        # Greedily add features
        remaining_features = features.copy()
        
        while len(keep_features) < max_features and remaining_features:
            best_feature = remaining_features[0]
            
            # If we already have features, pick the one least correlated with kept features
            if keep_features:
                avg_corr = {}
                for feature in remaining_features:
                    avg_corr[feature] = np.mean([
                        corr_matrix.loc[feature, kf] for kf in keep_features
                    ])
                
                # Get feature with minimum average correlation
                best_feature = min(avg_corr.keys(), key=lambda x: avg_corr[x])
            
            # Add best feature to keep list
            keep_features.append(best_feature)
            remaining_features.remove(best_feature)
            
            # Remove highly correlated features
            remaining_features = [
                f for f in remaining_features 
                if not np.any(corr_matrix.loc[f, keep_features] >= correlation_threshold)
            ]
        
        return keep_features
    
    def track_feature_stability(self, X: pd.DataFrame, y: pd.Series, 
                             feature_names: List[str], 
                             method: str = 'rf_importance') -> pd.DataFrame:
        """
        Track feature importance stability across different data subsets.
        
        Args:
            X: Feature DataFrame
            y: Target series
            feature_names: List of features to assess
            method: Feature importance method
            
        Returns:
            DataFrame with stability metrics
        """
        # Initialize stability tracker
        stability = {feature: [] for feature in feature_names}
        
        # Create random subsets
        for i in range(5):
            # Random subsample (80% of data)
            sample_idx = np.random.choice(len(X), size=int(0.8 * len(X)), replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Calculate importance on this subset
            if method == 'shap':
                importance = self.calculate_shap_values(X_sample[feature_names], y_sample)
            elif method == 'mutual_info':
                mi_values = mutual_info_regression(X_sample[feature_names], y_sample,
                                                  random_state=self.random_state)
                importance = {feature: mi for feature, mi in zip(feature_names, mi_values)}
            else:
                # Use Random Forest importance
                rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                rf.fit(X_sample[feature_names], y_sample)
                importance = {feature: imp for feature, imp in 
                              zip(feature_names, rf.feature_importances_)}
            
            # Normalize importance
            total = sum(importance.values())
            if total > 0:
                importance = {f: v/total for f, v in importance.items()}
            
            # Track importance for each feature
            for feature in feature_names:
                stability[feature].append(importance.get(feature, 0))
        
        # Calculate stability metrics
        result = []
        for feature in feature_names:
            values = stability[feature]
            
            result.append({
                'feature': feature,
                'mean_importance': np.mean(values),
                'std_importance': np.std(values),
                'cv_importance': np.std(values) / np.mean(values) if np.mean(values) > 0 else float('inf'),
                'min_importance': np.min(values),
                'max_importance': np.max(values)
            })
        
        # Convert to DataFrame and sort by mean importance
        return pd.DataFrame(result).sort_values('mean_importance', ascending=False)
